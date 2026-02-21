import einops
import torch

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


def convert_olmoe_weights(olmoe, cfg: HookedTransformerConfig):
    # As with Mixtral, OlMoE do NOT have bias
    state_dict = {}

    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    state_dict["embed.W_E"] = olmoe.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        # LayerNorms
        state_dict[f"blocks.{l}.ln1.w"] = olmoe.model.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln2.w"] = olmoe.model.layers[l].post_attention_layernorm.weight

        # Attention
        W_Q = olmoe.model.layers[l].self_attn.q_proj.weight
        W_K = olmoe.model.layers[l].self_attn.k_proj.weight
        W_V = olmoe.model.layers[l].self_attn.v_proj.weight
        W_O = olmoe.model.layers[l].self_attn.o_proj.weight

        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        # MoE
        state_dict[f"blocks.{l}.mlp.W_gate.weight"] = olmoe.model.layers[l].mlp.gate.weight

        # The mapping here from wn to W_{in/out/gate} is the same as Mixtral
        for e in range(cfg.num_experts):
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = (
                olmoe.model.layers[l].mlp.experts[e].up_proj.weight
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = (
                olmoe.model.layers[l].mlp.experts[e].gate_proj.weight
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = (
                olmoe.model.layers[l].mlp.experts[e].down_proj.weight
            )

    state_dict["ln_final.w"] = olmoe.model.norm.weight

    state_dict["unembed.W_U"] = olmoe.lm_head.weight.T

    return state_dict

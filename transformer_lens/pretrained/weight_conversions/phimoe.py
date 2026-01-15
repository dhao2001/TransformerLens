import einops
import torch

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


def convert_phimoe_weights(phi, cfg: HookedTransformerConfig):
    # As with Mixtral, Phi-MOE has no biases in many places.
    # However, it DOES have biases for attention projections and layer norms.
    state_dict = {}

    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    state_dict["embed.W_E"] = phi.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        # LayerNorms
        state_dict[f"blocks.{l}.ln1.w"] = phi.model.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = phi.model.layers[l].input_layernorm.bias
        state_dict[f"blocks.{l}.ln2.w"] = phi.model.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = phi.model.layers[l].post_attention_layernorm.bias

        # Attention
        W_Q = phi.model.layers[l].self_attn.q_proj.weight
        W_K = phi.model.layers[l].self_attn.k_proj.weight
        W_V = phi.model.layers[l].self_attn.v_proj.weight
        W_O = phi.model.layers[l].self_attn.o_proj.weight

        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            phi.model.layers[l].self_attn.q_proj.bias, "(n h)->n h", n=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn._b_K"] = einops.rearrange(
            phi.model.layers[l].self_attn.k_proj.bias, "(n h)->n h", n=cfg.n_key_value_heads
        )
        state_dict[f"blocks.{l}.attn._b_V"] = einops.rearrange(
            phi.model.layers[l].self_attn.v_proj.bias, "(n h)->n h", n=cfg.n_key_value_heads
        )
        state_dict[f"blocks.{l}.attn.b_O"] = phi.model.layers[l].self_attn.o_proj.bias

        # MoE
        state_dict[f"blocks.{l}.mlp.W_gate.weight"] = phi.model.layers[
            l
        ].block_sparse_moe.gate.weight

        # The mapping here from wn to W_{in/out/gate} is the same as Mixtral
        for e in range(cfg.num_experts):
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = (
                phi.model.layers[l].block_sparse_moe.experts[e].w3.weight
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = (
                phi.model.layers[l].block_sparse_moe.experts[e].w1.weight
            )
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = (
                phi.model.layers[l].block_sparse_moe.experts[e].w2.weight
            )

    state_dict["ln_final.w"] = phi.model.norm.weight
    state_dict["ln_final.b"] = phi.model.norm.bias

    state_dict["unembed.W_U"] = phi.lm_head.weight.T
    state_dict["unembed.b_U"] = phi.lm_head.bias

    return state_dict

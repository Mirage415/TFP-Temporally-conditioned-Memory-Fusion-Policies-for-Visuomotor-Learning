# TFP AdaLN Layer Architecture

This figure shows the model architecture, not the training flow. It focuses on where the memory-derived AdaLN condition
enters the suffix transformer stack in the current `tfp_adaln*` mainline.

Source: [docs/tfp_adaln_layer_architecture.mmd](../docs/tfp_adaln_layer_architecture.mmd)

```mermaid
flowchart TB
    subgraph Left["Temporal Memory Path"]
        O["Current chunk observation"] --> VE["SigLIP visual encoder"]
        O --> SP["state_proj"]
        VE --> VT["visual tokens"]
        SP --> ST["state token"]
        VT --> CAT["concat"]
        ST --> CAT
        Hprev["previous hidden h_{t-1}"] --> LTC["LTCEncoder.step"]
        DT["delta_t"] --> LTC
        CAT --> LTC
        LTC --> HT["updated hidden h_t"]
        HT --> MP["memory_adaln_proj"]
        MP --> MC["memory condition m_t"]
    end

    subgraph Top["Diffusion Time Path"]
        Tau["diffusion timestep t"] --> PE["sine-cosine pos emb"]
        PE --> TM1["time_mlp_in + swish"]
        TM1 --> TM2["time_mlp_out + swish"]
        TM2 --> TC["time condition z_t"]
        MC --> SUM["suffix_cond = z_t + m_t"]
        TC --> SUM
    end

    subgraph Main["pi0.5 Action Decoder Architecture"]
        Prefix["Prefix stream\nimage tokens + prompt tokens"] --> StackIn["Suffix transformer stack"]
        Suffix["Suffix stream\nnoisy action chunk tokens"] --> StackIn

        subgraph Block["Repeated suffix layer l = 1 ... L"]
            PA["pre_attention_norm\nAdaLN(cond=suffix_cond)"] --> ATTN["self-attention"]
            ATTN --> RES1["residual add"]
            RES1 --> PC["pre_cross_attention_norm\nAdaLN(cond=suffix_cond)"]
            PC --> XATTN["memory cross-attn branch\ninactive in tfp_adaln"]
            XATTN --> RES2["residual add"]
            RES2 --> PF["pre_ffw_norm\nAdaLN(cond=suffix_cond)"]
            PF --> MLP["feed-forward / MLP"]
            MLP --> RES3["residual add"]
        end

        StackIn --> Block
        Block --> StackOut["suffix hidden states"]
        StackOut --> OUT["action_out_proj"]
        OUT --> ACT["predicted action chunk"]
    end

    SUM --> PA
    SUM --> PC
    SUM --> PF

    Note["Current mainline tfp_adaln uses AdaLN-style conditioning in every suffix layer.\nThere are no explicit memory tokens in this model.\nThe cross-attention branch exists in the generic block implementation but is inactive here because memory_tokens=None and memory_cross_attn_layers=0."] --- Block
```

Code references:

- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:322)
- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:330)
- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:376)
- [src/openpi/models/gemma.py](../src/openpi/models/gemma.py:373)
- [src/openpi/models/gemma.py](../src/openpi/models/gemma.py:387)
- [src/openpi/models/gemma.py](../src/openpi/models/gemma.py:409)

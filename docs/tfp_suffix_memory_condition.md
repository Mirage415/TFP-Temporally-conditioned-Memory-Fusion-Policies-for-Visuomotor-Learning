# TFP Suffix Memory Condition

This diagram isolates the current `Pi0AdaLN` memory-conditioning path used by the `tfp_adaln*` configs.

Source: [docs/tfp_suffix_memory_condition.mmd](../docs/tfp_suffix_memory_condition.mmd)

```mermaid
flowchart LR
    subgraph Memory["Temporal Memory Branch"]
        A["Current hidden state h_t\nfrom LTCEncoder"] --> B["memory_adaln_proj"]
        B --> C["memory_cond\n[B, action_expert_width]"]
        D["chunk mask"] --> E["mask memory_cond"]
        C --> E
    end

    subgraph Diffusion["Diffusion Suffix Branch"]
        F["Target / sampled action chunk"] --> G["Add diffusion noise\nx_t"]
        H["timestep t"] --> I["posemb_sincos"]
        I --> J["time_mlp_in"]
        J --> K["swish"]
        K --> L["time_mlp_out"]
        L --> M["swish"]
        M --> N["adarms_cond"]
        G --> O["action_in_proj"]
        O --> P["suffix tokens"]
    end

    subgraph Merge["Condition Merge"]
        E --> Q["_merge_suffix_cond"]
        N --> Q
        Q --> R["suffix_cond\n= adarms_cond + memory_cond"]
    end

    subgraph Transformer["pi05 Action Expert"]
        S["prefix tokens\nimage + prompt"] --> T["PaliGemma.llm"]
        P --> T
        R --> T
        T --> U["suffix_out"]
        U --> V["action_out_proj"]
        V --> W["predicted action chunk\n[B, action_horizon, action_dim]"]
    end

    subgraph Train["Training"]
        W --> X["flow matching loss"]
        Y["target_actions"] --> X
    end

    subgraph Infer["Inference"]
        Z["cached prefix kv"] --> T
        AA["iterative denoising\n10 steps"] --> P
        W --> AB["final action chunk"]
    end
```

Relevant code:

- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:322)
- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:354)
- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:386)

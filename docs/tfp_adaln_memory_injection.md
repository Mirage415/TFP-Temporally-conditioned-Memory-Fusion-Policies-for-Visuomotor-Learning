# TFP AdaLN Memory Injection

This figure matches the current `tfp_adaln*` mainline model.

Unlike the older `tfp` variant, this version does **not** build explicit memory tokens for suffix cross-attention.
Instead, it projects the recurrent hidden state into a conditioning vector and injects it through the suffix decoder's
adaptive normalization pathway.

Source: [docs/tfp_adaln_memory_injection.mmd](../docs/tfp_adaln_memory_injection.mmd)

```mermaid
flowchart LR
    subgraph Temporal["Temporal Memory Update"]
        A["Current observation\nimages + state"] --> B["SigLIP visual encoding"]
        A --> C["state_proj"]
        B --> D["visual tokens"]
        C --> E["state token"]
        D --> F["concat for LTC input"]
        E --> F
        G["previous hidden\nh_{t-1}"] --> H["LTCEncoder.step"]
        I["delta_t"] --> H
        F --> H
        H --> J["updated hidden\nh_t"]
    end

    subgraph MemoryCond["Project Hidden to AdaLN Memory Condition"]
        J --> K["memory_adaln_proj"]
        K --> L["memory_cond"]
        M["chunk mask"] --> N["apply mask"]
        L --> N
        N --> O["masked memory_cond"]
    end

    subgraph TimeCond["Build Diffusion Time Condition"]
        P["diffusion timestep\nt"] --> Q["posemb_sincos"]
        Q --> R["time_mlp_in"]
        R --> S["swish"]
        S --> T["time_mlp_out"]
        T --> U["swish"]
        U --> V["adarms_cond"]
    end

    subgraph Merge["Fuse Conditions"]
        O --> W["suffix_cond\n= adarms_cond + memory_cond"]
        V --> W
    end

    subgraph Suffix["Action Suffix Decoder"]
        X["prefix tokens\nimage + prompt"] --> Y["PaliGemma suffix decoder"]
        Z["suffix tokens\nnoisy action chunk"] --> Y
        W --> Y
        Y --> AA["AdaLN / adaRMS modulation\ninside suffix layers"]
        AA --> AB["suffix_out"]
        AB --> AC["action_out_proj"]
        AC --> AD["predicted action chunk"]
    end

    AE["Note\nCurrent mainline does not create memory tokens.\nIt injects a single memory-derived conditioning vector\nthrough AdaLN-style modulation."] --- W
```

Relevant code:

- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:322)
- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:330)
- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:354)
- [src/openpi/models/gemma.py](../src/openpi/models/gemma.py:373)

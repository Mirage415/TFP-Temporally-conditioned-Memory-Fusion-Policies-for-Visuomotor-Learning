# TFP Pro Pipeline

This diagram reflects the current `tfp_pro` mainline temporal-memory model, which is the `tfp_adaln_curriculum`
family in [src/openpi/training/config.py](../src/openpi/training/config.py:765).

The Mermaid source lives in [docs/tfp_pro_pipeline.mmd](../docs/tfp_pro_pipeline.mmd).

```mermaid
flowchart TD
    subgraph Data["Data Pipeline"]
        A["LeRobot LIBERO episodes"] --> B["TemporalMemoryDataset"]
        B --> C["Window builder\nTBPTT windows"]
        C --> D["Per sample\n6 chunks x stride 6"]
        D --> E["Chunk fields\nimages, state, delta_t,\ntarget_actions[10], target_mask"]
    end

    subgraph Train["Training Loop"]
        E --> F["Load initial hidden\nfrom TemporalHiddenBank"]
        F --> G["jax.lax.scan over chunks"]
        G --> H["Chunk observation\npreprocess + tokenization"]
        H --> I["SigLIP image encoder"]
        H --> J["State projection"]
        I --> K["LTCEncoder.step"]
        J --> K
        K --> L["Hidden state h_t"]
        L --> M["memory_adaln_proj"]
        M --> N["memory condition"]
        H --> O["PaliGemma prefix\nimage tokens + prompt tokens"]
        H --> P["Diffusion suffix\nnoisy action chunk + time embedding"]
        O --> Q["pi05 action expert"]
        P --> Q
        N --> Q
        Q --> R["Predicted action chunk\n10 x 32"]
        E --> S["Supervision target\nfuture actions"]
        R --> T["Flow matching loss"]
        S --> T
        L --> U["Final hidden for window"]
        U --> V["Store back to\nTemporalHiddenBank"]
    end

    subgraph Infer["Stateful Inference"]
        W["Current observation"] --> X["Input transforms"]
        X --> Y["Compute delta_t from timestamp"]
        Y --> Z["LTCEncoder.step\nwith previous hidden"]
        Z --> AA["memory_adaln_proj"]
        AA --> AB["memory condition"]
        X --> AC["PaliGemma prefix"]
        AC --> AD["Diffusion sampler\n10 denoising steps"]
        AB --> AD
        AD --> AE["Output action chunk"]
        Z --> AF["Persist hidden state\ninside Policy"]
    end
```

## Render

If you have Mermaid CLI available elsewhere, render it with:

```bash
mmdc -i docs/tfp_pro_pipeline.mmd -o docs/tfp_pro_pipeline.svg
```

Relevant code paths:

- [src/openpi/training/temporal_memory_loader.py](../src/openpi/training/temporal_memory_loader.py:146)
- [src/openpi/models/temporal_memory.py](../src/openpi/models/temporal_memory.py:8)
- [src/openpi/models/pi0_adaln.py](../src/openpi/models/pi0_adaln.py:72)
- [scripts/train.py](../scripts/train.py:475)
- [src/openpi/policies/policy.py](../src/openpi/policies/policy.py:139)

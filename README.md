# Exact GLM-5.3-Flash-NVFP4 SGLang bundle for 4x RTX PRO 6000 Blackwell (SM120)

Locked, independently buildable representation of the live Pop!_OS deployment (`glm-5.3-flash-sglang-sm120`). Uses the exact `lmsysorg/sglang:glm-5.3-flash` image digest and preserves the effective server arguments and environment from the validated running container.

The 182 GiB checkpoint is intentionally **not** copied into the image. Compose mounts the verified local checkpoint read-only from `/mnt/llm_models/GLM-5.3-Flash-NVFP4` and reuses the writable SGLang cache at `/home/ser/.cache/sglang-glm53`. The six SM120 compatibility patches are **baked into the image** at their editable-install import paths, so no bind-mounted code is needed.

## Lock points

- Runtime image: `lmsysorg/sglang:glm-5.3-flash@sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290`
- Model: `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `9e0d74e3cef17f634e84fb8e2223707e02616290` (120 shards)
- Server model ID: `glm-5.3-flash` (`owned_by: sglang`)
- Context: 1,048,576 tokens; KV cache `fp8_e4m3`, 3,789,184 tokens per rank (~28 GB)
- TP/EP: 4 / 4
- MoE runner: **`flashinfer_cutlass`** — SM120-native CUTLASS fp4 MoE
- Attention: DSA backend with `flashinfer_sparse_mla` prefill/decode; linear attention via `triton` (KDA)
- Speculative decode: NEXTN (MTP), 5 steps, topk 1, 6 draft tokens
- Parsers: reasoning `glm45`, tool-call `glm47` — exactly what the model card prescribes
- CUDA graphs: full decode capture for batch sizes ≤ 8

## Why not flashinfer_cutedsl

FlashInfer's CuteDSL MoE stack hard-codes `supported_major_versions=[10]` (datacenter Blackwell sm100/sm103 tcgen05 kernels) in `flashinfer/jit/moe_utils.py::gen_moe_utils_module`. On consumer/pro workstation Blackwell (SM120, major version 12) startup dies during autotuning with:

```
RuntimeError: No supported CUDA architectures found for major versions [10].
```

No environment variable can fix this — the caller excludes major 12 by name. `flashinfer_cutlass` ships a dedicated `gen_cutlass_fused_moe_sm120_module` in this image and is validated here.

## Build and run

```bash
docker compose build --pull=false
docker compose up -d
curl http://127.0.0.1:8000/health            # container listens on 30000
```

Startup on this engine takes several minutes (182 GiB load + JIT + graph capture). `GET /health` returns 200 when ready; generation tests then show ~143 tok/s at low reasoning effort and ~208 tok/s at default effort on a single stream (NEXTN drafting).

## Client usage

```python
openai.chat.completions.create(
    model="glm-5.3-flash",
    messages=[{"role": "user", "content": "..."}],
    extra_body={
        "chat_template_kwargs": {
            "reasoning_effort": "low",   # 'low' | 'high'; unset => 'max'
            # "clear_thinking": True,
        }
    },
)
```

- Do **not** pass small `max_tokens`: under default (max) thinking effort the visible `content` field stays empty while the budget burns inside `<think>`. Omit it and let the model stop naturally.
- Sampling per the checkpoint's `generation_config.json`: temperature 1.0, top_p 0.95.
- Tool calls work by sending `tools` + `tool_choice` in the request; sglang needs no enable flag for this.

## Provenance

`manifest.json` records the base digest, model revision, hashes of the four core checkpoint metadata files, and sha256 of every baked patch. It is a configuration lock, not a replacement for validating all 120 model shards after transfer.

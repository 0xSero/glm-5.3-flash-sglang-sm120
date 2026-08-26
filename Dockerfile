# Exact GLM-5.3-Flash-NVFP4 runtime used on the Pop!_OS 4x RTX PRO 6000 Blackwell (SM120) host.
# The 182 GiB checkpoint is deliberately mounted by Compose instead of baked into this image.
# The six SM120 compatibility patches are baked into the image so the bundle has no bind-mounted code.
FROM lmsysorg/sglang:glm-5.3-flash@sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290

LABEL org.opencontainers.image.title="GLM-5.3-Flash NVFP4 SM120 SGLang runtime" \
      org.opencontainers.image.base.digest="sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290" \
      org.opencontainers.image.model="LibertAIDAI/GLM-5.3-Flash-NVFP4" \
      org.opencontainers.image.model.revision="9e0d74e3cef17f634e84fb8e2223707e02616290" \
      org.opencontainers.image.recipe.context="1048576" \
      org.opencontainers.image.recipe.model_id="glm-5.3-flash"

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_CUDA_ARCH_LIST=12.0a \
    FLASHINFER_CUDA_ARCH_LIST=12.0f \
    SGLANG_OPT_USE_TOPK_V2=0 \
    SGLANG_OPT_USE_TILELANG_INDEXER=1 \
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
    SGLANG_OPT_FP8_WO_A_GEMM=0

# SM120 compatibility patches, written over the editable-install sources at the same
# paths run-sglang.sh bind-mounts, so image behavior matches the live deployment.
COPY patches/sglang-glm5_next-debug.py /sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py
COPY patches/sglang-deepseek_nextn-glm53.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py
COPY patches/sglang-quant-utils-sm120.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/utils.py
COPY patches/sglang-modelopt-quant-sm120.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py
COPY patches/sglang-flash_mla_sm120-glm53.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
COPY patches/sglang-dsa_backend-glm53.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py

WORKDIR /sgl-workspace

ENTRYPOINT ["python3"]

# Validated 2026-08-26: flashinfer_cutlass MoE runner (SM120-native CUTLASS fp4 kernels).
# flashinfer_cutedsl CANNOT be substituted here: FlashInfer's CuteDSL MoE hard-codes
# supported_major_versions=[10] (datacenter Blackwell tcgen05), these cards are major 12.
CMD ["-m", "sglang.launch_server", \
     "--model-path", "/model", \
     "--served-model-name", "glm-5.3-flash", \
     "--tp-size", "4", \
     "--ep-size", "4", \
     "--context-length", "1048576", \
     "--quantization", "modelopt_fp4", \
     "--attention-backend", "dsa", \
     "--dsa-prefill-backend", "flashinfer_sparse_mla", \
     "--dsa-decode-backend", "flashinfer_sparse_mla", \
     "--linear-attn-backend", "triton", \
     "--kv-cache-dtype", "fp8_e4m3", \
     "--moe-runner-backend", "flashinfer_cutlass", \
     "--disable-shared-experts-fusion", \
     "--chunked-prefill-size", "8192", \
     "--max-prefill-tokens", "8192", \
     "--max-running-requests", "8", \
     "--mem-fraction-static", "0.90", \
     "--cuda-graph-max-bs-decode", "8", \
     "--speculative-algorithm", "NEXTN", \
     "--speculative-num-steps", "5", \
     "--speculative-eagle-topk", "1", \
     "--speculative-num-draft-tokens", "6", \
     "--speculative-adaptive", \
     "--media-url-max-file-size-mb", "1024", \
     "--reasoning-parser", "glm45", \
     "--tool-call-parser", "glm47", \
     "--host", "0.0.0.0", \
     "--port", "30000"]

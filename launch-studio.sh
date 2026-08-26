#!/usr/bin/env bash
# Studio-managed launcher for GLM-5.3-Flash NVFP4 on 4x RTX PRO 6000 Blackwell (SM120).
# Runs the exact-docker image in FOREGROUND so Local Studio supervises this process.
set -eu

docker rm -f glm-5.3-flash-sglang-sm120 >/dev/null 2>&1 || true

exec docker run --rm > /home/ser/glm53-flash-deploy/engine.log 2>&1 \
  --name glm-5.3-flash-sglang-sm120 \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  --security-opt label=disable \
  -p 8000:30000 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NVIDIA_VISIBLE_DEVICES=GPU-7b5db8b3-3a49-c0a7-b8e4-80dc1bd3e853,GPU-3cece4bc-432e-705e-7324-3f441d9cb4cc,GPU-fa982c97-64af-db6a-2ffb-08380e1f9375,GPU-c6ac75f2-cadf-6ff3-4cab-76c6033c1006 \
  -v /mnt/llm_models/GLM-5.3-Flash-NVFP4:/model:ro \
  -v /home/ser/.cache/sglang-glm53:/root/.cache:rw \
  -v /home/ser/glm53-flash-deploy/chat-template-mm.jinja:/chat-template.jinja:ro \
  glm53-flash-nvfp4-sglang:exact-20260826 \
  -m sglang.launch_server \
  --model-path /model \
  --served-model-name glm-5.3-flash \
  --tp-size 4 \
  --ep-size 4 \
  --context-length 1048576 \
  --quantization modelopt_fp4 \
  --attention-backend dsa \
  --dsa-prefill-backend flashinfer_sparse_mla \
  --dsa-decode-backend flashinfer_sparse_mla \
  --linear-attn-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-running-requests 8 \
  --mem-fraction-static 0.93 \
  --cuda-graph-max-bs-decode 8 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --speculative-adaptive \
  --media-url-max-file-size-mb 1024 \
  --enable-multimodal --chat-template /chat-template.jinja --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port 30000

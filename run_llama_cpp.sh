#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run llama.cpp server (GPU) — pass any HuggingFace model or local GGUF.
#
# Usage:
#   ./run_llama_cpp.sh ggml-org/gemma-3-1b-it-GGUF:Q8_0
#   ./run_llama_cpp.sh Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q4_K_M
#   ./run_llama_cpp.sh bartowski/Phi-4-mini-instruct-GGUF:Q4_K_M
#   ./run_llama_cpp.sh ./models/gemma-3-1b-it-Q8_0.gguf
#
# Options (env vars):
#   PORT=8090 ./run_llama_cpp.sh ...          # custom port (default: 8080)
#   CTX=4096 ./run_llama_cpp.sh ...           # custom context size (default: 4096)
#   NP=1 ./run_llama_cpp.sh ...              # parallel slots (default: 1, use 1 for VLM)
#   NAME=my-llm ./run_llama_cpp.sh ...        # custom container name
#   EMBED=1 ./run_llama_cpp.sh ...            # run as embedding server
#
# Stop:
#   docker stop assistant-llm

set -e

MODEL="${1:?Usage: $0 <user/repo:quant or path/to/model.gguf>}"
PORT="${PORT:-8080}"
CTX="${CTX:-4096}"
NP="${NP:-1}"
IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:b8095-r36.4-tegra-aarch64-cu126-22.04"

if [ "${EMBED:-0}" = "1" ]; then
    NAME="${NAME:-assistant-embed}"
    EXTRA_ARGS="--embeddings"
else
    NAME="${NAME:-assistant-llm}"
    EXTRA_ARGS=""
fi

# Stop existing container with same name
if [ "$(docker ps -aq -f name=^${NAME}$)" ]; then
    echo "Stopping existing $NAME..."
    docker stop "$NAME" > /dev/null 2>&1 || true
    docker rm "$NAME" > /dev/null 2>&1 || true
fi

# Detect local file vs HuggingFace repo
if [ -f "$MODEL" ]; then
    MODEL_DIR="$(cd "$(dirname "$MODEL")" && pwd)"
    MODEL_BASE="$(basename "$MODEL")"
    echo "Model : $MODEL (local)"
    echo "Port  : $PORT"
    echo ""
    docker run -d \
        --name "$NAME" \
        --runtime=nvidia \
        -p "${PORT}:8080" \
        -v "$MODEL_DIR:/models:ro" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        "$IMAGE" \
        llama-server \
        -m "/models/$MODEL_BASE" \
        --host 0.0.0.0 --port 8080 \
        -ngl 999 -c "$CTX" -np "$NP" -fa on --cache-reuse 256 $EXTRA_ARGS
else
    HF_CACHE="$HOME/.cache/huggingface"
    LLAMA_CACHE="$HOME/.cache/llama.cpp"
    mkdir -p "$HF_CACHE" "$LLAMA_CACHE"
    echo "Model : $MODEL (HuggingFace)"
    echo "Port  : $PORT"
    echo ""
    docker run -d \
        --name "$NAME" \
        --runtime=nvidia \
        -p "${PORT}:8080" \
        -v "$HF_CACHE:/root/.cache/huggingface" \
        -v "$LLAMA_CACHE:/root/.cache/llama.cpp" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        "$IMAGE" \
        llama-server \
        -hf "$MODEL" \
        --host 0.0.0.0 --port 8080 \
        -ngl 999 -c "$CTX" -np "$NP" -fa on --cache-reuse 256 $EXTRA_ARGS
fi

echo "✓ Container '$NAME' started."
echo ""
echo "  API  : http://localhost:${PORT}/v1/chat/completions"
echo "  Logs : docker logs -f $NAME"
echo "  Stop : docker stop $NAME"

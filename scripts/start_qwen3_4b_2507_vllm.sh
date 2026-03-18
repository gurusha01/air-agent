#!/bin/bash
pkill -f "vllm" 2>/dev/null || true
sleep 3

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/jarnav/hf_cache

echo "Starting vLLM with Qwen3-4B-Instruct-2507..."
nohup /home/jarnav/MLScientist/air-agent/.venv/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  > /tmp/vllm_2507.log 2>&1 &
echo "vLLM PID=$!"

for i in $(seq 1 24); do
  if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM ready after ${i}0s"
    curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; d=json.load(sys.stdin); print("Model:", d["data"][0]["id"])'
    break
  fi
  echo "  waiting... ($i/24)"
  sleep 10
done

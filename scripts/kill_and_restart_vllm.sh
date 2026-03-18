#!/bin/bash
# Kill all vLLM processes
for p in $(pgrep -f vllm); do
    kill -9 "$p" 2>/dev/null
done
sleep 5

# Check GPU memory is free
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/scratch/jarnav/hf_cache

echo "Starting vLLM with Qwen3-4B-Instruct-2507 (reduced seqs)..."
nohup /home/jarnav/MLScientist/air-agent/.venv/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  > /tmp/vllm_2507.log 2>&1 &
echo "vLLM started, PID=$!"

for i in $(seq 1 24); do
  if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM ready after ${i}x10s"
    curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; d=json.load(sys.stdin); print("Model:", d["data"][0]["id"])'
    break
  fi
  echo "  waiting... ($i/24)"
  sleep 10
done

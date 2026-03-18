#!/bin/bash
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Starting vLLM with max-model-len 32768..."

CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/jarnav/hf_cache \
  nohup /home/jarnav/MLScientist/air-agent/.venv/bin/python3 \
  -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 --trust-remote-code \
  >> /tmp/vllm_feb28.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID=$VLLM_PID"

# Wait for ready (up to 5 minutes)
for i in $(seq 1 30); do
    sleep 10
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM ready after ${i}x10s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM died. Last log:"
        tail -30 /tmp/vllm_feb28.log
        exit 1
    fi
    echo "  waiting... ($i/30)"
done

# Keep this step alive so Slurm cgroup doesn't kill the vLLM process
echo "Keeping srun step alive — waiting for vLLM (PID=$VLLM_PID) to exit..."
wait $VLLM_PID
echo "vLLM exited."

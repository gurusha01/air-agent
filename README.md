# air-agent

[![Release](https://img.shields.io/github/v/release/gurusha01/air-agent)](https://img.shields.io/github/v/release/gurusha01/air-agent)
[![Build status](https://img.shields.io/github/actions/workflow/status/gurusha01/air-agent/main.yml?branch=main)](https://github.com/gurusha01/air-agent/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/gurusha01/air-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/gurusha01/air-agent)
[![Commit activity](https://img.shields.io/github/commit-activity/m/gurusha01/air-agent)](https://img.shields.io/github/commit-activity/m/gurusha01/air-agent)
[![License](https://img.shields.io/github/license/gurusha01/air-agent)](https://img.shields.io/github/license/gurusha01/air-agent)

You need two repos for it 1. MLGym 2. VeRL. first clone them, they are already added in pyproject.toml so no need to install them as such. The main code is in `air` directory

## Setup

Clone MLGym and prime-rl repos:
```sh
git clone https://github.com/facebookresearch/MLGym ../MLGym &
git clone https://github.com/PrimeIntellect-ai/prime-rl ../prime-rl
```

The project structure should look like this
```
root/
├── air-agent/    (this repo)
├── MLGym/        (clone from https://github.com/facebookresearch/MLGym)
└── prime-rl/     (clone from https://github.com/PrimeIntellect-ai/prime-rl)
```

Install dependencies
```sh 
uv sync
```

Pull the MLGym docker image
```sh
docker pull aigym/mlgym-agent:latest
```

Copy over the data folder from MLGym repo
```sh 
mkdir data && cp -r ./../MLGym/data/* ./data
```


Setup .env for this repo and don't forget to edit the .env file to include the api key.
```sh
echo -e "MLGYM_CONFIG_ROOT=\"./../MLGym/configs\"\nMLGYM_WORKSPACE_PATH=\"./../MLGym/workspace\"\n\nOPENAI_API_KEY=\"<api-key>\"" > .env
```
## Testing
Run both commands on seperate shell to test the orchestrator directly without the full prime-rl distributed setup.
```sh
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 --trust-remote-code
```
Wait for the vLLM server finish booting and run:
```
uv run python -m air.prime_orchestrator \
    --task battleOfSexes \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --num-steps 10 \
    --batch-size 16 \
    --output-dir outputs/standalone_run
```

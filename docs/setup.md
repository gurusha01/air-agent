# Setup

Clone MLGym and prime-rl repos:
```sh
cd ..
git clone https://github.com/facebookresearch/MLGym
git clone https://github.com/PrimeIntellect-ai/prime-rl
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
mkdir data
cp -r ./../MLGym/data/* ./data
```
```
```


Setup .env
```sh
echo "MLGYM_CONFIG_ROOT=\"./../MLGym/configs\"\nMLGYM_WORKSPACE_PATH=\"./../MLGym/workspace\"\n\nOPENAI_API_KEY=\"<api-key>\"" > .env
```

Don't forget to edit the .env file to include the api key.

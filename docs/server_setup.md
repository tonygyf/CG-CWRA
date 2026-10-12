# Server setup

Clone the repository, then create the Python environment:

```bash
git clone <repository-url> CG-CWRA
cd CG-CWRA
bash scripts/bootstrap_server.sh
```

Before running a GPU job, install the PyTorch wheel compatible with the server's
CUDA driver. The bootstrap script intentionally does not download any dataset
or model checkpoint.

| Asset | Location |
| --- | --- |
| Raw / external datasets | `data/raw/`, `data/external/` |
| Preprocessed samples | `data/processed/` |
| Base checkpoints | `artifacts/pretrained/` |
| Hugging Face cache | `artifacts/hf_cache/` |
| Run checkpoints and results | `artifacts/checkpoints/`, `outputs/` |

The first implemented command is a configuration and parameter-count smoke test:

```bash
bash scripts/train.sh configs/train/example.yaml --dry-run
```

Generate the synthetic smoke dataset and run the standardized experiment suite:

```bash
python -m cgcwra.train.cli configs/train/research_smoke.yaml --prepare-smoke-data
bash scripts/train.sh configs/train/research_smoke.yaml
```

Write an environment report on the host or inside Docker:

```bash
bash scripts/verify_environment.sh
```

To train against TextMonkey or Mini-Monkey, add a model-specific integration
that obtains frozen ViT patch features, calls `CGCWRAAdapter`, and forwards the
returned tensor to that project's original connector. Do not modify frozen
backbone modules inside this repository.

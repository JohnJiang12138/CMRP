# Quick Start
## Package Dependencies

- Python 3.10.13
- PyTorch 2.0.1

for detailed information, please see ``requirements.txt``

## Preparations

Embedding model T5-Small is used.

```
git lfs install
git clone https://huggingface.co/google-t5/t5-small
```

## Preprocessing

Here we preprocess the datasets, prepare embeddings and calculate similarities.

```
cd ./preprocess
sh preprocess_WN18.sh
```

## Start Training

```
cd ../REINFORCE
sh run_kg_WN18.sh
```

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JohnJiang12138/CMRP">
    <img height="150" src="Logo.jpg?sanitize=true" />
  </a>
</div>

<h3 align="center">⭐️CMRP: Cross-modal Reinforced Prompting for Graph and Language Tasks⭐️</h3>


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

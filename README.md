Here's a revised version of your content with a more consistent style:

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JohnJiang12138/CMRP">
    <img height="150" src="Logo.jpg?sanitize=true" />
  </a>
</div>

<h3 align="center">⭐️CMRP: Cross-modal Reinforced Prompting for Graph and Language Tasks⭐️</h3>

<div align="center">

## Quick Start

### Package Dependencies

- Python 3.10.13
- PyTorch 2.0.1

For detailed information, please refer to `requirements.txt`.

### Preparations

We use the embedding model T5-Small.

```
git lfs install
git clone https://huggingface.co/google-t5/t5-small
```

### Preprocessing

Preprocess the datasets, prepare embeddings, and calculate similarities as follows:

```
cd ./preprocess
sh preprocess_WN18.sh
```

### Start Training

Initiate training with the following commands:

```
cd ../REINFORCE
sh run_kg_WN18.sh
```
</div>

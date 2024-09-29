<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JohnJiang12138/CMRP">
    <img height="150" src="Logo.jpg?sanitize=true" />
  </a>
</div>

<h3 align="center">⭐️CMRP: Cross-modal Reinforced Prompting for Graph and Language Tasks⭐️</h3>

<div align="center">
  
| **[Quick Start](#quick-start)** 
| **[Website](https://github.com/JohnJiang12138/CMRP)** | **[Paper](https://doi.org/10.1145/3637528.3671742)**
| **[Video](https://www.youtube.com/watch?v=QNq_jUVwO1s)**

![](https://img.shields.io/badge/Latest_version-v0.1-red)
![Testing Status](https://img.shields.io/badge/PyTorch-v2.0.1-red)
![Testing Status](https://img.shields.io/badge/license-MIT-blue)
![Testing Status](https://img.shields.io/badge/python->=3.10-red)

</div>


# Quick Start
## Package Dependencies

- Python 3.10.13
- PyTorch 2.0.1

for detailed information, please see ``requirements.txt``

```
git clone https://github.com/JohnJiang12138/CMRP.git
cd CMRP
conda create -n CMRP python=3.10.13
conda activate CMRP
pip install -r requirements.txt
```

## Preparations

During preprocessing, we use T5-Small for embedding.

```
git lfs install
git clone https://huggingface.co/google-t5/t5-small
```

Please replace the ``api_key`` and the ``requests`` detail with your customization.
In file``LLM_tasks/t1step2_llm_evaluate.py``, line 9 and line 73.

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
<h3 align="center">🌹Please Cite Our Work If Helpful:</h3>
<p align="center"><strong>Thanks! / 谢谢! / ありがとう! / merci! / 감사! / Danke! / спасибо! / gracias! ...</strong></p>

```
@inproceedings{jiang2024killing,
  title={Killing Two Birds with One Stone: Cross-modal Reinforced Prompting for Graph and Language Tasks},
  author={Jiang, Wenyuan and Wu, Wenwei and Zhang, Le and Yuan, Zixuan and Xiang, Jian and Zhou, Jingbo and Xiong, Hui},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1301--1312},
  year={2024}
}

```
<div align="center">
  <a href="https://github.com/JohnJiang12138/CMRP">
    <img height="1200" src="CMRP_KDD24Poster.jpg?sanitize=true" />
  </a>
</div>

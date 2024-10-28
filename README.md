# Generative Retrieval via Term Set Generation (TSGen)

This repository contains the implementation of the SIGIR'24 paper Generative Retrieval via Term Set Generation (TSGen).

## Quick Maps
- For training configurations of TSGen, check [train.yaml](src/data/config/mode/train.yaml) and [tsgen.yaml](src/data/config/tsgen.yaml)
- For the implementation of our proposed *permutation-invariant decoding* algorithm, check [index.py](src/utils/index.py), from line 1459 to line 2465 (we modify the `.generate()` method in huggingface transformers and implement it with a new class named `BeamDecoder`).

## Reproduction
- Environment
  - `python==3.9.12`
  - `torch==1.10.1`
  - `transformers==4.21.3`
  - `faiss==1.7.2`

- Data
```bash
# suppose you want to save the dataset at /data/TSGen
# NOTE: if you prefer another location, remember to set 'data_root' and 'plm_root' in src/data/config/base/_default.yaml accordingly
mkdir /data/TSGen
cd /data/TSGen
# download NQ320k dataset
wget https://huggingface.co/datasets/namespace-Pt/adon/resolve/main/NQ320k.tar.gz?download=true -O NQ320k.tar.gz
# untar the file, which results in the folder /data/TSGen/NQ320k
tar -xzvf NQ320k.tar.gz

# move to the code folder
cd TSGen/src
# preprocess the dataset, which results in the folder TSGen/src/data/cache/NQ320k/dataset
python -m scripts.preprocess base=NQ320k ++query_set=test

# move to the cache folder
cd TSGen/src/data/cache/NQ320k
# download the checkpoint and the term-set docid
wget https://huggingface.co/datasets/namespace-Pt/adon/resolve/main/tsgen.tar.gz?download=true -O TSGen.tar.gz
# untar the file, which results in the folder TSGen/src/data/cache/NQ320k/ckpts and TSGen/src/data/cache/NQ320k/codes
tar -xzvf NQ320k.tar.gz
```

- Evaluation
```bash
# evaluate with 100 beams
torchrun --nproc_per_node 8 run.py TSGen base=NQ320k mode=eval ++nbeam=100 ++eval_batch_size=20
```
The results should be similar to:
|MRR@10|MRR@100|Recall@1|Recall@10|Recall@100|
|0.771|0.774|0.708|0.889|0.948|

# Generative Retrieval via Term Set Generation (TSGen)

This repository contains the implementation of TSGen.

## Quick Maps
- For training configurations of TSGen, check [train.yaml](src/data/config/mode/train.yaml) and [tsgen.yaml](src/data/config/tsgen.yaml)
- For the implementation of our proposed *permutation-invariant decoding* algorithm, check [index.py](src/utils/index.py), from line 1459 to line 2465 (we modify the `.generate()` method in huggingface transformers and implement it with a new class named `BeamDecoder`).

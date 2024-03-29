# EgoExoLearn: Cross-View Temporal Action Segmentation

This is the codebase of cross-view temporal action segmentation.


## Requirements
* support Python 3.9, PyTorch 2.0.1, CUDA 11.8, CUDNN 8.7.0
* install all the library with: `pip install -r requirements.txt`
---

## Data Preparation
We follow MS-TCN to extract I3D RGB feature. Firstly, you should download from the main [README](../README.md) and unzip it to your folder.
The annotations of action anticipation and planning are at [./tas_annotation](./tas_annotation/).

## Usage


All scripts of domain adaption and zero-shot settings are at [./scripts](./scripts/). 
In order to launch one script to train the test the model, firstly, you should modify the `path_data` and `path_feat` to your folder. 

> We use `training`, `predict` or `eval` to control the behavior of scripts.

After that, you can launch one script through the following command example:

```bash
bash scripts/run_egobridge_baseline_ego_only.sh
```

It will train the temporal aciton segmentation model on ego-only view. So will other scripts.

---
### Acknowledgments

The codebase is based on [SSTDA](https://github.com/cmhungsteve/SSTDA).
We thank the authors for their efforts.

If you have any questions, feel free to contact Guo Chen (chenguo1177 <at> gmail.com)
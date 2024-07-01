# DyFADet: Dynamic Feature Aggregation for Temporal Action Detection (ECCV2024)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tridet-temporal-action-detection-with/temporal-action-localization-on-activitynet)](https://paperswithcode.com/sota/temporal-action-localization-on-activitynet?p=tridet-temporal-action-detection-with)

[Le Yang*+](https://github.com/yangle15), [Ziwei Zheng*](https://github.com/Ziwei-Zheng), [Yizeng Han](https://github.com/thuallen), Hao Cheng, Shiji Song, [Gao Huang](https://github.com/gaohuang), Fan Li

This repository contains the implementation of the paper, '[DyFADet: Dynamic Feature Aggregation for Temporal Action Detection](https://arxiv.org/pdf/2003.07326.pdf)'. 


<div align=center><img width="380" height="410" src=""/></div>


## Installation

1. Please ensure that you have installed PyTorch and CUDA. (This code requires PyTorch version >= 1.11. We use version=1.13.0 in our experiments)

2. Install the required packages by running the following command:

```shell
pip install  -r requirements.txt
```

3. Install NMS
```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

## Training

You can train your own model with the provided CONFIG files. The command for train is

```shell
CUDA_VISIBLE_DEVICES=0 python train.py ./configs/CONFIG_FILE --output OUTPUT_PATH
```

## Data Preparation

- We adopted the I3D features for THUMOS14, the TSP features for ActivityNet, the SlowFast features for Epic-Kitchen, and the SlowFast and EgoVLP features for Ego4D-MQ1.0 from [ActionFormer repository](https://github.com/happyharrycn/actionformer_release). To use these features, please download them from their link and unpack them into the data folder. For other features:

- For VideoMAEV2 features of **THUMOS14**, you can extract the features using the pre-trained VideoMAEv2 model as stated [here](https://github.com/sming256/OpenTAD/tree/main/configs/adatad). The experiments in our paper use the features extracted by VideoMAEv2-g. !!! We would like to express our gratitude to [Shuming Liu](https://github.com/sming256) for his help extracting the features!!! 

- For the SlowFast features of **HACS** dataset, please refer to [here](https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch) for more details about downloading and using the features. And the instruction about the VideoMAEv2 features for the HACS dataset can be found [here](https://github.com/dingfengshi/tridetplus).

- For the **FineAction** dataset, we adopt the VideoMAEv2 features from the [VideoMAEv2 repository](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/TAD.md). 


## Results
We provide a list of CONFIG files that allow you to easily build the architectures in the paper. These scripts are located in the `./tools` folder and include:

The mean average precision (mAP) results for each dataset are:

| Dataset         | 0.3 /0.5/0.1  | 0.7 /0.95  | Avg   | Pre-trained | Config |
|-----------------|-----------|------------|-------|-------------|--------|
| THUMOS14-I3D    | 84.0| 47.9 | 69.2  | checkpoint | thumos_i3d.yaml |
| THUMOS14-VM2-g  | 84.3| 50.2 | 70.5  | checkpoint | thumos_mae.yaml |
| ActivityNet-TSP | 58.1| 8.4  | 38.5  | checkpoint | anet_tsp.yaml   |
| HACS-SF         | 57.8| 11.8 | 39.2  | checkpoint | hacs_slowfast.yaml|
| HACS-VM2-g      | 64.0| 14.1 | 44.3  | checkpoint | hacs_mae.yaml.   |
| FineAction-VM2-g| 37.1| 5.9  | 23.8  | checkpoint | fineaction.yaml  |
| EPIC-KITCHEN-n  | 28.0| 20.8 | 25.0  | checkpoint | epic_slowfast_noun.yaml |
| EPIC-KITCHEN-v  | 26.8| 18.5 | 23.4  | checkpoint | epic_slowfast_verb.yaml |
| Ego4D-EV+SF     | 28.8| 16.9 | 22.8  | checkpoint |

*Note: We conduct all our experiments on a single Nvidia RTX4090 GPU and the training results may vary depending on the type of GPU and environments used.


## Test

We offer pre-trained models for each dataset, which you can download the chechpoints
from [Google Drive](https://drive.google.com/drive/folders/1eVROG6z-vHtm4AnXsh4N8ruUKkAidLqZ?usp=sharing). The command
for test is

```shell
CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/CONFIG_FILE PATH_TO_CHECKPOINT
```


### Citation
If you find this work useful or use our codes in your own research, please use the following bibtex:
```
@inproceedings{yang2020resolution,
  title={Resolution Adaptive Networks for Efficient Inference},
  author={Yang, Le and Han, Yizeng and Chen, Xi and Song, Shiji and Dai, Jifeng and Huang, Gao},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

### Contact
If you have any questions, please feel free to contact the authors. 

Le Yang: yangle15@xjtu.edu.cn

### Acknowledgments
Our code is built upon the codebase from [ActionFormer](https://github.com/happyharrycn/actionformer_release), [TriDet](https://github.com/dingfengshi/TriDet), [Detectron2](https://github.com/facebookresearch/detectron2), and we would like to express our
gratitude for their outstanding work.



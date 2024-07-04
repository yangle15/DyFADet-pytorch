# DyFADet: Dynamic Feature Aggregation for Temporal Action Detection (ECCV2024)
[![arXiv preprint](https://img.shields.io/badge/arxiv_2407.03197-blue%3Flog%3Darxiv)](https://arxiv.org/pdf/2407.03197) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dyfadet-dynamic-feature-aggregation-for/temporal-action-localization-on-hacs)](https://paperswithcode.com/sota/temporal-action-localization-on-hacs?p=dyfadet-dynamic-feature-aggregation-for)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dyfadet-dynamic-feature-aggregation-for/temporal-action-localization-on-fineaction)](https://paperswithcode.com/sota/temporal-action-localization-on-fineaction?p=dyfadet-dynamic-feature-aggregation-for)

This repository contains the implementation of the paper, '[DyFADet: Dynamic Feature Aggregation for Temporal Action Detection](https://arxiv.org/abs/2407.03197)'. 


<div align=center><img width="900" height="280" src="https://github.com/yangle15/DyFADet-pytorch/blob/main/pics/fig1.png"/></div>


## Installation

1. Please ensure that you have installed PyTorch and CUDA. (We use Pytorch=1.13.0 and CUDA=11.6 in our experiments.)

2. After you download the Repo, you need to install the required packages by running the following command:
```shell
pip install  -r requirements.txt
```

3. Install NMS
```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```


## Data Preparation

**HACS (SF features)**

- For the SlowFast features of the HACS dataset, please refer to [here](https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch) for the details about downloading and using the features. 

- Unpack the SlowFast feature into `/YOUR_DATA_PATH/`. You can find the processed annotation json files for the SlowFast feature in this Repo in `./data/hacs/annotations` folder.


**HACS (VideoMAEv2-g features)**

- This [Repo](https://github.com/dingfengshi/tridetplus) has provided the pre-extracted features of HACS using VideoMAEv2, you can download the features and unpack them into `/YOUR_DATA_PATH/`.

- The annotation json files is also the one in `./data/hacs/annotations` folder.


**THUMOS14 (I3D features)**

- Referring the procedure from [ActionFormer Repo](https://github.com/happyharrycn/actionformer_release/tree/main), you need to download the features (*thumos.tar.gz*) from [this Box link](https://uwmadison.box.com/s/glpuxadymf3gd01m1cj6g5c3bn39qbgr) or [this Google Drive link](https://drive.google.com/file/d/1zt2eoldshf99vJMDuu8jqxda55dCyhZP/view?usp=sharing) or [this BaiduYun link](https://pan.baidu.com/s/1TgS91LVV-vzFTgIHl1AEGA?pwd=74eh). The file includes I3D features, action annotations in json format, and external classification scores.

- Unpack Features and Annotations into `/YOUR_DATA_PATH/` and `/YOUR_ANNOTATION_PATH/`, respectively. 



**THUMOS14 (VideoMAEv2-g features).**

- You can extract the features using the pre-trained VideoMAEv2 model as stated [here](https://github.com/sming256/OpenTAD/tree/main/configs/adatad). The experiments in our paper use the features extracted by VideoMAEv2-g. **We would like to express our GREAT gratitude to [Shuming Liu](https://github.com/sming256) for his help extracting the features!!!**

- For the SlowFast features of **HACS** dataset, please refer to [here](https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch) for more details about downloading and using the features. And the instruction about the VideoMAEv2 features for the HACS dataset can be found [here](https://github.com/dingfengshi/tridetplus).

**ActivityNet 1.3**

- Referring the procedure from [ActionFormer Repo](https://github.com/happyharrycn/actionformer_release/tree/main), you need to download *anet_1.3.tar.gz* from [this Box link](https://uwmadison.box.com/s/aisdoymowukc99zoc7gpqegxbb4whikx) or [this Google Drive Link](https://drive.google.com/file/d/1VW8px1Nz9A17i0wMVUfxh6YsPCLVqL-S/view?usp=sharing) or [this BaiduYun Link](https://pan.baidu.com/s/1tw5W8B5YqDvfl-mrlWQvnQ?pwd=xuit). The file includes TSP features, action annotations in json format (similar to ActivityNet annotation format), and external classification scores.

- Unpack Features and Annotations into `/YOUR_DATA_PATH/` and `/YOUR_ANNOTATION_PATH/`, respectively.

- The used external classification scores in our experiments are in `./data/hacs/annotations/`.


**FineAction**

- The Pre-extracted features using VideoMAE V2-g can be downloaded [here](https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/features/fineaction_mae_g.tar.gz). Please refer the original [VideoMAEv2 repository](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/TAD.md) for more details.

- Unpack Features and Annotations into `/YOUR_DATA_PATH/` and `/YOUR_ANNOTATION_PATH/`, respectively.

- The used external classification scores in our experiments are in `./data/hacs/annotations/`.


## Training

You can train your own model with the provided CONFIG files. The command for train is

```shell
CUDA_VISIBLE_DEVICES=0 python train.py ./configs/CONFIG_FILE --output OUTPUT_PATH
```

You need to select a specific config files corresponding to different datasets. For the config json file, you need to further change the *json_file* variable to the path of your annotation file, and the *feat_folder* variable to the path of the downloaded dataset.

All the model can be trained on a single Nvidia RTX 4090 GPU (24GB).


## Evaluation

After training, you can test the obtained model by the following command:

```shell
CUDA_VISIBLE_DEVICES=0 python eval.py ./configs/CONFIG_FILE PATH_TO_CHECKPOINT
```

The mean average precision (mAP) results with the [pre-trained models](https://pan.baidu.com/s/1Aj-zLL4duNaX_GC4nJZ4Gg?pwd=wn4h) are :

| Dataset         | 0.3 /0.5/0.1  | 0.7 /0.95  | Avg   | Config |
|-----------------|-----------|------------|-------|-----------------|
| THUMOS14-I3D    | 84.0| 47.9 | 69.2  |  thumos_i3d.yaml |
| THUMOS14-VM2-g  | 84.3| 50.2 | 70.5  |  thumos_mae.yaml |
| ActivityNet-TSP | 58.1| 8.4  | 38.5  |  anet_tsp.yaml   |
| HACS-SF         | 57.8| 11.8 | 39.2  |  hacs_slowfast.yaml|
| HACS-VM2-g      | 64.0| 14.1 | 44.3  |  hacs_mae.yaml   |
| FineAction-VM2-g| 37.1| 5.9  | 23.8  |  fineaction.yaml  |
| EPIC-KITCHEN-n  | 28.0| 20.8 | 25.0  |  epic_slowfast_noun.yaml |
| EPIC-KITCHEN-v  | 26.8| 18.5 | 23.4  |  epic_slowfast_verb.yaml


### Citation
If you find this work useful or use our codes in your own research, please use the following bibtex:
```
@inproceedings{yang2024dyfadet,
  title={DyFADet: Dynamic Feature Aggregation for Temporal Action Detection},
  author={Yang, Le and Zheng, Ziwei and Han, Yizeng and Cheng, Hao and Song, Shiji and Huang, Gao and Li, Fan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### Contact
If you have any questions, please feel free to contact the authors. 

Ziwei Zheng: ziwei.zheng@stu.xjtu.edu.cn

Le Yang: yangle15@xjtu.edu.cn

### Acknowledgments
Our code is built upon the codebase from [ActionFormer](https://github.com/happyharrycn/actionformer_release), [TriDet](https://github.com/dingfengshi/TriDet), [Detectron2](https://github.com/facebookresearch/detectron2), and many other great Repos, we would like to express our gratitude for their outstanding work.



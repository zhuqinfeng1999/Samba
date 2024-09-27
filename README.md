# [Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model](https://doi.org/10.1016/j.heliyon.2024.e38495)

This is the official code repository for "Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model". {[Heliyon Paper](https://doi.org/10.1016/j.heliyon.2024.e38495)}
The paper has been accepted by Heliyon.

![image](https://github.com/user-attachments/assets/2f4eb159-db36-476e-aba9-e56ab07b5324)


[![GitHub stars](https://badgen.net/github/stars/zhuqinfeng1999/Samba)](https://github.com//zhuqinfeng1999/Samba)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2404.01705-b31b1b.svg)](https://arxiv.org/abs/2404.01705)

## Abstract

High-resolution remotely sensed images pose challenges to traditional semantic segmentation networks, such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). CNN-based methods struggle to handle high-resolution images due to their limited receptive field, while ViT-based methods, despite having a global receptive field, face challenges when processing long sequences. Inspired by the Mamba network, which is based on a state space model (SSM) to efficiently capture global semantic information, we propose a semantic segmentation framework for high-resolution remotely sensed imagery, named Samba. Samba utilizes an encoder-decoder architecture, with multiple Samba blocks serving as the encoder to efficiently extract multi-level semantic information, and UperNet functioning as the decoder. We evaluate Samba on the LoveDA, ISPRS Vaihingen, and ISPRS Potsdam datasets using the mIoU and mF1 metrics, and compare it with top-performing CNN-based and ViT-based methods. The results demonstrate that Samba achieves unparalleled performance on commonly used remotely sensed datasets for semantic segmentation. Samba is the first to demonstrate the effectiveness of SSM in segmenting remotely sensed imagery, setting a new performance benchmark for Mamba-based techniques in this domain of semantic segmentation. The source code and baseline implementations are available at https://github.com/zhuqinfeng1999/Samba.

## Segmentation Results

![image](https://github.com/user-attachments/assets/e1ce62b2-0f41-48ff-822a-c39b0f3780d6)

## Installation

### Requirements

Requirements: Ubuntu 20.04, CUDA 12.4

* Set up the mmsegmentation environment; we conduct experiments using the mmsegmentation framework. Please refer to https://github.com/open-mmlab/mmsegmentation.
* Install Mamba: execute pip install causal-conv1d==1.1.1 and pip install mamba-ssm
* Install apex-amp, pyyaml, timm, tlt

#### LoveDA datasets

* The LoveDA datasets can be found here https://github.com/Junjue-Wang/LoveDA.

* After downloading the dataset, you are supposed to put them into '/mmsegmentation/data/loveDA/'

* '/mmsegmentation/data/loveDA/'
- ann_dir
  - train
    - .png
  - val
    - .png
- img_dir
  - train
    - .png
  - val
    - .png

#### Model file and config file

- The model file Samba.py can be found in /mmsegmentation/mmseg/models/backbones/

- The config file samba_upernet.py for the combination of backbone and decoder head samba_upernet can be found in /mmsegmentation/configs/_base_/models

- The config file samba_upernet-15k_loveda-512x512_6e4.py for training can be found in /mmsegmentation/configs/samba/

## Training Samba

`bash tools/dist_train.sh /mmsegmentation/configs/samba/samba_upernet-15k_loveda-512x512_6e4.py 2 --work-dir /mmsegmentation/output/sambaupernet`

`bash tools/dist_train.sh /mmsegmentation/configs/samba/samba_upernet-15k_potsdam-512x512_6e4.py 2 --work-dir /mmsegmentation/output/sambaupernet`

`bash tools/dist_train.sh /mmsegmentation/configs/samba/samba_upernet-15k_vaihingen-512x512_6e4.py 2 --work-dir /mmsegmentation/output/sambaupernet`

## Testing Samba

`bash tools/dist_test.sh /mmsegmentation/configs/samba/samba_upernet-15k_loveda-512x512_6e4.py \ /mmsegmentation/output/sambaupernet/iter_15000.pth 2 --out /mmsegmentation/visout/sambaupernet`

`bash tools/dist_test.sh /mmsegmentation/configs/samba/samba_upernet-15k_potsdam-512x512_6e4.py \ /mmsegmentation/output/sambaupernet/iter_15000.pth 2 --out /mmsegmentation/visout/sambaupernet`

`bash tools/dist_test.sh /mmsegmentation/configs/samba/samba_upernet-15k_vaihingen-512x512_6e4.py \ /mmsegmentation/output/sambaupernet/iter_15000.pth 2 --out /mmsegmentation/visout/sambaupernet`

## Citation

If you find this work useful in your research, please consider cite:

```
@article{ZHU2024e38495,
title = {Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model},
journal = {Heliyon},
pages = {e38495},
year = {2024},
issn = {2405-8440},
doi = {https://doi.org/10.1016/j.heliyon.2024.e38495},
url = {https://www.sciencedirect.com/science/article/pii/S2405844024145264},
author = {Qinfeng Zhu and Yuanzhi Cai and Yuan Fang and Yihan Yang and Cheng Chen and Lei Fan and Anh Nguyen},
}

@article{zhu2024rethinking,
  title={Rethinking Scanning Strategies with Vision Mamba in Semantic Segmentation of Remote Sensing Imagery: An Experimental Study},
  author={Zhu, Qinfeng and Fang, Yuan and Cai, Yuanzhi and Chen, Cheng and Fan, Lei},
  journal={arXiv preprint arXiv:2405.08493},
  year={2024}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [Mamba](https://github.com/state-spaces/mamba), [SiMBA](https://github.com/badripatro/simba), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for making their valuable code publicly available.

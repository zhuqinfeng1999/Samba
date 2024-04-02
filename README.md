# [Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model](https://arxiv.org/abs/)

This is the official code repository for "Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model". {[Arxiv Paper](https://arxiv.org/abs/)} The paper will be released soon.

![image](https://github.com/zhuqinfeng1999/Samba/assets/34743935/723109f3-4e5b-45c2-ad4f-492a87277075)

[![GitHub stars](https://badgen.net/github/stars/zhuqinfeng1999/Samba)](https://github.com/zhuqinfeng1999/Samba)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv.svg)](https://arxiv.org/abs/)

## Abstract

High-resolution remotely sensed images poses a challenge for commonly used semantic segmentation methods such as Convolutional Neural Network (CNN) and Vision Transformer (ViT). CNN-based methods struggle with handling such high-resolution images due to their limited receptive field, while ViT faces challenges to handle long sequences. Inspired by Mamba, which adopts a State Space Model (SSM) to efficiently capture global semantic information, we propose a semantic segmentation framework for high-resolution remotely sensed images, named Samba. Samba utilizes an encoder-decoder architecture, with Samba blocks serving as the encoder for efficient multi-level semantic information extraction, and UperNet functioning as the decoder. We evaluate Samba on the LoveDA dataset, comparing its performance against top-performing CNN and ViT methods. The results reveal that Samba achieved unparalleled performance on LoveDA. This represents that the proposed Samba is an effective application of the SSM in semantic segmentation of remotely sensed images, setting a new benchmark in performance for Mamba-based techniques in this specific application. The source code and baseline implementations are available at https://github.com/zhuqinfeng1999/Samba.

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

`bash tools/dist_train.sh /mmsegmentation/configs/samba/samba_upernet-15k_loveda-512x512_6e4.py 2 --work-dir /mmsegmentation/output/sambaupernet --amp`

## Testing Samba

`bash tools/dist_test.sh /mmsegmentation/configs/samba/samba_upernet-15k_loveda-512x512_6e4.py \ /mmsegmentation/output/sambaupernet/iter_15000.pth 2 --out /mmsegmentation/visout/sambaupernet`

## Citation

If you find this work useful in your research, please consider cite:

The paper will be released soon.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of [Mamba](https://github.com/state-spaces/mamba), [SiMBA](https://github.com/badripatro/simba), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for making their valuable code publicly available.

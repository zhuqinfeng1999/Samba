# [Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model](https://arxiv.org/abs/2404.01705)

This is the official code repository for "Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model". {[Arxiv Paper](https://arxiv.org/abs/2404.01705)}

![image](https://github.com/zhuqinfeng1999/Samba/assets/34743935/723109f3-4e5b-45c2-ad4f-492a87277075)

[![GitHub stars](https://badgen.net/github/stars/zhuqinfeng1999/Samba)](https://github.com//zhuqinfeng1999/Samba)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2404.01705-b31b1b.svg)](https://arxiv.org/abs/2404.01705)

## Abstract

High-resolution remotely sensed images pose a challenge for commonly used semantic segmentation methods such as Convolutional Neural Network (CNN) and Vision Transformer (ViT). CNN-based methods struggle with handling such high-resolution images due to their limited receptive field, while ViT faces challenges in handling long sequences. Inspired by Mamba, which adopts a State Space Model (SSM) to efficiently capture global semantic information, we propose a semantic segmentation framework for high-resolution remotely sensed images, named Samba. Samba utilizes an encoder-decoder architecture, with Samba blocks serving as the encoder for efficient multi-level semantic information extraction, and UperNet functioning as the decoder. We evaluate Samba on the LoveDA, ISPRS Vaihingen, and ISPRS Potsdam datasets, comparing its performance against top-performing CNN and ViT methods. The results reveal that Samba achieved unparalleled performance on commonly used remote sensing datasets for semantic segmentation. Our proposed Samba demonstrates for the first time the effectiveness of SSM in semantic segmentation of remotely sensed images, setting a new benchmark in performance for Mamba-based techniques in this specific application. The source code and baseline implementations are available at https://github.com/zhuqinfeng1999/Samba.

## Segmentation Results

![对比](https://github.com/zhuqinfeng1999/Samba/assets/34743935/f2cac6e0-7669-4add-8739-79a3f5467603)

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
@article{zhu2024samba,
  title={Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model},
  author={Zhu, Qinfeng and Cai, Yuanzhi and Fang, Yuan and Yang, Yihan and Chen, Cheng and Fan, Lei and Nguyen, Anh},
  journal={arXiv preprint arXiv:2404.01705},
  year={2024}
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

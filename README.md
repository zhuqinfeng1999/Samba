# Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model

This is the official code repository for "Samba: Semantic Segmentation of Remotely Sensed Images with State Space Model". {Arxiv Paper}

![image](https://github.com/zhuqinfeng1999/Samba/assets/34743935/723109f3-4e5b-45c2-ad4f-492a87277075)

## Abstract

Remotely sensed images generally possess very high resolutions, making the semantic segmentation for remote sensing challenging. The commonly used methods for processing semantic segmentation are based on the Convolutional Neural Network (CNN) and Vision Transformer (ViT). However, due to the limitation of the receptive field in CNN, and the limitation of handling long sequences in ViT, it is difficult for the networks to handle high-resolution images. Inspired by Mamba, which is based on the State Space Model (SSM), capturing global semantic information with low computational complexity, we propose a semantic segmentation framework for remote sensing images, named Samba. The proposed framework utilizes an encoder-decoder architecture, employing Samba blocks as the encoder, which is designed for efficient image feature extraction to extract multi-level semantic information effectively, and UperNet as the decoder. We evaluate Samba on the LoveDA dataset, comparing it against top-performing CNN and ViT methods. The results reveal that Samba achieved unparalleled performance on LoveDA. This represents that the proposed Samba method is an effective application of the State Space Model in remote sensing image semantic segmentation, setting a benchmark for Mamba-based techniques in semantic segmentation of remotely sensed images. The source code and baseline implementations are available at https://github.com/zhuqinfeng1999/Samba.

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

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

We acknowledge all the authors of the employed public datasets, allowing the community to use these valuable resources for research purposes. We also thank the authors of Mamba, SiMBA, MMSegmentation for making their valuable code publicly available.

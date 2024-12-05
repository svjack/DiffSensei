<br />
<p align="center">
  <h1 align="center">MotionBooth: Motion-Aware Customized <br> Text-to-Video Generation</h1>
  <p align="center">
    <br />
    <a href="https://jianzongwu.github.io/"><strong>Jianzong Wu</strong></a>
    ·
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li</strong></a>
    ·
    <a href="https://zengyh1900.github.io/"><strong>Yanhong Zeng</strong></a>
    ·
    <a href="https://zhangzjn.github.io/"><strong>Jiangning Zhang</strong></a>
    .
    <a href="https://qianyuzqy.github.io/"><strong>Qianyu Zhou</strong></a>
    .
    <a href="https://github.com/ly015"><strong>Yining Li</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=T4gqdPkAAAAJ"><strong>Yunhai Tong</strong></a>
    .
    <a href="https://chenkai.site/"><strong>Kai Chen</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2406.17758'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/jianzongwu/MotionBooth'>
      <img src='https://img.shields.io/badge/Github-Code-blue?style=flat&logo=Github' alt='Code'>
    </a>
    <a href='https://jianzongwu.github.io/projects/motionbooth'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
  </p>
<br />

## Examples

**Customization and subject motion control**

<img src="assets/demo1.gif" width="600">

**Hybrid control on customization, subject and camera motion**

<img src="assets/demo2.gif" width="600">


## 🎉 News

- [2024-12-10] Checkpoint, dataset, and inference code are released!

## 📖 TL;DR

DiffSensei can generate vivid black-and-white manga panels with precise character and dialog layout control!

## 🛠️ Quick Start

### Installation

``` bash
# Create a new environment with Conda
conda create -n diffsensei python=3.11
conda activate diffsensei
# Install Pytorch and Diffusers related packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge diffusers transformers accelerate
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
# Third-party repo for running the gradio demo
pip install gradio-image-prompter
```

### Model Download

Download our DiffSensei model from [huggingface](https://huggingface.co/jianzongwu/DiffSensei) and place it in the `checkpoints` folder like this:

```
checkpoints
  |- diffsensei
    |- image_generator
      |- ...
    |- mllm
      |- ...
```


### Inference with Gradio

We provide gradio demo for inferencing DiffSensei.

``` bash
CUDA_VISIBLE_DEVICES=0 \
python -m scripts.demo.gradio \
  --config_path configs/model/diffsensei.yaml \
  --inference_config_path configs/inference/diffsensei.yaml \
  --ckpt_path checkpoints/diffsensei
```

*Please be patient. Try more prompts, characters, and random seeds, and download your favored manga panels!* 🤗

### The MangaZero Dataset

For license issues, we cannot directly share the images. Instead, we provide the manga image urls (in MangaDex) and annotations of our MangaZero dataset.
Note that the released version of MangaZero is about 3/4 of the full dataset used for training. The missing images is because some urls are not available. For similar usage for manga data, we strongly encourage everyone who is interested to collect their dataset freely from MangaDex, following the instruction of [MangaDex API](https://api.mangadex.org/docs/).

Please download MangaZero from [Huggingface](https://huggingface.co/datasets/jianzongwu/MangaZero).

After downloading the annotation file, please run `scripts/dataset/download_mangazero.py` to download and organize the images.


## Citation

```
```

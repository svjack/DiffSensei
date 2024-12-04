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

- [2024-6-28] Inference code, training code, and checkpoints are released!

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


### Pre-trained Model Preparation

We use [Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w) and [LaVie-base](https://huggingface.co/Vchitect/LaVie) for the base T2V models. Please download Zeroscope from the [official huggingface page](https://huggingface.co/cerspense/zeroscope_v2_576w). For LaVie, we provide a script to convert their original checkpoint into the format that is suitable for Diffusers. Please download the [LaVie-base](https://huggingface.co/Vchitect/LaVie) model and the [Stable-Diffusion-v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) checkpoint.

Then, organize the pre-trained models in the `checkpoints` folder.

```
checkpoints
  |- zeroscope_v2_576w
  |- stable-diffusion-v1-4
  |- lavie_base.pt
```

Then, run the following command to convert the checkpoint

``` bash
python -m scripts.convert_ckpts.convert_lavie_to_diffusers
``` 

Then, rename the `stable-diffusion-v1-4` folder to `lavie`. Additionally, you should replace the config file to LaVie's configs, following [checkpoint guide](docs/checkpoints.md).

The final checkpoint folder looks like this:

```
checkpoints
  |- zeroscope_v2_576w
  |- lavie
  |- lavie_base.pt (Not used anymore)
```

We use the converted lavie model for all the experiments.


### Inference with Gradio

For quick inference and re-producing the examples in paper, please download our trained customized checkpoints for the target subjects in [huggingface](https://huggingface.co/jianzongwu/MotionBooth). The names of the checkpoints correspond to the subject names in the MotionBooth dataset.

Please place the checkpoints in che `checkpoints` folder like this:

```
checkpoints
  |- customized
    |- zeroscope
      |- ...
    |- lavie
      |- ...
  |- zeroscope_v2_576w
  |- lavie
  |- lavie_base.pt (Not used anymore)
```

We use simple script files to indicate the subject and camera motion. We provide several examples in `data/scripts`. In these script files, the "bbox" controls the bounding box sequence for the subjects' motion, while the "camera speed" controls the corresponding camera motion speed.

We provide the inference script in `scripts/inference.py` for all types of MotionBooth applications. It uses [Accelerate PartialState](https://huggingface.co/docs/accelerate/index) to support multi GPU inference.

### MangaZero Data Download

We collect 26 objects from [DreamBooth](https://dreambooth.github.io/) and [CustomDiffusion](https://github.com/adobe-research/custom-diffusion) to perform the experiments in paper. These objects include pets, plushies, toys, cartoons, and vehicles. We also annotate masks for each image. We name it the MotionBooth dataset. Please download our dataset from [huggingface](https://huggingface.co/datasets/jianzongwu/MotionBooth).

Note that a few images from the original datasets are deleted because the low quality of the obtained masks. Additionally, a few images are resized and cropped to square shapes.

After downloading, please unzip and place the dataset under the `data` folder. It should look like this:

```
data
  |- MotionBooth
    |- images
      |- cat2
      |- ...
    |- masks
      |- cat2
      |- ...
  |- scripts
```




## 📢 Disclaimer

Please try more samples under different random seeds to have better results. 🤗

## Citation

```
article{wu2024motionbooth,
  title={MotionBooth: Motion-Aware Customized Text-to-Video Generation},
  author={Jianzong Wu and Xiangtai Li and Yanhong Zeng and Jiangning Zhang and Qianyu Zhou and Yining Li and Yunhai Tong and Kai Chen},
  journal={NeurIPS},
  year={2024},
}
```

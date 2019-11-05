<div align="center">
    <h1>TF-Unet</h1>
    <img href="#" src="./samples/u.jpg" />
    <p>General purpose U-Network implemented in Keras for image segmentation</p>
    <p>
        <a href="#getting-started">Getting started</a> •
        <a href="#training">Training</a> •
        <a href="#evaluation">Evaluation</a> •
        <a href="#contributing">Contributing</a>
    </p>
    <img src="https://travis-ci.com/juniorxsound/TF-Unet.svg?token=ztzi6EexNpaHGeSp1q8W&branch=master" />
    <img src="https://img.shields.io/badge/python-3.6-blue.svg" />
</div>

## Getting started

### Dependencies
To quickly get started make sure you have the following dependencies installed:
- [Docker (19 or newer)](https://docs.docker.com/install/) 📦
- [Make](https://www.gnu.org/software/make/) *[Optional macOS / Linux]* 🛠
- [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart) *[Optional for GPUs]* 🗜
</ul>

### Setup
Clone (or [download](https://github.com/juniorxsound/TF-Unet/archive/master.zip)) the repository and `cd` into it
```sh
git clone https://github.com/juniorxsound/TF-Unet.git && cd TF-Unet
```
Next build the Docker image by simply running `make build`
> The build process will pick either `Dockerfile.cpu` or `Dockerfile.gpu` based on your system

## Training

## Evaluation

## Thanks to
# (LVI) CIFAR-100 Image Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This project implements various image classification models for 
the CIFAR-100 dataset, including ResNet, ResNeXt, ViT, 
Swin Transformer, PyramidNet, and EfficientNet. 
It was developed as part of the Learning Vision Intelligence (LVI) 
course project.


## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Results](#results)
- [Visualizations](#visualizations)
- [Team Members](#team-members)
- [License](#license)

## Overview

The CIFAR-100 dataset consists of 60,000 32x32 color images 
in 100 classes. This project, developed as part of the Learning 
Vision Intelligence (LVI) course, 
aims to develop and compare high-performance image classification 
models using various state-of-the-art architectures.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/[your-username]/cifar100-classification.git
   cd lvi-cifar100-classifier-pytorch
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To train the model:

```sh
python train.py --epochs 200 --batch-size 128
```

To evaluate the model:

```sh
python evaluate.py --model-path path/to/saved/model.pth
```

## Dataset

The CIFAR-100 dataset is automatically downloaded by the PyTorch `torchvision` library. It includes:
- 50,000 training images
- 10,000 testing images
- 100 classes
- 32x32 pixel resolution

## Model Architecture

[Provide a brief description or diagram of your model architecture]

```
[You can include a simple ASCII diagram or code snippet here]
```

## Training

- **Optimizer**: Adam
- **Learning Rate**: 0.001 with cosine annealing
- **Batch Size**: 128
- **Epochs**: 200
- **Data Augmentation**: Random crop, horizontal flip, normalization

## Results

| Model | Top-1 Accuracy | Top-5 Accuracy | Super Top-1 Accuracy |
|-------|----------------|----------------|----------------------|
| ResNet | XX% | XX% | XX% |
| ResNeXt | XX% | XX% | XX% |
| ViT | XX% | XX% | XX% |
| Swin | XX% | XX% | XX% |
| PyramidNet | XX% | XX% | XX% |
| EfficientNet | XX% | XX% | XX% |

[You can add more details, graphs, or visualizations here]

## Visualizations

We have implemented various visualization functions in the `visualizations.py` file to help analyze and compare the performance of different models. These include:

- Loss graphs
- Accuracy graphs
- Loss landscapes

To generate visualizations:

```sh
python visualizations.py --model [model_name] --plot-type [plot_type]
```

Replace `[model_name]` with one of the implemented models and `[plot_type]` with either `loss`, `accuracy`, or `landscape`.

Example:
```sh
python visualizations.py --model resnet --plot-type loss
```

This will generate a loss graph for the ResNet model and save it in the `plots` directory.


## Team Members

This project was developed collaboratively by the following team members:

<a href="https://github.com/bigeco/lvi-cifar100-classifier-pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bigeco/lvi-cifar100-classifier-pytorch" />
</a>

(각자 역할 수정. (다 적기))
- **Lee Songeun**
  - Role: [e.g., Project Lead, Model Implementation]
  - GitHub: [@bigeco](https://github.com/bigeco)

- **Park Jihye**
  - Role: [e.g., Data Preprocessing, Model Training]
  - GitHub: [@park-ji-hye](https://github.com/park-ji-hye)

- **Song Daeun**
  - Role: [e.g., Results Analysis, Documentation]
  - GitHub: [@Song-Daeun](https://github.com/Song-Daeun)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue or contact [your-email@example.com].
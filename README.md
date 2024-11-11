# CoSyn: Context-Conditioned Joint Diffusion Model for Histopathology Nuclei Image-Label Pair Synthesis

This repository contains the official implementation of the paper:

**"Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model"**  

- ECCV version: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72624-8_9)
- Arxiv version: [Paper](https://arxiv.org/abs/2407.14434)

### Overview

We introduce a novel framework for co-synthesizing histopathology nuclei images and paired semantic labels using a context-conditioned joint diffusion model. Our method addresses the challenge of limited training data in multi-class histopathology nuclei analysis tasks by generating high-quality synthetic samples that incorporate spatial and structural context information.

### Key Features

- Context-conditioned diffusion model using nucleus centroid layouts and structure-related text prompts
- Concurrent synthesis of images, semantic labels, and distance maps for instance-wise nuclei labels
- Demonstrated effectiveness on multi-institutional, multi-organ, and multi-modality datasets

### Installation

```bash
git clone https://github.com/yourusername/CoSyn-ContextCondNucAug.git
cd CoSyn-ContextCondNucAug
pip install -r requirements.txt
```

### Acknowledgements

We would like to acknowledge the following projects that have contributed to our work:
This project is built upon the work of [GCDP](https://github.com/pmh9960/GCDP). We thank the authors for making their code available.

### Citation
If you find this work useful in your research, please consider citing our paper:
```bibtex
@InProceedings{10.1007/978-3-031-72624-8_9,
author="Min, Seonghui
and Oh, Hyun-Jic
and Jeong, Won-Ki",
editor="Leonardis, Ale{\v{s}}
and Ricci, Elisa
and Roth, Stefan
and Russakovsky, Olga
and Sattler, Torsten
and Varol, G{\"u}l",
title="Co-synthesis of Histopathology Nuclei Image-Label Pairs Using a Context-Conditioned Joint Diffusion Model",
booktitle="Computer Vision -- ECCV 2024",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="146--162",
abstract="In multi-class histopathology nuclei analysis tasks, the lack of training data becomes a main bottleneck for the performance of learning-based methods. To tackle this challenge, previous methods have utilized generative models to increase data by generating synthetic samples. However, existing methods often overlook the importance of considering the context of biological tissues (e.g., shape, spatial layout, and tissue type) Moreover, while generative models have shown superior performance in synthesizing realistic histopathology images, none of the existing methods are capable of producing image-label pairs at the same time. In this paper, we introduce a novel framework for co-synthesizing histopathology nuclei images and paired semantic labels using a context-conditioned joint diffusion model. We propose conditioning of a diffusion model using nucleus centroid layouts with structure-related text prompts to incorporate spatial and structural context information into the generation targets. Moreover, we enhance the granularity of our synthesized semantic labels by generating instance-wise nuclei labels using distance maps synthesized concurrently in conjunction with the images and semantic labels. We demonstrate the effectiveness of our framework in generating high-quality samples on multi-institutional, multi-organ, and multi-modality datasets. Our synthetic data consistently outperforms existing augmentation methods in the downstream tasks of nuclei segmentation and classification.",
isbn="978-3-031-72624-8"
}
```

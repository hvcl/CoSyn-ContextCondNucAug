# CoSyn: Context-Conditioned Joint Diffusion Model for Histopathology Nuclei Image-Label Pair Synthesis

This repository contains the official implementation of the paper:

**"Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model"**  
Accepted at ECCV 2024

## Overview

We introduce a novel framework for co-synthesizing histopathology nuclei images and paired semantic labels using a context-conditioned joint diffusion model. Our method addresses the challenge of limited training data in multi-class histopathology nuclei analysis tasks by generating high-quality synthetic samples that incorporate spatial and structural context information.

## Key Features

- Context-conditioned diffusion model using nucleus centroid layouts and structure-related text prompts
- Concurrent synthesis of images, semantic labels, and distance maps for instance-wise nuclei labels
- Demonstrated effectiveness on multi-institutional, multi-organ, and multi-modality datasets

## Installation

```bash
git clone https://github.com/yourusername/CoSyn-ContextCondNucAug.git
cd CoSyn-ContextCondNucAug
pip install -r requirements.txt

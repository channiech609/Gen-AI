# Gen-AI

To Start In Google Collab, run the following:
- !git clone https://github.com/channiech609/Gen-AI.git
- %cd Gen-AI
- !ls
- !pip install -r requirements.txt
- %run Code.py

## Project Objective
Our project aims to evaluate how different regularization techniques (L2, Dropout, and Batch Normalization) affect the performance of generative models. Specifically, we are focusing on image generation using Deep Convolutional Generative Adversarial Networks (DCGANs). The goal is to visually and quantitatively assess how these methods influence image quality, diversity, and training stability.


## Literature Review

We reviewed the original DCGAN paper by Radford et al. (2015), which provides a foundational architecture for image generation. Additionally, we explored literature on regularization techniques in deep learning, including:

- L2 regularization to prevent overfitting by penalizing large weights.

- Dropout for reducing co-adaptation of neurons and improving generalization.

- Batch Normalization for stabilizing training and accelerating convergence.

We also studied visualizations and evaluation techniques used in GAN research to inform our output comparison approach.

## Benchmarking based on our experiments
| Model | Image Quality(Visual) | Loss Curve Behavior | FID SCORE |
|---|---|---|---|
|Baseline (No regularizations)|Worst: High noise/mode collapse|Epochs 1-10: Loss_D drops rapidly from ~1.3 to near 0.05 by Epoch 2 and Loss_G surges from ~2.2 to ~7.8; Early discriminator dominance. Epoch 11: Loss_D spikes to 6.1, Loss_G crashes to 0.0014; Critical Collapse. Epochs 11-20: Loss_D fluctuates between 0.26–8.4,Loss_G between 0.68–14.9; Significant volatility. Epochs 20-50: Loss_D stabilizes near 0.001–0.01 and Loss_G plateaus around 6.0–7.0; Progressive Stabilization.|409.61|
|L2 Regularization|Good: Cleaner than baseline, minor artifacts|Epochs 1-15: Loss_D quickly becomes strong to 1.0 and Loss_G increases from 2.8 to >8.0; Discriminator dominates, generator struggles. Epoch 15: Loss_D suddenly fails to 0.0252; Loss_G spikes to 21.8705; Critical Collapse. Epochs 16-30: System rebalances, but losses keep fluctuating. Epochs 30-50: Relative stability emerges.|339.8|
|Dropout (p=0.3)|Best: Sharp details, diverse samples|Epochs 1-7: Loss_D drops rapidly to near-zero by epoch 5 and Loss_G spikes to extremely high by epoch 7; Discriminator Dominance. Epoch 8: Loss_D spikes to 4.3459 and Loss_G to 19.5167; Catastrophic Collapse. Epochs 9-50: Loss_D and Loss_G oscillate erratically; Persistent Instability.|329.63|
|BatchNorm|Poor: Blurry or repetitive patterns|Epochs 1-10: Loss_D starts strong with ~0.5 and becomes very strong >0.8 by epoch 5. Epochs 10-30: The losses oscillate significantly; Loss_D often reaches >0.9 while Loss_G drops very low (<0.2); Epochs 40-50: The system seems more balanced but still unstable.|398.83|

## Framework Selection
We are using PyTorch due to its flexibility, strong community support, and ease of integration with Google Colab. The official PyTorch DCGAN tutorial serves as the baseline implementation, which we adapt for our Fashion-MNIST dataset.

## EDA
We perform an exploratory data analysis(EDA) of the Fashion-MNIST dataset, which has 70,000 grayscale 28×28 images across 10 fashion categories. The EDA covers data inspection, cleaning, and descriptive statistics to understand dataset structure, check for missing values or duplicates, and summarize pixel intensity and class distributions. Visualization includes sample images per class, class distribution bar plots, and pixel intensity histograms. The dataset is balanced across classes with most pixel values near zero (background).

## Dataset Preparation
- Dataset: Fashion-MNIST (70,000 grayscale images, 10 classes)

- Preprocessing: Images are normalized to [-1, 1] and optionally resized to 64×64.

## Model Development
The generator and discriminator follow the DCGAN architecture.

We create four variants:

- Baseline (no regularization)

- L2 Regularization (weight decay = 0.01)

- Dropout (p = 0.3 in generator and discriminator)

- Batch Normalization (in generator and discriminator)

- Each model is modular and reusable with parameterized components for easy experimentation.


## Training & Fine-Tuning
All models are trained for 50 epochs using the Adam optimizer with learning rate = 0.0002, beta1 = 0.5, beta2 = 0.999

- Hyperparameters are consistent across all models to ensure fair comparison.

- Model checkpoints and sample outputs (64 images per model) are saved per epoch for analysis.


## Evaluation & Metrics
Qualitative: Visual inspection of output samples for image sharpness, diversity, and completeness.

Quantitative: Training loss curves and Fréchet Inception Distance (FID) scores to evaluate image quality and distribution similarity.

## References
- https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- https://arxiv.org/abs/1511.06434
- https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- https://github.com/zalandoresearch/fashion-mnist

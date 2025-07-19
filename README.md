# Gen-AI

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
|Baseline (No regularizations)|---|---|409.61|
|L2 Regularization|---|---|339.8|
|Dropout (p=0.3)|---|---|329.63|
|BatchNorm|---|---|398.83|




### Framework Selection
We are using PyTorch due to its flexibility, strong community support, and ease of integration with Google Colab. The official PyTorch DCGAN tutorial serves as the baseline implementation, which we adapt for our Fashion-MNIST dataset.


### Dataset Preparation
- Dataset: Fashion-MNIST (70,000 grayscale images, 10 classes)

- Preprocessing: Images are normalized to [-1, 1] and optionally resized to 64×64.

### Model Development
The generator and discriminator follow the DCGAN architecture.

We create four variants:

- Baseline (no regularization)

- L2 Regularization (weight decay = 0.01)

- Dropout (p = 0.3 in generator and discriminator)

- Batch Normalization (in generator and discriminator)

- Each model is modular and reusable with parameterized components for easy experimentation.


### Training & Fine-Tuning
All models are trained for 50 epochs using the Adam optimizer with learning rate = _ and beta1 = _

- Hyperparameters are consistent across all models to ensure fair comparison.

- Model checkpoints and sample outputs (20 images per model) are saved per epoch for analysis.


### Evaluation & Metrics
Qualitative: Visual inspection of output samples for image sharpness, diversity, and completeness.

Quantitative: Training loss curves and Fréchet Inception Distance (FID) scores to evaluate image quality and distribution similarity.

### References
https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://arxiv.org/abs/1511.06434
https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

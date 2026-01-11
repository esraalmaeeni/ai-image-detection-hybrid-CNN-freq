# AI-Generated Image Detection with Hybrid CNN-Frequency Fusion

This project implements a hybrid deep learning architecture to detect AI-generated images using spatial and frequency domain analysis, combined with adversarial training.

## Architecture Highlights

- **Spatial Branch**: CNN for spatial feature extraction
- **Frequency Branch**: Uses FFT to extract frequency-based features
- **Fusion Module**: Attention-based feature fusion
- **Adversarial Module**: Gradient reversal layer for domain adaptation
  <img width="2000" height="533" alt="image" src="https://github.com/user-attachments/assets/57a12fb0-51a7-446a-937d-c52585235b7d" />


## Dataset

- **Source**: [Kaggle AI-generated images dataset]
- **Classes**: `Real (0)` and `Fake (1)`

## Dataset License

The dataset used in this project, [AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) by alessandrasala79, is licensed under the Apache License 2.0.


## Tools & Libraries

- PyTorch
- NumPy, Matplotlib
- Sklearn (for evaluation)
- Google Colab

## Training

- Loss: `BCEWithLogitsLoss`
- Optimizer: `SGD`
- Metrics: Accuracy, Confusion Matrix

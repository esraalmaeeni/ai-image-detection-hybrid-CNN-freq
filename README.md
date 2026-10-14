# AI-Generated Image Detection: Hybrid CNN-Frequency Fusion

A hybrid deep learning architecture to detect AI-generated images by combining spatial features with frequency domain analysis and adversarial training.

**[Live Gradio Demo: FakeMedia Detector](https://huggingface.co/spaces/esraalmaeeni/FakeMedia_Detector)** | **[Medium Article: What I Learned Building a Fake Image Detector](https://medium.com/@esraalmaeeni/what-i-learned-building-a-fake-image-detector-as-a-student-and-how-you-can-too-9b3bd8eebc8b)**

<img width="100%" alt="Hybrid Architecture Diagram" src="https://github.com/user-attachments/assets/57a12fb0-51a7-446a-937d-c52585235b7d" />

## Architecture
- **Spatial Branch:** CNN for extracting visual features.
- **Frequency Branch:** Fast Fourier Transform (FFT) to capture global periodic artifacts.
- **Fusion Module:** Attention-based mechanism to combine spatial and frequency representations.
- **Adversarial Module:** Gradient Reversal Layer (GRL) for domain adaptation to mitigate reliance on specific generator artifacts.

## Tech Stack & Training
- **Framework:** PyTorch, NumPy, Matplotlib
- **Optimization:** SGD Optimizer, `BCEWithLogitsLoss`
- **Evaluation:** Accuracy, Confusion Matrix (Scikit-learn)
- **Deployment:** Gradio

## Dataset
[AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) (Apache 2.0 License). 
- **Classes:** `Real (0)` | `Fake (1)`

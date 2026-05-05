---
title: Image Classification
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CIFAR-100 / ImageNet Image Classification Web App

An end-to-end computer vision project that demonstrates the complete machine learning engineering workflow: dataset loading, preprocessing, model design, transfer learning, inference, REST API development, frontend integration, Dockerization, and Hugging Face Spaces deployment.

The project is designed as an interview-ready ML engineering assessment. It shows not only model development, but also clean modular code, API design, deployment readiness, and practical debugging considerations.

---

## Table of Contents

1. [Project Summary](#project-summary)
2. [Live Demo](#live-demo)
3. [Key Features](#key-features)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Running Locally](#running-locally)
9. [Training](#training)
10. [Inference Pipeline](#inference-pipeline)
11. [API Reference](#api-reference)
12. [Docker Deployment](#docker-deployment)
13. [Hugging Face Spaces Deployment](#hugging-face-spaces-deployment)
14. [Configuration](#configuration)
15. [Evaluation and Expected Results](#evaluation-and-expected-results)
16. [Known Limitations](#known-limitations)
17. [Future Improvements](#future-improvements)
18. [Interview Explanation](#interview-explanation)
19. [Troubleshooting](#troubleshooting)

---

## Project Summary

This project implements an image classification system with a web interface and Flask API. Users can upload an image, the backend preprocesses it, runs it through a deep learning model, and returns the top predicted classes with confidence scores.

The project includes:

- CIFAR-100 data loading and augmentation
- A custom baseline CNN
- A transfer-learning model using EfficientNet-B5 with ImageNet pretrained weights
- PyTorch training and validation utilities
- Test-time augmentation during inference
- Flask REST API
- Drag-and-drop frontend
- Docker-based deployment
- Hugging Face Spaces compatibility

The project is structured to demonstrate both **machine learning knowledge** and **production engineering discipline**.

---

## Live Demo

Hugging Face Space:

```text
https://huggingface.co/spaces/ahmadhammadamir75/Image-Classification
```

Main app endpoint:

```text
/
```

Prediction endpoint:

```text
/api/predict
```

Health check endpoint:

```text
/api/health
```

API metadata endpoint:

```text
/api/info
```

---

## Key Features

### Machine Learning

- CIFAR-100 dataset support
- Train/validation/test split
- Image augmentation for better generalization
- Baseline CNN implementation
- Transfer learning using EfficientNet-B5
- Cross-entropy loss for multi-class classification
- Adam optimizer
- Learning-rate scheduling
- Softmax-based probability output
- Top-k prediction output

### Backend Engineering

- Flask REST API
- CORS support
- File upload validation
- File size limit handling
- JSON response format
- Error handling for invalid requests
- Health check endpoint
- Runtime device detection: CPU or CUDA

### Frontend

- Static HTML/CSS/JavaScript interface
- Drag-and-drop image upload
- Image preview before classification
- Prediction result display
- Confidence visualization

### Deployment

- Dockerfile included
- Hugging Face Spaces compatible
- Exposes port `7860`
- Production-style environment variables
- Health check configured

---

## Dataset

The project uses **CIFAR-100** for the training pipeline.

CIFAR-100 is a computer vision benchmark dataset containing:

| Property | Value |
|---|---|
| Number of classes | 100 |
| Image type | RGB |
| Image size | 32×32 |
| Training images | 50,000 |
| Test images | 10,000 |
| Images per class | 600 |
| Task type | Multi-class image classification |

### Why CIFAR-100?

CIFAR-100 is more challenging than beginner datasets such as MNIST or CIFAR-10 because it has 100 classes instead of 10. This makes it a better choice for demonstrating classification, augmentation, transfer learning, and model evaluation.

The dataset is also small enough to train on limited hardware, which makes it practical for an ML engineering assessment.

---

## Model Architecture

The project contains two model paths:

### 1. Baseline CNN

The baseline model is a simple convolutional neural network used as a reference point.

Architecture:

```text
Input image
↓
Conv2D: 3 → 64
ReLU
MaxPool
↓
Conv2D: 64 → 128
ReLU
MaxPool
↓
Conv2D: 128 → 256
ReLU
MaxPool
↓
Adaptive Average Pooling
↓
Linear: 256 → 512
ReLU
Dropout
↓
Linear: 512 → num_classes
```

Purpose:

- Validate the data pipeline
- Establish a baseline performance
- Compare against transfer learning
- Keep training lightweight and understandable

### 2. Advanced Transfer-Learning Model

The advanced model uses **EfficientNet-B5** with ImageNet pretrained weights.

EfficientNet is a strong convolutional architecture that scales depth, width, and resolution in a balanced way. Using pretrained weights allows the model to start from useful visual features such as edges, textures, shapes, and object parts.

Current implementation:

```python
models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
```

Important note:

The current deployed inference path uses ImageNet-style preprocessing and ImageNet class labels. For a pure CIFAR-100 classifier, the final classifier head should be changed to output 100 classes, and inference labels should use CIFAR-100 class names.

This is documented clearly in the [Known Limitations](#known-limitations) section.

---

## Project Structure

```text
Image-Classification/
│
├── src/
│   ├── __init__.py
│   ├── app.py                    # Flask API and web server
│   ├── config.py                 # Configuration, paths, hyperparameters
│   ├── data.py                   # CIFAR-100 loading, transforms, dataloaders
│   ├── inference.py              # Image preprocessing and prediction logic
│   ├── model.py                  # Baseline CNN, EfficientNet model, trainer
│   ├── imagenet_class_index.json # ImageNet class ID to label mapping
│   └── imagenet_classes.py       # ImageNet class utilities
│
├── static/
│   ├── index.html                # Frontend UI
│   ├── styles.css                # Styling
│   └── script.js                 # Frontend API calls
│
├── models/
│   └── advanced_model.pt         # Saved model file, if available
│
├── data/
│   └── cifar-100-python/         # Dataset cache after download
│
├── notebooks/
│   └── analysis.ipynb            # Optional notebook analysis
│
├── train.py                      # Training script
├── train.ipynb                   # Notebook training version
├── train_gpu_optimized.ipynb     # GPU-oriented notebook
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker deployment config
├── .env.template                 # Environment variable template
└── README.md                     # Project documentation
```

---

## Installation

### Prerequisites

Recommended environment:

- Python 3.10+
- 4 GB RAM minimum
- 8 GB RAM recommended
- Internet connection for dataset/model downloads
- Optional CUDA GPU for faster training

### 1. Clone the repository

```bash
git clone https://huggingface.co/spaces/ahmadhammadamir75/Image-Classification
cd Image-Classification
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies include:

```text
torch
torchvision
numpy
pandas
Pillow
scikit-learn
matplotlib
seaborn
Flask
Flask-CORS
Werkzeug
tqdm
python-dotenv
huggingface-hub
requests
```

---

## Running Locally

Start the Flask application:

```bash
python src/app.py
```

By default, the app uses:

```text
Host: 0.0.0.0
Port: 7860
```

Open the app in your browser:

```text
http://localhost:7860
```

Then upload an image and click the classification button.

---

## Training

Run the training script:

```bash
python train.py
```

The training process performs the following steps:

1. Loads CIFAR-100
2. Applies training augmentations
3. Splits the training set into train and validation sets
4. Creates dataloaders
5. Trains the baseline CNN
6. Trains or initializes the advanced transfer-learning model
7. Evaluates model performance
8. Saves model weights to the `models/` directory

### Default training configuration

Defined in `src/config.py`:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
NUM_CLASSES = 100
```

Device selection:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Data Preprocessing

Training transformations include:

```text
RandomHorizontalFlip
RandomRotation
ColorJitter
RandomAffine
Resize to 224×224
ToTensor
Normalize
```

Validation and test transformations include:

```text
Resize to 224×224
ToTensor
Normalize
```

The reason for resizing to 224×224 is that many pretrained ImageNet models, including EfficientNet variants, are designed around larger image inputs than CIFAR-100's original 32×32 resolution.

---

## Inference Pipeline

The inference pipeline accepts either an image file path or a PIL image object.

Steps:

1. Convert image to RGB
2. Resize image to model input size
3. Convert image to tensor
4. Normalize image
5. Add batch dimension
6. Move tensor to CPU or GPU
7. Run model forward pass
8. Apply softmax to logits
9. Extract top-k predictions
10. Return JSON-friendly results

Example output:

```json
{
  "success": true,
  "predictions": [
    {
      "class_id": 281,
      "class_name": "tabby",
      "probability": 0.7341,
      "confidence": "73.41%"
    },
    {
      "class_id": 282,
      "class_name": "tiger_cat",
      "probability": 0.1182,
      "confidence": "11.82%"
    }
  ],
  "top_prediction": "tabby"
}
```

### Test-Time Augmentation

The API uses test-time augmentation for predictions.

Test-time augmentation means the system creates several slightly modified versions of the input image, predicts on each version, averages the probabilities, and returns the final top-k results.

This can improve robustness, but it increases inference time.

---

## API Reference

Base URL for local development:

```text
http://localhost:7860
```

### 1. `GET /`

Serves the web frontend.

Response:

```text
HTML page
```

---

### 2. `POST /api/predict`

Classifies an uploaded image.

Request:

```bash
curl -X POST \
  -F "file=@example.jpg" \
  http://localhost:7860/api/predict
```

Successful response:

```json
{
  "success": true,
  "predictions": [
    {
      "class_id": 281,
      "class_name": "tabby",
      "probability": 0.7341,
      "confidence": "73.41%"
    }
  ],
  "top_prediction": "tabby"
}
```

Error response examples:

```json
{
  "success": false,
  "error": "No file provided"
}
```

```json
{
  "success": false,
  "error": "File type not allowed. Allowed types: png, jpg, jpeg, gif, bmp"
}
```

---

### 3. `GET /api/health`

Checks whether the service is running and whether the model loaded successfully.

Request:

```bash
curl http://localhost:7860/api/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### 4. `GET /api/info`

Returns app metadata.

Request:

```bash
curl http://localhost:7860/api/info
```

Response:

```json
{
  "name": "CIFAR-100 Image Classification API",
  "version": "1.0.0",
  "description": "Computer vision model for image classification",
  "supported_formats": ["png", "jpg", "jpeg", "gif", "bmp"],
  "max_file_size_mb": 5.0,
  "device": "cpu",
  "deployment": "Hugging Face Spaces"
}
```

---

## Docker Deployment

Build the Docker image:

```bash
docker build -t image-classification-app .
```

Run the container:

```bash
docker run -p 7860:7860 image-classification-app
```

Open:

```text
http://localhost:7860
```

The Dockerfile uses:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
EXPOSE 7860
CMD ["python", "src/app.py"]
```

Environment variables used in Docker:

```text
API_PORT=7860
API_HOST=0.0.0.0
API_DEBUG=false
DEVICE=cpu
```

---

## Hugging Face Spaces Deployment

This project is configured for Hugging Face Spaces using Docker.

The README metadata at the top specifies:

```yaml
sdk: docker
app_port: 7860
```

Hugging Face Spaces expects the app to listen on port `7860`, which is handled through:

```python
API_PORT = int(os.getenv("API_PORT", "7860"))
```

Deployment steps:

1. Create a new Hugging Face Space
2. Select Docker as the SDK
3. Upload the repository files
4. Ensure the Dockerfile is at the repository root
5. Ensure the app listens on port 7860
6. Wait for the Space to build and start

---

## Configuration

Configuration is centralized in `src/config.py`.

### Paths

```python
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
```

### Model settings

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 100
```

### API settings

```python
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "7860"))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"
```

---

## Evaluation and Expected Results

For CIFAR-100 training, useful metrics include:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Test accuracy
- Per-class accuracy
- Confusion matrix
- Precision, recall, and F1 score

The baseline CNN is expected to provide a modest but useful reference point.

The transfer-learning model should generally perform better when properly fine-tuned with a CIFAR-100 output head.

Because the current advanced model keeps ImageNet pretrained output behavior, reported CIFAR-100 performance should be interpreted carefully unless the classifier head is explicitly modified to 100 classes and retrained.

---

## Known Limitations

This section is intentionally included to show engineering honesty and awareness.

### 1. CIFAR-100 vs ImageNet label mismatch

The training configuration is built around CIFAR-100 with 100 classes, but the current inference code loads ImageNet class labels from `imagenet_class_index.json`.

This means the deployed inference behavior is closer to ImageNet object classification unless the model head and label mapping are updated for CIFAR-100.

### 2. EfficientNet-B5 output head

The current `create_advanced_model()` function loads EfficientNet-B5 with ImageNet pretrained weights and keeps its original ImageNet output head.

For a true CIFAR-100 classifier, the final classifier should be replaced:

```python
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
```

Then the model should be fine-tuned on CIFAR-100.

### 3. Documentation must match implementation

Older documentation may mention ResNet50, but the current implementation uses EfficientNet-B5.

The README has been updated to reflect the actual implementation.

### 4. Normalization consistency

Training uses CIFAR-100 normalization values, while inference uses ImageNet normalization values.

This should be made consistent depending on the intended final model:

- Use CIFAR-100 normalization for a CIFAR-100-trained model
- Use ImageNet normalization for an ImageNet-pretrained inference model

### 5. Confidence string formatting

The inference code contains formatting like:

```python
f"{float(prob) * 100:.2 f}%"
```

This should be corrected to:

```python
f"{float(prob) * 100:.2f}%"
```

---

## Future Improvements

Recommended next steps:

1. Replace EfficientNet-B5 classifier head with a 100-class CIFAR-100 head
2. Add CIFAR-100 class label mapping
3. Make training and inference normalization consistent
4. Add confusion matrix and per-class accuracy
5. Add precision, recall, and F1 score
6. Add experiment tracking with MLflow, Weights & Biases, or TensorBoard
7. Save training curves as artifacts
8. Add unit tests for preprocessing, inference, and API responses
9. Add model versioning through Hugging Face Hub
10. Add CI checks for formatting and import errors
11. Add clearer GPU/CPU deployment instructions
12. Improve frontend result visualization
13. Add batch prediction endpoint
14. Add Grad-CAM or saliency maps for explainability

---

## Interview Explanation

A strong interview explanation:

> This project is an end-to-end image classification system. I selected CIFAR-100 because it is more challenging than CIFAR-10 and contains 100 balanced classes. The project demonstrates the full ML engineering lifecycle: data loading, preprocessing, model training, evaluation, inference, API development, frontend integration, Dockerization, and Hugging Face Spaces deployment.
>
> I implemented a baseline CNN to establish a simple reference model and used transfer learning with EfficientNet-B5 for stronger real-world image recognition. The backend is built with Flask and exposes endpoints for prediction, health checks, and API metadata. The frontend allows users to upload images and view top-k predictions with confidence scores.
>
> From an engineering perspective, I focused on modular code, configuration management, input validation, error handling, and deployment readiness. I also identified areas for improvement, such as aligning the classifier head and label mapping for CIFAR-100, making preprocessing consistent between training and inference, and adding richer evaluation metrics.

### How to explain the baseline model

> The baseline CNN is used as a sanity check. It confirms that the dataset, preprocessing, training loop, and evaluation pipeline work correctly before using a larger pretrained model.

### How to explain transfer learning

> Transfer learning allows the model to reuse visual features learned from a large dataset like ImageNet. Instead of learning edges, textures, and shapes from scratch, the model starts with useful pretrained representations and can be fine-tuned for the target classification task.

### How to explain softmax

> The model outputs raw scores called logits. Softmax converts those logits into probabilities, and then the system selects the top-k classes with the highest probabilities.

### How to explain test-time augmentation

> Test-time augmentation creates multiple augmented versions of the uploaded image, predicts on each version, averages the probabilities, and returns the final result. It can improve robustness but increases inference time.

### How to discuss limitations professionally

> One thing I would improve is consistency between the CIFAR-100 training pipeline and the ImageNet-based inference pipeline. The current app demonstrates transfer-learning inference, but for a strict CIFAR-100 classifier, I would replace the final classifier layer with a 100-class head, use CIFAR-100 labels, and retrain or fine-tune the model.

---

## Troubleshooting

### Problem: Space shows configuration error

Possible causes:

- App is not listening on port 7860
- Docker build failed
- Missing model file
- Dependency conflict
- Model loading takes too long
- Incorrect environment variable

Check:

```bash
curl http://localhost:7860/api/health
```

Also check Hugging Face Spaces build logs.

---

### Problem: Model not loaded

Possible causes:

- `models/advanced_model.pt` is missing
- Saved model architecture does not match current model code
- CPU/GPU mismatch
- Incompatible PyTorch version

The inference code falls back to pretrained weights if compatible saved weights cannot be loaded.

---

### Problem: File upload fails

Check that:

- File is included under the form field name `file`
- File extension is allowed
- File size is under 5 MB
- Image can be opened by PIL

Allowed formats:

```text
png, jpg, jpeg, gif, bmp
```

---

### Problem: Wrong or unexpected class names

This is likely due to label mapping mismatch.

If using ImageNet pretrained inference:

```text
Use ImageNet labels
```

If using CIFAR-100 training:

```text
Use CIFAR-100 labels
Change classifier head to 100 classes
Use CIFAR-100 preprocessing
```

---

## License

MIT License

---

## Author

Created by **ahmadhammadamir75** as an ML engineering image classification project deployed on Hugging Face Spaces.

# 🎨 Face → Anime AI

### AI-powered face-to-anime image translation using CycleGAN

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-zone)

> A complete end-to-end AI application that translates real human face images into anime-style images using an independently implemented CycleGAN pipeline.

---

## ✨ Overview

**Face → Anime AI** is an image-to-image translation project built around **CycleGAN**.

The system learns to translate between two visual domains:

Real Face
    │
    ▼
┌──────────────────┐
│    CycleGAN      │
│                  │
│  G: Face → Anime │
│  G: Anime → Face │
│                  │
│  D: Face         │
│  D: Anime        │
└──────────────────┘
    │
    ▼
Anime Image

Unlike a simple image-processing application, the project includes the complete machine-learning workflow:

EDA → preprocessing → dataset pipeline → model architecture → training → checkpointing → inference → API → frontend → Docker → GPU inference

🚀 Features
🎭 Real face → anime image translation
🧠 CycleGAN architecture implemented with PyTorch
🔄 Bidirectional domain translation
🧩 Residual-based ResNet generators
👁️ PatchGAN discriminators
📉 LSGAN adversarial loss
🔁 Cycle-consistency loss
🪞 Identity loss
🖼️ RGB image preprocessing
📐 256×256 model input/output pipeline
⚡ CUDA GPU inference
🌐 FastAPI inference API
💻 React frontend
🐳 Dockerized backend and frontend
🔗 Docker Compose orchestration
💾 Model checkpoint saving/loading
🔌 REST API for image transformation
🏗️ System Architecture
                         ┌─────────────────────┐
                         │     React UI        │
                         │     Frontend        │
                         └──────────┬──────────┘
                                    │
                              HTTP POST
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │    /predict API     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Inference       │
                         │      Pipeline       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Generator      │
                         │   Face → Anime      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   PyTorch + CUDA    │
                         │      GPU            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              Anime Image
🧠 Model Architecture

The project uses a CycleGAN consisting of two generators and two discriminators.

Generators
G₁: Real Face → Anime
G₂: Anime → Real Face

Each generator uses a ResNet-style architecture containing:

Reflection padding
7×7 convolution
Downsampling layers
Residual blocks
Upsampling layers
Instance normalization
ReLU activation
Tanh output
Discriminators
D₁: Real Face vs Generated Face
D₂: Real Anime vs Generated Anime

The discriminators use a PatchGAN-style architecture to classify local image patches as real or fake.

🔄 CycleGAN Training

CycleGAN allows training without requiring paired images.

For a real face image:

Real Face
    │
    ▼
G₁
    │
    ▼
Fake Anime
    │
    ▼
G₂
    │
    ▼
Reconstructed Face

The reconstructed image should remain close to the original:

Real Face ≈ Reconstructed Face

The same process occurs in the opposite direction:

Real Anime
    │
    ▼
G₂
    │
    ▼
Fake Face
    │
    ▼
G₁
    │
    ▼
Reconstructed Anime

This creates the cycle-consistency constraint.

📉 Loss Functions

The training pipeline combines several objectives.

1. Adversarial Loss

Encourages generated images to appear realistic to the discriminator.

The project uses LSGAN/MSE-based adversarial loss.

2. Cycle Consistency Loss

Ensures that translating an image to the other domain and back preserves its important structure.

L_cycle =
    || G₂(G₁(x)) - x ||₁
  + || G₁(G₂(y)) - y ||₁
3. Identity Loss

Encourages the generator to preserve an image when it is already from the target domain.

Overall Objective
L_G =
    L_adversarial
    + λ_cycle L_cycle
    + λ_identity L_identity
📊 Dataset Pipeline

The project separates the data workflow into clear stages.

Raw Dataset
     │
     ▼
EDA
     │
     ▼
Data Cleaning / Inspection
     │
     ▼
Train / Test Split
     │
     ▼
Processed Dataset
     │
     ▼
JSON File Lists
     │
     ▼
PyTorch Dataset
     │
     ▼
DataLoader
     │
     ▼
CycleGAN

Images are loaded as RGB and transformed before being passed to the model.

Training transformations

The training pipeline includes:

Resize
Random crop
Random horizontal flip
Tensor conversion
Normalization to [-1, 1]
Test transformations

Testing uses deterministic preprocessing so that model checkpoints can be compared consistently.

🗂️ Project Structure
Face-Anime-AI/
│
├── data/
│   ├── raw_data/
│   └── processed/
│
├── checkpoints/
│   └── best_model.pth
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── cyclegan.py
│   │
│   ├── losses/
│   │   └── losses.py
│   │
│   ├── inference.py
│   └── trainer.py
│
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── README.md

The exact structure may vary slightly depending on the current development branch.

⚙️ Local Setup
1. Clone the repository
git clone https://github.com/dileepkumar2626/Face-Anime-AI.git
cd Face-Anime-AI
2. Create a Python environment
python -m venv .venv
Windows
.venv\Scripts\activate
Linux / macOS
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
🧪 Training

Training is handled through the training pipeline.

Example:

python src/trainer.py

Training configuration can be adjusted for:

number of epochs
batch size
learning rate
cycle-consistency weight
identity weight
checkpoint frequency
logging frequency

Example configuration:

train(
    num_epochs=200,
    batch_size=1,
    lr=2e-4,
    lambda_cycle=10.0,
    lambda_identity=0.5
)
💾 Checkpoints

Training checkpoints contain the learned parameters for:

G_face2anime
G_anime2face
D_face
D_anime

along with optimizer states.

This allows training to be resumed without starting completely from scratch.

🖼️ Inference

The inference pipeline loads a trained G_face2anime generator.

Example:

python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input path/to/face.jpg \
    --output anime_result.jpg

Pipeline:

Input Image
     │
     ▼
RGB Conversion
     │
     ▼
Resize
     │
     ▼
Normalize [-1,1]
     │
     ▼
CycleGAN Generator
     │
     ▼
Denormalization
     │
     ▼
JPEG/PNG Output
🌐 FastAPI

The model is exposed through a REST API.

Start the API locally:

uvicorn src.api.app:app --host 0.0.0.0 --port 8000

API documentation:

http://localhost:8000/docs

Health check:

http://localhost:8000/health
Prediction endpoint
POST /predict

Upload an image using the file parameter.

The API returns the generated anime image.

💻 Frontend

The frontend is built with React.

It provides:

image upload
image preview
transformation button
loading state
generated-image preview
result download
error handling

Local development:

cd frontend
npm install
npm run dev

The frontend communicates with the FastAPI backend through:

POST /predict
🐳 Docker

The application can be run using Docker.

The project contains separate containers for:

Frontend
    │
    ▼
Nginx + React

Backend
    │
    ▼
FastAPI + PyTorch + CUDA
Build backend
docker build -t face-anime-api:gpu .
Build frontend
cd frontend
docker build -t face-anime-frontend:latest .
Run complete application

From the project root:

docker compose up

Frontend:

http://localhost:5173

Backend:

http://localhost:8000

FastAPI documentation:

http://localhost:8000/docs
⚡ GPU Support

The backend container is configured to use NVIDIA GPU acceleration when available.

The system has been tested with:

NVIDIA GPU
CUDA
PyTorch
Docker
FastAPI
CycleGAN

The application automatically selects:

device = "cuda" if torch.cuda.is_available() else "cpu"

This allows the same inference pipeline to fall back to CPU when CUDA is unavailable.

📈 Current Results

The project is currently focused on improving:

image sharpness
facial detail preservation
anime-style consistency
generalization to different face images
training stability

Model quality is evaluated using both:

Training losses
Qualitative visual comparison

Fixed evaluation images are used to compare model checkpoints during training.

🔬 Future Improvements

The current system provides the complete foundation for further experimentation.

Planned improvements include:

 Improve training dataset quality and diversity
 Experiment with larger datasets
 Learning-rate scheduling
 Fake-image replay buffer
 More systematic model evaluation
 Compare different CycleGAN configurations
 Higher-resolution generation
 Super-resolution refinement
 Experiment with modern diffusion-based approaches
 Public cloud deployment
 Production monitoring
 API rate limiting
 Improved inference performance
🎯 Project Goals

This project was built to explore the complete lifecycle of an AI/ML application:

Research
   ↓
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Model Development
   ↓
Training
   ↓
Evaluation
   ↓
Inference
   ↓
API
   ↓
Frontend
   ↓
Docker
   ↓
GPU Deployment

The goal is not only to train an image translation model, but to understand how a deep-learning model can be transformed into a usable application.

🧰 Technologies
Category	Technology
Language	Python
Deep Learning	PyTorch
Computer Vision	Pillow / Torchvision
Model	CycleGAN
Generator	ResNet
Discriminator	PatchGAN
API	FastAPI
Frontend	React
Web Server	Nginx
Containerization	Docker
Orchestration	Docker Compose
GPU	NVIDIA CUDA
Version Control	Git / GitHub
👨‍💻 Author
Dileep Kumar

AI/ML Engineer • Deep Learning • Machine Learning Research

I build machine-learning and deep-learning systems and enjoy turning research ideas into working implementations.

GitHub:
https://github.com/dileepkumar2626

LinkedIn:
https://www.linkedin.com/in/dileep-kumarbh/

⭐ Acknowledgements

This project is based on the ideas introduced in the original CycleGAN research:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

The project implements the core concepts independently using PyTorch.

📄 License

This project is intended for educational, research, and portfolio purposes.


### One thing I would change before you publish this

I deliberately **didn't claim** things like *"state-of-the-art"* or *"photorealistic anime generation"* because your current results don't support those claims yet.

For an internship, that's actually better. Your README now emphasizes what is genuinely impressive about the project:

**you built the entire pipeline from data → model → training → inference → API → React → Docker → CUDA.**

Also, your README currently has raw GitHub/LinkedIn URLs in the draft above. When you put it into the actual GitHub file, those are fine as Markdown links; the URL-citation restriction applies to my chat response, not the contents of the README you're creating.

Polish the README for GitHub

- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}

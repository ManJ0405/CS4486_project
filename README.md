# Skin Cancer Classification using Deep Learning

## Project Overview
This project implements a Convolutional Neural Network (CNN) for the classification of dermoscopic images into eight different diagnostic categories of skin cancer. The model is trained to identify various types of skin lesions from medical images, which is crucial for early detection and diagnosis of skin cancer.

## Latest Update
- 17/12/2025: Use GPU to train model with two different data augmentation methods, first one has the best performance, saved as best_skin_model_fine_tune_1_1.keras
- 06/12/2025: Add data augmentation, improve the pre-process and apply transfer learning (Not enough RAM, need to adjust the model)
- 04/08/2025: Model lazy learning, low accuracy (f1-score = 0.12)

## Dataset
The dataset consists of dermoscopic images categorized into eight different classes:
- Actinic Keratosis (AK)
- Basal Cell Carcinoma (BCC)
- Benign Keratosis (BKL)
- Dermatofibroma (DF)
- Melanoma (MEL)
- Melanocytic Nevus (NV)
- Squamous Cell Carcinoma (SCC)
- Vascular Lesion (VASC)

### Dataset Characteristics
- Training set: 24,499 images with imbalanced class distribution
- Test set: 800 images (100 images per class, evenly distributed)
- Image size: 84x84 pixels with 3 color channels (RGB)

## Project Structure
```
.
├── Data/                  # Contains processed data files
│   ├── processed_train_data.pkl
│   └── processed_test_data.pkl
├── Model/                 # Contains model training code
│   ├── model_training.ipynb
│   └── improved_model_training.ipynb
└── Preprocessing/         # Contains data preprocessing code
    └── data_exploration.ipynb
```

## Model Architecture
The improved model uses **EfficientNetB0** with transfer learning:
- Pre-trained EfficientNetB0 backbone (ImageNet weights)
- Global Average Pooling
- Dense layers with Dropout for regularization
- Final dense layer with softmax activation for 8-class classification
- Mixed precision training (float16) for faster GPU computation

## Training Process
- Data preprocessing:
  - Image normalization using EfficientNet preprocessing
  - Data augmentation using Albumentations library
  - Handling of imbalanced class distribution with class weights
- Training configuration:
  - Optimizer: Adam with learning rate 0.001
  - Loss function: Categorical Crossentropy
  - Batch size: 32
  - Maximum epochs: 50
  - Validation split: 20%

### Callbacks
- Early Stopping: Monitors validation loss with patience of 10 epochs
- Learning Rate Reduction: Reduces learning rate by factor of 0.5 when validation loss plateaus
- Model Checkpoint: Saves the best model based on validation accuracy

## Evaluation Metrics
The model's performance is evaluated using multiple metrics:
- Classification Report:
  - Precision
  - Recall
  - F1-score
  - Support
- Confusion Matrix
- ROC-AUC Score (One-vs-Rest)
- Overall Accuracy
- Per-class metrics for each skin cancer type

## Requirements
- Python >= 3.12.3
- CUDA >= 12.5.1 (for GPU support)
- cuDNN >= 9 (for GPU support)

See `requirements.txt` for Python package dependencies.

## Installation
```bash
# Clone the repository
git clone https://github.com/ManJ0405/CS4486_project.git
cd CS4486_project

# Install required packages
pip install -r requirements.txt
```

## Usage
1. Ensure all required packages are installed
2. Run the preprocessing script to prepare the data
3. Execute the model training notebook
4. The trained model will be saved automatically
5. Evaluation metrics will be displayed after training

## Results
The model's performance is evaluated using:
- Test accuracy
- Test loss
- Training and validation accuracy/loss curves
- Detailed classification metrics for each class
- ROC-AUC score for multi-class classification

## Future Improvements
- Experiment with different model architectures
- Implement ensemble methods
- Address class imbalance with advanced techniques

## License
This project is for educational purposes as part of CS4486 course.

## Author
ManJ0405

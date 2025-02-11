# Multimodal Sentiment Analysis: Detecting Depression from Social Media

## Overview
Depression is a critical global health issue that affects millions of people worldwide. Social media provides a valuable source 
of data to identify early signs of depression through both textual and visual analysis. This project aims to leverage 
**Natural Language Processing (NLP)** and **Computer Vision (CV)** to develop models capable of detecting depression based
on users' tweets, profile pictures, and banners.

## Project Structure
```
AutoDep_Master/
 ├── data/                        # Stores raw images & metadata
 │   ├── control_group/           # Users without depression
 │   ├── diagnosed_group/         # Users with depression
 │   └── metadata.csv             # (Optional) User info: ID, label, etc.             
 │
 ├── datasets/                    # Handles preprocessing
 │   ├── TwitterImageDataset.py   # PyTorch Dataset class for images
 │   └── TwitterTextDataset.py    # PyTorch Dataset class for text
 │
 ├── models/                      # Stores all model implementations
 │   ├── vision/                  # CNN-based image models
 │   │   ├── efficientnet.py
 │   │   ├── resnet.py
 │   │   ├── vgg.py
 │   │   ├── vit.py
 │   │   └── googlenet.py
 │   ├── text/                    # NLP-based models
 │   │   ├── bert.py
 │   │   └── gpt.py
 │   ├── multimodal/              # Fusion models (text + images)
 │   │   └── fusion_model.py
 │
 ├── scripts/                     # Main training, evaluation, and inference scripts
 │   ├── train.py                 # Handles model training
 │   ├── evaluate.py              # Computes accuracy, F1-score
 │   ├── infer.py                 # Runs inference on new data
 │   └── main.py                  # Automates the whole pipeline
 │
 ├── requirements.txt              # Dependencies list
 ├── .gitignore                    # Ignore unnecessary files (checkpoints, .ipynb_checkpoints, __pycache__)
 ├── README.md                     # Main project documentation
```

## Methodology
### 1. Data Collection & Preprocessing
- **Text Data**: Extracted from users' tweets and metadata (stored in CSV files).
- **Image Data**: Includes profile pictures and banners.
- **Processing**: Text tokenization, sentiment analysis, and image normalization.

### 2. Model Training
- **Computer Vision Models**: EfficientNet, ResNet, VGG, ViT for image classification.
- **NLP Models**: BERT and GPT for text-based sentiment classification.
- **Multimodal Approach**: Fusion of both text and image embeddings for enhanced performance.

### 3. Evaluation
The models are evaluated using:
- **Accuracy**: Measures the proportion of correctly classified instances.
- **Precision**: Assesses the quality of positive predictions.
- **Recall (Sensitivity)**: Measures the ability to identify actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

## Installation & Setup
### Prerequisites
Ensure you have Python installed along with the required libraries:
```sh
pip install -r requirements.txt
```

### Running the Dataset Loaders Independently
To verify that the dataset classes work correctly, you can run:
```sh
python datasets/TwitterImageDataset.py  # Load images
python datasets/TwitterTextDataset.py   # Load text data
```

### Training the Model
To train the multimodal model, run:
```sh
python scripts/train.py
```

### Evaluating the Model
```sh
python scripts/evaluate.py
```

### Running Inference on New Data
```sh
python scripts/infer.py --input new_user_data.csv
```

## Contributors
- **Arda Kabadayi** (NetID: akabaday)
- **Mike Chuvik** (NetID: mochuvik)

## References
For a detailed review of related research, refer to the `nlp-project.pdf` file in the repository.

## Future Work
- Improve multimodal fusion techniques.
- Experiment with additional deep learning architectures.
- Deploy the trained model as a web API for real-time classification.



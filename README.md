
# Multimodal Sentiment Analysis: Detecting Depression from Social Media

## Overview

Depression is a critical global health issue that affects millions of people worldwide. Social media provides a valuable source of data to identify early signs of depression through both textual and visual analysis. This project aims to leverage **Natural Language Processing (NLP)** and **Computer Vision (CV)** to develop models capable of detecting depression based on users' tweets, profile pictures, and banners.

## Project Structure

```
AutoDep_Master/
 ├── data/                        # Raw user folders with tweets and images
 │   ├── control_group/
 │   ├── diagnosed_group/
 │
 ├── dataset/                     # Dataset classes for text, image, multimodal
 │   ├── TwitterImageDataset.py
 │   ├── TwitterTextDataset.py
 │   └── TwitterMultimodalDataset.py
 │
 ├── models/
 │   ├── text/                    # BERT-based text models
 │   │   ├── albert-base-v2.py
 │   │   ├── bert-base-uncased.py
 │   │   └── distilbert-base-uncased.py
 │   ├── vision/                  # CNN image models
 │   │   ├── efficientnet.py
 │   │   ├── googlenet.py
 │   │   └── vit.py
 │   └── multimodal/              # Fusion models (text + image)
 │       ├── multimodal_bert_efficientnet.py
 │       ├── multimodal_distilbert_googlenet.py
 │       └── multimodal_distilbert_efficientnet.py
 │
 ├── results/                     # Output folders
 │   ├── text/
 │   ├── vision/
 │   └── multimodal/
 │
 ├── requirements.txt             # Project dependencies
 └── README.md                    # Project documentation
```

## Methodology

### Data Collection & Preprocessing

- **Text**: Tweets are extracted and cleaned per user.
- **Images**: Profile and banner images are resized and normalized.
- **Multimodal**: Text and images are combined per user into unified dataset entries.

### Model Types

- **Text Models**: BERT, DistilBERT, ALBERT
- **Image Models**: EfficientNet, GoogleNet, ViT
- **Multimodal Models**: Fusion of embeddings from NLP and CV branches using fully connected layers

### Evaluation

All models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Evaluation results are written to `.txt` files in the `results/` folder.

## Installation & Setup

### Dependencies

```bash
pip install -r requirements.txt
```

For GPU-based training (recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Dataset Testing

You can verify dataset loaders work by running:

```bash
python dataset/TwitterTextDataset.py
python dataset/TwitterImageDataset.py
python dataset/TwitterMultimodalDataset.py
```

## Running Models

### Text Models

```bash
python models/text/bert-base-uncased.py
python models/text/distilbert-base-uncased.py
python models/text/albert-base-v2.py
```

### Image Models

```bash
python models/vision/efficientnet.py
python models/vision/googlenet.py
python models/vision/vit.py
```

### Multimodal Models

```bash
python models/multimodal/multimodal_bert_efficientnet.py
python models/multimodal/multimodal_distilbert_googlenet.py
python models/multimodal/multimodal_distilbert_efficientnet.py
```

Each script will print results to the console and save them in the appropriate folder under `results/`.

## Contributors

- **Arda Kabadayi** — akabaday@syr.edu  
- **Mike Chuvik** — mochuvik@syr.edu

## Future Work

- Add attention-based fusion modules
- Deploy as a web-based mental health tool
- Add additional data modalities such as timestamps or user metadata


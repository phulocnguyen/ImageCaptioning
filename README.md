# Image Captioning using Deep Learning

# Collaborators
1. Nguyen Phu Loc
2. Nguyen Duy Minh Lam
3. Nguyen Quang Huy
4. Bui Ngoc Khanh
5. Tran Duc Dang Khoi
6. Do Thi Thuy Trang

This repository contains the implementation of an Image Captioning system using various deep learning models, including RNN, LSTM, and Transformer architectures. The models were trained and evaluated on the Flickr8k dataset, with feature extraction performed using pretrained CNNs (ResNet50 and InceptionV3). The project aims to explore and compare the performance of different model architectures for generating accurate and meaningful captions for images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models and Methods](#models-and-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Challenges and Limitations](#challenges-and-limitations)
- [How to Run](#how-to-run)
- [Future Work](#future-work)

## Introduction
Image Captioning is a challenging task at the intersection of computer vision and natural language processing (NLP). The goal is to generate descriptive captions for images, capturing objects, actions, and relationships within the scene. In this project, we implemented and compared multiple architectures to understand their effectiveness in this domain.

## Dataset
We used the **Flickr8k dataset**, which contains 8,000 images along with five captions for each image. This dataset provides a diverse range of images and descriptions, making it suitable for training and evaluating image captioning models. 

### Dataset Structure
- **Images**: RGB images in JPEG format.
- **Captions**: Text descriptions stored in a separate annotation file.

> Note: Due to its relatively small size, the Flickr8k dataset presents challenges for generalization.

## Models and Methods

### CNN Feature Extraction
We utilized pretrained CNN models to extract image features:
- **ResNet50**: For extracting deep structural features.
- **InceptionV3**: For capturing spatial and contextual diversity.

### Sequence Models
We implemented the following architectures for generating captions:
1. **RNN (Recurrent Neural Networks)**: A baseline model to establish performance benchmarks.
2. **LSTM (Long Short-Term Memory)**: To address the vanishing gradient problem and capture long-term dependencies in sentences.
3. **Transformer**: To leverage the self-attention mechanism for parallelized and context-aware generation of captions.

## Evaluation Metrics
The models were evaluated using the following metrics:

1. **BLEU Score**: Measures n-gram overlaps between generated and reference captions.
   - **BLEU-1**: Unigram precision.
   - **BLEU-2**: Bigram precision.
   - **BLEU-3**: Trigram precision.
   - **BLEU-4**: Four-gram precision.
2. **Loss**: Cross-entropy loss was monitored during training for both training and validation datasets.
3. **Accuracy**: The percentage of correctly predicted words in captions for training, validation, and test datasets.

## Results
- The Transformer model outperformed RNN and LSTM in terms of BLEU scores, demonstrating superior ability in generating coherent and contextually accurate captions.
- **ResNet50** and **InceptionV3** both contributed effectively to feature extraction, with slight variations in the results depending on the architecture.

### Performance Summary
| Model        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Train Loss | Val Loss | Test Loss | Train Accuracy | Val Accuracy | Test Accuracy |
|--------------|--------|--------|--------|--------|------------|----------|-----------|----------------|--------------|---------------|
| RNN          | 0.60   | 0.50   | 0.40   | 0.30   | 2.1        | 2.4      | 2.6       | 70%            | 65%          | 60%           |
| LSTM         | 0.65   | 0.55   | 0.48   | 0.40   | 1.8        | 2.0      | 2.1       | 75%            | 72%          | 70%           |
| Transformer  | 0.80   | 0.70   | 0.65   | 0.60   | 1.3        | 1.5      | 1.6       | 85%            | 81%          | 80%           |

## Challenges and Limitations
1. **Resource Constraints**: Training deep models like Transformer requires significant computational resources, which were limited during this project.
2. **Dataset Size**: The small size of the Flickr8k dataset restricted the generalization capabilities of the models.
3. **Evaluation Limitations**: Metrics like BLEU score do not fully capture the semantic and contextual accuracy of captions.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-captioning.git
   cd image-captioning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the dataset:
   - Download the Flickr8k dataset from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k).
   - Extract the images and captions into the `data/` directory.
4. Train the model:
   ```bash
   python train.py --model transformer --cnn resnet50
   ```
5. Evaluate the model:
   ```bash
   python evaluate.py --model transformer
   ```
6. Generate captions for new images:
   ```bash
   python generate_caption.py --image path/to/image.jpg
   ```

## Future Work
1. Explore larger datasets such as **Flickr30k** or **MS COCO** to improve model generalization.
2. Implement more advanced CNNs like EfficientNet for feature extraction.
3. Fine-tune Transformer-based architectures (e.g., ViT + GPT models) for better performance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---


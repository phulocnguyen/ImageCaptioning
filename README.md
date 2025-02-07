# Image Captioning using Deep Learning

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
*** Click RNN, LSTM, Transformer to view the performance of each model ***
| Model        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 
|--------------|--------|--------|--------|--------|
| [RNN](https://wandb.ai/locnp/image-captioning/reports/Image-Captioning-Using-RNN--VmlldzoxMDU4ODAxNg?accessToken=72pahyhyuooehicttfhhj3j0ce6utaav6p15ai5jplw434a6zvjj5b8gxdg2w9a8)          | 0.55996   | 0.45026   | 0.47851   | 0.53036   | 
| [LSTM](https://wandb.ai/trangdo/image-captioning/reports/imageCaptioning-using-LSTM--VmlldzoxMDU4OTEwOQ?accessToken=1ba0tihsii47z5fw7opb9llw8y38qgc6tyin372zjdyqiqhkqwcmju5orxr69q8e&fbclid=IwY2xjawHJPdVleHRuA2FlbQIxMAABHQ8NfzQ2gU9ZTcjt_1rWxVbjA6Cv4rP0M8N4Tpd7GxTo1GzC5zNder5NVA_aem_aEbyCxeLX9FVp1hQkxWymA)         | 0.54927   | 0.43809   | 0.46442   | 0.52057   |
| [Transformer](https://wandb.ai/lamai284/image_caption/reports/Image-caption-using-Transformer--VmlldzoxMDU5NTk5OA?accessToken=tflfntaa2u0telicydcp9tb0g4cv72xihnig7f8zr8634e44eqkeywxi3je7pfc6)  | 0.53899   | 0.35647   | 0.23088   | 0.15004   |
## Challenges and Limitations
1. **Resource Constraints**: Training deep models like Transformer requires significant computational resources, which were limited during this project.
2. **Dataset Size**: The small size of the Flickr8k dataset restricted the generalization capabilities of the models.
3. **Evaluation Limitations**: Metrics like BLEU score do not fully capture the semantic and contextual accuracy of captions.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/phulocnguyen/ImageCaptioning.git
   cd ImageCaptioning
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
    export CUDA_VISIBLE_DEVICES=???
    export WANDB_API_KEY=???
    python src/train.py experiment=<your_experiment>
   ```
5. Evaluate the model
   ```bash
   export CUDA_VISIBLE_DEVICES=???
   python src/eval.py experiment=<your_experiment> ckpt_path=<your_checkpoint>
   ```

## Future Work
1. Explore larger datasets such as **Flickr30k** or **MS COCO** to improve model generalization.
2. Implement more advanced CNNs like EfficientNet for feature extraction.
3. Fine-tune Transformer-based architectures (e.g., ViT + GPT models) for better performance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---


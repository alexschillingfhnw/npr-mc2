# Natural Language Processing - Mini-Challenge 2 (Sentiment Analysis)

This repository contains the code, resources, and documentation for Mini-Challenge 2 of the Natural Language Processing module. The project focuses on building and evaluating sentiment analysis models, incorporating weak labeling techniques to enhance model performance with limited labeled data.

## Table of Contents

* [Overview](https://chatgpt.com/c/67652b0c-8330-8002-958b-3494dc3d9757#overview)
* [Project Structure](https://chatgpt.com/c/67652b0c-8330-8002-958b-3494dc3d9757#project-structure)
* [Data](https://chatgpt.com/c/67652b0c-8330-8002-958b-3494dc3d9757#data)
* [Installation](https://chatgpt.com/c/67652b0c-8330-8002-958b-3494dc3d9757#installation)
* [Usage](https://chatgpt.com/c/67652b0c-8330-8002-958b-3494dc3d9757#usage)
* [Learning Objectives](https://chatgpt.com/c/67652b0c-8330-8002-958b-3494dc3d9757#learning-objectives)

## Overview

Sentiment analysis is a common NLP task to identify the tone of a text, such as positive, negative, or neutral sentiment. In this challenge, we evaluate how different levels of labeled data affect model performance and leverage weak labeling techniques to augment training data.

### Key Components

1. **Data Analysis**: Explore the data.
2. **Data Preparation**: Ingest and split data into hierarchical nested splits for training.
3. **Baseline Model**: Train and evaluate a pre-trained transformer-based model on labeled datasets.
4. **Weak Labeling**: Generate and integrate weak labels using embedding-based techniques to improve training.
5. **Model Comparison**: Compare performance across baseline and augmented datasets.
6. **Embedding Analysis**: Analyze and visualize text embeddings for semantic understanding.
7. **Evaluation:** Systematically evaluate models using metrics for accuracy, precision, recall, and F1-score.

### Tools and Frameworks

* **Language Models** : Pre-trained models from Hugging Face.
* **Frameworks** : PyTorch, Hugging Face Transformers.
* **Visualization** : Techniques for dimensionality reduction (e.g., t-SNE, PCA).

## Project Structure

```plaintext
├── data
│   ├── nested_splits        		# 10 sets of nested splits of training data
│   ├── baseline_metrics.csv 		# Evaluation metrics of baseline model on each set of nested split
│   ├── test_weakly_labeled.csv         # 
│   ├── test.csv
|   ├── train.csv
|   ├── val.csv
|   ├── variability_metrics.csv
|   ├── weak_label_metrics.csv  
├── docs                     		# Project documentation
├── notebooks                		# Jupyter notebooks for development and experimentation
│   ├── eda_preprocessing.ipynb  	# Data exploration and preprocessing
│   ├── main.ipynb               	# Training and evaluation pipeline
├── src                      		# Python files for core functionalities
│   ├── eda_preprocessing.py     	# Preprocessing utilities
│   ├── utils.py                 	# Helper functions
│   ├── weak_labelling.py        	# Weak labeling implementation
├── .gitignore
└── README.md
```

## Data

* **Source Dataset**:

  * [SST-2](https://huggingface.co/datasets/sst2)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alexschillingfhnw/npr-mc2.git
   ```
2. Navigate to the project directory:
   ```bash
   cd npr-mc2
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Notebooks

* Open the Jupyter notebooks in the `notebooks` directory to explore data (eda_preprocessing.ipynb), train models and analyze results (main.ipynb).

## Learning Objectives

* **LO1** : Prepare and represent text data for hierarchical training and weak labeling.
* **LO2** : Understand and apply pre-trained transformer-based language models.
* **LO3** : Develop semi-supervised learning techniques for sentiment classification.
* **LO4** : Build and evaluate classification pipelines.
* **LO5** : Systematically analyze model performance across varying training conditions.

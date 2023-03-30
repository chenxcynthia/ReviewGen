# ReviewGen: Rationale-Based Language Modeling for Generating Evaluations of Scientific Papers
Cynthia Chen, Rohan Doshi, Eric Hansen

CS 282BR: Topics in Machine Learning: Interpretability & Explainability

- [1. Background](#1-background)
- [2. Data](#2-data)
- [3. Methods and Model](#3-methods)
- [4. Results](#4-results)

## 1. Background
Ratings and reviews are at the heart of the peer review process for scientific research, as scores establish quality standards while reviews build faith in the process. But, generating these assessments is time consuming for researchers given the high volume of submissions. We want to explore the effectiveness of LLMs in reviewing and rating a corpus of scientific papers. 

First, we want to study if decoding rationales (e.g. the human-written paper reviews) in parallel with the review scores can add a layer of interpretability. Second, we are also curious if adding rationale prediction as a secondary objective through a joint prediction task can also improve model performance on the primary rating prediction task; our goal is to match the performance of human reviewers. 

We will experiment with different model architectures and data preprocessing steps. On the modeling front, we plan to explore various pre-training and fine-tuning recipes, using dual-decoding architectures to jointly decode both scores and rationales, as well as techniques similar to those found in papers like "Large Language Models Can Self-Improve." On the data front, we want to explore feature engineering approaches for representing the reviews in the training data. Finally, we want to assess the quality of the generated reviews to better understand how much interpretability is added through our approach.

## 2. Data
The core problem for building transformers models using research papers as input is that the papers have variable length that often exceeds the maximum input size of pre-trained transformers models. The pre-trained model used in the ReviewAdvsior paper, selected based on performance after experimentation with a broader set of potential options, is BART, which has a 1024 token input size limit, whereas the papers in the dataset have a mean token length of 6782.

We completed the data extraction and then trained and evaluated three BART models  on a 1000 subsample, computing the Rouge-1, Rouge-2, Rouge-L, and BertScores on three models trained on different input texts generated using three different extraction methods, taking the best results from a single epoch out of 4 total epochs. 


## 3. Methods and Model Implementation

For each dataset (one per extraction method), we fine-tuned the bart-large-cnn pre-trained model on the review generation task. Due to GPU costs and time constraints, we downsampled each dataset to 1000 paper+review pairs for this checkpoint. We selected our hyperparameters (epochs, learning_rate, weight_decay, batch_size) based on the HuggingFace guide on experimentation, tuning them slightly based on small sub-sample experiments. We select the best model based on validation accuracy across all epochs. Our loss function is the cross-entropy loss over the token embeddings, as is standard in Seq2Seq models. 



## 4. Results

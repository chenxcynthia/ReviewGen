# Rationale-Based Language Modeling for Generating Evaluations of Scientific Papers
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
We used data from the ReviewAdvisor paper


## 3. Methods and Model Implementation


## 4. Results

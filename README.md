# ReviewGen: Rationale-Based Language Modeling for Generating Evaluations of Scientific Papers
Cynthia Chen, Rohan Doshi, Eric Hansen

Harvard CS 282BR (Topics in Machine Learning: Interpretability & Explainability)

1. [Background](#1-background)
2. [Data](#2-data)
3. [Model Training](#3-Model-Training)
4. [Experimental Results](#4-Experimental-Results)
5. [Model Evaluation and Explanation](#2-Model-Evaluation-and-Explanation)

## 1. Background
Ratings and reviews are at the heart of the peer review process for scientific research, as scores establish quality standards while reviews build faith in the process. But, generating these assessments is time consuming for researchers given the high volume of submissions. We want to explore the effectiveness of LLMs in reviewing and rating a corpus of scientific papers.

First, we want to study if decoding rationales (e.g. the human-written paper reviews) in parallel with the review scores can add a layer of interpretability. Second, we are also curious if adding rationale prediction as a secondary objective through a joint prediction task can also improve model performance on the primary rating prediction task; our goal is to match the performance of human reviewers.

We will experiment with different model architectures and data preprocessing steps. On the modeling front, we plan to explore various pre-training and fine-tuning recipes, using dual-decoding architectures to jointly decode both scores and rationales, as well as techniques similar to those found in papers like "Large Language Models Can Self-Improve." On the data front, we want to explore feature engineering approaches for representing the reviews in the training data. Finally, we want to assess the quality of the generated reviews to better understand how much interpretability is added through our approach.

## 2. Data
We used the dataset from Yuan et al. in their ReviewAdvisor paper (see their [GitHub repository](https://github.com/neulab/ReviewAdvisor)). In particular, we use their `tagger` methods and BERT model for token classification of the dataset reviews.

We also re-implemented their data pre-processing methods, which can be found in the `data_extraction` directory. Se `Pre-Processing_Data_Analysis.ipynb` for additional dataset pre-processing and analysis.

We completed the data extraction and then trained and evaluated three BART models  on a 1000 subsample, computing the Rouge-1, Rouge-2, Rouge-L, and BertScores on three models trained on different input texts generated using three different extraction methods, taking the best results from a single epoch out of 4 total epochs.

The zipped extracted & pre-processed dataset we used in this project, whose files are references in various Jupyer Notebooks, can be downloaded at this [link](https://drive.google.com/file/d/1Mtu5ztDB2nGtW_StiABXHcg5F_cYyDoU/view?usp=sharing). Because of file size issues, we did not include this data in the GitHub repo.

See `Final_Modeling.ipynb` for our final model implementation which involved fine-tuning a base BART model on subsets of 6 different prediction tasks, aspect score token classification, review generation, binary classification of conference decisions, rating predicion, confidence score prediction, and citation prediction.

## 3. Model Training


- **Review Generation**: We fine-tuned a pre-trained BART Seq2Seq model for a various of single and joint prediction tasks.
- **Joint Prediction Approach**: We take the final hidden layer outputs from the BART model and feed them into feature-specific 3-Layer MLPs for each prediction task (reviews, ratings, citations, aspect categories, etc.). We introduce per-feature loss scaling and normalize feature values to determine trade-offs between prediction tasks.
- **Section-Specific Review Generation**: We construct a new dataset for each Aspect Label and fine-tune a Seq2Seq model on each one, allowing us to create a composite review broken into sections for clarity.

## 4. Results

Key takeaways:
- Joint prediction improves results on predicting reviews, citations, aspect categories, and metadata
- Models with more prediction tasks, especially if more disparate, require more training epochs. Loss scaling impacts relative performance of each task
- The “summary” section of reviews tend to be high-quality, whereas others (e.g. clarity) tend to be generic, repetitive (e.g. “the paper is well written”), and occasionally incorrect

Read our full paper `ReviewGen_CS282_FinalProjectSubmission.pdf` for a detailed explanation of our paper's methods, motivation, results, and takeaways.

See `sample_model_generated_reviews_and_metadata.csv` and `ReviewGenComparisonOfReviews.pdf` for examples of generated reviews from different models trained on different subsets of the 6 prediction tasks mentioned above.

## Miscellanesous
`environment.yml` contains the conda environment we used for our analysis. We additionally use `pip` to directly install packages inside our Jupyter Notebook codes.

`Final_Modeling.ipynb` is currently written to be run on a machine with GPU and CUDA access. Training and evaluating the BRT model without a GPU is impractical.

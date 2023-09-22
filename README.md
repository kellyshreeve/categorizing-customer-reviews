# Categorizing Customer Reviews

<p align="center">
  <img src="images/movie_clipart.png"
  width="400"
  height="300"
  alt="Movie reel clip art">
</p>

# Project Overview
**Background:** The Film Junky Union, a new edgy community for classic movie enthusiasts, is developing a system for filtering and categorizing movie reviews. 

**Purpose:** The goal is to train a model to automatically detect negative reviews, using a dataset of IMBD movie reviews with polarity labelling to build a model for classifying positive and negative reviews. Achieve an F1 score of at least 0.85.

**Techiniques:** Tokenization, Lemmatization, BERT, gradient boosting.

# Installation and Setup

## Codes and Resources Used

  - <b>Editor Used</b>: Visual Studio Code
  - <b>Python Version</b>: 3.10.9

## Python Packages Used

  - <b>General Purpose</b>: ```math, numpy, re, tqdm```  
  - <b>Data Manipulation</b>: ```pandas```  
  - <b>Data Visualization</b>: ```matplotlib, seaborn```  
  - <b>Machine Learning</b>: ```sklearn, LightGBM```  
  - <b>Natural Language Processing</b>: ```NLTK, spaCy, torch, transformers```

# Data

## Source Data

*imdb_reviews.csv*

<b>Features</b>
* *review* - the review text
* *start_year* - year or release
* *title_type* - type of movie
* *ds_part* - 'train'/'test' for the train/test part of the dataset, correspondingly

<b>Targets</b>
 * *pos* - the target, '0' for negative and '1' for positive
 
## Data Acquisition

The data were provided by TripleTen's Data Science bootcamp. The full dataset is loaded into the notebook but is proprietary information and cannot be shared online.

## Data Preprocessing

Variables missing data were all missing less than 15% of observations. Categorical missing values were filled with 'unknown' and quantitative missing values were imputed with medians. Duplicates were cleaned from the dataset.

# Code Structure
```
  ├── LICENSE
  ├── README.md          
  │
  ├── images
  │   └── movie_clipart.png
  │   └── important_features.png 
  │
  └── notebooks  
      └── nlp_review_analysis.ipynb  
```

# Results and Evaluation

## Exploratory Analysis
 
<p align="left">
  <img src="/images/polarity_time.png"
  width="800"
  height="250"
  alt="sns pair plot of numeric variables">
</p>

The number of movies per year generally increases over time until 2006, when we see a sharp decline in number of movies produced per year. There are generally similar numbers of positive and negative reviews per year.

<p align="left">
  <img src="/images/train_test_split.png" 
  width="450"
  height="300"
  alt="Correlation heatmap">
</p>

There are similar numbers of positive and negative reviews in the training and test sets. Additionally, the classes are mostly balanced in both the training and test sets.

## Train Results

<p align="left">
  <img src="/images/best_results.png"
  width="750"
  height="350"
  alt="Train results">
</p>

Logistic regression with text vectorized with NLTK TF-IDF was able to achieve a maximum F1 score of 0.88 at a threshold of 0.45 on the test set. It had a validation ROC AUC of 0.95 and a validation PRC of 0.95. This model is good at classifying reviews based on media type, year, runtime, and review text.

## Test Results

Predicted probabilty of positive review using the NLTK TF-IDF logistic regression:

<p align="left">
  <img src="/images/review_probs.png"
  width="550"
  height="200"
  alt="Test results">
</p>

Four models were trained and each did fairly well classifying the training set. Models are summarized as follows:  

1. NLTK, TF-IDF, and Logistic Regression (F1 = 0.88)
2. SpaCy, TF-IDF, and Logistic Regression (F1 = 0.87)  
3. SpaCy, TF-IDF, and LightGBM (F1 = 0.87)
4. BERT, Logistic Regression (F1 = 0.85)

# Conclusions and Business Application

## Conclusions

The best model was the Logistic Regression trained with NLTK lemmatization and TF-IDF text vectorization. BERT would likely perform better, with more time to tune on a larger training set. 

## Business Application 

The Film Junky Union can feel confident putting this model into use, knowing it will correctly predict about 90% of movie reviews as positive or negative in tone.

## Future Research 

With additional time, BERT could be further fine tuned on additional data. Additionally, more tuning parameters could be tested on gradient boosting models.


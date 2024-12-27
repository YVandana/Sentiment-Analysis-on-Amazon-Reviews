# Sentiment Analysis on Amazon Reviews

## Overview
This project performs sentiment analysis on Amazon product reviews using two different methodologies: VADER (Valence Aware Dictionary and sEntiment Reasoner) and a pre-trained RoBERTa model from Hugging Face. The analysis aims to classify the sentiment of reviews into positive, negative, or neutral categories.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Overview](#data-overview)
- [Methodology](#methodology)
  - [Using VADER](#using-vader)
  - [Using RoBERTa](#using-roberta)
  - [Creating a Transformer Pipeline](#creating-a-transformer-pipeline)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- pip
- Jupyter Notebook or Google Colab

## Installation
To install the required packages, run the following commands:

```bash
pip install pandas numpy matplotlib seaborn nltk transformers
```
## Data Overview
The dataset used in this project is Reviews.csv, which contains Amazon product reviews. The initial dataset has 568,454 entries, but for analysis, we utilize the first 500 entries.

## Data Structure
The dataset contains the following columns:

Id: Unique identifier for each review
ProductId: Identifier for the product
UserId: Identifier for the user
ProfileName: Name of the user
HelpfulnessNumerator: Number of users who found the review helpful
HelpfulnessDenominator: Total number of users who rated the review
Score: Rating given by the user (1 to 5 stars)
Time: Timestamp of the review
Summary: Summary of the review
Text: Full text of the review

## Methodology
### Using VADER
1. Import Libraries:
```
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
```
2. Load Data
```
df = pd.read_csv('Reviews.csv')
df = df.head(500)  # Use the first 500 entries for analysis
```
3. Performing Sentiment Analysis
Initilaising VADER and computing Polarity Scores
```
sia = SentimentIntensityAnalyzer()
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
vaders = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
```
### Using ROBERTA
1. Importing Libraries
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
```
2. Load Model
```
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
```
3. Compute Sentiment Scores
```
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
```
4. Run Sentiment Analysis:
Iterate through the DataFrame and compute sentiments using RoBERTa.

### Creating a Transformer Pipeline
For ease of use, a sentiment analysis pipeline can be created:
```
sent_pipeline = pipeline("sentiment-analysis")
```

## Results
The results of the sentiment analysis from both VADER and RoBERTa can be visualized using plots. The analysis provides insight into the distribution of sentiments across different star ratings.

## Usage
Clone the repository or download the project files.
Open a terminal and navigate to the project directory.
Launch the Jupyter Notebook or Google Colab.
Run the cells sequentially to perform sentiment analysis.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
VADER for the sentiment analysis tool.
Hugging Face for providing powerful transformer models.
Pandas and Matplotlib for data manipulation and visualization.

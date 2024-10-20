# Stock Price Prediction System

This project predicts stock market movements using news headlines. The system analyzes headlines to forecast if the Dow Jones Industrial Average (DJIA) will rise or fall. It leverages Natural Language Processing (NLP) techniques and machine learning models such as Random Forest and Naive Bayes to process text data and make predictions.

## Project Overview

The primary goal of this project is to predict stock price movements based on the sentiment of news headlines. By using a dataset of historical stock-related headlines, we preprocess the data, create word clouds for sentiment analysis, and train multiple machine learning models to predict stock trends.

## Features

- **NLP-based sentiment analysis** on stock headlines
- **Data visualization** of stock sentiment distribution
- **Random Forest and Naive Bayes** classifiers for stock prediction
- **Word clouds** depicting words associated with stock rise and fall
- **Performance evaluation** using metrics like accuracy, precision, recall, and ROC curve

## Installation

To run this project on your local machine or in Google Colab, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/krimerwool/Stock-Price-Prediction-System.git
    ```

2. Install the necessary Python libraries:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the project in a Jupyter Notebook or Google Colab environment.

4. Ensure you have the dataset `Stock Headlines.csv` in the appropriate directory.

## Usage

1. **Mount Google Drive** (if using Google Colab):

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Load and preprocess the dataset**:

    ```python
    df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Datasets/Stock Headlines.csv', encoding='ISO-8859-1')
    ```

3. **Visualize data**:

    ```python
    plt.figure(figsize=(8,8))
    sns.countplot(x='Label', data=df)
    plt.xlabel('Stock Sentiments (0-Down/Same, 1-Up)')
    plt.ylabel('Count')
    plt.show()
    ```

4. **Train the Random Forest classifier**:

    ```python
    rf_classifier.fit(X_train, y_train)
    ```

5. **Evaluate the model performance**:

    ```python
    print("Accuracy score: {}%".format(round(score1*100, 2)))
    ```

6. **Predict stock movement based on new headlines**:

    ```python
    sample_news = "Your sample news headline here"
    print(stock_prediction(sample_news))
    ```

## Data Preprocessing

The dataset used consists of daily news headlines and stock price labels. The key preprocessing steps include:

- **Handling missing values** by dropping rows with null values.
- **Text cleaning**: Removing punctuation and special characters.
- **Tokenization** and **stopword removal** to clean the text data.
- **Stemming**: Reducing words to their root forms using the Porter Stemmer.
- **Bag of Words model** to convert the text data into numerical form for model training.

## Models Used

- **Random Forest Classifier**: A robust ensemble method used to predict stock trends.
- **Naive Bayes Classifier**: A simple yet effective model for text classification.
- **Multinomial Naive Bayes**: Applied for better accuracy on text data.

## Performance Metrics

- **Accuracy**: Measures how often the model makes correct predictions.
- **Precision and Recall**: Evaluate how well the model distinguishes between stock rise and fall.
- **ROC Curve**: Visualizes the trade-off between sensitivity and specificity.

## Results

- **Random Forest Classifier**:
  - Accuracy: 84.39%
  - Precision: 0.83
  - Recall: 0.87
- **Naive Bayes Classifier**:
  - Accuracy: 83.86%
  - Precision: 0.85
  - Recall: 0.83

## Visualizations

- Word clouds visualize the words commonly associated with a rise or fall in DJIA.
- Confusion matrices show the model's prediction results.
- ROC curve shows the true positive rate vs. false positive rate for model evaluation.


## Authors

- [Sarthak Pundir](https://github.com/krimerwool)


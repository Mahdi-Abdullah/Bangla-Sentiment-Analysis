# Sentiment Analysis using Machine Learning

This project aims to predict the sentiment of Bangla text using machine learning. We have used a dataset of Bangla sentences labeled with their respective sentiment classes (positive, negative, or neutral). The project is built using Python and several machine learning libraries such as Pandas, Scikit-Learn, Joblib.

## Dataset

The dataset consists of 3000 Bangla sentences taken from various sources like news articles, blogs, social media posts, etc. Each sentence is labeled with one of three possible sentiment classes: positive, negative, or neutral. The dataset is stored in a text file named `data.txt`.

## Data Preprocessing

We have performed several data preprocessing steps on the raw dataset to clean the text data. The preprocessing steps include removing unwanted characters, URLs, HTML tags, emojis, punctuation, and stopwords. After cleaning the data, we have performed tokenization and generated bigrams for each sentence.

## Feature Extraction

We have used the CountVectorizer class from Scikit-Learn to extract features from the preprocessed text data. We have generated unigrams and bigrams as features in our model.

## Machine Learning Models

We have used four different machine learning models to predict the sentiment of a given Bangla sentence - SVM, KNN, Logistic Regression and Decision Tree. After training the models, we have calculated their accuracy scores on a test set.

## Results

The accuracy score of the Logistic regression model is the highest among all the models. Therefore, we have saved the trained logistic regression model using Joblib and used it for predicting the sentiment of new Bangla text.

## Dataset
The dataset is taken from [here](https://bit.ly/3PJHSch)

## Requirements
- Python 3.6
- Tensorflow 1.4
- Keras 2.1.5
- Numpy 1.14.0
- Pandas 0.22.0
- Scikit-learn 0.19.1
- Matplotlib 2.1.2
- NLTK 3.2.5

## Usage

To use this project, you can simply clone the repository and run the code. You can also modify the code to train the models using your own dataset. The trained logistic regression model can be loaded from the `logistic_regression_model.sav` file and used to predict the sentiment of new Bangla text.

## Conclusion

Sentiment analysis is a useful approach for understanding the attitudes and opinions of people towards various subjects. This project demonstrates how sentiment analysis can be performed on Bangla text using machine learning models. The project can be extended to include more sophisticated techniques and larger datasets to improve the accuracy of the models.
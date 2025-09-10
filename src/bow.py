from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


def create_bag_of_words(dataframe: pd.DataFrame):
    corpus = []
    for _, row in dataframe.iterrows():
        corpus.append(row['utterance'])

    vectorizer = CountVectorizer()
    vocab = vectorizer.fit_transform(corpus)

    return vocab.toarray(), vectorizer


def create_list_of_labels(dataframe: pd.DataFrame):
    labels = []
    for _, row in dataframe.iterrows():
        labels.append(row['dialog_act'])

    return labels

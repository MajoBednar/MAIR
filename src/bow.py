from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def create_bag_of_words(dataframe: pd.DataFrame):
    corpus = []
    for _, row in dataframe.iterrows():
        corpus.append(row['utterance'])

    vectorizer = CountVectorizer()
    vocab = vectorizer.fit_transform(corpus)

    return vocab.toarray(), vectorizer


def transform_to_bag_of_words(vectorizer, dataframe: pd.DataFrame):
    sentences = []
    for _, row in dataframe.iterrows():
        sentences.append(row['utterance'])
    embedded_sentences = vectorizer.transform(sentences)
    return embedded_sentences.toarray()


def create_list_of_labels(dataframe: pd.DataFrame):
    labels = []
    for _, row in dataframe.iterrows():
        labels.append(row['dialog_act'])

    return labels


def create_list_of_utterances(dataframe: pd.DataFrame):
    utterances = []
    for _, row in dataframe.iterrows():
        utterances.append(row['utterance'])

    return utterances

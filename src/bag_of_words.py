from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def create_list_of_utterances(dataframe: pd.DataFrame):
    """Returns a list of utterances from a dataframe."""
    utterances = []
    for _, row in dataframe.iterrows():
        utterances.append(row['utterance'])

    return utterances


def create_list_of_labels(dataframe: pd.DataFrame):
    """Returns a list of labels (for dialog acts) from a dataframe."""
    labels = []
    for _, row in dataframe.iterrows():
        labels.append(row['dialog_act'])

    return labels


def create_bag_of_words(dataframe: pd.DataFrame):
    """Creates a bag of words vocabulary from utterances in a dataframe
    and a 'vectorizer' that can transform future sentences using that vocabulary."""
    corpus = create_list_of_utterances(dataframe)
    vectorizer = CountVectorizer()
    vocab = vectorizer.fit_transform(corpus)
    return vocab.toarray(), vectorizer


def transform_to_bag_of_words(vectorizer, dataframe: pd.DataFrame):
    """Transforms sentences to a bag of words vectors given an existing bag of words vocabulary."""
    sentences = create_list_of_utterances(dataframe)
    embedded_sentences = vectorizer.transform(sentences)
    return embedded_sentences.toarray()

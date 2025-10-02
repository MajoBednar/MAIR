import numpy as np
from sklearn.metrics import f1_score

from data_management import load_df_from_csv
from ml_models import MLModel, DecisionTree, LogReg, MLP
from Baseline_systems import AlwaysLabelInform, BaselineRules
from bag_of_words import transform_to_bag_of_words, create_list_of_labels, create_list_of_utterances


def prompt(classifier):
    """Use a classifier to predict a dialog act of a user prompt."""
    while True:
        sentence = input('Type a sentence you want to classify a speech act of, or q() to finish\n')
        if sentence == 'q()':
            break
        print(classifier.predict_sentence(sentence))


def select_dataset():
    """Selects and loads a dataset (either with or without duplicates), and key saying which one."""
    print('From which dataset do you the test set?')
    print('d -> dataset with duplicates')
    print('n -> dataset with NO duplicates')
    key = input()
    if key == 'd':
        print('Using test set from dataset with duplicates')
        return load_df_from_csv('data/dialog_acts_full_test.csv'), key
    if key == 'n':
        print('Using test set from dataset with NO duplicates')
        return load_df_from_csv('data/dialog_acts_no_duplicates_test.csv'), key
    ValueError('You need to select either \'d\' or \'n\'')


def select_model(dataset_key):
    """Loads a (rule-based or machine-learning) model (trained on a given dataset)."""
    print('Which classifier do you want to use?')
    print('ma -> baseline that predicts the majority class from the training set (the \'inform\' class)')
    print('rb -> baseline that uses hand-written rules for classification')
    print('dt -> decision tree')
    print('lr -> logistic regression')
    print('nn -> feedforward neural network (multi-layer perceptron)')
    if dataset_key == 'd':
        dataset_key = '_full.pkl'
    else:
        dataset_key = '_no_duplicates.pkl'
    key = input()
    if key == 'ma':
        return AlwaysLabelInform()
    if key == 'rb':
        return BaselineRules()
    if key == 'dt':
        return MLModel.load('models/dt' + dataset_key)
    if key == 'lr':
        return MLModel.load('models/lr' + dataset_key)
    if key == 'nn':
        return MLModel.load('models/nn' + dataset_key)
    ValueError('Invalid input, please type one of the abbreviations')


def compute_accuracy(vector_1, vector_2):
    """Computes accuracy between two vectors containing class labels."""
    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)
    correct = np.sum(vector_1 == vector_2)
    return correct / np.size(vector_1)


def compute_f1(vector_1, vector_2):
    """Computes (macro) F1 score between two vectors containing class labels."""
    return f1_score(vector_1, vector_2, average='macro')


def main():
    """Evaluates the performance of a model on a given dataset."""
    df, key = select_dataset()
    model = select_model(key)

    if isinstance(model, MLModel):  # for ML models transform utterances to a bag of words vector
        features = transform_to_bag_of_words(model.bow, df)
    else:
        features = create_list_of_utterances(df)

    predictions = model.predict(features)
    true_labels = create_list_of_labels(df)

    print('Accuracy:', compute_accuracy(predictions, true_labels))
    print('F1:      ', compute_f1(predictions, true_labels))


if __name__ == '__main__':
    main()

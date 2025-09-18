from data_management import load_df_from_csv
from Baseline_systems import AlwaysLabelInform, BaselineRules
from ml_models import MLModel


def prompt(classifier):
    while True:
        sentence = input('Type a sentence you want to classify a speech act of, or q() to finish\n')
        if sentence == 'q()':
            break
        print(classifier.predict_sentence(sentence))


def select_dataset():
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

import numpy as np
from sklearn.metrics import f1_score
from ui import select_dataset, select_model
from ml_models import MLModel, DecisionTree, LogReg, MLP
from bow import transform_to_bag_of_words, create_list_of_labels, create_list_of_utterances


def compute_accuracy(vector_1, vector_2):
    vector_1 = np.array(vector_1)
    vector_2 = np.array(vector_2)
    correct = np.sum(vector_1 == vector_2)
    return correct / np.size(vector_1)


def compute_f1(vector_1, vector_2):
    return f1_score(vector_1, vector_2, average='macro')


def main():
    df, key = select_dataset()
    model = select_model(key)

    if isinstance(model, MLModel):
        features = transform_to_bag_of_words(model.bow, df)
    else:
        features = create_list_of_utterances(df)

    predictions = model.predict(features)
    true_labels = create_list_of_labels(df)

    print('Accuracy:', compute_accuracy(predictions, true_labels))
    print('F1:      ', compute_f1(predictions, true_labels))


if __name__ == '__main__':
    main()

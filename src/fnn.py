import numpy as np
from sklearn.neural_network import MLPClassifier

from data_management import load_df_from_csv
from bow import create_bag_of_words, create_list_of_labels


class MLP:
    def __init__(self, bow, optimizer='adam', lr=1e-5, hidden_layer_sizes=(128, 64)):
        self.model = MLPClassifier(solver=optimizer, alpha=lr, hidden_layer_sizes=hidden_layer_sizes)
        self.bow = bow

    def train(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def predict_sentence(self, sentence: str):
        embedded_sentence = self.bow.transform([sentence.lower()])
        return self.model.predict(embedded_sentence.toarray())


def main():
    df = load_df_from_csv('data/dialog_acts_full_train.csv')
    features, vectorizer = create_bag_of_words(df)
    print('feature vector len:', len(features[0]))
    # print(features[0])
    labels = create_list_of_labels(df)

    mlp = MLP(vectorizer)
    mlp.train(features, labels)
    print(mlp.predict([np.zeros(721, dtype=np.uint32), np.ones(721, dtype=np.uint32)]))
    print(mlp.predict_sentence('hello and bye'))


if __name__ == '__main__':
    main()

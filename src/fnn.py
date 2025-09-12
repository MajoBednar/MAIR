from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from data_management import load_df_from_csv
from bow import create_bag_of_words, create_list_of_labels, transform_to_bag_of_words
from testing import prompt


class MLP:
    def __init__(self, bow, optimizer='adam', lr=0.001, alpha=1e-5, hidden_layer_sizes=(128, 64)):
        self.model = MLPClassifier(
            solver=optimizer,
            learning_rate_init=lr,
            alpha=alpha,
            hidden_layer_sizes=hidden_layer_sizes
        )
        self.bow = bow

    def train(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def predict_sentence(self, sentence: str):
        embedded_sentence = self.bow.transform([sentence.lower()])
        return self.model.predict(embedded_sentence.toarray())

    def cross_validation(self, features, labels):
        scores = cross_val_score(self.model, features, labels, cv=5)
        print(scores)

    def evaluate_accuracy(self, test_features, test_labels):
        print(self.model.score(test_features, test_labels))


def main():
    df = load_df_from_csv('data/dialog_acts_no_duplicates_train.csv')
    features, vectorizer = create_bag_of_words(df)
    print('feature vector len:', len(features[0]))
    # print(features[0])
    labels = create_list_of_labels(df)

    mlp = MLP(vectorizer)
    mlp.cross_validation(features, labels)
    # mlp.train(features, labels)
    #
    # df_test = load_df_from_csv('data/dialog_acts_no_duplicates_test.csv')
    # features_test = transform_to_bag_of_words(vectorizer, df_test)
    # labels_test = create_list_of_labels(df_test)
    # predictions = mlp.predict(features_test)
    #
    # prompt(mlp)


if __name__ == '__main__':
    main()

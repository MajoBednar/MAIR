from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from data_management import load_df_from_csv
from bow import create_bag_of_words, create_list_of_labels, transform_to_bag_of_words
from testing import prompt


class MLModel:
    def __init__(self, bow):
        self.bow = bow
        self.model = None
        self.name = 'Unknown'

    def train(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def predict_sentence(self, sentence: str):
        embedded_sentence = self.bow.transform([sentence.lower()])
        return self.model.predict(embedded_sentence.toarray())

    def cross_validation(self, features, labels):
        scores = cross_val_score(self.model, features, labels, cv=5)
        print(self.name, scores)

    def evaluate_accuracy(self, test_features, test_labels):
        print(self.model.score(test_features, test_labels))


class LogReg(MLModel):
    def __init__(self, bow):
        super().__init__(bow)
        self.model = LogisticRegression()
        self.name = 'Logistic Regression'


class DecisionTree(MLModel):
    def __init__(self, bow):
        super().__init__(bow)
        self.model = DecisionTreeClassifier()
        self.name = 'Decision Tree'


class MLP(MLModel):
    def __init__(self, bow, optimizer='adam', lr=0.001, alpha=1e-5, hidden_layer_sizes=(128, 64)):
        super().__init__(bow)
        self.model = MLPClassifier(
            solver=optimizer,
            learning_rate_init=lr,
            alpha=alpha,
            hidden_layer_sizes=hidden_layer_sizes
        )
        self.name = 'Multi-Layer Perceptron'


def main():
    df = load_df_from_csv('data/dialog_acts_no_duplicates_train.csv')
    features, vectorizer = create_bag_of_words(df)
    print('feature vector len:', len(features[0]))
    # print(features[0])
    labels = create_list_of_labels(df)

    mlp = MLP(vectorizer)
    lr = LogReg(vectorizer)
    dt = DecisionTree(vectorizer)
    mlp.cross_validation(features, labels)
    lr.cross_validation(features, labels)
    dt.cross_validation(features, labels)

    mlp.train(features, labels)

    df_test = load_df_from_csv('data/dialog_acts_no_duplicates_test.csv')
    features_test = transform_to_bag_of_words(vectorizer, df_test)
    labels_test = create_list_of_labels(df_test)
    predictions = mlp.predict(features_test)

    prompt(mlp)


if __name__ == '__main__':
    main()

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

from data_management import load_df_from_csv
from bow import create_bag_of_words, create_list_of_labels


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

    def tune_hyperparams(self, features, labels, hyperparam_grid):
        grid_search = GridSearchCV(self.model, hyperparam_grid, cv=5, scoring='accuracy', verbose=2)
        grid_search.fit(features, labels)
        self.model = grid_search.best_estimator_
        print('Best hyperparameters for', self.name, 'are:')
        print(grid_search.best_params_)
        print('Cross-validation accuracy was:', grid_search.best_score_)

    def save(self, path: str):
        """Use .pkl as extension."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


class LogReg(MLModel):
    def __init__(self, bow):
        super().__init__(bow)
        self.model = LogisticRegression()
        self.name = 'Logistic Regression'

    def tune_hyperparams(self, features, labels, hyperparam_grid=None):
        if hyperparam_grid is None:
            hyperparam_grid = {
                'C': [0, 0.01, 0.1, 0.5, 1, 5, 10, 20]
            }
        super().tune_hyperparams(features, labels, hyperparam_grid)


class DecisionTree(MLModel):
    def __init__(self, bow):
        super().__init__(bow)
        self.model = DecisionTreeClassifier()
        self.name = 'Decision Tree'

    def tune_hyperparams(self, features, labels, hyperparam_grid=None):
        if hyperparam_grid is None:
            hyperparam_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 20]
            }
        super().tune_hyperparams(features, labels, hyperparam_grid)


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

    def tune_hyperparams(self, features, labels, hyperparam_grid=None):
        if hyperparam_grid is None:
            hyperparam_grid = {
                'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
                'learning_rate_init': [0.001, 0.01],
                'batch_size': [32, 64]
            }
        super().tune_hyperparams(features, labels, hyperparam_grid)


def main():
    # df = load_df_from_csv('data/dialog_acts_no_duplicates_train.csv')
    # features, vectorizer = create_bag_of_words(df)
    # labels = create_list_of_labels(df)
    #
    # nn = MLP(vectorizer)
    #
    # nn.tune_hyperparams(features, labels)
    # nn.save('models/nn_no_duplicates.pkl')

    df = load_df_from_csv('data/dialog_acts_full_train.csv')
    features, vectorizer = create_bag_of_words(df)
    labels = create_list_of_labels(df)

    nn = MLP(vectorizer)

    nn.tune_hyperparams(features, labels)
    nn.save('models/nn_full.pkl')


if __name__ == '__main__':
    main()

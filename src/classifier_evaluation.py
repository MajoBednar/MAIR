from ui import select_dataset, select_model
from ml_models import MLModel
from bow import transform_to_bag_of_words, create_list_of_labels, create_list_of_utterances


def main():
    df, key = select_dataset()
    model = select_model(key)

    if isinstance(model, MLModel):
        features = transform_to_bag_of_words(model.bow, df)
    else:
        features = create_list_of_utterances(df)

    predictions = model.predict(features)
    true_labels = create_list_of_labels(df)

    print(predictions)
    print(true_labels)


if __name__ == '__main__':
    main()

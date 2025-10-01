import pandas as pd
import random


def load_data_as_df(path: str):
    dialog_acts = []
    utterances = []

    with open(path, 'r') as file:
        for line in file:
            # Split on first space to separate dialog_act from utterance
            parts = line.split(' ', 1)
            dialog_act = parts[0]
            utterance = parts[1].lower()  # also convert to lowercase

            dialog_acts.append(dialog_act)
            utterances.append(utterance)

    dataframe = pd.DataFrame({
        'dialog_act': dialog_acts,
        'utterance': utterances
    })

    return dataframe


def remove_duplicates(dataframe):
    df_without_duplicates = dataframe.drop_duplicates(subset=['utterance'], keep='first')
    return df_without_duplicates


def split_and_save_datasets(dataframe, path: str, name: str):
    df_shuffled = dataframe.sample(frac=1).reset_index(drop=True)

    split_index = int(0.85 * df_shuffled.shape[0])
    print('Original len:', df_shuffled.shape[0])
    df_train = df_shuffled.iloc[:split_index]
    df_test = df_shuffled.iloc[split_index:]
    print('Train len:', df_train.shape[0])
    print('Test len:', df_test.shape[0])

    df_train.to_csv(path + '/' + name + '_train.csv')
    df_test.to_csv(path + '/' + name + '_test.csv')


def display_information_training_set(df):
    label_counts = df['dialog_act'].value_counts()
    print(label_counts)

    df['word_count'] = df['utterance'].str.split().str.len()
    print(f'Shortest utterance has {df["word_count"].min()} words.')
    print(f'Longest utterance has {df["word_count"].max()} words.')


def load_df_from_csv(path: str):
    return pd.read_csv(path)


def add_new_property(path: str, property_name: str, property_values: list[str]):
    df = load_df_from_csv(path)
    print(df.head(10), '\n')
    df[property_name] = [random.choice(property_values) for _ in range(len(df))]
    print(df.head(10))
    df.to_csv(path, index=False)


if __name__ == '__main__':
    df_duplicates = load_df_from_csv('data/dialog_acts_full_train.csv')
    df_no_duplicates = load_df_from_csv('data/dialog_acts_no_duplicates_train.csv')

    display_information_training_set(df_duplicates)
    print('-----')
    display_information_training_set(df_no_duplicates)

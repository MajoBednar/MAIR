import pandas as pd
import random


def load_data_as_df(path: str):
    """Creates a dataframe of dialog acts and utterances from a dialog_acts.dat file."""
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
    """Removes duplicate utterances from a dataframe."""
    df_without_duplicates = dataframe.drop_duplicates(subset=['utterance'], keep='first')
    return df_without_duplicates


def split_and_save_datasets(dataframe, path: str, name: str):
    """Shuffles and splits a dataframe to 85% train and 15% test sets. Also saves them to .csv files."""
    df_shuffled = dataframe.sample(frac=1).reset_index(drop=True)
    split_index = int(0.85 * df_shuffled.shape[0])

    df_train = df_shuffled.iloc[:split_index]
    df_test = df_shuffled.iloc[split_index:]

    df_train.to_csv(path + '/' + name + '_train.csv')
    df_test.to_csv(path + '/' + name + '_test.csv')


def display_information_training_set(df):
    """Displays the dialog_act label counts and min and max utterance lengths of a dataframe."""
    label_counts = df['dialog_act'].value_counts()
    print(label_counts)

    df['word_count'] = df['utterance'].str.split().str.len()
    print(f'Shortest utterance has {df["word_count"].min()} words.')
    print(f'Longest utterance has {df["word_count"].max()} words.')


def load_df_from_csv(path: str):
    """Loads .csv file to a dataframe."""
    return pd.read_csv(path)


def add_new_property(path: str, property_name: str, property_values: list[str]):
    """Adds a new property filled with given property_values at random to a .csv file."""
    df = load_df_from_csv(path)
    df[property_name] = [random.choice(property_values) for _ in range(len(df))]
    df.to_csv(path, index=False)


if __name__ == '__main__':
    df_duplicates = load_df_from_csv('data/dialog_acts_full_train.csv')
    df_no_duplicates = load_df_from_csv('data/dialog_acts_no_duplicates_train.csv')

    display_information_training_set(df_duplicates)
    print('-----')
    display_information_training_set(df_no_duplicates)

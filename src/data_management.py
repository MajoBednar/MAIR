import pandas as pd


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


def load_df_from_csv(path: str):
    return pd.read_csv(path)


if __name__ == '__main__':
    df = load_data_as_df('data/dialog_acts.dat')
    df_no_duplicates = remove_duplicates(df)

    split_and_save_datasets(df, 'data', 'dialog_acts_full')
    split_and_save_datasets(df_no_duplicates, 'data', 'dialog_acts_no_duplicates')

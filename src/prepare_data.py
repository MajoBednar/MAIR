import pandas as pd


def invert_q3(score):
    return 8 - score


def prepare_df(dataframe):
    new_df = dataframe.copy()
    for idx, row in new_df.iterrows():
        for trial in ['T1', 'T2', 'T3', 'T4']:
            new_df.at[idx, 'Q3 ' + trial] = invert_q3(row['Q3 ' + trial])

    conditions_df = pd.DataFrame()

    for idx, row in new_df.iterrows():
        score_no = 0
        score_exp = 0
        for trial in ['T1', 'T2', 'T3', 'T4']:
            if row['State ' + trial] == 'Explicit confirmation':
                for question in ['Q1 ', 'Q2 ', 'Q3 ']:
                    score_exp += row[question + trial]
            else:
                for question in ['Q1 ', 'Q2 ', 'Q3 ']:
                    score_no += row[question + trial]
        new_df.at[idx, 'cExp'] = score_exp / 2
        conditions_df.at[idx, 'cExp'] = score_exp / 2
        new_df.at[idx, 'cNo'] = score_no / 2
        conditions_df.at[idx, 'cNo'] = score_no / 2

    return new_df, conditions_df


if __name__ == '__main__':
    path = 'data/Results_experiment_MAIR.csv'
    df = pd.read_csv(path)
    # print(df)
    processed_df, cond_df = prepare_df(df)
    print(processed_df)
    print(cond_df)

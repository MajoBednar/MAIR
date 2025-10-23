import pandas as pd
from scipy.stats import shapiro, ttest_rel


def normality_test(dataframe):
    print('Shapiro-Wilk test:')
    stat, p = shapiro(dataframe['cExp'])
    print(f"Experimental: W={stat:.3f}, p={p:.3f}")

    stat, p = shapiro(dataframe['cNo'])
    print(f"Control: W={stat:.3f}, p={p:.3f}")

    diff = dataframe['cExp'] - dataframe['cNo']
    stat, p = shapiro(diff)
    print(f"Difference: W={stat:.3f}, p={p:.3f}")


def invert_q3(score):
    return 8 - score


def prepare_df(dataframe=pd.read_csv('data/Results_experiment_MAIR.csv')):
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

    print(f"Explicit Confirmation: M = {cond_df['cExp'].mean():.2f}, SD = {cond_df['cExp'].std(ddof=1):.2f}")
    print(f"No Confirmation: M = {cond_df['cNo'].mean():.2f}, SD = {cond_df['cNo'].std(ddof=1):.2f}")

    normality_test(cond_df)
    t_stat, p_val = ttest_rel(cond_df['cExp'], cond_df['cNo'])
    print(f'T-test: t={t_stat}, p={p_val}')

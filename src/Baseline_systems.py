import pandas as pd

"""
Reads the dialog data and puts it in a dataframe with two collumns:
dialog_act and utturance.
"""
def read_dialog_file():
    filepath = 'dialog_acts.dat'

    dialog_acts = []
    utterances = []
    
    with open(filepath, 'r') as file:
        for line in file:
            # Split on first space to separate dialog_act from utterance
            parts = line.split(' ', 1)
            dialog_act = parts[0]
            utterance = parts[1] 

            dialog_acts.append(dialog_act)
            utterances.append(utterance)

    df = pd.DataFrame({
        'dialog_act': dialog_acts,
        'utterance': utterances
    })

    return df


"""
Function that splits the data in 80% train data and 20% test data.
First shuffles the data so that the split is random.
"""
def split_data(df):
    #Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42)
    
    #Calculate split index
    split_index = int(len(df_shuffled) * 0.8)
    
    #Split the data
    train_data = df_shuffled.iloc[:split_index]
    test_data = df_shuffled.iloc[split_index:]
    
    #Reset indices for clean df
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    return train_data, test_data


"""
Function that removes the utturances that appear multiple times in the data.
"""
def remove_duplicate_utaturances(df):
    df_without_duplicates = df.drop_duplicates(subset=['utterance'], keep='first')
    
    return df_without_duplicates


"""
Function that displays all different dialog acts and how often they occur.
Output is given as a pandas series.
"""
def dialog_act_counter(df):
    # Count the occurrences of each dialog act
    act_counts = df['dialog_act'].value_counts()
    
    return act_counts


"""
A baseline system that, regardless of the content of the utterance, always assigns the majority class of in the data.
--> Majority class is inform found in dialog_act_counter.
"""
def always_label_inform(df):
    #Number of total utterances
    total = len(df)
    
    #Number of'inform' utturances
    correct_labeled = len(df[df['dialog_act'] == 'inform'])
    
    performance = correct_labeled/total 
    return performance


"""
Method that implements baseline rules found from empirical evulation of the dataset.
Applying these rules gives and approximate performance of 0.86
"""
def baseline_rules(df):
    # Number of total utterances 
    compare_df = df.copy()
    total = len(df)

    # Set standard label to inform 
    df['dialog_act'] = 'inform'

    # If yes is in utterance label affirm 
    df.loc[df['utterance'].str.contains('yes', case=False, na=False), 'dialog_act'] = 'affirm'
    
    # If goodbye or bye is in utterance label bye 
    df.loc[df['utterance'].str.contains('bye|goodbye', case=False, na=False), 'dialog_act'] = 'bye'

    # If the word no is in utterance label negate 
    df.loc[df['utterance'].str.contains(r'\bno\b', case=False, na=False), 'dialog_act'] = 'negate'

    # If thank is in utterance label thankyou 
    df.loc[df['utterance'].str.contains('thank', case=False, na=False), 'dialog_act'] = 'thankyou'

    # If cough, unintelligible, noise or sil is in utterance label null 
    df.loc[df['utterance'].str.contains('cough|unintelligible|noise|sil', case=False, na=False), 'dialog_act'] = 'null'

    # If Phone number, address, post code or postcode is in utterance label request 
    df.loc[df['utterance'].str.contains('phone number|address|post code|what|postcode', case=False, na=False), 'dialog_act'] = 'request'

    # If another is in utternce label requalts 
    df.loc[df['utterance'].str.contains('another', case=False, na=False), 'dialog_act'] = 'requalts'

    # Calculate the correct labeled dialog acts
    matches = (df["dialog_act"] == compare_df["dialog_act"])
    
    correct_labeled = matches.sum()

    performance = correct_labeled/total 
    return performance

def main():
    #Create dataframes and output length 
    dialog_data = read_dialog_file()
    print('number of rows:', len(dialog_data))
    dialog_data_wo_duplicates = remove_duplicate_utaturances(dialog_data)
    print('Number of rows without duplicates:', len(dialog_data_wo_duplicates))

    train_data, test_data = split_data(dialog_data)

    print('training length: ',len(train_data),'testing length: ', len(test_data))
    #Display dialoge acts and the number of occurances such that first baseline can be achieved
    #print(dialog_act_counter(train_data))
    print('The performance of always assign majority class mehtod is: ', always_label_inform(test_data))

if __name__ == "__main__":
    main()

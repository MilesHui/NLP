# test
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from wissee_ai_projects.projects.Financial_Headlines_and_Twit_sentiment_analysis.src.parameters import finphrase_dir, tweet_dir
import json
import re
from tqdm.notebook import tqdm
import random

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parameters import finphrase_dir, tweet_dir, twit_dir



## Select consensus level for finphrase
# filename = 'Sentences_66Agree.txt'
# filename = 'Sentences_50Agree.txt'
filename = 'Sentences_75Agree.txt'
# filename = 'Sentences_AllAgree.txt'

def load_finphrase(filename):
    ''' Clean FinancialPhrasebank data
        Input:
            - filename
        Output:
            - a dataframe for the loaded financial phase bank data
    '''
    df = pd.read_csv(finphrase_dir + filename,
                     sep='\@',
                     engine='python',
                     header=None,
                     names=['sentence','label'])
    print('Total number of record in the file: ', df.shape[0])
    df.drop_duplicates(inplace=True)
    print('Total number of record after dropping duplicates: ', df.shape[0])
    print('Missing label: ', df['label'].isnull().sum())
    df.reset_index(inplace=True, drop=True)
    # df = pd.get_dummies(df, columns=['label'])
    return df

# load fin dataset
def load_data_fin():

    fin = load_finphrase(filename)

    # Encode the label
    le = LabelEncoder()
    le.fit(fin['label'])
    print(list(le.classes_))
    fin['label'] = le.transform(fin['label'])
    return fin


# load tweet
def load_tweet(filename):
    ''' Clean FinancialPhrasebank data
        Input:
            - filename
        Output:
            - a dataframe for the loaded financial phase bank data
    '''
    with open(tweet_dir + filename, 'r') as f:
        twits = json.load(f)

    print(twits['data'][:10])
    print("The number of twits is: ", len(twits['data']))
    messages = [twit['message_body'] for twit in twits['data']]
    # Since the sentiment scores are discrete, we'll scale the sentiments to 0 to 4 for use in the network
    sentiments = [twit['sentiment'] + 2 for twit in twits['data']]

    print('Sample Messages: \n', messages[:10])
    return messages, sentiments





def preprocess(message):
    """
    This function takes a string as input, then performs these operations:
        - lowercase
        - remove URLs
        - remove ticker symbols
        - removes punctuation
        - tokenize by splitting the string on whitespace
        - removes any single character tokens

    Parameters
    ----------
        message : The text message to be preprocessed.

    Returns
    -------
        tokens: The preprocessed text into tokens.
    """
    # Lowercase the twit message
    text = message.lower()

    # Replace URLs with a space in the message
    text = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', text)

    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub('\$[a-zA-Z0-9]*', ' ', text)

    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub('\@[a-zA-Z0-9]*', ' ', text)

    # Replace everything not a letter with a space
    text = re.sub('[^a-zA-Z]', ' ', text)

    return text

# load twit data
def load_data_twit():

    tweet_filename = 'twits.json'
    messages, sentiments = load_tweet(tweet_filename)
    # Process for all messages
    preprocessed = [preprocess(message) for message in tqdm(messages)]

    tmp_dict = {'org message': messages, 'sentence': preprocessed, 'label': sentiments}
    tmp_df = pd.DataFrame(tmp_dict)

    # Ignore tweets having less than 10 words
    tmp_df = tmp_df.loc[tmp_df['sentence'].apply(lambda x: len(x.split())) >= 10]
    tmp_df.reset_index(drop=True, inplace=True)

    # Take 0, 2, 4 and update them to 0, 1, 2
    twit = tmp_df.loc[(tmp_df['label'] == int(0)) | (tmp_df['label'] == int(2)) | (tmp_df['label'] == int(4))]

    def update_label(x):
        if x == int(2):
            return int(1)
        elif x == int(4):
            return int(2)
        else:
            return int(0)

    twit['label'] = twit['label'].apply(lambda x: update_label(x))

    # Balancing the data
    n_negative = sum(1 for each in twit['label'] if each == 0)
    n_neutral = sum(1 for each in twit['label'] if each == 1)
    n_positive = sum(1 for each in twit['label'] if each == 2)
    N_examples = twit.shape[0]

    balanced = {'org message': [], 'sentence': [], 'label': []}

    # Keep probability
    # As the negative has the least number of data, trim neutral and positive
    keep_prob_neutral = n_negative / n_neutral
    keep_prob_positive = n_negative / n_positive
    # keep_prob = 1

    for i, row in tqdm(twit.iterrows(), total=twit.shape[0]):
        if row['sentence'].strip() == "":
            continue
        elif (row['label'] == 0) or ((row['label'] == 1) and (random.random() < keep_prob_neutral)) or (
                (row['label'] == 2) and (random.random() < keep_prob_positive)):
            balanced['org message'].append(row['org message'])
            balanced['sentence'].append(row['sentence'])
            balanced['label'].append(row['label'])

    twit = pd.DataFrame(balanced)
    return twit

# load ticker data
def load_data_ticker():

    # ticker data
    ticker = pd.read_csv(twit_dir + 'youtube_reddit_ticker_data_for_test_labeled_na.csv')
    ticker = ticker.rename(columns={"clean_text": "sentence"})
    ticker_abr = ticker[['sentence','baseline_sentiment_score', 'label']]
    return ticker_abr

def load_data_reddit():
    reddit = pd.read_excel("/home/yongfeng/wissee_ai_projects/projects/financial_sentiment_analysis_yongfeng/Ticker_Sentiment_Pseudo_Labeling/results/reddit_combined.xlsx")
    reddit = reddit.rename(columns={'reddit_text': 'sentence', 'sentiment': 'label'})
    reddit = reddit[['sentence', 'label']]
    train, test = train_test_split(reddit)
    return train, test



import datetime
import sys
import os
import numpy as np
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import pandas as pd

def average_len_reddit_postings(reddit_data):
    """
    calculate the average length of reddit postings
    """
    df = reddit_data.copy()
    ave_leg = df["text"].apply(lambda x: len(x.split())).mean()
    print(ave_leg)

def delete_url(reddit_data, ticker, name):
    """
    delete reddit postings with urls
    """

    df = reddit_data.copy()
    df['http'] = df["text"].str.find(r'http')
    # df_url = df.loc[(df['http'] != -1)]
    # df_url.to_excel(path + '/results/{}/reddit_{}_{}_only_url.xlsx'.format(ticker, ticker, name))

    df = df[df['http'] == -1]
    df = df.drop(columns=['http'])
    # df.to_excel(path+'/results/{}/reddit_{}_{}_removed_no_url.xlsx'.format(ticker, ticker, name))

    return df


def delete_multiple_entity_mentions(reddit_data, entity_list, ticker, name):
    """
    delete postings with multiple tickers in posting
    """
    df = reddit_data.copy()
    df['entity'] = df["tickers"].str.replace('|', '', regex=False).str.replace('$', '', regex=False)
    for entity in entity_list:
        df['entity'] = df["entity"].str.replace(entity, '', regex=True)
    df = df[df['entity'].str.len() == 0 ]
    df = df.drop(columns=['entity'])
    # df.to_csv(path+'/results/{}/reddit_{}_{}_removed.csv'.format(ticker, ticker, name))
    return df

def delete_length_1_100(reddit_data, column, ticker, name):
    """
    delete postings with length equals to 1 or larger than 200
    """
    df = reddit_data.copy()
    df['len'] = df[column].apply(lambda x: len(x.split()))
    df = df[df['len'] > 1 ]
    df = df[df['len'] < 100 ]
    df = df.drop(columns=['len'])
    # df.to_csv(path+'/results/{}/reddit_{}_{}_removed.csv'.format(ticker, ticker, name))
    return df


def lambda_delete_question(sentences):
    """
    lambda function in pandas to delete sentences in postings with question marks
    """
    new_sentences = []
    for sentence in sentences:
        if sentence.find(r'?') == -1:
            new_sentences.append(sentence)
    return new_sentences


def lambda_count_sent(sentences):
    """
    lambda function in pandas to calculate number of sentences in postings
    """
    return len(sentences)




def lambda_combine_sent(sentences):
    """
    lambda function in pandas to combine sentences that are tokenized

    """
    new_sentences= ''
    for sentence in sentences:
        new_sentences = new_sentences + sentence + ' '
    return new_sentences


def delete_question_sentence(reddit_data, ticker, name):
    """
    delete sentences in postings with question marks
    """
    df = reddit_data.copy()
    df['sentences'] = df.text.apply(sent_tokenize)
    df["sentence_no_question"] = df.apply(lambda x: lambda_delete_question(x['sentences']), axis=1)
    df['question_mark_location'] = df["text"].str.find(r'?')
    df['question'] = df['question_mark_location']>0


    df['num_sen'] = df.apply(lambda x: lambda_count_sent(x['sentence_no_question']), axis = 1)
    df = df[df['num_sen'] != 0]
    df['new_text'] = df.apply(lambda x: lambda_combine_sent(x['sentence_no_question']), axis = 1)
    df = df.drop(columns=['sentences', "sentence_no_question", 'num_sen','question_mark_location'])

    return df

def lambda_separate_sentences(new_len, text):
    """
    lambda function to separate sentences to make sure all with length less than 60
    """
    new_text = ''
    remaining_text = ''
    if new_len:
        text_sens = sent_tokenize(text)
        length = 0
        for sen in text_sens:
            length = length + len(sen.split())
            if length < 60:
                new_text = new_text + sen + ' '
            else:
                remaining_text = remaining_text + sen + ' '
    else:
        new_text = text
    return remaining_text, new_text

def separate_sentences_iter(reddit_data, ticker, name):
    """
    iterator that separate sentences
    """
    df = reddit_data.copy()
    df['new_text_over_60'] = (df['new_text'].apply(lambda x: len(x.split()))>60)
    df['remaining_sent'], df['new_text'] = zip(*df.apply(lambda x: lambda_separate_sentences(x['new_text_over_60'], x["new_text"]), axis = 1))

    flt_returned = ~(df['remaining_sent'].str.len() == 0)
    d = df[flt_returned].loc[df[flt_returned].index,:].reset_index(drop=True)
    dc = d.copy()
    d['new_text'] = d['remaining_sent']
    d['remaining_sent'] = ''
    dc['remaining_sent'] = ''
    # idx_duplicate = d.duplicated(keep="first")
    dfc = pd.concat([d, dc, df[~flt_returned]])
    dfc = dfc.drop(columns=['remaining_sent'])

    # dfc.to_excel(path + '/results/{}/test1.xlsx'.format(ticker))
    return dfc





def separate_sentences(reddit_data, ticker, name):
    """
    function to separate sentences in posting
    """
    df = reddit_data.copy()
    df['separated'] = df['new_text'].apply(lambda x: len(x.split())) > 60
    temp = df['new_text'].apply(lambda x: len(x.split())) > 60
    n = 0
    while (np.sum(temp)>0 and n < 3):
        df = separate_sentences_iter(df, ticker, name)
        temp = df['new_text'].apply(lambda x: len(x.split())) > 60
        n = n+1
    try: # in case np.sum(temp)<0 directly, then there will be no column new_text_over_60
        df = df.drop(columns=['new_text_over_60'])
    except:
        pass

    return df

def lambda_delete_more_then_2_sentences(text):
    text_sens = sent_tokenize(text)
    return len(text_sens)




def delete_more_then_2_sentences(reddit, ticker,name):
    df = reddit.copy()
    df['length_of_sentences'] = df['new_text'].apply(lambda x: lambda_delete_more_then_2_sentences(x))
    df = df[df['length_of_sentences'] <= 2]
    df = df.drop(columns=['length_of_sentences'])
    return df



def main_function(reddit_data, ticker, name):
    """
    pipeline function
    """
    reddit = reddit_data.copy()
    reddit = delete_url(reddit, ticker, name)
    reddit = delete_multiple_entity_mentions(reddit, entity_list, ticker, name)
    reddit = delete_question_sentence(reddit, ticker, name) # create new column called new_text
    reddit = delete_length_1_100(reddit, 'new_text', ticker, name)
    reddit = delete_more_then_2_sentences(reddit, ticker,name)
    if reddit.shape[0]==0:
        print('There are no rows remain after deleting url, multiple entities, and too short or too long postings')
        return 0
    # plt.hist(reddit_len['len'], density=True)
    # plt.title('reddit_{}_{}'.format(ticker, name))
    # plt.savefig(path + '/results/{}/reddit_{}_{}.png'.format(ticker, ticker, name))
    # average_len_reddit_postings(reddit_len)
    # reddit_no_question = delete_question_sentence(reddit, ticker, name)

    # reddit_result = separate_sentences(reddit_no_question, ticker, name)
    # reddit_result = reddit_result.sample(min(reddit_result.shape[0], 500),random_state=42)
    reddit.insert(0, 'reddit_text', reddit['new_text'])
    reddit.insert(0, 'sentiment', '')
    reddit = reddit.drop(columns=['new_text', 'Unnamed: 0', 'Unnamed: 0.1', '0'])
    # reddit_result_len = delete_length_1_200(reddit_result, 'reddit_text', ticker, name)

    # create excel writer
    #writer = pd.ExcelWriter(path + '/results/output/reddit_{}_{}_result.xlsx'.format(ticker, ticker, name))
    # write dataframe to excel sheet named 'marks'
    reddit.to_excel(path + '/results/reddit_posting_preprocessed/{}_{}.xlsx'.format(ticker, name),
                    sheet_name='{}_{}_result'.format(ticker, name),
                           index=False)
                           # columns=['new_text', 'text', 'tickers', 'sentiment_score', 'question',
                           #          'separated'], index=False)


    return reddit


if __name__ == "__main__":

    path = os.getcwd()
    tickers = pd.read_csv("wissee_tickers_by_coverage_07222021.csv")
    tickers = tickers['ticker'].tolist()

    for ticker in tickers:
        for name in ['peak', 'valley']:
            # reddit_data = pd.read_excel(path + '/results/reddit_posting/{}_{}.xlsx'.format(ticker, name))
            # entity_list = [ticker]
            # main_function(reddit_data, ticker, name)

            try:
                reddit_data = pd.read_excel(path + '/results/reddit_posting/{}_{}.xlsx'.format(ticker, name))
                entity_list = [ticker]
                main_function(reddit_data, ticker, name)
            except:
                print('maybe {} {} dataset des not exist'.format(ticker, name))







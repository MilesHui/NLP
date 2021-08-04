import numpy as np
import random
import sys
import pandas as pd
import torch
from bert import test_transformer, predict_transformer
from parameters import weights_dir


# Set Random Seed
random.seed(42)
np.random.seed(42)
rand_seed = 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Hyperparameters
batch_size = 32
max_seq_length = 96



def test_bert(test, weight):

    x_test = np.array(test['sentence'])
    y_test = np.array(test['label'].astype(int).values)

    # load data
    if torch.cuda.is_available():
        bert_model = torch.load(weights_dir+weight)
    else:
        bert_model = torch.load(weights_dir + weight, map_location=torch.device('cpu'))
    pred = predict_transformer(bert_model, x_test, batch_size, max_seq_length)
    return pred



if __name__ == "__main__":

    # try:
    #     if sys.argv[1] == 'fin':
    #         weight = 'fin_bert_model_4.pt'
    #         weight_name = 'newsheadline'
    # except:
    #     weight = 'twit_bert_model_0.pt'
    #     weight_name = 'stocktwits'

# input data
# d = {'sentence': ['With the new production plant the company would increase its capacity to meet the expected increase '
#                   'in demand and would improve the use of raw materials and therefore increase the production profitability .'],
#      'label': [2]}
# data = pd.DataFrame(data=d)
# pred = test_bert(data, weight)
# print(pred)

    path = "/home/yongfeng/wissee_ai_projects/projects/financial_sentiment_analysis_yongfeng/Ticker_Sentiment_Pseudo_Labeling/results/"
    for name in ['peak', 'valley']:
        df = pd.read_excel(path + "/reddit_{}_combined.xlsx".format(name))
        data = pd.DataFrame()
        data['sentence'] = df['reddit_text'].astype(str)
        data['label'] = np.zeros_like(df['reddit_text'])

        for w in ['fin', 'twit']:
            if w =='fin':
                weight = 'fin_bert_model_4.pt'
                weight_name = 'newsheadline'
            elif w == 'twit':
                weight = 'twit_bert_model_0.pt'
                weight_name = 'stocktwits'

            pred = test_bert(data, weight)
            pred = np.array(pred)


            df['label_{}'.format(weight_name)] = pred
            df['label_{}'.format(weight_name)] = df['label_{}'.format(weight_name)].map({"Negative": -1, "Neutral": 0, "Positive":1})

            # change order of columns
            # reddit_text = df['reddit_text']
            # sentiment_score = df['sentiment_score']
        label_newsheadline = df['label_newsheadline']
        label_stocktwits = df['label_stocktwits']

        df.drop(labels=['label_newsheadline',  'label_stocktwits'], axis=1, inplace=True)

        df.insert(3, 'label_newsheadline', label_newsheadline)
        df.insert(3, 'label_stocktwits', label_stocktwits)

        df.to_excel(path + '/reddit_{}_combined_result_with_label.xlsx'.format(name),index=False)

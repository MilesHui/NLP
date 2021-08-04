import os
import re
import logging
import pandas as pd

path = os.getcwd()
tickers = pd.read_csv("wissee_tickers_by_coverage_07222021.csv")
tickers = tickers['ticker'].tolist()
for name in ['valley', 'peak']:
    result = pd.DataFrame()
    for ticker in tickers:
        try:
            reddit_data = pd.read_excel(path + '/results/reddit_posting_preprocessed/{}_{}.xlsx'.format(ticker, name))
            result = result.append(reddit_data)
        except:
            print('{} {} dataset does not exist'.format(ticker, name))
    # result = result.sample(min(300, result.shape[0]))
    result.to_excel(path + '/results/reddit_{}_combined.xlsx'.format(name),index=False)

result_peak_valley = pd.DataFrame()
for name in ['valley', 'peak']:
    df = pd.read_excel(path + '/results/reddit_{}_combined.xlsx'.format(name))
    if name == 'valley':
        df['sentiment'] = -1
    elif name == 'peak':
        df['sentiment'] = 1
    result_peak_valley = result_peak_valley.append(df)

result_peak_valley.to_excel(path + '/results/reddit_combined.xlsx', index=False)

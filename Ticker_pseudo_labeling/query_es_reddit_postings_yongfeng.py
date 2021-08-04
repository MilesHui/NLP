"""
query reddit ES index
"""
import datetime
import sys
import os
import re
import logging
import pandas as pd
# import pytz
# from dateutil import parser
sys.path.insert(1, "/home/yongfeng/wissee_ai_projects/")
from common.utils import log_util
from common.utils.elasticsearch_helper import scroll_search
from common.data_streaming.reddit_trending_tickers.utils.process_tickers import ProcessTickers
from common.utils.reddit_helper.RedditQueryCompiler import RedditQueryCompiler
from common.utils.time_helper import convert_time
from datetime import date, timedelta
script_name = "query_es_reddit_postings"

def query_corpus_by_ticker(ticker: str = None,
                           start_time: str = '2020-01-01',
                           end_time: str = '2021-01-26'):
    """
    query corpus by ticker from es: reddit_postings

    :param end_time:
    :param start_time:
    :param ticker:
    renamed 'query_corpus_by_ticker' to query_corpus
    """
    logger = logging.getLogger(script_name)

    index = 'reddit_postings'

    T = ProcessTickers()

    # compile ticker keywords
    def filter_by_keyword_match(text):
        if keyword_regex_case_sensitive and keyword_regex_case_insensitive:
            if re.search(keyword_regex_case_sensitive, text) or re.search(keyword_regex_case_insensitive, text):
                return True
        elif keyword_regex_case_sensitive:
            if re.search(keyword_regex_case_sensitive, text):
                return True
        elif keyword_regex_case_insensitive:
            if re.search(keyword_regex_case_insensitive, text):
                return True

        return False

    if ticker:
        keywords = T.get_ticker_keywords(ticker=ticker)
    else:
        keywords = []
        # print("no keywords found, return None")
        # return

    if len(keywords) > 0:
        regex_patterns = T.compile_ticker_regex(ticker=ticker)
        keyword_regex_case_sensitive = regex_patterns.get("case_sensitive", None)
        # keyword_regex_case_sensitive = None
        keyword_regex_case_insensitive = regex_patterns.get("case_insensitive", None)
    else:
        keyword_regex_case_sensitive = None
        keyword_regex_case_insensitive = None

    ## compile search query
    msg = "ticker is {}, keywords is {}".format(ticker, keywords)
    log_util.info(logger=logger, msg=msg)
    reddit_query_compiler = RedditQueryCompiler(keywords=keywords,
                                                start_time=start_time,
                                                end_time=end_time)

    full_query = reddit_query_compiler.full_query

    #### enable scroll search
    records = scroll_search.scroll_search(query=full_query, index=index, page_size=1000, explain=False)
    data = pd.DataFrame(records)
    if not data.empty:
        # ADDED to filter text using keyword regex, because searching from elasticsearch,
        # e.g. '$FOR' will search 'for' without considering $, here we use regex to strict search
        if keyword_regex_case_insensitive or keyword_regex_case_sensitive:
            data["text_match"] = data["text"].apply(lambda x: filter_by_keyword_match(x))
            data = data[data["text_match"]]
    return data




def judge_close_market(created_at, peak):
    peak_date_format = datetime.datetime.strptime(peak, '%Y-%m-%d')
    date = convert_time(created_at, out_time_format='date')
    # print('peak', peak)
    # print('peak_date_format', peak_date_format)
    # print('date', date)
    # print('peak_date_format_day', peak_date_format.day)
    # print('date.day ', date.day)
    if date.day == peak_date_format.day:
        if (date.hour > 0 and date.hour < 9):
            return 1
        else:
            return 0
    elif date.day == peak_date_format.day - 1:
        if (date.hour > 16 and date.hour < 24):
            return 1
        else:
            return 0




def get_peak_reddit_posting(peak_data, peak_ind):
    peak_date = peak_data[peak_data.columns[0]]
    # print(peak_date)
    results = []

    for peak in peak_date:
        peak_date_format = datetime.datetime.strptime(peak, '%Y-%m-%d')

        # last day
        peak_last_day_date_format = peak_date_format- datetime.timedelta(days=1)
        peak_last_day = peak_last_day_date_format.strftime('%Y-%m-%d')

        peak_next_day_date_format = peak_date_format+ datetime.timedelta(days=1)
        peak_next_day = peak_next_day_date_format.strftime('%Y-%m-%d')

        # This day
        data_today = query_corpus_by_ticker(ticker=ticker, start_time=peak, end_time=peak)
        if len(data_today) != 0 :
            data_today['peak_date'] = peak
            data_today['est_time'] = data_today['created_at'].apply(convert_time)
            data_today['after_market'] = data_today['created_at'].apply(lambda x: (judge_close_market(x, peak = peak)))
            # if len(data_today[data_today['after_market']==1]) != 0:
            results.append(data_today[data_today['after_market']==1])

        # last day
        data_last_day = query_corpus_by_ticker(ticker=ticker, start_time=peak_last_day, end_time=peak_last_day)
        if len(data_last_day) != 0:
            data_last_day['peak_date'] = peak
            data_last_day['est_time'] = data_last_day['created_at'].apply(convert_time)
            data_last_day['after_market'] =  data_last_day['created_at'].apply(lambda x: (judge_close_market(x, peak = peak)))
            # if len(data_last_day[data_last_day['after_market']==1]) != 0:
            results.append(data_last_day[data_last_day['after_market']==1])
            # results.append(data_last_day)

        # next day
        data_next_day = query_corpus_by_ticker(ticker=ticker, start_time=peak_next_day, end_time=peak_next_day)
        if len(data_next_day) != 0:
            data_next_day['peak_date'] = peak
            data_next_day['est_time'] = data_next_day['created_at'].apply(convert_time)
            data_next_day['after_market'] =  data_next_day['created_at'].apply(lambda x: (judge_close_market(x, peak = peak)))
            # print(len(data_next_day[data_next_day['after_market']==1]))
            # if len(data_next_day[data_next_day['after_market']==1]) != 0:
            results.append(data_next_day[data_next_day['after_market']==1])
            # results.append(data_last_day)
        # data = pd.concat([data_today, data_last_day])
        #
        # results.append(data)

    try:
        appended_results = pd.concat(results)
        merged_results = pd.merge(appended_results, peak_data, left_on='peak_date', right_on='Unnamed: 0', how='left')
        #
        # save csv
        if peak_ind == True:
            # appended_results.to_excel(path+'/results/{}/reddit_{}_peak.csv'.format(ticker, ticker))
            merged_results.to_excel(path+'/results/reddit_posting/{}_peak.xlsx'.format(ticker))
        elif peak_ind == False:
            # appended_results.to_csv(path+'/results/{}/reddit_{}_valley.csv'.format(ticker, ticker))
            merged_results.to_excel(path+'/results/reddit_posting/{}_valley.xlsx'.format(ticker))
        return merged_results.shape[0]
    except:
        pass





# def get_reddit_posting(ticker, sdate, edate, year):
#     rng = pd.date_range(sdate,edate-timedelta(days=1),freq='d')
#     peak_date = rng.format(formatter=lambda x: x.strftime('%m/%d/%Y'))
#     results = []
#     for peak in peak_date:
#         data = query_corpus_by_ticker(ticker=ticker, start_time=peak, end_time=peak)
#         results.append(data)
#     # convert to est time
#     appended_results = pd.concat(results)
#     # save csv
#     appended_results.to_excel(path+'/results/{}/reddit_{}_{}.xlsx'.format(ticker, ticker, year))




if __name__ == "__main__":

    path = os.getcwd()
    tickers = pd.read_csv("wissee_tickers_by_coverage_07222021.csv")
    tickers = tickers['ticker'].tolist()
    lengths = []
    for ticker in tickers:
        print('Now extract reddit postings for {}'.format(ticker))
        for peak_ind in [True, False]:
            if peak_ind == True:
                print('Now extract peak reddit postings for {}'.format(ticker))
                try:
                    peak_data = pd.read_csv(path + '/results/ticker_price/{}_peak.csv'.format(ticker))
                    length_peak = get_peak_reddit_posting(peak_data, peak_ind)
                except:
                    print('Oops! Can not find this ticker {}'.format(ticker))





            elif peak_ind == False:
                print('Now extract valley reddit postings for {}'.format(ticker))
                try:
                    peak_data = pd.read_csv(path + '/results/ticker_price/{}_valley.csv'.format(ticker))
                    length_valley = get_peak_reddit_posting(peak_data, peak_ind)
                except:
                    print('Oops! Can not find this ticker {}'.format(ticker))


        lengths.append({'ticker': ticker, 'len of peak': length_peak, 'len of valley': length_valley})

    lengths_result = pd.DataFrame(lengths)
    lengths_result.to_excel('results/reddit_posting/all_peaks_valleys.xlsx', index=False)


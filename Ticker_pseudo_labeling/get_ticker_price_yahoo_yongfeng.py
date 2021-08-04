import datetime
import pandas as pd
import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data
import sys
from scipy.signal import find_peaks, find_peaks_cwt

def peak_detection(data=pd.Series(dtype='float64'), height=0, prominence=None,threshold = None, distance = None,
                   peak_type='peak',
                   topN=None):
    """
    :param data:
    :param height:
    :param prominence:
    :param threshold:
    :param distance:
    :param peak_type: peak (default), valley
    :param return_position_only: if True, return position only; otherwise, return a pd series containing time and value
    :param topN: if not empty, select topN peaks
    :return:
    """

    if not prominence:
        prominence = height / 2.0### emperical settings: half of height
    if peak_type.lower() in ['valley',"valleys"]:
        valleys = find_peaks(data*-1, height=height*-1, prominence=prominence, threshold=threshold, distance=distance)
        heights = valleys[1]['peak_heights']*-1## reverse it back
        pos = valleys[0]

        found_valleys = dict(zip(data.index[pos], heights))
        found_valleys = pd.Series(found_valleys)
        if topN:
            found_valleys.sort_values(ascending=True, inplace=True)
            found_valleys = found_valleys[:topN]
        return found_valleys
    else:
        peaks = find_peaks(data, height=height, prominence=prominence,threshold=threshold,distance=distance)
        heights = peaks[1]['peak_heights']
        pos = peaks[0]
        found_peaks = dict(zip(data.index[pos], heights))
        found_peaks = pd.Series(found_peaks)
        if topN:
            found_peaks.sort_values(ascending=False, inplace=True)
            found_peaks = found_peaks[:topN]

        return found_peaks



def choose_peaks_valleys_no_prior(peaks, valleys, days):
    peaks = peaks.copy().to_frame()
    peaks['ind'] = 1
    valleys = valleys.copy().to_frame()
    valleys['ind'] = -1
    com_peaks = pd.concat([peaks, valleys]).sort_index()
    com_peaks['date'] = com_peaks.index

    days = datetime.timedelta(days)
    com_peaks['date_before'] = com_peaks['date'] - days
    com_peaks['ind_new'] = None

    for index, row in com_peaks.iterrows():
        temp = com_peaks.loc[(com_peaks['date'] >= row['date_before']) & (com_peaks['date'] < row['date']) &
                             (com_peaks['ind'] != row['ind']), :]
        if temp.shape[0] != 0:
            com_peaks.loc[index, 'ind_new'] = 0
        else:
            com_peaks.loc[index, 'ind_new'] = row['ind']

    peaks = com_peaks.loc[com_peaks['ind_new']==1, 0]
    valleys = com_peaks.loc[com_peaks['ind_new'] == -1, 0]
    return peaks, valleys






def get_ticker(ticker, start_date, end_date, up, down, prior_date):

    data = get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")
    data['return'] = None
    data['return'] = (data['open'] - data['close'].shift(1))/data['close'].shift(1)

    df = pd.Series(data['return'])
    df.index = data.index

    # calculate quantile
    quantile_up = round(df.quantile(up/100)*100, 2)
    quantile_down = round(df.quantile(down/100)*100, 2)

    peaks_detected = peak_detection(df,peak_type='peak')
    peaks = peaks_detected[peaks_detected >= df.quantile(up/100)]
    # peaks = peaks_detected
    valleys_detected = peak_detection(df, peak_type='valley')
    valleys = valleys_detected[valleys_detected <= df.quantile(down/100)]
    # valleys = valleys_detected
    # save peaks and valleys
    # peaks = peaks.to_frame(name='peak_return')
    # valleys = valleys.to_frame(name='peak_return')
    peaks, valleys = choose_peaks_valleys_no_prior(peaks, valleys, prior_date)

    data_peak = pd.merge(peaks, data, left_index=True, right_index=True, how='left')
    # data_peak.to_excel('results/ticker_price/{}_peak.xlsx'.format(ticker))
    data_peak.to_csv('results/ticker_price/{}_peak.csv'.format(ticker))
    print('{} with peak quantile of {}%'.format(ticker, quantile_up))
    data_valley = pd.merge(valleys, data, left_index=True, right_index=True, how='left')
    # data_valley.to_excel('results/ticker_price/{}_valley.xlsx'.format(ticker))
    data_valley.to_csv('results/ticker_price/{}_valley.csv'.format(ticker))
    print('{} with valley quantile of {}%'.format(ticker, quantile_down))


    return df, peaks, valleys, quantile_up, quantile_down

def create_fig(ticker, start_date, end_date, df, peaks, valleys, up, down, prior_date):
    # Save the peak plots
    fig, ax = plt.subplots()
    ax.plot(df.index, df.values,
            color="black", linestyle='-')
    ax.scatter(x=peaks.index,
               y=peaks.values,
               color='green')

    ax.grid(True)
    plt.xticks(rotation=90)

    plt.title('{} peak from {} to {} with quantile of {}%'.format(ticker, start_date, end_date, round(df.quantile(up/100)*100, 2)))
    plt.savefig('results/ticker_price_graph/{}_peak.png'.format(ticker))
    plt.close()


    # Save the valley plots
    fig, ax = plt.subplots()
    ax.plot(df.index, df.values,
            color="black", linestyle='-')
    ax.scatter(x=valleys.index,
               y=valleys.values,
               color='green')

    ax.grid(True)
    plt.xticks(rotation=90)

    plt.title('{} valley from {} to {} with quantile of {}%'.format(ticker, start_date, end_date,round(df.quantile(down/100)* 100, 2) ))
    plt.savefig('results/ticker_price_graph/{}_valley.png'.format(ticker))
    plt.close()



if __name__ == "__main__":

    start_date = "01/01/2020"
    end_date = "07/01/2021"
    quantile_result = []
    tickers = pd.read_csv("wissee_tickers_by_coverage_07222021.csv")
    tickers = tickers['ticker'].tolist()

    up = 95
    down = 5
    prior_date = 30
    for ticker in tickers:
    # for ticker in ['tsla']:
        try:
            df, peaks, valleys, quantile_90, quantile_10 = get_ticker(ticker, start_date, end_date, up, down, prior_date)
            quantile_result.append({'ticker': ticker, 'quantile up': quantile_90, 'quantile down': quantile_10,
                                    'num of peaks': len(peaks), 'num of valleys': len(valleys)})
            create_fig(ticker, start_date, end_date, df, peaks, valleys, up, down, prior_date)
        except AssertionError as e:
            print('Oops! Can not find this ticker {}'.format(ticker))
            quantile_result.append({'ticker': ticker, 'quantile up': 'NA', 'quantile down': 'NA',
                                    'num of peaks': 'NA', 'num of valleys': 'NA'})
        except KeyError:
            print('Oops! Something wrong with this ticker {}'.format(ticker))
            quantile_result.append({'ticker': ticker, 'quantile up': 'NA', 'quantile down': 'NA',
                                    'num of peaks': 'NA', 'num of valleys': 'NA'})


    quantile_result = pd.DataFrame(quantile_result)
    quantile_result.to_excel('results/ticker_price/all_peaks_valleys.xlsx', index=False)
    print('Done')





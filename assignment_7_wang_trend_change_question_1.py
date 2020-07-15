import pandas as pd
import numpy as np
from scipy.stats import linregress
from collections import deque
import matplotlib.pyplot as plt

def trading_strategy_with_stats(trading_df, prediction_label, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    prediction_label: the label for which we're going to trade off of
    returns: A df of trades made based on Predicted Labels. Also gives whether transactions were made along with other stats.
    '''
    # The daily balance we will be using
    total_balance = weekly_balance
    trading_history = deque()
    trading_days_index = 0
    # Boolean indicating whether we have a position or not
    position = False
    # The number of shares
    shares = 0
    # String indicating what position was taken
    taken_position = ''
    while(trading_days_index < len(trading_df.index)):
        trading_label = trading_df.iloc[trading_days_index][[prediction_label]].values[0]
        if trading_days_index > 0:
            previous_trading_label = trading_df.iloc[trading_days_index-1][[prediction_label]].values[0]
        else:
            previous_trading_label = None
        trading_value =  trading_df.iloc[trading_days_index][['Close']].values[0]
        # If we have money and no position, we immediately take a position based on the day
        if total_balance != 0 or shares != 0:
            if not position:
                # If it's the last day and we have no position, we can't trade. Otherwise we trade
                if trading_days_index != len(trading_df.index)-1:
                    position = True
                    # A buy and hold
                    if  trading_label == 'GREEN':
                        # If I have more than $100, I will only trade with $100
                        if total_balance > 100:
                            shares = np.divide(100, trading_value)
                            total_balance = total_balance - 100
                        # If I have less than 100, I can only trade with as much as I have
                        else:
                            shares = np.divide(total_balance, trading_value)
                            total_balance = 0
                        taken_position = 'LONG'
                    # A short
                    else:
                        # I can always sell short because I can borrow money
                        # I essentially need to buy these shares back
                        shares = np.negative(np.divide(100, trading_value))
                        # Add 100 dollars of shares short
                        total_balance = total_balance + 100
                        taken_position = 'SHORT'
                else:
                    taken_position = 'NONE'
            # Otherwise we have a position
            else:
                # If it's the last day and we have a position, we must close out the position
                if trading_days_index == len(trading_df.index) - 1:
                    # If I'm long on the last day
                    if shares > 0:
                        total_balance = total_balance + np.multiply(shares, trading_value)
                        taken_position = 'LONG TO SELL'
                    # If I'm short on the last day
                    elif shares < 0:
                        buy_back_shares = np.abs(shares)
                        total_balance = total_balance - np.multiply(buy_back_shares, trading_value)
                        taken_position = 'SHORT TO BUY'
                else:
                    # If we have previous trading days that are the same, we can keep going
                    if (trading_label == 'GREEN' and previous_trading_label == 'GREEN') or (trading_label == 'RED' and previous_trading_label == 'RED'):
                        taken_position = 'NONE'
                    # We have a short position from yesterday, and we must close it by buying shares back and subtracting from our total
                    elif trading_label == 'GREEN' and previous_trading_label == 'RED':
                        buy_back_shares = np.abs(shares)
                        total_balance = total_balance - np.multiply(buy_back_shares, trading_value)
                        shares = 0
                        position = False
                        taken_position = 'SHORT TO BUY'
                    # We have a long position from yesterday, and we must close it by selling shares and adding to our total
                    elif trading_label == 'RED' and previous_trading_label == 'GREEN':
                        total_balance = total_balance + np.multiply(shares, trading_value)
                        shares = 0
                        position = False
                        taken_position = 'LONG TO SELL'

        # Regardless of whether we made a trade or not, we append the weekly cash and week over
        trading_history.append([trading_df.iloc[trading_days_index][['Year']].values[0],
                    trading_df.iloc[trading_days_index][['Week_Number']].values[0],
                    trading_df.iloc[trading_days_index][['Close']].values[0],
                    total_balance, taken_position, shares])
        # If we have no money, then we stop trading
        trading_days_index = trading_days_index+1
                
        
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Week_Number', 'Price', 'Balance', 'Position', 'Shares'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df


def get_pnl(pnl_table):
    '''
    pnl_table: the dataframe our profit and loss
    return: a tuple of pnls from short and long
    '''
    # Calculate all the rows where we have a LONG to LONG TO SELL
    # We have to normalize indices to subtract properly
    long_open = pnl_table.query('Position in ["LONG"]')
    long_open.reset_index(drop=True, inplace=True)
    long_to_sell = pnl_table.query('Position in ["LONG TO SELL"]')
    long_to_sell.reset_index(drop=True, inplace=True)

    shares_bought_vector = pd.DataFrame(np.multiply(long_open[["Shares"]].values.astype(float), long_open[["Price"]].values.astype(float)), columns=['Balance'])
    # Do the same for short positions
    short_open = pnl_table.query('Position in ["SHORT"]')
    short_open.reset_index(drop=True, inplace=True)
    short_to_buy = pnl_table.query('Position in ["SHORT TO BUY"]')
    short_to_buy.reset_index(drop=True, inplace=True)
    # PNL from longs is the total balance we have at the end of a long transaction minus the balance we had in cash after first making that long transaction 
    # minus the initial seed money to buy shares (sometimes less than 100)
    
    pnl_from_long = long_to_sell[['Balance']] - long_open[['Balance']] - shares_bought_vector

    # PNL from short is the balance we have at the end minus the difference between the open balance and our profit from the initial short
    pnl_from_short = short_to_buy[['Balance']] - (short_open[['Balance']] - 100)
    return pnl_from_long, pnl_from_short


def get_windowed_slice_and_fit(df, W=5):
    '''
    df: the dataframe where we are splitting into time series
    return: dataframe with prediction labels
    '''
    all_predictions = deque()
    df_values = df['Close']
    df_values.reset_index(drop=True, inplace=True)
    for i in range(W-1, len(df_values)-2):
        # Calculate coefficients given a W
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(i-W+1, i+1), df_values.loc[i-W+1:i].values.T.astype(float))
        # X value is the the predicted X value, eg taking a window of 5, X = 6
        X = i + 1
        prediction = np.add(np.multiply(slope, X), intercept)
        # Previous W
        p_w = df_values.loc[i]
        if prediction > p_w:
            all_predictions.append('GREEN')
        else:
            all_predictions.append('RED')
    df_prediction = pd.DataFrame(np.array(all_predictions), columns=['Regression Predictions'])
    df_prediction.index += W+1

    return df_prediction

def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'
    # Read from that file for answering our questions
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    print('\nQuestion 1:')
    print('In this problem, I will effectively trade by labeling each week as either GREEN or RED depending on the following criteria: ')
    print('If P*_W+1 > P_W: This week is GREEN.')
    print('If P*_W+1 < P_W: This week is RED.')
    print('If GREEN, and there is no previous position, we will open a long position')
    print('If GREEN, and the previous week is GREEN, we will hold on to our long position.')
    print('If GREEN, and the previous week is RED, we will buy the number of shares from the previous day.')
    print('If RED, and there is no previous position, we sell short our stock with 100/P_W number of shares.')
    print('If RED, and the previous week is GREEN, we will close the long position.')
    print('If RED, and the previous week is RED, we will continue to hold our position')
    print('Additional complexity is added here with the position that we have changing over time. I will start with $100.')
    print('I am using Close price instead of Adj Close price due to dividends')
    print('Assumptions:')
    print('If I have more than $100 then I will only use $100 to open a position')
    print('If I have less than $100, I will only use that amount of money to trade')
    print('Question 1 takes some time to run, so I have split up the scripts for the first assignment.')

    pnl_deque = deque()
    for W in range(5, 31):
        print('Processing W = {} out of 30'.format(W))
        df_trading_days = pd.DataFrame(df_2018.iloc[W+1: len(df_2018.index)][['Year', 'Week_Number', 'Close']], columns=['Year', 'Week_Number', 'Close']).join(get_windowed_slice_and_fit(df_2018, W))
        df_trading_days.reset_index(drop=True, inplace=True)
        # This gives the pnl table
        pnl = trading_strategy_with_stats(df_trading_days, 'Regression Predictions')
        # Grab the means
        pnl_long, pnl_short = get_pnl(pnl)
        total_pnl = pnl_long.append(pnl_short)
        pnl_deque.append(np.round(total_pnl.mean(), 2))

    plt.plot(np.arange(5, 31), np.array(pnl_deque))
    plt.title('Moving Window Trading Strategy WMT')
    plt.xlabel('W')
    plt.ylabel('Average P/L per trade ($)')
    plt.savefig(fname='Q_1_W_vs_PnL')
    plt.show()
    plt.close()
    # Have to add 5 to it since our initial moving window is 5
    print('Best W* is {}'.format(np.add(np.argmax(np.array(pnl_deque)), 5)))

if __name__ == "__main__":
    main()
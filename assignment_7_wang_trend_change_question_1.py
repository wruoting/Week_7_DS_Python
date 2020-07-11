import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import linregress
from collections import deque
import matplotlib.pyplot as plt

def print_confusion_matrix(Y_2019, confusion_matrix_df):
    '''
    Y_2019: input vector for the confusion matrix
    confusion_matrix_df: the input confusion df
    '''
    total_data_points = len(Y_2019)
    true_positive_number = confusion_matrix_df['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print("True positive rate: {}%".format(true_positive_rate))
    print("True negative rate: {}%".format(true_negative_rate))


def make_trade(cash, open, close):
    '''
    cash: float of cash on hand
    open: float of open price
    close: float of close price
    returns: The cash made from a long position from open to close
    '''
    shares = np.divide(cash, open)
    return np.multiply(shares, close)

def trading_strategy(trading_df, prediction_label, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    prediction_label: the label for which we're going to trade off of
    returns: A df of trades made based on Predicted Labels
    '''
    # The daily balance we will be using
    total_balance = weekly_balance
    trading_history = deque()
    trading_days_index = 0
    # Boolean indicating whether we have a position or not
    position = False
    # The number of shares
    shares = 0
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
                    # A short
                    else:
                        # I can always sell short because I can borrow money
                        # I essentially need to buy these shares back
                        shares = np.negative(np.divide(100, trading_value))
                        # Add 100 dollars of shares short
                        total_balance = total_balance + 100
            # Otherwise we have a position
            else:
                # If it's the last day and we have a position, we must close out the position
                if trading_days_index == len(trading_df.index) - 1:
                    # If I'm long on the last day
                    if shares > 0:
                        total_balance = total_balance + np.multiply(shares, trading_value)
                    # If I'm short on the last day
                    elif shares < 0:
                        buy_back_shares = np.abs(shares)
                        total_balance = total_balance - np.multiply(buy_back_shares, trading_value)
                else:
                    # If we have previous trading days that are the same, we can keep going (No code necessary)
                    # We have a short position from yesterday, and we must close it by buying shares back and subtracting from our total
                    if trading_label == 'GREEN' and previous_trading_label == 'RED':
                        buy_back_shares = np.abs(shares)
                        total_balance = total_balance - np.multiply(buy_back_shares, trading_value)
                        shares = 0
                        position = False
                    # We have a long position from yesterday, and we must close it by selling shares and adding to our total
                    elif trading_label == 'RED' and previous_trading_label == 'GREEN':
                        total_balance = total_balance + np.multiply(shares, trading_value)
                        shares = 0
                        position = False

        # Regardless of whether we made a trade or not, we append the weekly cash and week over
        trading_history.append([trading_df.iloc[trading_days_index][['Year']].values[0],
                    trading_df.iloc[trading_days_index][['Week_Number']].values[0],
                    total_balance])
        # If we have no money, then we stop trading
        trading_days_index = trading_days_index+1
                
        
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Week_Number', 'Balance'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df



def get_windowed_slice_and_fit(df, W=5):
    '''
    df: the dataframe where we are splitting into time series
    return: dataframe with prediction labels
    '''
    all_predictions = deque()
    df_values = df['Close']

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
    iris_dataset = 'iris.data'

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

    pnl_deque = deque()
    for W in range(5, 31):
        print('Processing W = {} out of 30'.format(W))
        df_trading_days = pd.DataFrame(df_2018.iloc[W+1: len(df_2018.index)][['Year', 'Week_Number', 'Close']], columns=['Year', 'Week_Number', 'Close']).join(get_windowed_slice_and_fit(df_2018, W))
        df_trading_days.reset_index(drop=True, inplace=True)
        # this gives the pnl table. We only care about the last day
        pnl = trading_strategy(df_trading_days, 'Regression Predictions')
        pnl_deque.append(np.round(np.subtract(pnl.iloc[-1]['Balance'], 100), 2))

    plt.plot(np.arange(5, 31), np.array(pnl_deque))
    plt.title('Moving Window Trading Strategy WMT')
    plt.xlabel('W')
    plt.ylabel('P/L ($)')
    plt.savefig(fname='Q_1_W_vs_PnL')
    plt.show()
    plt.close()
    # Have to add 5 to it since our initial moving window is 5
    print('Best W* is {}'.format(np.add(np.argmax(np.array(pnl_deque), 5))))

if __name__ == "__main__":
    main()
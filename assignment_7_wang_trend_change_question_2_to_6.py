import pandas as pd
import numpy as np
from scipy.stats import linregress
from collections import deque
import matplotlib.pyplot as plt
from assignment_7_wang_trend_change_question_1 import get_pnl, trading_strategy_with_stats

def calculate_average_number_of_days_per_trade(df):
    '''
    df: the dataframe where we are finding the average 
    return: tuple with the average number of days for a long and average number of days for a short position
    '''
    # Split the dataframes into multiple dataframes based on LONG and SHORT
    opening_indicies_long = df.index[df['Position'] == 'LONG']
    closing_indicies_long = df.index[df['Position'] == 'LONG TO SELL']
    opening_indicies_short = df.index[df['Position'] == 'SHORT']
    closing_indicies_short = df.index[df['Position'] == 'SHORT TO BUY']
    long_positions_days = closing_indicies_long - opening_indicies_long + 1
    short_positions_days = closing_indicies_short - opening_indicies_short + 1

    return np.average(long_positions_days), np.average(short_positions_days)

def get_windowed_slice_and_fit_with_params(df, W=5):
    '''
    df: the dataframe where we are splitting into time series
    return: dataframe with prediction labels, r_values, p_values, std_err
    '''
    all_predictions = deque()
    r_values = deque()
    p_values = deque()
    std_errs = deque()
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
        r_values.append(r_value)
        p_values.append(p_value)
        std_errs.append(std_err)
        if prediction > p_w:
            all_predictions.append('GREEN')
        else:
            all_predictions.append('RED')
    df_prediction = pd.DataFrame(np.array([np.array(all_predictions), np.array(r_values), np.array(p_values), np.array(std_errs)]).T,
        columns=['Regression Predictions', 'R Values', 'P Values', 'Std Errors'])
    df_prediction.index += W+1

    return df_prediction


def question_2(df_trading_days, df_year, year, fname, W):
    '''
    df_trading_days: the dataframe for our trading days
    df_year: dataframe from the year we are trading with
    year: the year we are working with
    fname: name of the file we are opening/saving
    W: the range we want to work with
    return: None
    '''
    fig, ax1 = plt.subplots()
    # We have to start from day 21 of trading
    x_values_regression = df_trading_days.index.values + W
    x_values_trading_days = df_year.index.values
    color = 'tab:blue'
    ax1.set_xlabel('Day of {}'.format(year))
    ax1.set_ylabel('Price', color=color)
    ax1.plot(x_values_trading_days, df_year['Close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('R Values', color=color)
    ax2.plot(x_values_regression, df_trading_days[['R Values']].astype(float).values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.grid(True)
    fig.set_size_inches(10, 8)
    plt.title('R Squared values in relation to closing prices')
    plt.savefig(fname=fname)
    plt.show()
    plt.close()

def question_3(pnl_table):
    '''
    pnl_table: the dataframe our profit and loss
    return: None
    '''
    # Every time a position is opened is counted
    all_open_positions = pnl_table.query('Position in ["LONG", "SHORT"]').count()['Position']
    all_long_positions = pnl_table.query('Position in ["LONG"]').count()['Position']
    all_short_positions = pnl_table.query('Position in ["SHORT"]').count()['Position']
    print('The number of total positions opened is {}'.format(all_open_positions))
    print('The number of long positions opened is {}'.format(all_long_positions))
    print('The number of short positions opened is {}'.format(all_short_positions))


def main():
    ticker='WMT'
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    # Need to reset index on df_2019
    df_2019.reset_index(drop=True, inplace=True)
    
    print('\nQuestion 2:')
    print('The best W* from the previous problem was 20')
    W = 20
    fname_2019 = 'Q_2_R_Squared_to_Days'
    df_trading_days = pd.DataFrame(df_2019.iloc[W+1: len(df_2019.index)][['Year', 'Week_Number', 'Close']], columns=['Year', 'Week_Number', 'Close']).join(get_windowed_slice_and_fit_with_params(df_2019, W))
    df_trading_days.reset_index(drop=True, inplace=True)
    question_2(df_trading_days, df_2019, 2019, fname_2019, W)
    print('The average R squared value is: {}'.format(np.round(np.average(df_trading_days[['R Values']].astype(float)), 2)))
    print('The R squared value describes the correlation coefficient of a relationship between the day of the year and price. A strong negative correlation indicates a ')
    print('negative price movement over time, and a strong positive correlation indicates a price movement upwards over time. Periods of the stock where there are clear drops shows ')
    print('a negative correlation, but only after a window of time. The R squared movement is lagged due to the moving window effect. However, the movements of the R squared values and prices ')
    print('can still be observed. There is an average R squared of 0.3 which indicates that the stock trends up with time, as seen by the ticker movement.')

    print('\nQuestion 3:')
    # Create some regression prediction tables
    pnl_table = trading_strategy_with_stats(df_trading_days, 'Regression Predictions')
    # Every time a position is opened is counted
    question_3(pnl_table)

    print('\nQuestion 4:')
    pnl_from_long, pnl_from_short = get_pnl(pnl_table)
    print("Average P/L from longs: ${}".format(np.round(pnl_from_long['Balance'].mean(), 2)))
    print("Average P/L from shorts: ${}".format(np.round(pnl_from_short['Balance'].mean(), 2)))

    print('\nQuestion 5:')
    print('I am including the day that I start the position and the day that I close the position')
    average_long_days, average_short_days = calculate_average_number_of_days_per_trade(pnl_table)
    print('Average long days: {}'.format(np.round(average_long_days, 2)))
    print('Average short days: {}'.format(np.round(average_short_days, 2)))

    print('\nQuestion 6:')
    print('Comparing to Question 2: ')
    fname_2018 = 'Q_6_R_Squared_to_Days_2018'
    df_trading_days_2018 = pd.DataFrame(df_2018.iloc[W+1: len(df_2018.index)][['Year', 'Week_Number', 'Close']], columns=['Year', 'Week_Number', 'Close']).join(get_windowed_slice_and_fit_with_params(df_2018, W))
    df_trading_days_2018.reset_index(drop=True, inplace=True)
    question_2(df_trading_days_2018, df_2018, 2018, fname_2018, W)
    print('The same kind of pattern exists between the two years. R squared values change with a lag ')
    print('Comparing to Question 3: ')
    # Create some regression prediction tables
    pnl_table_2018 = trading_strategy_with_stats(df_trading_days_2018, 'Regression Predictions')
    # Every time a position is opened is counted
    question_3(pnl_table_2018)
    print('There were slightly more opened positions in 2018 compared to 2019 (40 to 34), with significantly more short positions in 2018 (25 to 15)')
    print('Comparing to Question 4: ')
    pnl_from_long, pnl_from_short = get_pnl(pnl_table_2018)
    print("Average P/L from longs: ${}".format(np.round(pnl_from_long['Balance'].mean(), 2)))
    print("Average P/L from shorts: ${}".format(np.round(pnl_from_short['Balance'].mean(), 2)))
    print('Compared to 2019, 2018 saw profits, which means that this model fitted for 2018 works for 2018, but not as well for 2019. This indicates that the window used for 2019 was most likely ')
    print('not optimal. A different window would most likely provide better results.')
    print('Comparing to Question 5: ')
    average_long_days, average_short_days = calculate_average_number_of_days_per_trade(pnl_table_2018)
    print('Average long days: {}'.format(np.round(average_long_days, 2)))
    print('Average short days: {}'.format(np.round(average_short_days, 2)))
    print('The number of average long and short days are not significantly different. This could indicate that there is a relationship between the window and the average trading days.')
    print('Overall, the patterns between 2018 and 2019 carry over to be not significantly different (at least from a heuristic perspective). This indicates that the method we are trading with ')
    print('is being carried over. This makes sense, since we are using an optimal window to trade with between years. However, 2019 showed losses while 2018 showed gains with the same window for ')
    print('trading. This just means that the model does not work well across these two years.')



if __name__ == "__main__":
    main()
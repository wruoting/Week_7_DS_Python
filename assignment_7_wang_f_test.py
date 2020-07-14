import pandas as pd
import numpy as np
from assignment_7_wang_lin_regression import Stats
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import f as fisher_f
from sklearn.linear_model import LinearRegression

def calculate_f_distribution_significance(L, L_1, L_2, n):
    '''
    L: the SSE of the regression
    L_1: the SSE of the first split data set
    L_2: the SSE of the second split data set
    n: the number of entries
    return: the fisher statistic
    '''
    return (L - (L_1 + L_2) / 2) * ((n-4) / (L_1 + L_2))


def calculate_sse(dataframe, split_index):
    '''
    dataframe: the dataframe that requires calculation by splitting indices
    return: a tuple of each sse
    '''
    if split_index == 0 or split_index == len(dataframe)-1:
        raise Exception("Cannot calculate SSE of a split with no two lines")
    df_1 = dataframe[0:split_index]
    df_1.reset_index(drop=True, inplace=True)
    df_2 = dataframe[split_index+1:-1]
    df_2.reset_index(drop=True, inplace=True)
    df_len = len(dataframe.index.values)
    stats_1 = Stats(df_1.index.values.reshape(split_index,), df_1[["Close"]].values.reshape(split_index,))
    stats_1.fit()
    stats_2 = Stats(df_2.index.values.reshape(df_len-split_index-2,), df_2[["Close"]].values.reshape(df_len-split_index-2,))
    stats_2.fit()
    return stats_1.sse(), stats_2.sse()


def calculate_min_sse_split_index(dataframe):
    '''
    dataframe: dataframe in which we will iterate through and find all linear regressions based on a split on indices
    return: a tuple of the optimal index to split the array by and individual sse values
    '''
    sse_index = deque()
    sse_value_split = deque()
    sse_values_individual = deque()
    # Split indices from 2 to n-2
    # My assumption is that 
    for split_index in range(2, len(dataframe.index.values)-3):
        std_err_1, std_err_2 = calculate_sse(dataframe, split_index)
        sse_index.append(split_index)
        sse_value_split.append(std_err_1 + std_err_2)
        sse_values_individual.append([std_err_1, std_err_2])
    sse_index = np.array(sse_index)
    sse_value_split = np.array(sse_value_split)
    sse_values_individual = np.array(sse_values_individual)
    return sse_index[np.argsort(sse_value_split)[0]], sse_values_individual[np.argsort(sse_value_split)[0]]

def split_data(dataframe, p=0.1):
    '''
    dataframe: the dataframe for our trading year
    return: the minimum SSE days for each month
    '''
    # Split dataframe for the year into months
    months = np.unique(dataframe[['Month']].values)
    group_by_month = dataframe.groupby('Month')

    indices = deque()
    sse1 = deque()
    sse2 = deque()
    f_values = deque()
    p_values = deque()
    is_critical = deque()
    for month in months:
        df_month = group_by_month.get_group(month)
        df_len = len(df_month.index.values)
        # The main regression set
        linear_regression_model = Stats(df_month.index.values.reshape(df_len,), df_month[["Close"]].values.reshape(df_len,))
        linear_regression_model.fit()
        split_index, individual_values = calculate_min_sse_split_index(df_month)
        f_value = calculate_f_distribution_significance(linear_regression_model.sse(), individual_values[0], individual_values[1], df_len)
        p_value = fisher_f.cdf(f_value, 2, df_len-4)
        # Column vectors for df
        f_values.append(f_value)
        indices.append(split_index)
        sse1.append(individual_values[0])
        sse2.append(individual_values[1])
        p_values.append(p_value)
        is_critical.append(p_value < p)
    df_array = np.array([months.T, np.array(indices).T, np.array(sse1).T, np.array(sse2).T, np.array(f_values).T, np.array(p_values).T, np.array(is_critical).T]).T
    return pd.DataFrame(df_array, columns=["Month", "Split Index", "SSE_1", "SSE_2", "F values", "P values", "Is Critical"])

def main():
    ticker='WMT'
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    # Need to reset index on df_2019
    df_2019.reset_index(drop=True, inplace=True)
    print('I will be using close instead of adj close')
    # Process
    # 1. Take a set of days n, and split into 1...k and k+1 ... n
    # 2. Construct 2 regressions and calculate the loss functions for the SSE
    # 3. Look for a k that minimizes the loss functions
    # 4. Construct an F statistic
    # Assumptions:
    # K must be: 2 < k < n-1, because we cannot have a split of say 1 and 2....n, since one point cannot make a linear regression
    print('Question 1:')
    print('For the year 2018:')
    print('The candidate days are listed as the "Split Index" of the table below. A split index of 10 would indicate that the month is split into 10 and n-10 days')
    data_2018_f_table = split_data(df_2018)
    print(data_2018_f_table)
    print('The critical months were:')
    print(data_2018_f_table[data_2018_f_table["Is Critical"] == 1])
    print('For the year 2019:')
    print('The candidate days are listed as the "Split Index" of the table below. A split index of 10 would indicate that the month is split into 10 and n-10 days')
    data_2019_f_table = split_data(df_2019)
    print(data_2019_f_table)
    print('The critical months were:')
    print(data_2019_f_table[data_2019_f_table["Is Critical"] == 1])

    print('Question 2:')
    print('In 2018, there is significant change in price trending for the following months: {}'.format(data_2018_f_table[data_2018_f_table["Is Critical"] == 1][["Month"]].values.T.flatten()))
    print('In 2019, there is significant change in price trending for the following months: {}'.format(data_2019_f_table[data_2019_f_table["Is Critical"] == 1][["Month"]].values.T.flatten()))

    print('Question 3')
    print('There are more changes in year 2019, or year 2. This could imply a more volatile year of trading, as there are more significant shifts in trend.')
if __name__ == "__main__":
    main()
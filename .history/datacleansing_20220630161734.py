import pandas as pd
import numpy as np


# get the new dataframe of weekday pedesdrian
# according to given year and hour
def getCountHourly(df, year, hour):
    df_new = df[(df['Year'] == year) & (df['Time'] == hour) & (df['Day']
                .isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))]
    return df_new


# get the median, mean, maximum, minimum of the given dataframe according to 'Hourly_Counts' column
def summaryHourlyCount(df, time):
    result = pd.DataFrame(columns=['time', 'median', 'mean', 'max', 'min'])
    df = df.groupby(['Month', 'Mdate'], as_index=False)["Hourly_Counts"].sum()
    median = np.median(df["Hourly_Counts"])
    mean = np.mean(df["Hourly_Counts"])
    max = np.max(df["Hourly_Counts"])
    min = np.min(df["Hourly_Counts"])
    data = {'time':time, 'median':median, 'mean': mean, 'max':max, 'min':min}
    result = result.append(data, ignore_index=True)

    return result

# get the hourly count of pedestrians based on given conditions
def sum_by_year(filename, condition):
    df = pd.read_csv(filename)
    df_2021 = df[df['Year'] == 2021]
    df_2022 = df[df['Year'] == 2022]
    count_2021 = df_2021.groupby(condition)["Hourly_Counts"].sum()
    count_2022 = df_2022.groupby(condition)["Hourly_Counts"].sum()
    return (count_2021, count_2022)

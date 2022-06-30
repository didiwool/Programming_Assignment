import pandas as pd
import numpy as np
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

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


# data cleansing helper for question 5, 7, 8
def dailyCount(df):
    df = df.groupby('Day').mean()
    df = pd.DataFrame(df.to_dict()['Hourly_Counts'].items(), columns= ["Day","Hourly Counts"])
    df.Day = df.Day.astype("category")
    df.Day = df.Day.cat.set_categories(WEEK)
    df = df.sort_values("Day") 
    df = df.set_index(df["Day"], drop = True)
    return df

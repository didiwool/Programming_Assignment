import pandas as pd
import numpy as np


# get the new dataframe of weekday pedesdrian
# according to given year and hour
def getCountHourly(df, year, hour):
    df_new = df[(df['Year'] == year) & (df['Time'] == hour) & (df['Day']
                .isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))]
    return df_new


# get the median, mean, maximum, minimum of the given dataframe according to 'Hourly_Counts' column
def summaryHourlyCount(df):
    result = pd.DataFrame(columns=['median', 'mean', 'max', 'min'])
    df = df.groupby(['Month', 'Mdate'], as_index=False)["Hourly_Counts"].sum()
    median = np.median(df["Hourly_Counts"])
    mean = np.mean(df["Hourly_Counts"])
    max = np.max(df["Hourly_Counts"])
    min = np.min(df["Hourly_Counts"])
    
    result['median'] = median
    result['mean'] = mean
    result['max'] = max
    result['min'] = min

    return result


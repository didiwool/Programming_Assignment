from datacleansing import *
from graphplot import *
import pandas as pd
import numpy as np

# the generalized function of question 1
def pedestrianStats(filename, year, hours):
    df = pd.read_csv(filename)
    new_df = pd.DataFrame(columns=['time', 'median', 'mean', 'max', 'min'])

    for time in hours:
        hourly_df = getCountHourly(df, year, time)
        result = summaryHourlyCount(hourly_df, time)
        new_df = pd.concat([new_df, result], ignore_index=True)

    return new_df
from datacleansing import *
from graphplot import *
import pandas as pd
import numpy as np

# the generalized function of question 1
def pedestrianStats(filename, year, hours):
    df = pd.read_csv(filename)
    new_df = pd.DataFrame(columns=['median', 'mean', 'max', 'min'])
    for time in hours:
        df = getCountHourly(df, year, time)
        result = summaryHourlyCount(df)
        new_df.append(result, ignore_index=True)

    return new_df
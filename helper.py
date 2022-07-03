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
    df = pd.DataFrame(df.to_dict()['Hourly_Counts'].items(), columns= ["Day","Hourly Counts"])
    df.Day = df.Day.astype("category")
    df.Day = df.Day.cat.set_categories(WEEK)
    df = df.sort_values("Day") 
    df = df.set_index(df["Day"], drop = True)
    return df

# function for preparing the train data or test data in Q10
def dataForCount(df, nearby, sensor_id, start_time, end_time):
    rain_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April"]) ) & (df.Date_Time.dt.strftime('%m-%d') != '04-30')].sort_values(by = ['Date_Time'])['Rainfall amount (millimetres)'])
    solar_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April"]) ) & (df.Date_Time.dt.strftime('%m-%d') != '04-30')].sort_values(by = ['Date_Time'])['Daily global solar exposure (MJ/m*m)'])
    temp_prev = np.array(df[(df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April"])) & (df.Date_Time.dt.strftime('%m-%d') != '04-30')].sort_values(by = ['Date_Time'])['Maximum temperature (Degree C)'])
    sensor3_past1 = np.array(df[(df.Sensor_ID == sensor_id) & (df.Time >= start_time-1) & (df.Time <= end_time-2) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April"])) & (df.Date_Time.dt.strftime('%m-%d') != '01-01')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    nearby_past1 = np.array(df[(df.Sensor_ID == nearby) & (df.Time >= start_time-1) & (df.Time <= end_time-2) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April"])) & (df.Date_Time.dt.strftime('%m-%d') != '01-01')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    sensor3_pastday = np.array(df[(df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month.isin(["January", "February", "March", "April"])) & (df.Date_Time.dt.strftime('%m-%d') != '04-30')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    
    return (rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1, sensor3_pastday)



def testdataForCount(df, nearby, sensor_id, start_time, end_time, day):
    dictDay = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    if day == 'Monday':
        yesterday = 'Sunday'
    else:
        for k, v in dictDay.items():
            if v == dictDay[day] - 1:
                yesterday = k
    

    rain_prev = np.array(df[(df.Day == yesterday) & (df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month == 'May' ) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Rainfall amount (millimetres)'])
    solar_prev = np.array(df[(df.Day == yesterday) & (df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month == 'May' ) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Daily global solar exposure (MJ/m*m)'])
    temp_prev = np.array(df[(df.Day == yesterday) & (df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month == 'May' ) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Maximum temperature (Degree C)'])
    sensor3_past1 = np.array(df[(df.Day == day) & (df.Sensor_ID == sensor_id) & (df.Time >= start_time-1) & (df.Time <= end_time-2) & (df.Year ==2022) &(df.Month == 'May' ) & (df.Date_Time.dt.strftime('%m-%d') != '05-01')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    nearby_past1 = np.array(df[(df.Day == day) & (df.Sensor_ID == nearby) & (df.Time >= start_time-1) & (df.Time <= end_time-2) & (df.Year ==2022) &(df.Month == 'May' ) & (df.Date_Time.dt.strftime('%m-%d') != '05-01')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    sensor3_pastday = np.array(df[(df.Day == yesterday) & (df.Sensor_ID == sensor_id) & (df.Time >= start_time) & (df.Time <= end_time-1) & (df.Year ==2022) &(df.Month == 'May' ) & (df.Date_Time.dt.strftime('%m-%d') != '05-31')].sort_values(by = ['Date_Time'])['Hourly_Counts'])
    print(rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1, sensor3_pastday)
    return (rain_prev, solar_prev, temp_prev, sensor3_past1, nearby_past1, sensor3_pastday)


def computeDistance(distance, merge_df):
    for day in merge_df['Date_Time'].unique():
        count_sensor1 = merge_df[merge_df.Date_Time == day]['Hourly_Counts_x']
        count_sensor2 = merge_df[merge_df.Date_Time == day]['Hourly_Counts_y']    
        distance[day] = np.linalg.norm(np.array(count_sensor1)-np.array(count_sensor2))

    return distance


def pearsonDistance(pearson_coef, compare_merged):
    for day in compare_merged['Date_Time'].unique():
        count_sensor1 = compare_merged[compare_merged.Date_Time == day]['Hourly_Counts_x']
        count_sensor2 = compare_merged[compare_merged.Date_Time == day]['Hourly_Counts_y']     
        pearson_coef[day] = np.abs(np.corrcoef(np.array(count_sensor1),np.array(count_sensor2))[1,0])
    
    return pearson_coef


def diffConclusion(e_distance):
    max_day = max(e_distance, key=e_distance.get) 
    max_change = e_distance[max_day]
    min_day = min(e_distance, key=e_distance.get) 
    min_change = e_distance[min_day]
    print("Day with the most change is " + str(max_day) + ", and the greatest change is " + str(round(max_change)) + ".")
    print("Day with the least change is " + str(min_day) + ", and the least change is " + str(round(min_change)) + ".")



def dataframeToDict(df, drop):
    """
    Convert dataframe df to dictionary using date_key as key, drop the columns listed in the list drop.
    """
    df["date_key"] = df["Year"].astype(str) + df["Month"].astype(str) + df["Day"].astype(str)
    df.drop(drop, axis = 1, inplace = True)
    df_dict = list(df.set_index(df.date_key).drop("date_key", axis = 1).to_dict().values())[0]
    return df_dict
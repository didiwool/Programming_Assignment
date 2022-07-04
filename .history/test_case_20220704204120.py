import re
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch
import pandas.testing
from graphplot import bar_with_title, time_series_for_two, \
    unusual_day_plot
from helper import get_count_hourly, summary_hourly_count, \
    daily_count, data_for_count, test_data_for_count, \
    compute_distance, pearson_distance, \
    diff_conclusion, find_extreme_item
from solution import data_cleansing, pedestrian_stats, pedestrian_scatter, \
    pedestrian_hist, sensor_hist, pedestrian_hist_rain, \
    pedestrian_hist_rain_temp, \
    time_series_sensor, model_for_count, unusual_day, daily_difference, \
    sensor_correlation, invest_covid_travel, invest_lockdown


COUNT_TEST = "count_test.csv"
RAIN_TEST = "rainfall_test.csv"
TEMP_TEST = "temperature_test.csv"
SOLAR_TEST = "solar_test.csv"

class TestExtremeItem(unittest.TestCase):

    def test_normal(self):
        dict = {'Monday': 30, 'Tuesday': 10, 'Wednesday': 20, 'Thursday': 50}
        result = find_extreme_item(dict)
        self.assertEqual(result, ('Thursday', 'Tuesday'))

    def test_empty(self):
        dict = {}
        result = find_extreme_item(dict)
        self.assertEqual(result, ('', ''))



class TestDataCleansing(unittest.TestCase):

    def test_files(self):
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        df_target = pd.DataFrame(columns=['ID', 'Date_Time', \
            'Year', 'Month', 'Mdate', 'Day', 'Time', 'Sensor_ID', \
            'Sensor_Name', 'Hourly_Counts', 'date_key', 'Rainfall amount (millimetres)', \
            'Maximum temperature (Degree C)', 'Daily global solar exposure (MJ/m*m)'])
        count_file = pd.read_csv(COUNT_TEST)
        df_target[['ID', 'Date_Time', 'Year', 'Month', 'Mdate', 'Day', \
            'Time', 'Sensor_ID', 'Sensor_Name', 'Hourly_Counts']] = count_file
        df_target['Date_Time'] = pd.to_datetime(df_target['Date_Time'])
        df_target['date_key'] = '202252'
        df_target['Rainfall amount (millimetres)'] = 0.0
        df_target['Maximum temperature (Degree C)'] = 22.1
        df_target['Daily global solar exposure (MJ/m*m)'] = 9.9
        self.assertEqual(True, dataframe.equals(df_target))

    def test_wrong_file(self):
        with self.assertRaises(IOError):
            data_cleansing('wrong.csv', RAIN_TEST, TEMP_TEST, SOLAR_TEST)


class TestHourlyCount(unittest.TestCase):

    def test_0_am(self):
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        result = get_count_hourly(dataframe, 2022, 0)    
        time = result.iloc[0]['Time'].astype(int)

        self.assertEqual(time, 0)


    def test_12_pm(self):
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)

        with self.assertRaises(IndexError):
            get_count_hourly(dataframe, 2022, 24)


class TestSummaryHourlyCount(unittest.TestCase):

    def test_max(self):
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        summary = summary_hourly_count(dataframe, 12).iloc[0]['max']

        self.assertEqual(summary, float(dataframe['Hourly_Counts'].sum()))
    
    def test_mean(self):
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        summary = summary_hourly_count(dataframe, 12).iloc[0]['mean']

        self.assertEqual(summary, float(dataframe['Hourly_Counts'].sum()))
    
    def test_error(self):
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        with self.assertRaises(KeyError):
            summary_hourly_count(dataframe, 12).iloc[0]['meaning']


class TestDiffConclusion(unittest.TestCase):

    def test_euclidean(self):
        dict = {'Monday': 10, 'Tuesday': 25, 'Wednesday': 30}
        with patch('builtins.print') as mocked_print:
            diff_conclusion(dict, 'Euclidean distance')
            mocked_print.assert_called_with("Day with the "+
                "least Euclidean distance is Monday, and the value is 10.")

    def test_pearson(self):
        dict = {'Monday': 50, 'Tuesday': 25, 'Wednesday': 30}
        with patch('builtins.print') as mocked_print:
            diff_conclusion(dict, 'Pearson correlation coefficient')
            mocked_print.assert_called_with("Day with the "+
                "least Pearson correlation coefficient is Tuesday, and the value is 25.")

    
class TestDailyCount(unittest.TestCase):

    def test_count(self):
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        df_new = daily_count(dataframe)
        df_new.drop('Day', axis=1, inplace=True)
        df_new = df_new.reset_index()
        self.assertEqual(True, df_new['Hourly Counts'].equals(dataframe['Hourly_Counts']))




if __name__ == '__main__':
    unittest.main()

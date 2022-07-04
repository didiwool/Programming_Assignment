"""
Test File for the code.
"""
import pandas as pd
import unittest
from unittest.mock import patch
from helper import get_count_hourly, summary_hourly_count, \
    daily_count, diff_conclusion, find_extreme_item
from solution import data_cleansing


COUNT_TEST = "count_test.csv"
RAIN_TEST = "rainfall_test.csv"
TEMP_TEST = "temperature_test.csv"
SOLAR_TEST = "solar_test.csv"


class TestExtremeItem(unittest.TestCase):
    """
    test the function of find_extreme_item in helper.py
    """
    def test_normal(self):
        """
        test using a dictionary as input
        """
        dictionary = {'Monday': 30, 'Tuesday': 10, 'Wednesday': 20, 'Thursday': 50}
        result = find_extreme_item(dictionary)
        self.assertEqual(result, ('Thursday', 'Tuesday'))

    def test_empty(self):
        """
        test using an empty dictionary
        """
        dictionary = {}
        result = find_extreme_item(dictionary)
        self.assertEqual(result, ('', ''))



class TestDataCleansing(unittest.TestCase):
    """
    test data_cleansing function in solution.py
    """
    def test_files(self):
        """
        test the correctness of function using new csv files
        """
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
        """
        test using wrong file names as input of function
        """
        with self.assertRaises(IOError):
            data_cleansing('wrong.csv', RAIN_TEST, TEMP_TEST, SOLAR_TEST)


class TestHourlyCount(unittest.TestCase):
    """
    test the function of get_hourly_count in helper.py
    """
    def test_0_am(self):
        """
        test by input of 0am which is a valid input
        """
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        result = get_count_hourly(dataframe, 2022, 0)
        time = result.iloc[0]['Time'].astype(int)

        self.assertEqual(time, 0)


    def test_12_pm(self):
        """
        test by input of 12pm which is not a valid input
        """
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        with self.assertRaises(IndexError):
            get_count_hourly(dataframe, 2022, 24)


class TestSummaryHourlyCount(unittest.TestCase):
    """
    test the function of summary_hourly_count in helper.py
    """
    def test_max(self):
        """
        test the 'max' part of the output
        """
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        summary = summary_hourly_count(dataframe, 12).iloc[0]['max']

        self.assertEqual(summary, float(dataframe['Hourly_Counts'].sum()))

    def test_mean(self):
        """
        test the 'mean' part of the output
        """
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        summary = summary_hourly_count(dataframe, 12).iloc[0]['mean']

        self.assertEqual(summary, float(dataframe['Hourly_Counts'].sum()))

    def test_error(self):
        """
        test with a wrong index name, should raise a key error
        """
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        with self.assertRaises(KeyError):
            summary_hourly_count(dataframe, 12).iloc[0]['meaning']


class TestDiffConclusion(unittest.TestCase):
    """
    test the function of diff_conclusion in helper.py
    """
    def test_euclidean(self):
        """
        test a euclidean output, using the patch package to
        compare the last print output
        """
        dictionary = {'Monday': 10, 'Tuesday': 25, 'Wednesday': 30}
        with patch('builtins.print') as mocked_print:
            diff_conclusion(dict, 'Euclidean distance')
            mocked_print.assert_called_with("Day with the "+
                "least Euclidean distance is Monday, and the value is 10.")

    def test_pearson(self):
        """
        test a pearson coefficient output
        """
        dict = {'Monday': 50, 'Tuesday': 25, 'Wednesday': 30}
        with patch('builtins.print') as mocked_print:
            diff_conclusion(dict, 'Pearson correlation coefficient')
            mocked_print.assert_called_with("Day with the "+
                "least Pearson correlation coefficient is Tuesday, and the value is 25.")


class TestDailyCount(unittest.TestCase):
    """
    test the function of daily_count in helper.py
    """
    def test_count(self):
        """
        test the 'hourly counts' column after aggregation
        """
        dataframe = data_cleansing(COUNT_TEST, RAIN_TEST, TEMP_TEST, SOLAR_TEST)
        df_new = daily_count(dataframe)
        df_new.drop('Day', axis=1, inplace=True)
        df_new = df_new.reset_index()
        self.assertEqual(True, df_new['Hourly Counts'].equals(dataframe['Hourly_Counts']))




if __name__ == '__main__':
    unittest.main()

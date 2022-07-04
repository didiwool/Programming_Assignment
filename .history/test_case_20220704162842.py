import pandas as pd
import numpy as np
import unittest
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

class TestExtremeItem(unittest.TestCase):

    def test_normal(self):
        dict = {'Monday': 30, 'Tuesday': 10, 'Wednesday': 20, 'Thursday': 50}
        result = find_extreme_item(dict)
        self.assertEqual(result, ('Thursday', 'Tuesday'))

    def test_empty(self):
        dict = {}
        result = find_extreme_item(dict)
        self.assertEqual(result, ('', ''))


if __name__ == '__main__':
    unittest.main()

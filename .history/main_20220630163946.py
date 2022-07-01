from helper import *
from graphplot import *
from solution import *
import pandas as pd
import numpy as np

# files for reading, change the route name or file name if necessary
count_file = "count2021-2022.csv"
rain_file = "rainfall-all-years/IDCJAC0009_086338_1800_Data.csv"
temp_file = "temperature-all-years/IDCJAC0010_086338_1800_Data.csv"
solar_file = "solar-all-years/IDCJAC0016_086338_1800_Data.csv"

#  data cleansing part
df = data_cleansing(count_file, rain_file, temp_file, solar_file)

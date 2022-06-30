from helper import *
from graphplot import *
from solution import *
import pandas as pd
import numpy as np

#  data cleansing part
count_file = "data/count2021-2022.csv"
rain_file = "data/IDCJAC0009_086338_1800_Data.csv"
temp_file = "data/IDCJAC0010_086338_1800_Data.csv"
solar_file = "data/IDCJAC0016_086338_1800_Data.csv"

df = data_cleansing(count_file, rain_file, temp_file, solar_file)
print(df)
from datacleansing import *
from graphplot import *
from solution import *
import pandas as pd
import numpy as np

#  work out question 1 based on requirement 
df = pedestrianStats("count2021-2022.csv", 2022, [8, 13, 17])
df
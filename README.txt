**Purpose**
This program is created to run basic data wrangling, plotting, and analysis for pedestrian activity in Melbourne for years 2021 and 2022,
using the data from Hourly Counts Sensors from various points in the city, as well as Bureau of Meteorology data on rainfall, temperature, and solar exposure. 

**How to use this?**

Python version: 3.9.10
Required packages: pandas, numpy, collections, tabulate, matplotlib, sklearn, seaborn
Run the file main.py and the results of each problem in the assignment will be either printed in the command line or saved to the relative path. 

**Structure and Modules**

1.) *Helper Module and Graphplot Module*

The Helper Module has been created to contain functions that can be used to get commonly used information that is used for the solutions for each question. 
Example of this is the weekly pedestrian hourly count, basic statistical information such as the mean weekly count. 

The Graphplot Module has been created for repeatedly used plotting codes. 

These information and functions are made to be used and plugged into the Solution Module. Refer to the docstrings for descriptions of each function and other relevant variables.

2.) *Solution Module*

This contains the functions for solving each of the questions in the assignment. General information needed such as weekly counts have already been derived through the *Helper Function*, which can be imported into the *Solution Module*.

Refer to the docstrings for descriptions of each function and other relevant variables.

3.) *Main Module*

The *Main Module* summarises the outputs and answers to all questions in the assignment. Computations and plots can be derived through importing the *Solution Module* for the answers. Load in the relevant CSV files for the data. Filenames may be changed based on the necessity. 

**Testing**
Tested Functions: 
find_extreme_items, data_cleansing, get_hourly_count, summary_hourly_count, diff_conclusion, daily_count

Code for running the test code: python3 test_case.py
12 tests, all passed

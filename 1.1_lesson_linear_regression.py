"""Exercise: Simple Data Plotting
The aim of this exercise is to plot TV Ads vs Sales based on the Advertisement dataset which should look similar to the graph given below. 

Instructions:
Read the Advertisement data and view the top rows of the dataframe to get an understanding of the data and the columns.

Select the first 7 observations and the columns TV and sales  to make a new data frame.

Create a scatter plot of the new data frame TV budget vs sales .
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

# "Advertising.csv" containts the data set used in this exercise
data_filename = 'Advertising.csv'

# Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)

# Get a quick look of the data
df.iloc[0:7]

### edTest(test_pandas) ###
# Create a new dataframe by selecting the first 7 rows of
# the current dataframe
df_new = df.head(7) # type: ignore

# Print your new dataframe to see if you have selected 7 rows correctly
print(df_new)

# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df_new['TV'], df_new['Sales'], s=10, color='blue')

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel("TV budget")
plt.ylabel("Sales")

# Add plot title 
plt.title("Exercise plot")


"""Post-Exercise Question
Instead of just plotting seven points, experiment to plot all points."""

# Your code here
# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df['TV'], df['Sales'], s=10, color="red")

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel("TV budget")
plt.ylabel("Sales")

# Add plot title 
plt.title("Post-Exercise plot")
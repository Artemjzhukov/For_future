import seaborn as sbn
import pandas as pd
import matplotlib.pyplot as plt # import matplotlib
from scipy import stats
import numpy as np
#FILTER DATA & AUTO NAMING

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1
Lower_Fence = Q1 - (1.5 * IQR)
Upper_Fence = Q3 + (1.5 * IQR)
df[(df["Age"] < Lower_Fence) |(df["Age"] > Upper_Fence)]

#Filter out the outlier data and print only the potential data. To do so, just negate the above result using ~ operator
df[~((df["Age"] < Lower_Fence) |(df["Age"] > Upper_Fence))]

# Sort df by titles
df_sorted = df.sort_values(by=('Titles'), ascending=False)

# Make a programmatic title
team_with_most_titles = df_sorted['Team'][0] # get team with most titles
most_titles = df_sorted['Titles'][0] # get the number of max titles
title = 'The {} have the most titles with {}'.format(team_with_most_titles, most_titles) # create title
print(title)

-------------------------
# find the index of the greatest value in list y
index_of_max_y = y.index(max(y))
# determine the most sold item
most_sold_item = x[index_of_max_y]
plt.title('{} Produce the Most Sales Revenue'.format(most_sold_item)) # create programmatic title

###BOX

sbn.boxplot(df['Age'])

plt.boxplot(y) # generate boxplot
plt.title(title) # programmatic title
plt.show() # print plot

###LINE

# Create the plot
plt.plot(x, y, '*:b') # plot items sold (y) by month (x)
plt.xlabel('Month') # label x-axis
plt.ylabel('Items Sold') # label y-axis
plt.title('Items Sold has been Increasing Linearly') # add plot title
plt.show() # print plot

plt.figure(figsize=(10,5)) # increase plot size
plt.plot(x, y, 'D-k') # connect markers with a solid line
plt.plot(x, y2, '--r', label='x squared')  # add a line for y2
plt.xlabel('Linearly Spaced Numbers') # add x axis label
plt.ylabel('y Value') # add y axis label
plt.title('As x increases, \nx Cubed (black) increases \nat a Greater Rate than \nx Squared (red)', fontsize=22) # increase font size and multi-line
plt.legend(loc='upper left') # create a plot legend and place it in the upper left
plt.show()

##BAR

plt.bar(x, y) # plot revenue by group
plt.xlabel('Item Type') # x-axis label
plt.ylabel('Sales Revenue ($)') # y-axis label
plt.title('{} Produce the Most Sales Revenue'.format(most_sold_item)) # create programmatic title
plt.show() # print the plot

plt.bar(df_sorted['Team'], df_sorted['Titles'], color='red') # plot titles by team and make bars red
plt.xlabel('Team') # create x label
plt.ylabel('Number of Championships') # create y label
plt.xticks(rotation=45) # rotate x tick labels 45 degrees
plt.title(title) # title
plt.savefig('Titles_by_Team') # save figure to present working directory
plt.savefig('Titles_by_Team', bbox_inches='tight') # fix the cropping issue
plt.show() # print plot

#HORISONTAL BAR
plt.barh(x, y) # turn the plot horizontal

##HISTOGRAM
plt.hist(y, bins=20)
plt.xlabel('y Value')
plt.ylabel('Frequency')
plt.title(normal_YN) # programmatic plot title
plt.show()

#SCATTER
plt.scatter(x, y) # generate scatterplot
plt.xlabel('Weight') # label x-axis
plt.ylabel('Height') # label y-axis
plt.title(title) # set programmatic title
plt.show() # print plot

###SUB PLOTS
fig, axes = plt.subplots() # create figure and set of axes ????????????
fig, axes = plt.subplots(nrows=1, ncols=2)
--------------
# Enlarge the figure size and Save the figure
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,8)) # for figure size
# line plot (top left)
axes[0,0].plot(Items_by_Week['Week'], Items_by_Week['Items_Sold']) # for line plot
axes[0,0].set_xlabel('Week')
axes[0,0].set_ylabel('Items Sold')
axes[0,0].set_title('Line')
# Bar plot (top right)
axes[0,1].bar(Items_by_Week['Week'], Items_by_Week['Items_Sold']) # for bar plot
axes[0,1].set_xlabel('Week')
axes[0,1].set_ylabel('Items Sold')
axes[0,1].set_title('Bar')
# Horizontal bar plot (middle left)
axes[1,0].barh(Items_by_Week['Week'], Items_by_Week['Items_Sold']) # for horizontal bar plot
axes[1,0].set_xlabel('Items Sold')
axes[1,0].set_ylabel('Week')
axes[1,0].set_title('Horizontal Bar')
# Histogram (middle right)
axes[1,1].hist(y, bins=20) # for histogram
axes[1,1].set_xlabel('y')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Histogram') 
# Scatterplot (bottom left)
axes[2,0].scatter(Weight_by_Height['Height'], Weight_by_Height['Weight']) # for scatterplot
axes[2,0].set_xlabel('Height')
axes[2,0].set_ylabel('Weight')
axes[2,0].set_title('Scatter')
# Box-and-Whisker
axes[2,1].boxplot(y) # for Box-and-Whisker
axes[2,1].set_title('Box-and-Whisker')
plt.tight_layout() # prevent plot overlap
fig.savefig('Six_Subplots') # save figure

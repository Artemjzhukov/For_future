import pandas as pd    
import numpy as np
#reading the data into the dataframe into the object data
df = pd.read_csv("https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter01/Data/Marketing_subscription_prediction_latest_ed
df = pd.read_csv("https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter01/Data/german_credit_data.csv")
df1 = pd.read_csv('https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter01/Data/mark.csv',header = 0)
df2 = pd.read_csv('https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter01/Data/student.csv',header = 0)    
df = pd.read_csv('weather.csv')
                 
# Create a list for x
x = ['Boston Celtics','Los Angeles Lakers', 'Chicago Bulls', 'Golden State Warriors', 'San Antonio Spurs']
# Create a list for y
y = [17, 16, 6, 6, 5]
# Put into a data frame so we can sort them
df = pd.DataFrame({'Team': x,'Titles': y})       
                 
y = np.random.normal(loc=0, scale=0.1, size=100) # 100 numbers with mean of 0 and standard deviation of 0.1                 
                 
                 ## INFO AND DESCRIPTION
                 
#Finding number of rows and columns
print("Number of rows and columns : ",df.shape)
df.shape
                 
#Basic Statistics of each column
df.describe().transpose()
                 
#Basic Information of each column
df.info()
df.head()
df.tail()                 
                 

#Print the index names of the data frame. Using the command df.index
df.index
#Make the address column as an index and reset it back to the original data frame.
df.set_index('Address', inplace=True)     
#Undo                 
df.reset_index(inplace=True) 
                 
#Frequency distribution of Education column
df.education.value_counts()   
#Find the frequency distribution of each categorical column      
df_categorical.Grade.value_counts()
df_categorical.Gender.value_counts()                 
                 
#Find the distinct unique values in a *** column
df.education.unique()  
df_categorical['Grade'].unique()
                 
 


                 
                 
df.isna().sum()                 
#finding the data types of each column and checking for null
null_ = df.isna().any()
dtypes = df.dtypes
sum_na_ = df.isna().sum()
info = pd.concat([null_,sum_na_,dtypes],axis = 1,keys = ['isNullExist','NullSum','type'])
info                 

                 ## DROP AND FILL
                 

#Impute the numerical data of the age column with its mean                
mean_age = df.age.mean()
df.age.fillna(mean_age,inplace=True)   
#Impute the numerical data of duration column with its median
median_duration = df.duration.median() 
df. duration.fillna(median_duration,inplace=True)
#Impute the categorical data of the contact column with its mode.
mode_contact = df.contact.mode()[0]
df.contact.fillna(mode_contact,inplace=True)                 
df1 = df['Price']                 

X = df.drop('Price', axis=1)                 
#removing Null values
df = df.dropna()
#Total number of null in each column
df.isna().sum()
                 
df.iloc[0:4,0:3]
df.loc[0:4,["Avg. Area Income", "Avg. Area House Age"]]                 
               

                 ## SEPARATE
                 
df.education.unique()
#Let us group "basic.4y", "basic.9y" and "basic.6y" together and call them "basic".
#To do so, we can use replace function from pandas
df.education.replace({"basic.9y":"Basic","basic.6y":"Basic","basic.4y":"Basic"},inplace=True)

df_categorical.Grade.replace({"1st Class":1, "2nd Class":2, "3rd Class":3}, inplace= True)                 
df_categorical.Gender.replace({"Male":0,"Female":1}, inplace= True)           
                 
#Select and perform a suitable encoding method for the data
#Select all the non numeric data using select_dtypes function
data_column_category = df.select_dtypes(exclude=[np.number]).columns
#Select all the numeric data using select_dtypes function
data_column_numeric = df.select_dtypes(include=[np.number]).columns
df_onehot_category_frame = pd.get_dummies(df[data_column_category])
final_encoded_df = pd.concat([df[data_column_numeric],df_onehot_category_frame],axis=1)
final_encoded_df.head()

# dummy code 'Summary'                 
df_dummies = pd.get_dummies(df, drop_first=True)   
# shuffle df_dummies
df_shuffled = shuffle(df_dummies, random_state=42)  
# split df_shuffled into X and y
DV = 'Rain' # Save the DV as DV
X = df_shuffled.drop(DV, axis=1) # get features (X)
y = df_shuffled[DV] # get DV (y)                 
                 
#Select all the columns which are not numeric, using the following code and implement further.
data_column_category = df.select_dtypes(exclude=[np.number]).columns  
df[data_column_category].head()
                 
###Performing bucketing using the pd.cut() function on the marks column and displaying the top 10 columns.
df['bucket'] = pd.cut(df['marks'],5,labels = ['Poor','Below_average','Average','Above_Average','Excellent'])
              
                 
###import the LabelEncoder class
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#Creating the object instance
label_encoder = LabelEncoder()
for i in data_column_category:
    df[i] = label_encoder.fit_transform(df[i])   
                 
###Performing Onehot Encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(df[data_column_category])
#Creating a dataframe with encoded data with new column name
onehot_encoded_frame = pd.DataFrame(onehot_encoded, columns = onehot_encoder.get_feature_names(data_column_category))
df_onehot_getdummies = pd.get_dummies(df1[data_column_category], prefix=data_column_category)
data_onehot_encoded_data = pd.concat([df_onehot_getdummies,df1[data_column_number]],axis = 1)

###Perform the StandardScaler
from sklearn.preprocessing import StandardScaler 
std_scale = StandardScaler().fit_transform(df)
scaled_frame = pd.DataFrame(std_scale,columns=df.columns)

###Perform the Normalization scaling. To do so, use MinMaxScaler() class from sklearn.preprocessing and implement fit_transorm() method                 
from sklearn.preprocessing import MinMaxScaler 
norm_scale = MinMaxScaler().fit_transform(df)
scaled_frame = pd.DataFrame(norm_scale,columns=df.columns)                 
                 
#Find the categorical column and separate out with different dataframe. To do so, use select_dtypes() function from pandas dataframe
df_categorical = df.select_dtypes(exclude=np.number)                 
                 
###Splitting the data with train and test
#Segregating Independent and Target variable
X=final_encoded_df.drop(columns='y')
y=final_encoded_df['y']
from sklearn. model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

###Clear mistakes <> (max&min) in dataset
sbn.boxplot(df['Age'])

Q1 = df["Age"].quantile(0.25)
Q3 = df["Age"].quantile(0.75)
IQR = Q3 - Q1                 
Lower_Fence = Q1 - (1.5 * IQR)
Upper_Fence = Q3 + (1.5 * IQR)
df[(df["Age"] < Lower_Fence) |(df["Age"] > Upper_Fence)]

#Filter out the outlier data and print only the potential data. To do so, just negate the above result using ~ operator
df[~((df["Age"] < Lower_Fence) |(df["Age"] > Upper_Fence))]                 

                 ##MERGE

#Perform data integration to both the dataframe with respect to the column ‘Student_id’ key word ‘pd.merge()’
df = pd.merge(df1, df2, on = 'Student_id')

                 ##NORMALIZE
                 
# run the shapiro wilk test
from scipy.stats import shapiro
shap_w, shap_p = shapiro(y)

# set up some logic
if shap_p > 0.05:
    normal_YN = 'Fail to reject the null hypothesis. Data is normally distributed.'
else:
    normal_YN = 'Null hypothesis is rejected. Data is not normally distributed.'
print(normal_YN)
-----------------------------------------------------------                 
# Calculate shapiro wilk p-value
from scipy.stats import shapiro
shap_w, shap_p = shapiro(y)
print(shap_p)

# Get outliers
# convert to z-scores
from scipy.stats import zscore
y_z_scores = zscore(y) # convert y into z scores

# get the number of scores with absolute value of 3 or more
total_outliers = 0
for i in range(len(y_z_scores)):
    if abs(y_z_scores[i]) >= 3:
        total_outliers += 1
print(total_outliers)
           
# set up some logic for the title
if shap_p > 0.05:
    title = 'Normally distributed with {} outlier(s).'.format(total_outliers)
else:
    title = 'Not normally distributed with {} outlier(s).'.format(total_outliers)
print(title)  
-----------------------------------------------------                 
# calculate pearson correlations
from scipy.stats import pearsonr
correlation_coeff, p_value = pearsonr(x, y)
print(correlation_coeff)

# Set up some logic
if correlation_coeff == 1.00:
    title = 'There is a perfect positive linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff >= 0.8:
    title = 'There is a very strong, positive linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff >= 0.6:
    title = 'There is a strong, positive linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff >= 0.4:
    title = 'There is a moderate, positive linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff >= 0.2:
    title = 'There is a weak, positive linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff > 0:
    title = 'There is a very weak, positive linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff == 0:
    title = 'There is no linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff <= -0.8:
    title = 'There is a very strong, negative linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff <= -0.6:
    title = 'There is a strong, negative linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff <= -0.4:
    title = 'There is a moderate, negative linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
elif correlation_coeff <= -0.2:
    title = 'There is a weak, negative linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
else: 
    title = 'There is a very weak, negative linear relationship (r = {0:0.2f}).'.format(correlation_coeff)
print(title)                 

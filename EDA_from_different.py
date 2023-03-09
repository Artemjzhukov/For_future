import pandas as pd    
import numpy as np
#reading the data into the dataframe into the object data
df = pd.read_csv("https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter01/Data/Marketing_subscription_prediction_latest_ed

                 ## INFO AND DESCRIPTION
                 
#Finding number of rows and columns
print("Number of rows and columns : ",df.shape)
df.shape
                 
#Basic Statistics of each column
df.describe().transpose()
                 
#Basic Information of each column
df.info()
                 

#Print the index names of the data frame. Using the command df.index
df.index
#Make the address column as an index and reset it back to the original data frame.
df.set_index('Address', inplace=True)     
#Undo                 
df.reset_index(inplace=True) 
                 
#Frequency distribution of Education column
df.education.value_counts()   
                 
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
                 
#After grouping, this is the column
df.education.unique()  
                 
#Select and perform a suitable encoding method for the data
#Select all the non numeric data using select_dtypes function
data_column_category = df.select_dtypes(exclude=[np.number]).columns
#Select all the numeric data using select_dtypes function
data_column_numeric = df.select_dtypes(include=[np.number]).columns
df_onehot_category_frame = pd.get_dummies(df[data_column_category])
final_encoded_df = pd.concat([df[data_column_numeric],df_onehot_category_frame],axis=1)
final_encoded_df.head()
                 
#Splitting the data with train and test
#Segregating Independent and Target variable
X=final_encoded_df.drop(columns='y')
y=final_encoded_df['y']
from sklearn. model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                 

                 

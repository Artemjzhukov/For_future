# split X and y into testing and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# scale X_train and X_test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# generate predictions on the test data
predictions = model.predict(X_test)

# plot correlation of predicted and actual values
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.scatter(y_test, predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values (r = {0:0.2f})'.format(pearsonr(y_test, predictions)[0], 2))
plt.show()

# plot distribution of residuals
import seaborn as sns
from scipy.stats import shapiro
sns.distplot((y_test - predictions), bins = 50)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Histogram of Residuals (Shapiro W p-value = {0:0.3f})'.format(shapiro(y_test - predictions)[1]))
plt.show()

# compute metrics and put into a dataframe
from sklearn import metrics
import numpy as np
metrics_df = pd.DataFrame({'Metric': ['MAE', 
                                      'MSE', 
                                      'RMSE', 
                                      'R-Squared'],
                          'Value': [metrics.mean_absolute_error(y_test, predictions),
                                    metrics.mean_squared_error(y_test, predictions),
                                    np.sqrt(metrics.mean_squared_error(y_test, predictions)),
                                    metrics.explained_variance_score(y_test, predictions)]}).round(3)
print(metrics_df)
# generate predicted probabilities of yes
predicted_prob = model.predict_proba(X_test)[:,1]

# generate predicted classes
predicted_class = model.predict(X_test_scaled)

# generate predicted classes
predicted_class = model.predict(X_test)

# evaluate performance with confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
cm = pd.DataFrame(confusion_matrix(y_test, predicted_class))
cm['Total'] = np.sum(cm, axis=1)
cm = cm.append(np.sum(cm, axis=0), ignore_index=True)
cm.columns = ['Predicted No', 'Predicted Yes', 'Total']
cm = cm.set_index([['Actual No', 'Actual Yes', 'Total']])
print(cm)



# generate a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_class))

###GRID

# Specify the hyperparameter space
import numpy as np
grid = {'criterion': ['mse','mae'],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_impurity_decrease': np.linspace(0.0, 1.0, 10),
        'bootstrap': [True, False],
        'warm_start': [True, False]}

# Instantiate the GridSearchCV model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
model = GridSearchCV(RandomForestRegressor(), grid, scoring='explained_variance', cv=5)

# Fit to the training set
model.fit(X_train_scaled, y_train)
# Print the tuned parameters
best_parameters = model.best_params_
print(best_parameters)
------------------------------------------------
import numpy as np
grid = {'penalty': ['l1', 'l2'],
        'C': np.linspace(1, 10, 10)}

# instantiate GridSearchCV model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
model = GridSearchCV(LogisticRegression(solver='liblinear'), grid, scoring='f1', cv=5)
# fit the gridsearch model
model.fit(X_train, y_train)
# print the best parameters
best_parameters = model.best_params_
print(best_parameters)
--------------------------------
# instantiate grid
import numpy as np
grid = {'C': np.linspace(1, 10, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# instantiate GridSearchCV model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
model = GridSearchCV(SVC(gamma='auto'), grid, scoring='f1', cv=5)
# fit the gridsearch model
model.fit(X_train_scaled, y_train)
# print the best parameters
best_parameters = model.best_params_
print(best_parameters)
------------------------------
# Specify the hyperparameter space
import numpy as np
grid = {'criterion': ['gini', 'entropy'],
        'min_weight_fraction_leaf': np.linspace(0.0, 0.5, 10),
        'min_impurity_decrease': np.linspace(0.0, 1.0, 10),
        'class_weight': [None, 'balanced'],
        'presort': [True, False]}

# Instantiate the GridSearchCV model
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
model = GridSearchCV(DecisionTreeClassifier(), grid, scoring='f1', cv=5)
# Fit to the training set
model.fit(X_train_scaled, y_train)
# Print the tuned parameters
best_parameters = model.best_params_
print(best_parameters)

### LINEAR

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train[['Humidity']], y_train)
intercept = model.intercept_
coefficient = model.coef_
print('Temperature = {0:0.2f} + ({1:0.2f} x Humidity)'.format(intercept, coefficient[0]))

predictions = model.predict(X_test[['Humidity']])

###LOGISTIC
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
intercept = model.intercept_
coefficients = model.coef_
coef_list = list(coefficients[0,:])
# put coefficients in a df with feature name
coef_df = pd.DataFrame({'Feature': list(X_train.columns),
                        'Coefficient': coef_list})
print(coef_df)

### DECISION TREE
# access the 'Tree__criterion' value
print(best_parameters['criterion'])

# instantiate model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight=best_parameters['class_weight'],
                               criterion=best_parameters['criterion'],
                               min_impurity_decrease=best_parameters['min_impurity_decrease'],
                               min_weight_fraction_leaf=best_parameters['min_weight_fraction_leaf'],
                               presort=best_parameters['presort'])

# scale X_train and fit model
model.fit(X_train_scaled, y_train)

# extract feature_importances attribute
print(model.feature_importances_)

# plot feature importance in descending order
import pandas as pd
import matplotlib.pyplot as plt
df_imp = pd.DataFrame({'Importance': list(model.feature_importances_)}, index=X.columns)
# sort dataframe
df_imp_sorted = df_imp.sort_values(by=('Importance'), ascending=True)
# plot these
df_imp_sorted.plot.barh(figsize=(5,5))
plt.title('Relative Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('Variable')
plt.legend(loc=4)
plt.show()

### RANDOM FOREST
# instantiate model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(criterion=best_parameters['criterion'],
                              max_features=best_parameters['max_features'],
                              min_impurity_decrease=best_parameters['min_impurity_decrease'],
                              bootstrap=best_parameters['bootstrap'],
                              warm_start=best_parameters['warm_start'])

# fit model
model.fit(X_train_scaled, y_train)

# plot feature importance in descending order
import pandas as pd
import matplotlib.pyplot as plt
df_imp = pd.DataFrame({'Importance': list(model.feature_importances_)}, index=X.columns)
# sort dataframe
df_imp_sorted = df_imp.sort_values(by=('Importance'), ascending=True)
# plot these
df_imp_sorted.plot.barh(figsize=(5,5))
plt.title('Relative Feature Importance')
plt.xlabel('Relative Importance')
plt.ylabel('Variable')
plt.legend(loc=4)
plt.show()

### K-MEANS CLUSTER
# standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # create StandardScaler() object
scaled_features = scaler.fit_transform(df_shuffled) # fit scaler model and transform df_shuffled

# instantiate an empty dataframe
import pandas as pd
labels_df = pd.DataFrame()

# Build 100 models
from sklearn.cluster import KMeans
for i in range(100):
    model = KMeans(n_clusters=2)
    model.fit(scaled_features) # fit model
    labels = model.labels_ # get predicted labels
    labels_df['Model_{}_Labels'.format(i+1)] = labels # put the labels into the empty df

# calculate mode for each row
row_mode = labels_df.mode(axis=1)

# assign the row_mode array as a column in labels_df
labels_df['row_mode'] = row_mode

# preview the data
print(labels_df.head(5))
-------------------------------------------------
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)

# fit model
model.fit(scaled_features)

# get predicted labels
labels = model.labels_

# see how many of each label we have
import pandas as pd
pd.value_counts(labels)

# add label to df_shuffled
df_shuffled['Predicted_Cluster'] = labels
print(df_shuffled.head(5))

mean_inertia_list = [] # create a list for the average inertia at each n_clusters
for x in range(1, 11): # loop through n_clusters 1-10
    inertia_list = [] # create a list for each individual inertia value at n_cluster
    for i in range(100):
        model = KMeans(n_clusters=x) # instantiate model
        model.fit(scaled_features) # fit model
        inertia = model.inertia_ # get inertia
        inertia_list.append(inertia) # append inertia to inertia_list
    # moving to the outside loop
    mean_inertia = np.mean(inertia_list) # get mean of inertia list
    mean_inertia_list.append(mean_inertia) # append mean_inertia to mean_inertia_list
print(mean_inertia_list) 

# plot inertia by n_clusters
import matplotlib.pyplot as plt
x = list(range(1, len(mean_inertia_list)+1))
y = mean_inertia_list
plt.plot(x, y)
plt.title('Mean Inertia by n_clusters')
plt.xlabel('n_clusters')
plt.xticks(x)
plt.ylabel('Mean Inertia')
plt.show()

###Mean Inertia by Cluster After PCA Transformation

# instantiate PCA model
from sklearn.decomposition import PCA
model = PCA()
# fit model
model.fit(scaled_features)
# get proportion of explained variance in each component
explained_var_ratio = model.explained_variance_ratio_
# print the explained variance ratio
print(explained_var_ratio)
from sklearn.decomposition import PCA
model = PCA(n_components=best_n_components) # remember, best_n_components = 6
# fit model and transform scaled_features into best_n_components
df_pca = model.fit_transform(scaled_features)

# fit 100 models for each n_clusters 1-10
from sklearn.cluster import KMeans
import numpy as np
mean_inertia_list_PCA = [] # create a list for the average inertia at each n_clusters
for x in range(1, 11): # loop through n_clusters 1-10
    inertia_list = [] # create a list for each individual inertia value at n_cluster
    for i in range(100):
        model = KMeans(n_clusters=x) # instantiate model
        model.fit(df_pca) # fit model
        inertia = model.inertia_ # get inertia
        inertia_list.append(inertia) # append inertia to inertia_list
    # moving to the outside loop
    mean_inertia = np.mean(inertia_list) # get mean of inertia list
    mean_inertia_list_PCA.append(mean_inertia) # append mean_inertia to mean_inertia_list
    
# print mean_inertia_list_PCA
print(mean_inertia_list_PCA)  

# plot inertia by n_clusters with both lines
import matplotlib.pyplot as plt
x = list(range(1,len(mean_inertia_list_PCA)+1))
y = mean_inertia_list_PCA
y2 = mean_inertia_list 
plt.plot(x, y, label='PCA')
plt.plot(x, y2, label='No PCA')
plt.title('Mean Inertia by n_clusters for Original Features and PCA Transformed Features')
plt.xlabel('n_clusters')
plt.xticks(x)
plt.ylabel('Inertia')
plt.legend()
plt.show()

###LDA MODEL
# instantiate LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()

# fit the model on the training data
model.fit(X_train_scaled, y_train)

# compute explained ratio by component
model.explained_variance_ratio_

# transform the training features to the training components
X_train_LDA = model.transform(X_train_scaled) 

# transform the testing features to the testing components
X_test_LDA = model.transform(X_test_scaled) 

# create a random forest model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier() 

# fit the model on the training components
model.fit(X_train_LDA, y_train) 

# generate predictions on the testing components
predictions = model.predict(X_test_LDA) 

# style the confusion matrix
from sklearn.metrics import confusion_matrix 
import pandas as pd
import numpy as np
cm = pd.DataFrame(confusion_matrix(y_test, predictions))
cm['Total'] = np.sum(cm, axis=1)
cm = cm.append(np.sum(cm, axis=0), ignore_index=True)
cm.columns = ['Predicted 1', 'Predicted 2', 'Predicted 3', 'Total']
cm = cm.set_index([['Actual 1', 'Actual 2', 'Actual 3', 'Total']])
print(cm)

# to get the accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

###SCALER
# scale X_train and X_test
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # instantiate StandardScaler model
X_train_scaled = scaler.fit_transform(X_train) # transform X_train to z-scores
X_test_scaled = scaler.transform(X_test) # transform X_test to z-scores

### HCA MODEL
# create linkage model
from scipy.cluster.hierarchy import linkage 
model = linkage(scaled_features, method='complete')

import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(10,5))
plt.title('Dendrogram for Glass Data')
dendrogram(model,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()

# get labels
from scipy.cluster.hierarchy import fcluster 
labels = fcluster(model, t=9, criterion='distance')
print(labels)

# assign labels array as a column in df_shuffled
df_shuffled['Predicted_Cluster'] = labels

# preview data
print(df_shuffled.head(5))

-----------------------
import numpy as np
cum_sum_explained_var = np.cumsum(model.explained_variance_ratio_)
print(cum_sum_explained_var)

# set a threshold for % of variance in the data to preserve
threshold = .95
for i in range(len(cum_sum_explained_var)):
    if cum_sum_explained_var[i] >= threshold:
        best_n_components = i+1
        break
    else:
        pass
    
# print the best number of n_components
print('The best n_components is {}'.format(best_n_components))

### XGB 

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/adult-data.csv", names=['age', 'workclass', 'education-num',
                                                 'occupation', 'capital-gain', 'capital-loss',
                                                 'hours-per-week', 'income'])
data.tail()
data['workclass'] = LabelEncoder().fit_transform(data['workclass'])
data['occupation'] = LabelEncoder().fit_transform(data['occupation'])
data['income'] = LabelEncoder().fit_transform(data['income'])

X = data.copy()
X.drop("income", inplace = True, axis = 1)
Y = data.income

X_train, X_test = X[:int(X.shape[0]*0.8)].values, X[int(X.shape[0]*0.8):].values
Y_train, Y_test = Y[:int(Y.shape[0]*0.8)].values, Y[int(Y.shape[0]*0.8):].values

train = xgb.DMatrix(X_train, label=Y_train)
test = xgb.DMatrix(X_test, label=Y_test)

param = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'binary:hinge'}
num_round = 50
model = xgb.train(param, train, num_round)

preds = model.predict(test)

accuracy = accuracy_score(Y[int(Y.shape[0]*0.8):].values, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
---------------------
train = xgb.DMatrix(X_train, label=Y_train)
test = xgb.DMatrix(X_test, label=Y_test)

test_error = {}
for i in range(20):
    param = {'max_depth':i, 'eta':0.1, 'silent':1, 'objective':'binary:hinge'}
    num_round = 50
    model_metrics = xgb.cv(param, train, num_round, nfold = 10)
    test_error[i] = model_metrics.iloc[-1]['test-error-mean']

plt.scatter(test_error.keys(),test_error.values())
plt.xlabel('Max Depth')
plt.ylabel('Test Error')
plt.show()


param = {'max_depth':4, 'eta':0.1, 'silent':1, 'objective':'binary:hinge'}
num_round = 300
model_metrics = xgb.cv(param, train, num_round, nfold = 10)

plt.scatter(range(300),model_metrics['test-error-mean'], s = 0.7, label = 'Test Error')
plt.scatter(range(300),model_metrics['train-error-mean'], s = 0.7, label = 'Train Error')
plt.legend()
plt.show()

param = {'max_depth':4, 'eta':0.1, 'silent':1, 'objective':'binary:hinge'}
num_round = 100
model = xgb.train(param, train, num_round)
preds = model.predict(test)
accuracy = accuracy_score(Y[int(Y.shape[0]*0.8):].values, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

Accuracy: 79.77%

model.save_model('churn-model.model')
-----------------------------------------
train = xgb.DMatrix(X_train, label=Y_train)
test = xgb.DMatrix(X_test, label=Y_test)

param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softmax', 'num_class': 3}
num_round = 5
model = xgb.train(param, train, num_round)

preds = model.predict(test)

accuracy = accuracy_score(Y[int(Y.shape[0]*0.8):].values, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

Accuracy: 89.77%
------------------------------------------
train = xgb.DMatrix(X_train, label=Y_train)
test = xgb.DMatrix(X_test, label=Y_test)

test_error = {}

for i in range(20):
    param = {'max_depth':i, 'eta':0.1, 'silent':1, 'objective':'binary:hinge'}
    num_round = 50
    model_metrics = xgb.cv(param, train, num_round, nfold = 10)
    test_error[i] = model_metrics.iloc[-1]['test-error-mean']

plt.scatter(test_error.keys(),test_error.values())
plt.xlabel('Max Depth')
plt.ylabel('Test Error')
plt.show()

test_error = {}

for i in range(1,100,5):
    param = {'max_depth':9, 'eta':0.001*i, 'silent':1, 'objective':'binary:hinge'}
    num_round = 500
    model_metrics = xgb.cv(param, train, num_round, nfold = 10)
    test_error[i] = model_metrics.iloc[-1]['test-error-mean']

lr = [0.001*(i) for i in test_error.keys()]
plt.scatter(lr,test_error.values())
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.show()

param = {'max_depth':9, 'eta':0.04, 'silent':1, 'objective':'binary:hinge'}
num_round = 500
model_metrics = xgb.cv(param, train, num_round, nfold = 10)

plt.scatter(range(500),model_metrics['test-error-mean'], s = 0.7, label = 'Test Error')
plt.scatter(range(500),model_metrics['train-error-mean'], s = 0.7, label = 'Train Error')
plt.legend()
plt.show()

list(model_metrics['test-error-mean']).index(min(model_metrics['test-error-mean'])) 

139

-----------------------------------------------

data[['workclass', 'occupation', 'income']] = data[['workclass', 'occupation', 'income']].apply(lambda x: x.str.strip())

label_dict = defaultdict(LabelEncoder)

data[['workclass', 'occupation', 'income']] = data[['workclass', 'occupation', 'income']].apply(lambda x: label_dict[x.name].fit_transform(x))

def save_obj(var, file):
    with open(file + '.pkl', 'wb') as f:
        pickle.dump(var, f, pickle.HIGHEST_PROTOCOL)
def load_obj(file):
    with open(file + '.pkl', 'rb') as f:
        return pickle.load(f)

save_obj(label_dict, 'income_labels')

X = data.copy()
X.drop("income", inplace = True, axis = 1)
Y = data.income

X_train, X_test = X[:int(X.shape[0]*0.8)].values, X[int(X.shape[0]*0.8):].values
Y_train, Y_test = Y[:int(Y.shape[0]*0.8)].values, Y[int(Y.shape[0]*0.8):].values

train = xgb.DMatrix(X_train, label=Y_train)
test = xgb.DMatrix(X_test, label=Y_test)

param = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'binary:hinge'}
num_round = 50
model = xgb.train(param, train, num_round)

preds = model.predict(test)

accuracy = accuracy_score(Y[int(Y.shape[0]*0.8):].values, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

Accuracy: 83.66%

model.save_model('income-model.model')

### KERAS
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=7))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size=8)

Epoch 1/5
352/352 [==============================] - 0s 1ms/step - loss: 3.3547 - acc: 0.4318 
Epoch 2/5
352/352 [==============================] - 0s 190us/step - loss: -5.6614 - acc: 0.6449
Epoch 3/5
352/352 [==============================] - 0s 179us/step - loss: -5.6614 - acc: 0.6449
Epoch 4/5
352/352 [==============================] - 0s 185us/step - loss: -5.6614 - acc: 0.6449
Epoch 5/5
352/352 [==============================] - 0s 170us/step - loss: -5.6614 - acc: 0.6449

<keras.callbacks.History at 0x268a58a5470>

preds = model.predict(X_test, batch_size=128)

accuracy = accuracy_score(Y_test, preds.astype(int))
print("Accuracy: %.2f%%" % (accuracy * 100.0))

Accuracy: 80.68%
------------------------
filepath="avocado-{epoch:02d}-{val_loss:.2f}.hdf5"
model_ckpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
es = EarlyStopping(monitor='val_loss', min_delta=1, patience=5, verbose=1)
callbacks = [model_ckpt, es]

model = Sequential()
model.add(Dense(units=16, activation='relu', input_dim=13))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_55 (Dense)             (None, 16)                224       
_________________________________________________________________
dense_56 (Dense)             (None, 8)                 136       
_________________________________________________________________
dense_57 (Dense)             (None, 1)                 9         
=================================================================
Total params: 369
Trainable params: 369
Non-trainable params: 0
_________________________________________________________________

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=40, batch_size=32, callbacks = callbacks)






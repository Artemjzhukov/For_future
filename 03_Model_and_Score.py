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

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding

Using TensorFlow backend.

%matplotlib inline

data = pd.read_csv('data/avocado.csv')

data.shape

(18249, 14)

data['Day'], data['Month'] = data.Date.str[:2], data.Date.str[3:5]

data = data.drop(['Unnamed: 0', 'Date'], axis = 1)

data.T

data = data.dropna()

label_dict = defaultdict(LabelEncoder)

data[['region', 'type', 'Day', 'Month', 'year']] = data[['region', 'type', 'Day', 'Month', 'year']].apply(lambda x: label_dict[x.name].fit_transform(x))

X = data
y = X.pop('AveragePrice')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

Bring data in the proper format for the model

cat_cols_dict = {col: list(data[col].unique()) for col in ['region', 'type', 'Day', 'Month', 'year']}

train_input_list = []
test_input_list = []

for col in cat_cols_dict.keys():
    raw_values = np.unique(data[col])
    value_map = {}
    for i in range(len(raw_values)):
        value_map[raw_values[i]] = i       
    train_input_list.append(X_train[col].map(value_map).values)
    test_input_list.append(X_test[col].map(value_map).fillna(0).values)


other_cols = [col for col in data.columns if (not col in cat_cols_dict.keys())]
train_input_list.append(X_train[other_cols].values)
test_input_list.append(X_test[other_cols].values)

cols_out_dict = {
    'region': 12, 
    'type': 1, 
    'Day': 10, 
    'Month': 3, 
    'year': 1
}

inputs = []
embeddings = []

for col in cat_cols_dict.keys():
    
    inp = Input(shape=(1,), name = 'input_' + col)
    embedding = Embedding(len(cat_cols_dict[col]), cols_out_dict[col], input_length=1, name = 'embedding_' + col)(inp)
    embedding = Reshape(target_shape=(cols_out_dict[col],))(embedding)
    inputs.append(inp)
    embeddings.append(embedding)


input_numeric = Input(shape=(8,))
embedding_numeric = Dense(16)(input_numeric) 
inputs.append(input_numeric)
embeddings.append(embedding_numeric)

x = Concatenate()(embeddings)
x = Dense(16, activation='relu')(x)
x = Dense(4, activation='relu')(x)
output = Dense(1, activation='linear')(x)

model = Model(inputs, output)

model.compile(loss='mse', optimizer='adam')

WARNING:tensorflow:From C:\Users\Maedr3\AppData\Roaming\Python\Python36\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.

model.summary()

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_region (InputLayer)       (None, 1)            0                                            
__________________________________________________________________________________________________
input_type (InputLayer)         (None, 1)            0                                            
__________________________________________________________________________________________________
input_Day (InputLayer)          (None, 1)            0                                            
__________________________________________________________________________________________________
input_Month (InputLayer)        (None, 1)            0                                            
__________________________________________________________________________________________________
input_year (InputLayer)         (None, 1)            0                                            
__________________________________________________________________________________________________
embedding_region (Embedding)    (None, 1, 12)        648         input_region[0][0]               
__________________________________________________________________________________________________
embedding_type (Embedding)      (None, 1, 1)         2           input_type[0][0]                 
__________________________________________________________________________________________________
embedding_Day (Embedding)       (None, 1, 10)        310         input_Day[0][0]                  
__________________________________________________________________________________________________
embedding_Month (Embedding)     (None, 1, 3)         36          input_Month[0][0]                
__________________________________________________________________________________________________
embedding_year (Embedding)      (None, 1, 1)         4           input_year[0][0]                 
__________________________________________________________________________________________________
input_1 (InputLayer)            (None, 8)            0                                            
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 12)           0           embedding_region[0][0]           
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 1)            0           embedding_type[0][0]             
__________________________________________________________________________________________________
reshape_3 (Reshape)             (None, 10)           0           embedding_Day[0][0]              
__________________________________________________________________________________________________
reshape_4 (Reshape)             (None, 3)            0           embedding_Month[0][0]            
__________________________________________________________________________________________________
reshape_5 (Reshape)             (None, 1)            0           embedding_year[0][0]             
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 16)           144         input_1[0][0]                    
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 43)           0           reshape_1[0][0]                  
                                                                 reshape_2[0][0]                  
                                                                 reshape_3[0][0]                  
                                                                 reshape_4[0][0]                  
                                                                 reshape_5[0][0]                  
                                                                 dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 16)           704         concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 4)            68          dense_2[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            5           dense_3[0][0]                    
==================================================================================================
Total params: 1,921
Trainable params: 1,921
Non-trainable params: 0
__________________________________________________________________________________________________

model.fit(train_input_list, y_train, validation_data = (test_input_list, y_test), epochs=50, batch_size=32)

WARNING:tensorflow:From C:\Users\Maedr3\AppData\Roaming\Python\Python36\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Train on 12774 samples, validate on 5475 samples
Epoch 1/50
12774/12774 [==============================] - 2s 124us/step - loss: 53295180743.5138 - val_loss: 286530513.3150
Epoch 2/50
12774/12774 [==============================] - 1s 59us/step - loss: 121049092.4849 - val_loss: 43968444.5480
Epoch 3/50
12774/12774 [==============================] - 1s 60us/step - loss: 26135176.1193 - val_loss: 15038244.8321
Epoch 4/50
12774/12774 [==============================] - 1s 59us/step - loss: 10282040.9579 - val_loss: 7312467.7255
Epoch 5/50
12774/12774 [==============================] - 1s 57us/step - loss: 5610835.5616 - val_loss: 4212715.4917
Epoch 6/50
12774/12774 [==============================] - 1s 57us/step - loss: 3284640.9104 - val_loss: 2626078.3302
Epoch 7/50
12774/12774 [==============================] - 1s 63us/step - loss: 1985170.2991 - val_loss: 1682499.5763
Epoch 8/50
12774/12774 [==============================] - 1s 61us/step - loss: 1256837.4144 - val_loss: 1206512.3684
Epoch 9/50
12774/12774 [==============================] - 1s 61us/step - loss: 846565.4335 - val_loss: 876602.0336
Epoch 10/50
12774/12774 [==============================] - 1s 68us/step - loss: 625909.4434 - val_loss: 681300.6102
Epoch 11/50
12774/12774 [==============================] - 1s 74us/step - loss: 479072.8697 - val_loss: 538276.9503
Epoch 12/50
12774/12774 [==============================] - 1s 63us/step - loss: 374273.3134 - val_loss: 428596.6480
Epoch 13/50
12774/12774 [==============================] - 1s 65us/step - loss: 287107.1753 - val_loss: 336243.7636
Epoch 14/50
12774/12774 [==============================] - 1s 59us/step - loss: 212684.3274 - val_loss: 261381.1963
Epoch 15/50
12774/12774 [==============================] - 1s 60us/step - loss: 155402.7377 - val_loss: 201733.5198
Epoch 16/50
12774/12774 [==============================] - 1s 64us/step - loss: 111050.9949 - val_loss: 155582.0523
Epoch 17/50
12774/12774 [==============================] - 1s 60us/step - loss: 78947.7107 - val_loss: 118516.8250
Epoch 18/50
12774/12774 [==============================] - 1s 61us/step - loss: 55657.5692 - val_loss: 91547.3390
Epoch 19/50
12774/12774 [==============================] - 1s 59us/step - loss: 40087.8331 - val_loss: 70461.5736
Epoch 20/50
12774/12774 [==============================] - 1s 60us/step - loss: 29210.3017 - val_loss: 54778.5208
Epoch 21/50
12774/12774 [==============================] - 1s 61us/step - loss: 22371.3657 - val_loss: 42820.1845
Epoch 22/50
12774/12774 [==============================] - 1s 61us/step - loss: 17217.7532 - val_loss: 32748.5839
Epoch 23/50
12774/12774 [==============================] - 1s 63us/step - loss: 13032.9318 - val_loss: 24105.4347
Epoch 24/50
12774/12774 [==============================] - 1s 59us/step - loss: 9814.6165 - val_loss: 17514.5128
Epoch 25/50
12774/12774 [==============================] - 1s 59us/step - loss: 7470.5298 - val_loss: 12964.9630
Epoch 26/50
12774/12774 [==============================] - 1s 60us/step - loss: 5728.4827 - val_loss: 9616.4998
Epoch 27/50
12774/12774 [==============================] - 1s 58us/step - loss: 4438.0689 - val_loss: 7191.6616
Epoch 28/50
12774/12774 [==============================] - 1s 56us/step - loss: 3449.0771 - val_loss: 5125.3246
Epoch 29/50
12774/12774 [==============================] - 1s 58us/step - loss: 2779.7216 - val_loss: 3599.9340
Epoch 30/50
12774/12774 [==============================] - 1s 64us/step - loss: 2263.7153 - val_loss: 2589.4246
Epoch 31/50
12774/12774 [==============================] - 1s 76us/step - loss: 1874.7155 - val_loss: 1800.4863
Epoch 32/50
12774/12774 [==============================] - 1s 78us/step - loss: 1548.2751 - val_loss: 1273.8217
Epoch 33/50
12774/12774 [==============================] - 1s 75us/step - loss: 1260.1104 - val_loss: 915.2101
Epoch 34/50
12774/12774 [==============================] - 1s 86us/step - loss: 1016.3263 - val_loss: 666.3749
Epoch 35/50
12774/12774 [==============================] - 1s 82us/step - loss: 817.2215 - val_loss: 498.6022
Epoch 36/50
12774/12774 [==============================] - 1s 71us/step - loss: 649.8555 - val_loss: 382.1295
Epoch 37/50
12774/12774 [==============================] - 1s 78us/step - loss: 511.6337 - val_loss: 283.7043
Epoch 38/50
12774/12774 [==============================] - 1s 89us/step - loss: 398.0049 - val_loss: 211.8547
Epoch 39/50
12774/12774 [==============================] - 1s 87us/step - loss: 312.4168 - val_loss: 157.8509
Epoch 40/50
12774/12774 [==============================] - 2s 157us/step - loss: 244.0374 - val_loss: 116.0204
Epoch 41/50
12774/12774 [==============================] - 2s 179us/step - loss: 188.7864 - val_loss: 81.1109
Epoch 42/50
12774/12774 [==============================] - 2s 186us/step - loss: 142.0455 - val_loss: 56.6849
Epoch 43/50
12774/12774 [==============================] - 2s 126us/step - loss: 106.0369 - val_loss: 39.0254
Epoch 44/50
12774/12774 [==============================] - 2s 147us/step - loss: 76.6613 - val_loss: 19.0574
Epoch 45/50
12774/12774 [==============================] - 2s 143us/step - loss: 56.2175 - val_loss: 10.9867
Epoch 46/50
12774/12774 [==============================] - 2s 149us/step - loss: 36.9431 - val_loss: 6.1915
Epoch 47/50
12774/12774 [==============================] - 2s 121us/step - loss: 23.6332 - val_loss: 4.5866
Epoch 48/50
12774/12774 [==============================] - 2s 144us/step - loss: 14.4377 - val_loss: 3.9991
Epoch 49/50
12774/12774 [==============================] - 2s 137us/step - loss: 8.7404 - val_loss: 3.5535
Epoch 50/50
12774/12774 [==============================] - 2s 129us/step - loss: 5.2511 - val_loss: 3.1125

<keras.callbacks.History at 0x237da91dbe0>

model.evaluate(test_input_list, y_test)

5475/5475 [==============================] - 0s 54us/step

3.112497390557642

embedding_region = model.get_layer('embedding_region').get_weights()[0]
embedding_Day = model.get_layer('embedding_Day').get_weights()[0]
embedding_Month = model.get_layer('embedding_Month').get_weights()[0]

label_dict['region'].inverse_transform(cat_cols_dict['region'])

array(['Albany', 'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston',
       'BuffaloRochester', 'California', 'Charlotte', 'Chicago',
       'CincinnatiDayton', 'Columbus', 'DallasFtWorth', 'Denver',
       'Detroit', 'GrandRapids', 'GreatLakes', 'HarrisburgScranton',
       'HartfordSpringfield', 'Houston', 'Indianapolis', 'Jacksonville',
       'LasVegas', 'LosAngeles', 'Louisville', 'MiamiFtLauderdale',
       'Midsouth', 'Nashville', 'NewOrleansMobile', 'NewYork',
       'Northeast', 'NorthernNewEngland', 'Orlando', 'Philadelphia',
       'PhoenixTucson', 'Pittsburgh', 'Plains', 'Portland',
       'RaleighGreensboro', 'RichmondNorfolk', 'Roanoke', 'Sacramento',
       'SanDiego', 'SanFrancisco', 'Seattle', 'SouthCarolina',
       'SouthCentral', 'Southeast', 'Spokane', 'StLouis', 'Syracuse',
       'Tampa', 'TotalUS', 'West', 'WestTexNewMexico'], dtype=object)

pca = PCA(n_components=2)
Y = pca.fit_transform(embedding_region[:25])
plt.figure(figsize=(8,8))
plt.scatter(-Y[:, 0], -Y[:, 1])
for i, txt in enumerate((label_dict['region'].inverse_transform(cat_cols_dict['region']))[:25]):
    plt.annotate(txt, (-Y[i, 0],-Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points')
plt.show()

pca = PCA(n_components=2)
Y = pca.fit_transform(embedding_Day)
plt.figure(figsize=(8,8))
plt.scatter(-Y[:, 0], -Y[:, 1])
for i, txt in enumerate(label_dict['Day'].inverse_transform(cat_cols_dict['Day'])):
    plt.annotate(txt, (-Y[i, 0],-Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points')
plt.show()

pca = PCA(n_components=2)
Y = pca.fit_transform(embedding_Month)
plt.figure(figsize=(8,8))
plt.scatter(-Y[:20, 0], -Y[:20, 1])
for i, txt in enumerate(label_dict['Month'].inverse_transform(cat_cols_dict['Month'])):
    plt.annotate(txt, (-Y[i, 0],-Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points')
    if i == 20:
        break
plt.show()


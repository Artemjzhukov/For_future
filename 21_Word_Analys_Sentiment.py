import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
import numpy as np

import xgboost as xgb
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.metrics import accuracy_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, BatchNormalization, Embedding, Flatten

MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16

data = pd.read_csv('../../chapter 8/data/movie_reviews.csv', encoding='latin-1')

data.shape

(25000, 2)

data.SentimentText = data.SentimentText.str.lower()

def clean_str(string):
    
    string = re.sub(r"https?\://\S+", '', string)
    string = re.sub(r'\<a href', ' ', string)
    string = re.sub(r'&amp;', '', string) 
    string = re.sub(r'<br />', ' ', string)
    string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', string)
    string = re.sub('\d','', string)
    string = re.sub(r"can\'t", "cannot", string)
    string = re.sub(r"it\'s", "it is", string)
    return string

data.SentimentText = data.SentimentText.apply(lambda x: clean_str(str(x)))

pd.Series(' '.join(data['SentimentText']).split()).value_counts().head(10)

movie    43558
film     39095
it       30659
one      26509
is       20355
like     20270
good     15099
the      13913
time     12682
even     12656
dtype: int64

stop_words = stopwords.words('english') + ['movie', 'film', 'time']
stop_words = set(stop_words)
remove_stop_words = lambda r: [[word for word in word_tokenize(sente) if word not in stop_words] for sente in sent_tokenize(r)]
data['SentimentText'] = data['SentimentText'].apply(remove_stop_words)

model = Word2Vec(
        data['SentimentText'].apply(lambda x: x[0]),
        iter=10,
        size=16,
        window=5,
        min_count=5,
        workers=10)

model.wv.most_similar('fun')

[('entertaining', 0.8260140419006348),
 ('lighten', 0.825722336769104),
 ('laughs', 0.8177958726882935),
 ('enjoy', 0.790296733379364),
 ('enjoyable', 0.7853423357009888),
 ('plenty', 0.7833274602890015),
 ('comedy', 0.7706939578056335),
 ('funny', 0.7564221620559692),
 ('definitely', 0.7507157325744629),
 ('guaranteed', 0.7493278980255127)]

 

model.wv.save_word2vec_format('movie_embedding.txt', binary=False)

def combine_text(text):    
    try:
        return ' '.join(text[0])
    except:
        return np.nan

data.SentimentText = data.SentimentText.apply(lambda x: combine_text(x))

data = data.dropna(how='any')

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(list(data['SentimentText']))
sequences = tokenizer.texts_to_sequences(data['SentimentText'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Found 77348 unique tokens.

reviews = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

def load_embedding(filename, word_index , num_words, embedding_dim):
    embeddings_index = {}
    file = open(filename, encoding="utf-8")
    for line in file:
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:])
        embeddings_index[word] = coef
    file.close()
    
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, pos in word_index.items():
        if pos >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[pos] = embedding_vector
    return embedding_matrix

embedding_matrix = load_embedding('movie_embedding.txt', word_index, len(word_index), EMBEDDING_DIM)

X_train, X_test, y_train, y_test = train_test_split(reviews, pd.get_dummies(data.Sentiment), test_size=0.2, random_state=9)

inp = Input((MAX_SEQUENCE_LENGTH,))
embedding_layer = Embedding(len(word_index),
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False)(inp)
model = Flatten()(embedding_layer)
model = BatchNormalization()(model)
model = Dropout(0.10)(model)
model = Dense(units=256, activation='relu')(model)
model = Dense(units=64, activation='relu')(model)
model = Dropout(0.5)(model)
predictions = Dense(units=2, activation='softmax')(model)
model = Model(inputs = inp, outputs = predictions)

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics = ['acc'])

model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 100)               0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 100, 16)           1237568   
_________________________________________________________________
flatten_1 (Flatten)          (None, 1600)              0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 1600)              6400      
_________________________________________________________________
dropout_1 (Dropout)          (None, 1600)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               409856    
_________________________________________________________________
dense_2 (Dense)              (None, 64)                16448     
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 130       
=================================================================
Total params: 1,670,402
Trainable params: 429,634
Non-trainable params: 1,240,768
_________________________________________________________________

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=256)

preds = model.predict(X_test)

accuracy_score(np.argmax(preds, 1), np.argmax(y_test.values, 1))

0.7634

y_actual = pd.Series(np.argmax(y_test.values, axis=1), name='Actual')
y_pred = pd.Series(np.argmax(preds, axis=1), name='Predicted')
pd.crosstab(y_actual, y_pred, margins=True)

Predicted 	0 	1 	All
Actual 			
0 	1774 	679 	2453
1 	504 	2043 	2547
All 	2278 	2722 	5000

review_num = 111
print("Review: \n"+tokenizer.sequences_to_texts([X_test[review_num]])[0])
sentiment = "Positive" if np.argmax(preds[review_num]) else "Negative"
print("\nPredicted sentiment = "+ sentiment)
sentiment = "Positive" if np.argmax(y_test.values[review_num]) else "Negative"
print("\nActual sentiment = "+ sentiment)

Review: 
love love love another absolutely superb performance miss beginning end one big treat n't rent buy

Predicted sentiment = Positive

Actual sentiment = Positive
-------------------------------------------------
import re
import pandas as pd

data = pd.read_csv('../../chapter 8/data/movie_reviews.csv', encoding='latin-1')

data.shape

(25000, 2)

string = data.SentimentText[0] + '<br /><br />- review Jamie Robert Ward (http://www.invocus.net)'

string

"first think another Disney movie, might good, it's kids movie. watch it, can't help enjoy it. ages love movie. first saw movie 10 8 years later still love it! Danny Glover superb could play part better. Christopher Lloyd hilarious perfect part. Tony Danza believable Mel Clark. can't help, enjoy movie! give 10/10!<br /><br />- review Jamie Robert Ward (http://www.invocus.net)"

len(string)

377

Remove hyperlinks from the data

string = re.sub(r"https?\://\S+", '', string)

string

"first think another Disney movie, might good, it's kids movie. watch it, can't help enjoy it. ages love movie. first saw movie 10 8 years later still love it! Danny Glover superb could play part better. Christopher Lloyd hilarious perfect part. Tony Danza believable Mel Clark. can't help, enjoy movie! give 10/10!<br /><br />- review Jamie Robert Ward ("

string = re.sub(r'<br />', ' ', string)

string

"first think another Disney movie, might good, it's kids movie. watch it, can't help enjoy it. ages love movie. first saw movie 10 8 years later still love it! Danny Glover superb could play part better. Christopher Lloyd hilarious perfect part. Tony Danza believable Mel Clark. can't help, enjoy movie! give 10/10!  - review Jamie Robert Ward ("

string = re.sub('\d','', string)

string

"first think another Disney movie, might good, it's kids movie. watch it, can't help enjoy it. ages love movie. first saw movie   years later still love it! Danny Glover superb could play part better. Christopher Lloyd hilarious perfect part. Tony Danza believable Mel Clark. can't help, enjoy movie! give /!  - review Jamie Robert Ward ("

string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', '', string)

string

"first think another Disney movie might good it's kids movie watch it can't help enjoy it ages love movie first saw movie   years later still love it Danny Glover superb could play part better Christopher Lloyd hilarious perfect part Tony Danza believable Mel Clark can't help enjoy movie give    review Jamie Robert Ward "

string = re.sub(r"can\'t", "cannot", string)

string

"first think another Disney movie might good it's kids movie watch it cannot help enjoy it ages love movie first saw movie   years later still love it Danny Glover superb could play part better Christopher Lloyd hilarious perfect part Tony Danza believable Mel Clark cannot help enjoy movie give    review Jamie Robert Ward "

string = re.sub(r"it\'s", "it is", string)

string

'first think another Disney movie might good it is kids movie watch it cannot help enjoy it ages love movie first saw movie   years later still love it Danny Glover superb could play part better Christopher Lloyd hilarious perfect part Tony Danza believable Mel Clark cannot help enjoy movie give    review Jamie Robert Ward '

len(string)

324

Find all words that start with a capital letter

re.findall(r"[A-Z][a-z]*", string)

['Disney',
 'Danny',
 'Glover',
 'Christopher',
 'Lloyd',
 'Tony',
 'Danza',
 'Mel',
 'Clark',
 'Jamie',
 'Robert',
 'Ward']

Finds 1 and 2 letter words

re.findall(r"\b[A-z]{1,2}\b", string)

['it', 'is', 'it', 'it', 'it']

-------------------------------------------
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

from sklearn.model_selection import train_test_split

data = pd.read_csv('../../chapter 8/data/movie_reviews.csv', encoding='latin-1')

data.shape

(25000, 2)

data.SentimentText = data.SentimentText.str.lower()

def clean_str(string):
    
    string = re.sub(r"https?\://\S+", '', string)
    string = re.sub(r'\<a href', ' ', string)
    string = re.sub(r'&amp;', 'and', string) 
    string = re.sub(r'<br />', ' ', string)
    string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', string)
    string = re.sub('\d','', string)
    string = re.sub(r"can\'t", "cannot", string)
    string = re.sub(r"it\'s", "it is", string)
    return string

data.SentimentText = data.SentimentText.apply(lambda x: clean_str(str(x)))

pd.Series(' '.join(data['SentimentText']).split()).value_counts().head(10)

movie    43558
film     39095
it       30659
one      26509
is       20355
like     20270
good     15099
the      13913
time     12682
even     12656
dtype: int64

stop_words = stopwords.words('english') + ['movie', 'film', 'time']
stop_words = set(stop_words)
remove_stop_words = lambda r: [[word for word in word_tokenize(sente) if word not in stop_words] for sente in sent_tokenize(r)]
data['SentimentText'] = data['SentimentText'].apply(remove_stop_words)

def combine_text(text):    
    try:
        return ' '.join(text[0])
    except:
        return np.nan

data.SentimentText = data.SentimentText.apply(lambda x: combine_text(x))

data = data.dropna(how='any')

tokenizer = Tokenizer(num_words=250)
tokenizer.fit_on_texts(list(data['SentimentText']))
sequences = tokenizer.texts_to_sequences(data['SentimentText'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Found 77348 unique tokens.

reviews = pad_sequences(sequences, maxlen=200)

with open('tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, 
                        handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)

data.SentimentText[124]
------------------------------------------
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

C:\Users\Maedr3\Anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
  warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
Using TensorFlow backend.

plt.rcParams['figure.figsize'] = [10, 10]

data = pd.read_csv('../../chapter 8/data/movie_reviews.csv', encoding='latin-1')

data.shape

(25000, 2)

data.SentimentText = data.SentimentText.str.lower()

def clean_str(string):
    
    string = re.sub(r"https?\://\S+", '', string)
    string = re.sub(r'\<a href', ' ', string)
    string = re.sub(r'&amp;', '', string) 
    string = re.sub(r'<br />', ' ', string)
    string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', string)
    string = re.sub('\d','', string)
    string = re.sub(r"can\'t", "cannot", string)
    string = re.sub(r"it\'s", "it is", string)
    return string

data.SentimentText = data.SentimentText.apply(lambda x: clean_str(str(x)))

pd.Series(' '.join(data['SentimentText']).split()).value_counts().head(10)

movie    43558
film     39095
it       30659
one      26509
is       20355
like     20270
good     15099
the      13913
time     12682
even     12656
dtype: int64

stop_words = stopwords.words('english') + ['movie', 'film', 'time']
stop_words = set(stop_words)
remove_stop_words = lambda r: [[word for word in word_tokenize(sente) if word not in stop_words] for sente in sent_tokenize(r)]
data['SentimentText'] = data['SentimentText'].apply(remove_stop_words)

data['SentimentText'][0] 

[['first',
  'think',
  'another',
  'disney',
  'might',
  'good',
  'kids',
  'watch',
  'help',
  'enjoy',
  'ages',
  'love',
  'first',
  'saw',
  'years',
  'later',
  'still',
  'love',
  'danny',
  'glover',
  'superb',
  'could',
  'play',
  'part',
  'better',
  'christopher',
  'lloyd',
  'hilarious',
  'perfect',
  'part',
  'tony',
  'danza',
  'believable',
  'mel',
  'clark',
  'help',
  'enjoy',
  'give']]

data['SentimentText'] = data['SentimentText'].apply(lambda x: x[0])

model = Word2Vec(
        data['SentimentText'],
        iter=10,
        size=100,
        window=5,
        min_count=5,
        workers=10)

vocab = list(model.wv.vocab)

len(vocab)

28838

model.wv.most_similar('insight')

[('insights', 0.7341898679733276),
 ('perspective', 0.7136699557304382),
 ('understanding', 0.6958176493644714),
 ('humanity', 0.6425720453262329),
 ('complexity', 0.6353663206100464),
 ('overwhelming', 0.6318362951278687),
 ('courage', 0.6294285655021667),
 ('ambiguity', 0.6231480836868286),
 ('appreciation', 0.6217454671859741),
 ('importance', 0.6216951012611389)]

model.wv.similarity(w1='happy', w2='sad')

0.5401219168065229

model.wv.similarity(w1='violent', w2='brutal')

0.8172468019549712

word_limit = 200
X = model[model.wv.vocab][:word_limit]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)[:word_limit]
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()


                       
-------------------------------------------------------

from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.utils import plot_model

Using TensorFlow backend.

model = Sequential()
    
model.add(Conv2D(48, (3, 3), activation='relu', padding='same', input_shape=(50,50,1)))    
model.add(Conv2D(48, (3, 3), activation='relu'))    
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.10))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 50, 50, 48)        480       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 48, 48, 48)        20784     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 24, 24, 48)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 24, 24, 48)        192       
_________________________________________________________________
dropout_1 (Dropout)          (None, 24, 24, 48)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 27648)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               14156288  
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 1026      
=================================================================
Total params: 14,178,770
Trainable params: 14,178,674
Non-trainable params: 96
_________________________________________________________________

plot_model(model, to_file='model.png', show_shapes=True)

 

import pandas as pd
import pandas_profiling

data = pd.read_csv("../../data/telco-chrun.csv")

pandas_profiling.ProfileReport(data)

                       

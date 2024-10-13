import tensorflow as tf
import pandas as pd
import re
import string

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential

# Load data
data = pd.read_csv('IMDB_Dataset.csv')

# The first 500 reviews are going to be used for testing

data = data.sample(frac=1)
data_train = data.iloc[100:]
data_test = data.iloc[:100]

# Preprocess data

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    remove_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(remove_html, '[%s]' % re.escape(string.punctuation), '')

max_features = 10000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(data_train['review'])

# convert text to sequences

X_train = tokenizer.texts_to_sequences(data_train['review'])

X_test = tokenizer.texts_to_sequences(data_test['review'])

# pad sequences

maxlen = max([len(x) for x in X_train])

print('Max length:', maxlen)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')

# Arrays of labels with 0 being negative and 1 being positive

Y_train = data_train['sentiment'].map({'positive': 1, 'negative': 0}).values
Y_test = data_test['sentiment'].map({'positive': 1, 'negative': 0}).values

# Build the model

embed_size = 128

input = Input(shape=(maxlen,))

model = Sequential([
  Embedding(max_features, embed_size), # Embedding layer: 10000 words and 128 dimensions
  LSTM(60, return_sequences=True),
  GlobalMaxPool1D(),
  Dense(50, activation='relu'),
  Dropout(0.1),
  Dense(2, activation='sigmoid') # 2 classes: positive or negative
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model

batch_size = 100
epochs = 3

# Split the test data into validation and evaluation sets
validation_split = 0.5
validation_size = int(len(X_test) * validation_split)

X_val = X_test[:validation_size]
Y_val = Y_test[:validation_size]

X_eval = X_test[validation_size:]
Y_eval = Y_test[validation_size:]

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val))

# Test the model using the evaluation set

score = model.evaluate(X_eval, Y_eval, batch_size=batch_size)

print(f'Test loss: {score[0]} - Test accuracy: {score[1]}')

# Test the model using plain text reviews

sample_reviews_plain = [
  'I think that the acting was strong in Avatar.',
  'The film didnt start for 10 minutes but it was worth the wait.', 
  'I love this movie',
  'I hate this movie',
  'I do not like this movie',
  'I do not hate this movie',
  'I would recommend this movie to my friends',
]

sample_reviews = tokenizer.texts_to_sequences(sample_reviews_plain)
sample_reviews = sequence.pad_sequences(sample_reviews, maxlen=maxlen, padding='post')

predictions = model.predict(sample_reviews)
predicted_classes = (predictions > 0.5).astype("int32")

for review, prediction in zip(sample_reviews_plain, predicted_classes):
  print(f'Review: {review} - Sentiment: {"Positive" if prediction[1] == 1 else "Negative"}')
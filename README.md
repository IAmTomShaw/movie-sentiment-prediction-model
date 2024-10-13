# Movie Sentiment Analysis

This project aims to build a sentiment analysis model using TensorFlow and Keras to classify movie reviews as positive or negative.

## Table of Contents
- [Installation](#installation)
- [Download the Data](#download-the-data)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
  ```bash
  git clone https://github.com/IAmTomShaw/movie-sentiment.git
  cd movie-sentiment
  ```

2. Install the requirements.txt file:
  ```bash
  pip install -r requirements.txt
  ```

## Download the Data

Download the IMDB dataset from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project directory. You will need to unzip the file and rename it to `IMDB_Dataset.csv`.

## Usage

1. Load the data:
  ```python
  data = pd.read_csv('IMDB_Dataset.csv')
  ```

2. Preprocess the data and split into training and testing sets:
  ```python
  data = data.sample(frac=1)
  data_train = data.iloc[100:]
  data_test = data.iloc[:100]
  ```

3. Tokenize and pad the sequences:
  ```python
  tokenizer = Tokenizer(num_words=10000)
  tokenizer.fit_on_texts(data_train['review'])
  X_train = tokenizer.texts_to_sequences(data_train['review'])
  X_test = tokenizer.texts_to_sequences(data_test['review'])
  maxlen = max([len(x) for x in X_train])
  X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
  X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')
  ```

## Model Architecture

The model consists of the following layers:
- Embedding layer
- LSTM layer
- GlobalMaxPool1D layer
- Dense layer with ReLU activation
- Dropout layer
- Dense layer with sigmoid activation

## Training

Train the model with the following parameters:
- Batch size: 100
- Epochs: 3
- Validation split: 50% of the test data

You can adjust these parameters as needed.

```python
model.fit(X_train, Y_train, batch_size=100, epochs=3, validation_data=(X_val, Y_val))
```

## Evaluation

Evaluate the model using the evaluation set:
```python
score = model.evaluate(X_eval, Y_eval, batch_size=100)
print(f'Test loss: {score[0]} - Test accuracy: {score[1]}')
```

## Testing

Test the model using plain text reviews:
```python
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
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
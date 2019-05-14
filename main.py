import tensorflow as tf
from tensorflow import keras

import numpy as np

#Download 50 000 movie reviews as data
#Note: 10 000 most common words are kept 
#Rest are discarded to simplify training
imdb = keras.datasets.imdb;
(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words = 10000);

#Confirms Data loaded correctly
print("Loaded training entries: {}, Loaded labels: {}".format(len(trainData), len(trainLabels)));

#Dict mapping words to integers
wordIndex = imdb.get_word_index();

#Saves first 4 indices for special cases
wordIndex = {key :(value + 3) for key, value in wordIndex.items()};
#Assigns special cases to first 4 indices
wordIndex["<PAD>"] = 0;
wordIndex["<START>"] = 1;
wordIndex["<UNK>"] = 2;  # unknown
wordIndex["<UNUSED>"] = 3;

#Dict mapping integers to words
reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()]);

def decode_review(text):
  '''
  Converts movie review stored as integers to text
  Input: Movie review text stored as list of ints
  Output: String mapping ints to 10 000 most common words
  '''
  return ' '.join([reverseWordIndex.get(i, '?') for i in text]);

#This pads the training data to be the same
#length as the longest array (movie review).
trainData = keras.preprocessing.sequence.pad_sequences(trainData, #Dataset being padded
value = wordIndex["<PAD>"], #Value appended to review
padding = 'post', #Padding goes to end
maxlen = 256); #Review wordcount <= 256

#Pads testing data
testData = keras.preprocessing.sequence.pad_sequences(testData,
value = wordIndex["<PAD>"],
padding = 'post',
maxlen = 256);

#Input tensor shape used (number of movie reviews)
vocabSize = 10000
#Creating model and adding layers
net = keras.Sequential();
#Input layer goes from 10 000 connections to 16
net.add(keras.layers.Embedding(vocabSize, 16));
#Standardises output length to handle variable inputs
net.add(keras.layers.GlobalAveragePooling1D());
#Dense 16-neuron layer 
#Note: numNeurons determines freedom network 
#has in representing data
net.add(keras.layers.Dense(16, activation='relu'));
#Output with sigmoid activation for binary classification
net.add(keras.layers.Dense(1, activation='sigmoid'));
#Confirms network architecture
net.summary();

net.compile(
  #Note: Uses past gradients to predict new ones
  optimizer='adam',
  #Note: Better than MES for binary probabilities
  loss='binary_crossentropy',
  #Checks how accurate the network is for progress
  metrics=['accuracy']);

#Saves first 10 000 reviews and labels for validation
#Note: Prevents overfitting
dataValidation = trainData[:10000];
labelsValidation = trainLabels[:10000];
#Saves last 15 000 reviews and labels for training
partialTrainData = trainData[10000:];
partialTrainLabels = trainLabels[10000:];

#Accuracy tracked with validation dataset.
history = net.fit(
  #Training data and labels
  partialTrainData,
  partialTrainLabels,
  #Trains 40 epochs
  epochs = 40,
  #Batches of 512 reviews each
  batch_size = 512,
  #Validates after training
  validation_data = (dataValidation, labelsValidation),
  #Amount of progress shown during training
  #Note: 0 = None, 1 = Progress bar, 2 = Every Epoch
  verbose = 1);

#Evaluates accuracy of network
results = net.evaluate(testData, testLabels);
print(results);
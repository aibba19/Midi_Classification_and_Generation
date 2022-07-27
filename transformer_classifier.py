from keras.layers import Flatten
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import Sequential
from transformer_block import TransformerBlock
from token_and_position_embedding import TokenAndPositionEmbedding
#from sklearn.metrics import f1_score, accuracy_score

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import backend as K

physical_devices = tf.config.experimental.list_physical_devices('CPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class TransformerClassifier:

  def __init__(self, input_shape, vocabulary_size, config, maxlen=12050):
    # opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
    self.set_config(config)
    opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
    vocabulary_size = vocabulary_size
    latent_dim = 128
    num_heads = 4
    num_classes = 4
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(input_shape,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocabulary_size+1, latent_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(latent_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # input = layers.Input(shape=input_shape, name='input')
    # x = layers.Embedding(vocabulary_size + 1, latent_dim, name='embed')(input)
    # x = layers.LSTM(self.nneurons, return_sequences=True)(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.LSTM(self.nneurons, return_sequences=False)(x)
    # x = Flatten()(x)
    # x = layers.Dropout(0.2)(x)
    #
    # output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # input = layers.Input(shape=input_shape, name='input')
    ## embedding_layer = TokenAndPositionEmbedding(maxlen, vocabulary_size, latent_dim)
    ## x = embedding_layer(input)
    ## transformer_block = TransformerBlock(latent_dim, num_heads=3, ff_dim=64)
    ## x = transformer_block(x)
    # x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(32, activation="relu")(x)
    # x = layers.Dropout(0.2)(x)
    # output = layers.Dense(num_classes, activation="softmax", name='output')(x)

    # Creating and compiling the model
    self.loss = "categorical_crossentropy"
    self.model = keras.Model(inputs=inputs, outputs=outputs)
    self.model.compile(opt, loss=self.loss, metrics=["accuracy"])
    # self.model = Sequential()
    # self.model.add(layers.Embedding(vocabulary_size + 1, latent_dim))
    # self.model.add(layers.LSTM(64))
    # self.model.add(layers.Dense(num_classes, activation='softmax'))
    # self.model.compile(opt, loss=self.loss, metrics=["accuracy"])
    # Setting checkpoint
    #self.checkpoint_path = "transformer_training_1/cp.ckpt"
    self.checkpoint_path = os.path.join('models', 'attention_nes', 'cp.ckpt')
    self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

  def load_model(self):
    self.model.load_weights(self.checkpoint_path)

  def set_config(self, config):
    self.nneurons = config['neurons']
    self.epochs = config['epochs']
    self.batch_size = config['batch_size']

  def fit(self, x_train, x_val, y_train, y_val):
    history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                             epochs=self.epochs,
                             validation_data=(x_val, y_val))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.show()



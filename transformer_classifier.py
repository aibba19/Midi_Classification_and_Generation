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


    self.set_config(config)
    opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
    vocabulary_size = vocabulary_size
    latent_dim = 128
    num_heads = 4
    num_classes = 6
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

    # Creating and compiling the model
    self.loss = "categorical_crossentropy"
    self.model = keras.Model(inputs=inputs, outputs=outputs)
    self.model.compile(opt, loss=self.loss, metrics=["accuracy"])

    self.checkpoint_path = os.path.join('models', 'attention_nes', 'cp-{epoch:04d}.ckpt')
    self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

  def load_model(self):
    self.model.load_weights(self.checkpoint_path)

  def set_config(self, config):
    self.nneurons = config['neurons']
    self.epochs = config['epochs']
    self.batch_size = config['batch_size']

  def fit(self, x_train, x_val, y_train, y_val):

    cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  monitor='val_accuracy',
                                                  mode='max',
                                                  save_best_only=True
                                                  )

    self.model.save_weights(self.checkpoint_path.format(epoch=0))

    history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                             epochs=self.epochs,
                             validation_data=(x_val, y_val),
                             callbacks=[cp_callback])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Training loss')
    plt.ylabel('Validation loss')

    plt.savefig('Loss_graph.png')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Training accuracy')
    plt.ylabel('Validation accuracy')

    plt.savefig('Accuracy_graph.png')
    plt.show()

    self.model.save(r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\models\transformer\saved_model')



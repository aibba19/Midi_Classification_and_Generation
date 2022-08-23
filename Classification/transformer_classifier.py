from keras.callbacks import ReduceLROnPlateau
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

  def __init__(self, input_shape, vocabulary_size, config):

    self.set_config(config)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    vocabulary_size = vocabulary_size
    latent_dim = 128
    num_heads = 4
    num_classes = 5
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(input_shape,))
    embedding_layer = TokenAndPositionEmbedding(input_shape, vocab_size=vocabulary_size+1, embed_dim=latent_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(latent_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    # Creating and compiling the model
    #self.loss = "categorical_crossentropy"
    self.loss = "binary_crossentropy"
    self.model = keras.Model(inputs=inputs, outputs=outputs)
    self.model.compile(opt, loss=self.loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name='cat_acc')])

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
                                                  monitor='val_cat_acc',
                                                  mode='max',
                                                  save_best_only=True
                                                  )

    early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_cat_acc', patience=10, verbose=1)

    self.model.save_weights(self.checkpoint_path.format(epoch=0))

    history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                             epochs=self.epochs,
                             validation_data=(x_val, y_val),
                             callbacks=[cp_callback, early_cb])

    # x = np.linspace(0, 10, 1)
    #
    # fig, ax = plt.subplots()
    # ax.plot(x, history.history['loss'], label='Training')
    # ax.plot(x, history.history['val_loss'], label='Validation')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Loss')
    # leg = ax.legend()
    # ax.legend(frameon=False, loc='upper center', ncol=2)
    #
    # fig.savefig('Loss_graph.png')
    # fig.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(x, history.history['cat_acc'], label='Training')
    # ax.plot(x, history.history['val_cat_acc'], label='Validation')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Categorical accuracy')
    # leg = ax.legend()
    # ax.legend(frameon=False, loc='upper center', ncol=2)
    #
    # fig.savefig('auc_graph.png')
    # fig.show()

    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig('loss_graph.png')
    plt.show()

    plt.plot(history.history['cat_acc'], label='Training')
    plt.plot(history.history['val_cat_acc'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.savefig('accuracy_graph.png')
    plt.show()

    #self.model.save(r'C:\Users\andri\Desktop\Tesi\Midi_Classification_and_Generation\models\transformer\saved_model')
    self.model.save(os.path.join('models', 'attention_nes', 'saved_model'))
    print("Model Saved")



from keras.layers import Flatten, Dropout
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

from token_and_position_embedding import TokenAndPositionEmbedding
from transformer_block import TransformerBlock

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# AttentionNES: Class used for training the music generator
#     - input_shape: used for the Input layer
#     - vocabulary_sizes: size of each (musical) word vocabulary; it is important for the Embedding layer
class LSTM:

    def __init__(self, input_shape, vocabulary_size):

        print("Input shape for LSTM Model: ", input_shape )
        print("Vocabulary size for LSTM Model: ", vocabulary_size)
        
        # Compiling the model
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        #This dimension probably come from
        #piano_roll : np.ndarray, shape=(128,times.shape[0])
        #from pretty midi lib
        latent_dim = 128

        #latent_dim = 50
        
        num_classes = 5

        # Creating and compiling the model

        #input = layers.Input(shape=input_shape, name='input', dtype="int32")
        
        #The second parameter is the output dimension of the layer
        #x = layers.Embedding(vocabulary_size + 1, latent_dim, name='embed',  mask_zero= True)(input)
        
        #x = layers.Embedding(vocabulary_size + 1,num_classes,name='embed')(input)

        #x = layers.LSTM(latent_dim, return_sequences=True)(x)
        #x = Flatten()(x)
        #x = Dropout(0.5)(x)
        #First argument is the number of classes in output
        #output = layers.Dense(num_classes, activation='softmax', name='output')(x)

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

        #MIA PROVA
        input = layers.Input(shape=input_shape, name='input', dtype="int32")
        x = layers.Embedding(vocabulary_size + 1, latent_dim, name='embed', mask_zero=True)(input)
        x = layers.Bidirectional(tf.keras.layers.LSTM(latent_dim))(x)
        x = layers.Dense(latent_dim, activation='relu')(x)
        output = layers.Dense(num_classes, activation='sigmoid', name='output')(x)


        # Creating and compiling the model
        self.loss = 'binary_crossentropy'
        self.model = keras.Model(input, output)
        self.model.compile(opt, loss=self.loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name='cat_acc')])

        # Setting checkpoint
        self.checkpoint_path = os.path.join('models', 'lstm', 'cp.ckpt')
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)

    def fit(self, x_train, x_test, y_train, y_test):
        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      monitor='val_cat_acc',
                                                      mode='max',
                                                      save_best_only=True
                                                      )

        early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_cat_acc', patience=10, verbose=1)

        history = self.model.fit(x_train, y_train,
                                 batch_size=128,
                                 epochs=100,
                                 validation_data=(x_test, y_test),
                                 callbacks=[cp_callback])

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

        self.model.save(os.path.join('models', 'lstm', 'saved_model'))
        print("Model Saved")

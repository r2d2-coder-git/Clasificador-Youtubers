
#Sistema operativo
import os as os
#Tensorflow 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins import projector
#Pandas library is good for analyzing tabular data. You can use it for exploratory data analysis, statistics, visualization.
import pandas as pd
#Scikit-learn is a collection of advanced machine-learning algorithms for Python. It also is built upon Numpy and SciPy.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#KERAS es una API escrita en python que permite acceder a otros frameworks que desarrollan redes neuronales como TensorFlow.
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
from keras.engine.saving import load_model
#Librería para plotear resultados de modelos.
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#Libreria para preprocesado de texto.
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#Libreria numpy
import numpy as np
#Librerías para configuración de hiperparámetros.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
#Preprocesado de etiquetas
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
#json
import json
#VARIABLES GLOBALES DEL MODELO
fichero_com = 'redNeuronal/data/comentarios.csv'
maxlen = 200
embedding_dim = 100
vocab_size = 0
embedding_matrix = np.zeros((vocab_size, embedding_dim))


#Método para encontrar los vectores de un vocabulario dentro de un conjunto words embedding preentrenado.
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath, encoding='utf8') as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

# define baseline model
def baseline_model(optimizer='adam', loss='categorical_crossentropy'):
	# Se crea el modelo con 4 capas, una embedding, una piscina para reducir las carecterísticas, y 2 densas.
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    #model.summary()
    return model

def main():
    #Se lee el fichero csv de comentarios
    df = pd.read_csv(fichero_com, names=['comentarios', 'categoria'], sep=',')
    #El primer valor es el nombre de la columna
    comentarios = df['comentarios'].values[1:]
    Y = df['categoria'].values[1:]
    #Mostrar distribución de datos.
    serie = pd.Series(df['categoria'].values[1:])
    distribucion_clases = serie.value_counts()

    categorias = list(distribucion_clases.keys())
    print(categorias)
    values = distribucion_clases.values
    plt.pie(values, autopct='%1.1f%%', labels=categorias)
    plt.title("Distribución de clases")
    plt.axis('equal')
    #plt.show()

    # Codficar las variables de salida como enteros.
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # Convertir los enteros a variables dummy (one hot encoding)
    dummy_y = np_utils.to_categorical(encoded_Y)
    dummy_y = np.argmax(dummy_y, axis=-1)

    print (dummy_y)
    #Dividir el dataset en conjunto de training y test
    sentences_train, sentences_test, y_train, y_test = train_test_split(
    comentarios, dummy_y, test_size=0.25, random_state=1000)
    print(y_train)
    #Embedding words
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    
    global vocab_size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    print("el tamaño del vocabulario es: ",vocab_size)
    print("el tamaño training antes tokenizar es: " , len(sentences_train))
    print("El tamaño del conjunto de entrenamiento despues tokenizar: ", len(X_train))
    print("Tamaño de y " , len(y_train))

    #Se añade un padding para que todas las frases se vean representadas por vectores del mismo tamaño.
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    #Creamos la matriz con los vectores preentrenados de palabras
    global embedding_matrix
    embedding_matrix = create_embedding_matrix(
    'redNeuronal/data/embeddings-m-model.vec',
    tokenizer.word_index, embedding_dim)
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("Tamaño del vocabulario cubierto por nuestros vectores preentrenados " ,nonzero_elements / vocab_size)
    #Entrenamiento
    estimator = KerasClassifier(build_fn=baseline_model, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    #Hiperparámetros
    optimizers = ['adam',"rmsprop"] #rmsprop  
    epochs = [30,50,100]
    batches = [10,20,30]
    #Entrenamiento parrilla de hiperparámetros
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy', cv=kfold, verbose =0, n_jobs=3) #Uso de todas las CPUs menos 2
    grid_result = grid.fit(X_train, y_train)
    #Modelo
    estimator = grid.best_estimator_
    print(grid.best_params_)
    print(grid.best_score_)
    #Predicciones de los ejemplos de test
    estimator.model.save('path_to_my_model.h5')
    np.save("xtest.npy", X_test)
    np.save("ytest.npy", y_test)
    
    #PROYECTOR DE EMBEDDINGS
    # Set up a logs directory, so Tensorboard knows where to look for files
    #Cogemos las palabras en el tokenizer
    string_json = tokenizer.get_config()['word_docs']
    palabras = json.loads(string_json).keys()

    log_dir='./redNeuronal/proyector_emb/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'metadata.tsv'), "w",encoding='UTF-8') as f:
        for subpalabra in palabras:
            f.write("{}\n".format(subpalabra))
        for unknown in range(1, vocab_size - len(subpalabra)):
            f.write("unknown #{}\n".format(unknown))
    # Save the weights we want to analyse as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, so
    # we will remove that value.
    weights = tf.Variable(estimator.model.layers[0].get_weights()[0][1:])
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)
    
if __name__ == '__main__':
    main()

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

from sklearn.metrics import confusion_matrix

#VARIABLES GLOBALES DEL MODELO
fichero_com = 'redNeuronal/data/comentarios.csv'
maxlen = 200
embedding_dim = 100
vocab_size = 0
embedding_matrix = np.zeros((vocab_size, embedding_dim))

#Función para mostrar en 2 gráficas la historia del entrenamiento y la validación.
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

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
def baseline_model():
	# Se crea el modelo con 4 capas, una embedding, una piscina para reducir las carecterísticas, y 2 densas.
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(4, activation='sigmoid'))
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

    # Codficar las variables de salida como enteros.
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # Convertir los enteros a variables dummy (one hot encoding)
    dummy_y = np_utils.to_categorical(encoded_Y)

    #Dividir el dataset en conjunto de training y test
    sentences_train, sentences_test, y_train, y_test = train_test_split(
    comentarios, dummy_y, test_size=0.25, random_state=1000)

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
    model = baseline_model()
    history = model.fit(X_train,y_train, epochs =200, batch_size=20, verbose = 0,validation_split=0.33 )
    #Limpiar la sesión para próximos fit.
    clear_session()
    #Evaluación del modelo.
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    #Dibujar gráficas.
    plot_history(history)


    #estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
    #kfold = KFold(n_splits=10, shuffle=True)
    #results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    #print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    #estimator.fit(X_train, y_train)
    #estimator.model.predict(X_test)
   # print(estimator.score(X_test, y_test))
    

if __name__ == '__main__':
    main()

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
fichero_com = 'keras/data/comentarios.csv'
maxlen = 100
embedding_dim = 50
vocab_size = 0
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

# define baseline model
def baseline_model():
	# Se crea el modelo con 4 capas, una embedding, una piscina para reducir las carecterísticas, y 2 densas.
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model

def main():
    df = pd.read_csv(fichero_com, names=['comentarios', 'categoria'], sep=',')
    #El primer valor es el nombre de la columna
    comentarios = df['comentarios'].values[1:]
    Y = df['categoria'].values[1:]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    sentences_train, sentences_test, y_train, y_test = train_test_split(
    comentarios, dummy_y, test_size=0.25, random_state=1000)

    #Embedding words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    global vocab_size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    print(X_train)
    print(dummy_y)
    print("el tamaño del vocabulario es: ",vocab_size)
    print("el tamaño training antes tokenizar es: " , len(sentences_train))
    print("El tamaño del conjunto de entrenamiento despues tokenizar: ", len(X_train))
    print("Tamaño de y " , len(y_train))

    #Se añade un padding para que todas las frases se vean representadas por vectores del mismo tamaño.
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


    #Entrenamiento
    estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    estimator.fit(X_train, y_train)
    estimator.model.predict(X_test)
    print(estimator.score(X_test, y_test))

if __name__ == '__main__':
    main()

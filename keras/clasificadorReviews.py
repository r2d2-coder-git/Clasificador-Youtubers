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

filepath_dict = {'yelp':   'data/yelp_labelled.txt',
                 'amazon': 'data/amazon_cells_labelled.txt',
                 'imdb':   'data/imdb_labelled.txt'}

df_list = []
#Para cada fichero de opiniones se va creando un dataframe que se añade a una lista de dataframes que después se une con el concat en un sólo dataframe.
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)


#Cogemos las opiniones de la página yelp.com (1000 opiniones)
df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

#Dividimos las opiniones en un 75% de entrenamiento y un 25% de test. Con 0 o 1 como variables de salida predictora.
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

#Aplicamos la técnica BOW. A través del vocabulario sacado de las opiniones del train, sacar los vectores con la cuenta de palabras para train y test.
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
print(X_test.__len__)

#Creamos un clasificador de regresión logística y entrenamos el modelo con el trainset y la salida y.
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy:", score)

#Entrenamiento y accuracy para todas las páginas de reviews.
for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train)
    X_test  = vectorizer.transform(sentences_test)


    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print('Accuracy for {} data: {:.4f}'.format(source, score))


#KERAS 

#RED NEURONAL CON COUNTVECTORIZER
#Crear la red neuronal con una capa intermedia de 10 nodos y una de salida.
input_dim = X_train.shape[1]  # Number of features
print (input_dim)
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()
#Hiperparámetros de la red neuronal, entrenamiento del modelo y validación.

history = model.fit(X_train, y_train,
                    epochs=15,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

#Limpiar la sesión para próximos fit.
clear_session()
#Evaluación del modelo.
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#Plotear las gráficas la primera gráfica muestra el accuracy de train y test en función del número de épocas y la segunda gráfica va mostrando los resultados de la función de pérdidas 
# También en función del número de épocas. Un factor importante es ver cuando la función de pérdidas para el conjunto de test vuelve a subir para ver que estamos incurriendo en overfitting 
# en el modelo.

#plot_history(history)


##RED NEURONAL CON WORD EMBEDDING
#Preprocesado de texto, tokenizar las palabras.
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print("el tamaño del vocabulario es: ",vocab_size)
print("El tamaño del conjunto de entrenamiento es: ", len(sentences_train))
maxlen = 100

#Se añade un padding para que todas las frases se vean representadas por vectores del mismo tamaño.
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(len(X_test[0]))
embedding_dim = 50

model = Sequential()
#Input_dim : Tamaño del diccionario
#Output_dim : Tamaño de los vectores sin ceros como paddings
#Input_length : Maxima longitud de los vectores con ceros de padding.
#Weights : Son los pesos que nos dan en el conjunto de embedding preentrenado.
#Trainable : True indica que esos pesos preentrenados se sigan entrenando con nuestras frases, False indica que los pesos para las palabras del vocabulario son siempre los de preentramiento.
#model.add(layers.Embedding(input_dim=vocab_size, 
#                           output_dim=embedding_dim, 
#                           input_length=maxlen))
#Si deseas conectar una capa densa directamente a una capa de incrustación, primero debe aplanar 
# la matriz de salida 2D a un vector 1D utilizando la capa aplanada.
#model.add(layers.Flatten())
#También se puede utilizar una piscina de datos temporales en la que se guarde el valor máximo de todas las características. (Un vector de 50 posiciones con los valores máximos de entre
# todas las palabras.) Con esto reducimos el número de características un montón, ya que en la capa de flatten tenemos 5000 características y en con esta pool tenemos solo 50 características 
# por frase.
#Creamos la matriz de words embedding preentrenadas. Que incluye nuestro vocabulario acompañado de los vectores.
embedding_matrix = create_embedding_matrix(
    'data/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)

model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#plot_history(history)


##CONVOLUTIONAL NEURAL NETWORKS (CNN)

embedding_dim = 100

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
#Capa convolucional, parámetros:
#Número de filtros para cada oración
#Número de kernels: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
#Función de activación.
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#ENTRENAMIENTO DE MODELOS CON DISTINTOS HIPERPARÁMETROS 
# Main settings
epochs = 20
embedding_dim = 50
maxlen = 100
output_file = 'data/output.txt'

# Run grid search for each source (yelp, amazon, imdb)
for source, frame in df.groupby('source'):
    print('Running grid search for data set :', source)
    sentences = df['sentence'].values
    y = df['label'].values

    # Train-test split
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    # Tokenize words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # Parameter grid for grid search
    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
    model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=10,
                            verbose=False)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
    grid_result = grid.fit(X_train, y_train)

    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)

    # Save and evaluate results
    prompt = input(f'finished {source}; write to file and proceed? [y/n]')
    if prompt.lower() not in {'y', 'true', 'yes'}:
        break
    with open(output_file, 'a',encoding='utf8') as f:
        s = ('Running {} data set\nBest Accuracy : '
             '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(
            source,
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)
        print(output_string)
        f.write(output_string)

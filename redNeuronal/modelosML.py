#Sistema operativo
import os as os
#Tensorflow 
import tensorflow as tf
from tensorboard.plugins import projector
#Pandas library is good for analyzing tabular data. You can use it for exploratory data analysis, statistics, visualization.
import pandas as pd
#Scikit-learn is a collection of advanced machine-learning algorithms for Python. It also is built upon Numpy and SciPy.
from sklearn.model_selection import PredefinedSplit
#KERAS es una API escrita en python que permite acceder a otros frameworks que desarrollan redes neuronales como TensorFlow.
from keras.models import Sequential
from keras import layers
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
#Preprocesado de etiquetas
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV
#json
import json
#random
import random
#Guardar hiperparámetros
from keras.callbacks import CSVLogger

#VARIABLES GLOBALES DEL MODELO

#Fichero de comentarios
fichero_com = 'redNeuronal/data/comentarios_sin_duplicados.csv'

#Máxima longitud de comentario
maxlen = 200

#Tamaño de embeddings
embedding_dim = 100

#Tamaño del vocabulario
vocab_size = 0

#Matriz de embedding preentrenados
embedding_matrix = np.zeros((vocab_size, embedding_dim))

#Tipo de red neuronal a entrenar.
# 0: Red normal
# 1: CNN
# 2: LSTM
tipo_red = 0

#Ficheros de datos
fichero_train = 'resultados_modelos/data/train_df.csv'
fichero_validation =  'resultados_modelos/data/validation_df.csv'
fichero_test =  'resultados_modelos/data/test_df.csv'
fichero_embedding = 'redNeuronal/data/embeddings-m-model.vec'

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
def baseline_model(tipo,optimizer='adam', loss='categorical_crossentropy'):
	# Se crea el modelo con 4 capas, una embedding, una piscina para reducir las carecterísticas, y 2 densas.
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
    #Red convolucional
    if (tipo == 1):
        model.add(layers.Conv1D(128, 5, activation='relu'))
    #Red recurrente
    elif (tipo == 2):
        model.add(layers.LSTM(units = 100, return_sequences=True, recurrent_dropout=0.2))
        model.add(layers.LSTM(units = 100, return_sequences=True))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model
        
def array_to_dataframe(x_test):
    x_test_df = pd.DataFrame({"X_test":[x_test[0]]})
    for i in range(len(x_test)-1):
        x_test_df = x_test_df.append({'X_test':x_test[i+1]}, ignore_index=True)
    return x_test_df

def train_test_validation_split(df, categorias):
    column_names = df.columns
    df_train = pd.DataFrame(columns=column_names)
    df_validation = pd.DataFrame(columns=column_names)
    df_test = pd.DataFrame(columns=column_names)
    #Recorremos las categorías para separar los nombres de las canales en proporción.
    for categoria in categorias:

        category_rows = df.loc[df['categoria']== categoria]
        names_channels = list(category_rows['nombre_canal'].unique())
        num_channels = len(names_channels)
        #Separamos los nombres de los canales segun el porcentaje de train, validation y test
        names_train = random.sample(names_channels,round(num_channels*0.6))
        remaining_channels = list(set(names_channels)-set(names_train))
        names_validation = random.sample(remaining_channels, round(num_channels*0.2))
        names_test = list(set(remaining_channels)- set(names_validation))
        #Asignamos las filas correspondientes a cada conjunto
        for name_train in names_train:
            rows_train = df.loc[df['nombre_canal'] == name_train]
            df_train = df_train.append(rows_train, ignore_index= False)

        for name_validation in names_validation:
            rows_validation = df.loc[df['nombre_canal'] == name_validation]
            df_validation = df_validation.append(rows_validation, ignore_index= False)

        for name_test in names_test:
            rows_test = df.loc[df['nombre_canal'] == name_test]
            df_test = df_test.append(rows_test, ignore_index= False)
    return df_train,df_validation,df_test

def main():
    #Se lee el fichero csv de comentarios
    df = pd.read_csv(fichero_com, names=['comentarios', 'id_comentarios', 'nombre_canal', 'categoria'], sep=',')
    #La primera fila son los nombres de las columnas
    df = df.drop(0, axis=0)
    df.reset_index(drop=True, inplace=True)
    Y = df['categoria'].values[:]
    #Mostrar distribución de datos.
    serie = pd.Series(df['categoria'].values[1:])
    distribucion_clases = serie.value_counts()
    categorias = list(distribucion_clases.keys())

    #Gráfica de tarta con la distribución por categorías.
    values = distribucion_clases.values
    plt.pie(values, autopct='%1.1f%%', labels=categorias)
    plt.title("Distribución de clases")
    plt.axis('equal')
    plt.show()
    
    # Codficar las variables de salida como enteros.
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    # Convertir los enteros a variables dummy (one hot encoding)
    dummy_y = np_utils.to_categorical(encoded_Y)
    dummy_y = np.argmax(dummy_y, axis=-1) #Para que no esté en one hot encoding y ver métricas
    #Guardar un diccionario con las variables en string asociadas con el encoding correspondiente.
    topicos_encoded = dict(zip(dummy_y,Y))
    np.save("topicos_encoded.npy",topicos_encoded)
    
    #Añadimos la columna dummy_y al dataFrame.
    df = df.assign(dummy_y = dummy_y)
    
    if not os.path.isfile(fichero_train):
        #Dividir el dataset en conjunto de training test y validacion
        train_df, validation_df, test_df = train_test_validation_split(df, categorias)
    else:
        print("Cargando conjuntos de datos...")
        train_df = pd.read_csv(fichero_train, names=['comentarios', 'id_comentarios', 'nombre_canal', 'categoria', 'dummy_y'], sep=',')
        index_names = train_df[train_df['dummy_y'] == 'dummy_y'].index
        train_df.drop(index_names, inplace=True)

        validation_df = pd.read_csv(fichero_validation, names=['comentarios', 'id_comentarios', 'nombre_canal', 'categoria', 'dummy_y'], sep=',')
        index_names = validation_df[validation_df['dummy_y'] == 'dummy_y'].index
        validation_df.drop(index_names, inplace=True)

        test_df = pd.read_csv(fichero_test, names=['comentarios', 'id_comentarios', 'nombre_canal', 'categoria', 'dummy_y'], sep=',')
        index_names = test_df[test_df['dummy_y'] == 'X_test'].index
        test_df.drop(index_names, inplace=True)

    #Cambiamos el tipo de datos a np.int64
    train_df = train_df.astype({"dummy_y": np.int64})
    validation_df = validation_df.astype({"dummy_y": np.int64})
    if not os.path.isfile('resultados_modelos/data/test_df.csv'):
        test_df = test_df.astype({"dummy_y": np.int64})

    #Juntamos el training y la validacion y preparamos el cross validation
    train_val_df = pd.concat([train_df,validation_df],ignore_index = True)
    lenTrain = len(train_df.index)
    lenValidation = len(validation_df.index)
    splitCV = [0] * lenTrain + [-1] * lenValidation
    
    sentences_train = train_val_df['comentarios']
    y_train = train_val_df['dummy_y']
    sentences_test = test_df['comentarios']
    
    #Embedding words
    tokenizer = Tokenizer(num_words=60000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    
    global vocab_size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    print("El tamaño del vocabulario es: ",vocab_size)

    #Se añade un padding para que todas las frases se vean representadas por vectores del mismo tamaño.
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    x_test_df = array_to_dataframe(X_test)
    #Añadimos los vectores de 200 posiciones a cada comentario de test para predecir en predecirModelo.py.
    test_df.reset_index(drop=True, inplace=True)
    test_df = pd.concat([test_df,x_test_df], axis=1)

    #Creamos la matriz con los vectores preentrenados de palabras 
    global embedding_matrix
    embedding_matrix = create_embedding_matrix(
    fichero_embedding,
    tokenizer.word_index, embedding_dim)
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    print("Tamaño del vocabulario cubierto por vectores preentrenados: " ,nonzero_elements / vocab_size)

    #Entrenamiento
    estimator = KerasClassifier(build_fn=baseline_model, tipo=2, verbose=0)
    ps = PredefinedSplit(test_fold = splitCV)
    kfold = ps

    #Hiperparámetros
    optimizers = ['rmsprop' ,'adam'] #rmsprop  
    epochs = [30,50,75] 
    batches = [256,512,1024] 

    #Entrenamiento parrilla de hiperparámetros
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='accuracy', cv=kfold, verbose =0, n_jobs=3, error_score = "raise") #Uso de todas las CPUs menos 2
    #Logger para guardar hiperparámetros
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    grid.fit(X_train, y_train,callbacks=[csv_logger])
    #Modelo
    estimator = grid.best_estimator_

    #Mejores hiperparámetros
    print(grid.best_params_)
    print(grid.best_score_)
    
    #Guardamos el modelo y guardamos el dataframe de test si no está guardado aún.
    estimator.model.save('model.h5')
    if not os.path.isfile(fichero_train):
        test_df.to_csv('test_df.csv', index=False,encoding='utf-8', sep= ',')
        train_df.to_csv('train_df.csv', index=False,encoding='utf-8', sep= ',')
        validation_df.to_csv('validation_df.csv', index=False,encoding='utf-8', sep= ',')

    #PROYECTOR DE EMBEDDINGS
    #Cogemos las palabras en el tokenizer
    string_json = tokenizer.get_config()['word_docs']
    palabras = json.loads(string_json).keys()
    #Directorio donde se ponen los ficheros para configurar el modelo de visión de las palabras.
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

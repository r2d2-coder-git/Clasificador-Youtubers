# load numpy array from npy file
import numpy as np
import pandas as pd
#keras
from keras.engine.saving import load_model
from keras.utils import np_utils
#sklearn
import sklearn.metrics as metrics
#Contador 
from collections import Counter
#Visualización de matrices de confusión
import seaborn as sns
import matplotlib.pyplot as plt

import collections

fichero_test = "resultados_modelos/data/test_df.csv"
modelo_actual = "red_normal"

#Convertir los elementos de una lista que son string en vectores con un formato específico.
def string_to_vector(list_strings):
    list_vector = []
    for vector in list_strings:
        vector = vector.replace("[",'')
        vector = vector.replace("]",'')
        vector = vector.replace("\n",'')
        list_int = np.array(list(map(int,vector.split())))
        list_vector.append(list_int)
        
    return np.asarray(list_vector)

#Función que asigna etiquetas a los canales y extrae metricas.
def asignarEtiquetas(df, prediccion_string):
    canales = df.nombre_canal.unique()
    fich_resultados = open('resultados.txt','w', encoding='utf-8')
    aciertos = 0
    categorias_originales =[]
    categorias_predichas = []
    #Bucle que extrae información de cada canal.
    for canal in canales:
        filas = df.loc[df['nombre_canal'] == canal]
        categoria = filas.categoria.unique()[0]
        fich_resultados.write("CANAL DE " + canal + " Categoria: " + categoria + '\n')
        #Etiquetado del canal
        predicciones = np.array(filas.predicciones_string)
        num_comentarios = len(predicciones)
        desglose_resultados = dict(Counter(predicciones))
        num_aciertos = desglose_resultados[categoria]
        etiqueta_mejor = max(desglose_resultados, key=desglose_resultados.get)
        fich_resultados.write("CLASIFICACIÓN DEL CANAL:\n")
        fich_resultados.write("En el canal de " + canal + " hay un total de " + str(num_aciertos) + " aciertos de un total de " + str(num_comentarios) + " comentarios." + 
        " Mayor categoría asignada: " + etiqueta_mejor + " con un total de " + str(desglose_resultados.get(etiqueta_mejor)) + " comentarios.\n")
        fich_resultados.write("Accuracy: " +str(num_aciertos/num_comentarios) + '\n')
        #Ver cuantos canales se han etiquetado correctamente.
        if etiqueta_mejor == categoria:
            fich_resultados.write("ACIERTO EN CLASIFICACIÓN DE CANAL!\n\n")
            aciertos+=1
        else:
            fich_resultados.write("FALLO EN CLASIFICACIÓN DE CANAL :(\n\n")
        #Asignamos al array de canales las etiquetas predichas 
        categorias_originales.append(categoria)
        categorias_predichas.append(etiqueta_mejor)

    fich_resultados.write("MATRIZ DE CONFUSION POR CANALES\n")
    matrix_canales = metrics.confusion_matrix(categorias_originales, categorias_predichas)
    sns.heatmap(matrix_canales, annot=True, xticklabels=prediccion_string, yticklabels=prediccion_string)
    plt.show()
    fich_resultados.write(np.array2string(matrix_canales) + '\n\n')
    report_canales = metrics.classification_report(categorias_originales, categorias_predichas, digits = 3)
    fich_resultados.write(report_canales+ '\n')

    fich_resultados.write("PORCENTAJE DE ACIERTOS EN CLASIFICACIÓN: " + str(aciertos/len(canales)*100) + '%\n\n')

    fich_resultados.close()
    return None

def main():
    # Cargamos los conjuntos de test y el mejor modelo entrenado.
    test_df = pd.read_csv(fichero_test, names=['comentarios', 'nombre_canal', 'categoria','dummy_y','X_test'], sep=',')
    
    
    index_names = test_df[test_df['dummy_y'] == 'dummy_y'].index
    test_df.drop(index_names, inplace=True)
    print(test_df)
    #Agrupamos por canales
    test_df = test_df.sort_values('nombre_canal')
    #Pandas convierte los arrays en strings.
    X_test_aux = test_df['X_test'].to_numpy()
    #Convertimos los string que representan los comentarios en vectores.
    X_test = string_to_vector(X_test_aux)
    y_test = test_df ['dummy_y'].to_numpy()
    print("Conjunto de test " , len(X_test))
    print("Conjunto de train", len(y_test))
    #Cargamos el diccionario de asignación tópico-claseInt.
    topicos_encoded = np.load('resultados_modelos/' + modelo_actual + '/topicos_encoded.npy',allow_pickle='TRUE').item()
    #Cargamos el modelo
    model = load_model('resultados_modelos/' + modelo_actual + '/model.h5')

    #Hacemos predict de X_test
    predicciones = model.predict(X_test)
    clases_predichas = predicciones.argmax(axis=-1)
    #Precisión en el test y cambiar a one hot encoding para model.evaluate
    y_test_one_hot = np_utils.to_categorical(y_test)
    _, accuracy_test = model.evaluate(X_test, y_test_one_hot, verbose = False)
    print("Testing Accuracy:  {:.4f}".format(accuracy_test))

    #Convertir las clases predichas en strings y las clases del dataframe también.
    prediccion_string = list(map(topicos_encoded.get,clases_predichas))
    y_test = [int(x) for x in y_test]
    y_test_string = list(map(topicos_encoded.get,y_test))
    #Añadimos los resultados al dataframe para comprobar métricas por canal.
    test_df = test_df.assign(y_test_string=y_test_string,predicciones_string=prediccion_string,predicciones_int=clases_predichas)

    #Función que genera el fichero resultados.txt con la información de las predicciones por canal de youtube.
    od = collections.OrderedDict(sorted(topicos_encoded.items()))
    topicos_sorted = [v for k,v in od.items()]
    asignarEtiquetas(test_df, topicos_sorted)
    #Analisis de los comentarios en global.
    matrix_comentarios = metrics.confusion_matrix(y_test_string, prediccion_string)
    sns.heatmap(matrix_comentarios, fmt = 'd', annot=True, xticklabels=topicos_sorted, yticklabels=topicos_sorted)
    plt.show()
    sns.heatmap(matrix_comentarios/np.sum(matrix_comentarios), annot=True, 
            fmt='.2%', cmap='Blues', xticklabels=topicos_sorted, yticklabels=topicos_sorted)
    plt.show()
    report_comentarios = metrics.classification_report(y_test_string, prediccion_string, digits = 3)
    #Añadir los resultados al fichero.
    fich_resultados = open('resultados.txt','a', encoding='utf-8')
    fich_resultados.write("MATRIZ DE CONFUSIÓN DE TODOS LOS COMENTARIOS DE TEST\n\n")
    fich_resultados.write(np.array2string(matrix_comentarios) + '\n\n')
    fich_resultados.write("MÉTRICAS DE TODOS LOS COMENTARIOS\n\n")
    fich_resultados.write(report_comentarios+ '\n')

    
if __name__ == '__main__':
    main()
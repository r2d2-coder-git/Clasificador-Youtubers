
# load numpy array from npy file
import numpy as np
#keras
from keras.engine.saving import load_model
from keras.utils import np_utils
#sklearn
import sklearn.metrics as metrics

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

    
def main():
    # Cargamos los conjuntos de test y el mejor modelo entrenado.
    X_test = np.load('xtest.npy')
    y_test = np.load('ytest.npy')
    print(X_test.shape)
    print(y_test.shape)
    model = load_model("path_to_my_model.h5")

    #Hacemos predict de X_test
    predicciones = model.predict(X_test)
    clases_predichas = predicciones.argmax(axis=-1)
    #Precisión en el test y cambiar a one hot encoding para model.evaluate
    y_test_one_hot = np_utils.to_categorical(y_test)
    score, accuracy_test = model.evaluate(X_test, y_test_one_hot, verbose = False)
    print("Testing Accuracy:  {:.4f}".format(accuracy_test))
    #Matriz de confusión con los resultados de test y las clases predichas.
    matrix = metrics.confusion_matrix(y_test, clases_predichas)
    print(matrix)
    #Informe de porcentajes por clase
    nombre_clases = list(map(str,set(clases_predichas)))
    print(metrics.classification_report(y_test, clases_predichas,target_names = nombre_clases, digits=3))
if __name__ == '__main__':
    main()
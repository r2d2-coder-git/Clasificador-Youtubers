
# load numpy array from npy file
import numpy as np
import pandas as pd
#keras
from keras.engine.saving import load_model
from keras.utils import np_utils
#sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
def plot_history(history):
    acc = history['accuracy'].values
    loss = history['loss'].values
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def main():
    evolution_normal = "./resultados_modelos/red_normal/training.log"
    df = pd.read_csv(evolution_normal, names=['epoch', 'accuracy', 'loss'], sep=',')
    df = df.drop(0, axis=0)
    df.reset_index(drop=True, inplace=True)
    plot_history(df)


if __name__ == '__main__':
    main()

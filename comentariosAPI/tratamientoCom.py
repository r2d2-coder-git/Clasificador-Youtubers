import pandas as pd
import csv
def main():
    fichero_com = 'redNeuronal/data/comentarios.csv'
    #Se lee el fichero csv de comentarios
    etiquetas_nuevas = {"id_comentarios": "id_comment", "nombre_canal": "channel_name", "categoria": "category" }
    df = pd.read_csv(fichero_com, names=['comentarios', 'id_comentarios', 'nombre_canal', 'categoria'], sep=',')
    print(len(df.index))

    #Para training 
    df_sin_duplicados = df.drop_duplicates(subset = ['comentarios','nombre_canal','categoria'], keep = 'last').reset_index(drop=True)
    df_sin_duplicados.to_csv("./redNeuronal/data/comentarios_sin_duplicados.csv",index=False, quoting=csv.QUOTE_ALL)


    #Para corpus
    df_corpus = df_sin_duplicados.drop(['comentarios'],axis = 1)
    df_corpus.rename(columns = etiquetas_nuevas, inplace = True)
    df_corpus.to_csv("corpus.csv",index = False, quoting=csv.QUOTE_ALL)

    print(len(df_corpus.index))


if __name__ == '__main__':
    main()

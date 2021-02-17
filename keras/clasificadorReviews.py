#Pandas library is good for analyzing tabular data. You can use it for exploratory data analysis, statistics, visualization.
import pandas as pd
#Scikit-learn is a collection of advanced machine-learning algorithms for Python. It also is built upon Numpy and SciPy.
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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
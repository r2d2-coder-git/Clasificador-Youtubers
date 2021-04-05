#Especificar fichero de credenciales.
CLIENT_SECRETS_FILE = "comentariosAPI/credencialesAPI.json"

#Determinar alcance de acceso
SCOPES = ["https://www.googleapis.com/auth/youtube",
          "https://www.googleapis.com/auth/youtube.force-ssl",
          "https://www.googleapis.com/auth/youtubepartner"]
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


#Crear un servicio que contacte con la API de Youtube.
import os 
import pickle
import csv
import io
import re
import emoji
import google.oauth2.credentials
 
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

#Crear Servicio de conexión a la API de youtube con credenciales en caché (con un archivo token.pickle)
def get_authenticated_service():
    credentials = None
    if os.path.exists('comentariosAPI/token.pickle'):
        with open('comentariosAPI/token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Comprobar si las credenciales son válidas
    if not credentials or not credentials.valid:
        # Comprueba si han caducado las credenciales
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
 
        # Guarda en un fichero pickle las credenciales para la siguiente ejecución.
        with open('comentariosAPI/token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
 
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

##Coger información del canal con el parámetro contentDetails en función de si el identificador es idChannel o el nombre de usuario del canal de youtube.
## Devuelve el id de las subidas de vídeo.
def coger_id_upload(conn,id):
    #Probamos buscando con un petición por nombre de usuario.
    respuesta = conn.channels().list(
    forUsername=id,
    part="contentDetails"
    ).execute()
    num_canales = respuesta['pageInfo']['totalResults']
    #Si no existe el nombre de usuario probamos por identificador de canal
    if (num_canales == 0):
        respuesta = conn.channels().list(
        id=id,
        part="contentDetails"
        ).execute()
        num_canales = respuesta['pageInfo']['totalResults']
        #Si existe devolvemos el id del canal y el id de las subidas.
        if (num_canales > 0):
            return id,respuesta['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    #Si es el nombre de usuario buscamos el id del canal y lo devolvemos junto con el id de las subidas.
    id = respuesta['items'][0]['id']
    return id, respuesta['items'][0]['contentDetails']['relatedPlaylists']['uploads']


#Función para coger la categoría de un canal
def categoria_canal (conn, id_canal):
    res = channels_response = conn.channels().list(
        id=id_canal,
        part="topicDetails"
        ).execute()
    return res['items'][0]['topicDetails']['topicCategories']

#Función para coger los últimos num_videos por medio del identificador de subidas del canal de youtube.
def coger_videos(conn, playlist_id, num_videos):
    id_videos = []
    next_page_token = None 
    titulos_videos = []
    for i in range(num_videos):
        res = conn.playlistItems().list(playlistId = playlist_id, part = 'snippet', maxResults = 1, pageToken = next_page_token).execute()
        id_videos.append(res['items'][0]['snippet']['resourceId']['videoId'])
        titulos_videos.append(res['items'][0]['snippet']['title'])
        next_page_token = res.get('nextPageToken')
        #Comprobar si hay más páginas de vídeos.
        if next_page_token is None:
            break
    return titulos_videos, id_videos

#Función para coger los comentarios de los videos asociados a id_videos y devolver cada comentario con la categoria del canal.
def coger_comentarios(conn,id_videos,categoria):
    comentarios = []
    next_page_token = None
    
    for id_video in id_videos:
        results = conn.commentThreads().list(part = 'snippet', videoId = id_video, textFormat = 'plainText').execute()
        while results:
            for item in results['items']:
                #Poner límite de comentarios por canal
                if (len(comentarios) == 250):
                    return comentarios
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                #Quitamos los emojis de los comentarios y eliminamos solo los comentarios con emojis
                comentario_limpio = deEmojify(comment)
                if comentario_limpio != "":
                    comentarios.append([comentario_limpio,categoria])

            #Comprobar si hay más páginas de comentarios
            if 'nextPageToken' in results:
                next_page_token = results['nextPageToken']
                results = conn.commentThreads().list(part = 'snippet', videoId = id_video, textFormat = 'plainText', pageToken = next_page_token).execute()
            else:
                break
                
    return comentarios

#Función que escribe el la lista de tripletas (titulo, id video, comentarios) en un fichero csv.
def write_to_csv(final):
    with io.open('./redNeuronal/data/comentarios.csv', "w", encoding="utf-8", newline="") as comments_file:
        comments_writer = csv.writer(comments_file)
        comments_writer.writerow(['Comentarios', 'Categoria'])
        for row in final:
            comments_writer.writerow(row)

#Quedarse con la última palabra de una url
def ultima_palabra_url(url):
    index = url.rfind('/')
    return url[index+1:]

#Concatenar las diferentes categorias de un canal.
def concatenar_categorias(list):
    result= ''
    for element in list:
        result += str(element)+" "
    return result

#Coger urls de los ficheros
def leer_canales(fich_canales):
    f = open(fich_canales, "r")
    lineas = f.readlines()
    print (lineas)
    f.close()
    return lineas
#Aplanar listas de un solo nivel y listas de listas
def aplanar_lista(lst):
    return sum(lst, [])
#Separar url y categoria 
def separar_info_canal(info):
    if(info[-1] == '\n'):
        url_categoria = info[:-1].split(sep=',')
    else:
        url_categoria = info.split(sep=',')
    url = url_categoria[0]
    categoria = url_categoria[1]
    return url,categoria
    
#Limpiar comentarios de emojis y otros simbolos
def deEmojify(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def main():
    comentarios_total = []
    fichero_canales ='comentariosAPI/canales.txt'
    info_canal = leer_canales(fichero_canales)
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    youtube = get_authenticated_service()
    num_videos = 5
    for info in info_canal:
        #Cogemos la url y la categoria del canal
        url, categoria = separar_info_canal(info)
        id = ultima_palabra_url(url)
        #Cogemos el id de las subidas del canal.
        id_canal, id_uploads = coger_id_upload(youtube, id)
        #Cogemos los títulos de los num_videos últimos y los ids de los num_videos últimos.
        titulos_videos, id_videos = coger_videos(youtube,id_uploads, num_videos)
        #Cogemos los comentarios de los vídeos junto con la categoria del canal (máximo 250 por canal).
        comentarios = coger_comentarios(youtube,id_videos, categoria)
        comentarios_total.append(comentarios)
    #Escribimos los comentarios junto con la categoria del canal en un csv.
    write_to_csv(aplanar_lista(comentarios_total))

if __name__ == '__main__':
    main()


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
import google.oauth2.credentials
 
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

#Crear Servicio de conexión a la API de youtube con credenciales en caché (con un archivo token.pickle)
def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
 
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
 
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

##Coger información del canal con el parámetro contentDetails en función de si el identificador es idChannel o el nombre de usuario del canal de youtube.
## Devuelve el id de las subidas de vídeo.
def coger_id_upload(conn,tipo_id):
    search_channel_name = ""
    channels_response=""
    id_user =""
    if tipo_id == 'u':
        search_channel_name = input('Introduce el usuario: ')
        channels_response = conn.channels().list(
        forUsername=search_channel_name,
        part="contentDetails"
        ).execute()
        search_channel_name = channels_response['items'][0]['id']
    elif tipo_id == 'i':
        search_channel_name = input('Introduce el id del canal: ')
        channels_response = conn.channels().list(
        id=search_channel_name,
        part="contentDetails"
        ).execute()
    else: 
        print('Tipo de identificador no soportado.')
        return None
    ##Cogemos todas las playlist del canal y cogemos aquella que incluye todas las subidas del usuario.
    return search_channel_name,channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

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
    return dict(zip(titulos_videos, id_videos))

#Función para coger los comentarios de los videos asociados a id_videos.
def coger_comentarios(conn,id_videos):
    comentarios_aux = []
    comentarios = []
    next_page_token = None
    
    for id_video in id_videos:
        results = conn.commentThreads().list(part = 'snippet', videoId = id_video, textFormat = 'plainText').execute()
        while results:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comentarios_aux.append(comment)

            #Comprobar si hay más páginas de comentarios
            if 'nextPageToken' in results:
                next_page_token = results['nextPageToken']
                results = conn.commentThreads().list(part = 'snippet', videoId = id_video, textFormat = 'plainText', pageToken = next_page_token).execute()
            else:
                break
        comentarios.append(comentarios_aux)
        comentarios_aux = []
    
    return dict(zip(id_videos, comentarios))

"""
def info_videos (conn, id_videos):
    for id_video in id_videos:
        results = conn.videos().list(part = 'snippet', id = id_video).execute()
        categoria = results['items'][0]['snippet']['categoryId']
        print(categoria)
        results = conn.videoCategories().list(part = 'snippet', id = categoria).execute()
        print(results['items'])
    return 
"""

#Función que escribe el la lista de tripletas (titulo, id video, comentarios) en un fichero csv.
def write_to_csv(final):
    with io.open('comentarios.csv', "w", encoding="utf-8", newline="") as comments_file:
        comments_writer = csv.writer(comments_file)
        comments_writer.writerow(['Categorias','Titulo', 'Video ID', 'Comentarios'])
        comments_writer.writerow(final)

def main():
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    youtube = get_authenticated_service()
    num_videos = 1
    tipo = input('¿Introduce usuario o id?[U/I]: ')
    tipo = tipo.lower()
    #Cogemos el id de las subidas del canal.
    id_canal,id_uploads = coger_id_upload(youtube,tipo)
    categorias = categoria_canal(youtube,id_canal)
    #categorias = categorias[0] + " "+ categorias[1]
    print(categorias)
    #Cogemos los títulos de los num_videos últimos y los ids de los num_videos últimos.
    titulos_idVideos = coger_videos(youtube,id_uploads, num_videos)
    titulos_videos = titulos_idVideos.keys()
    id_videos = list(titulos_idVideos.values())
    #Cogemos los comentarios de los vídeos.
    comentarios = coger_comentarios(youtube,id_videos)
    #TODO Solo coge la primera categoria de la lista.
    final = list(zip(categorias,titulos_videos,id_videos,comentarios.values()))
    write_to_csv(final)

if __name__ == '__main__':
    main()


# Especificar fichero de credenciales.
import os
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory
from string import punctuation
import emoji
import re
import io
import csv
import pickle
import langid

CLIENT_SECRETS_FILE = "comentariosAPI/credencialesAPI.json"

# Determinar alcance de acceso
SCOPES = ["https://www.googleapis.com/auth/youtube",
          "https://www.googleapis.com/auth/youtube.force-ssl",
          "https://www.googleapis.com/auth/youtubepartner"]
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

# Diccionario para buscar los canales de youtube por keywords.
diccionario_topicos = {
    'informatica': ['informatica', 'programacion', 'codigo', 'web'],
    'cocina': ['cocina', 'comida', 'cocinar', 'plato de comida'],
    'politica': ['politica', 'economia', 'politico', 'politicos', 'gobierno'],
    'dibujo': ['dibujo', 'dibujar', 'pintura', 'artista']
}

fichero_comentarios = './redNeuronal/data/comentarios.csv'
fichero_canales_restantes = "canales_sin_leer-dibujo.txt"

# Crear Servicio de conexión a la API de youtube con credenciales en caché (con un archivo token.pickle)
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

    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)

# Coger id de canales por keyword y sacamos un diccionario
# {topico1: {id_canal1, id_canal2}
# topico2: {id_canal1, id_canal2}}
def buscar_canales(conn, palabras_topicos, num_canales, keys):
    # Lista auxiliar para coger los ids de un topico.
    id_canales = []
    # Lista de canales agrupadas por topicos.
    canales_topicos = []
    for keywords in palabras_topicos:
        id_canales = []
        results = conn.search().list(q=keywords, part="snippet",
                              type='channel', maxResults=num_canales).execute()
        while results:
            for item in results['items']:
                id_canales.append(item['snippet']['channelId'])
            # Comprobar si hay más canales de ese topico
            if 'nextPageToken' in results:
                next_page_token = results['nextPageToken']
                results = conn.search().list(q=keywords, part="snippet", type='channel',
                                      maxResults=num_canales, pageToken=next_page_token).execute()
            else:
                break
        canales_topicos.append(id_canales)
        print("Primera busqueda encontrado ", len(id_canales))
    return dict(zip(keys, canales_topicos))

# Coger información del canal con el parámetro contentDetails en función de si el identificador es idChannel o el nombre de usuario del canal de youtube.
# Devuelve el id de las subidas de vídeo.
def coger_id_upload(conn, id):
    # Probamos buscando con un petición por nombre de usuario.
    respuesta = conn.channels().list(
        forUsername=id,
        part="contentDetails"
    ).execute()
    num_canales = respuesta['pageInfo']['totalResults']
    # Si no existe el nombre de usuario probamos por identificador de canal
    if (num_canales == 0):
        respuesta = conn.channels().list(
            id=id,
            part="contentDetails"
        ).execute()
        num_canales = respuesta['pageInfo']['totalResults']
        # Si existe devolvemos el id del canal y el id de las subidas y el nombre del canal.
        if (num_canales > 0):
            respuesta_nombre_canal = conn.channels().list(
                id=id,
                part="brandingSettings"
            ).execute()
            nombre_canal = respuesta_nombre_canal['items'][0]['brandingSettings']['channel']['title']
            return id, respuesta['items'][0]['contentDetails']['relatedPlaylists']['uploads'], nombre_canal
    # Si es el nombre de usuario buscamos el id del canal y lo devolvemos junto con el id de las subidas y el nombre del canal.
    respuesta_nombre_canal = conn.channels().list(
        forUsername=id,
        part="brandingSettings"
        ).execute()
    if respuesta_nombre_canal != None:
        nombre_canal = respuesta_nombre_canal['items'][0]['brandingSettings']['channel']['title']
    id = respuesta['items'][0]['id']
    return id, respuesta['items'][0]['contentDetails']['relatedPlaylists']['uploads'], nombre_canal


# Función para coger la categoría de un canal
def categoria_canal(conn, id_canal):
    res = channels_response = conn.channels().list(
        id=id_canal,
        part="topicDetails"
    ).execute()
    return res['items'][0]['topicDetails']['topicCategories']

# Función para coger los últimos num_videos por medio del identificador de subidas del canal de youtube.
def coger_videos(conn, playlist_id, num_videos):
    id_videos = []
    next_page_token = None
    titulos_videos = []
    results = None
    try:
        results = conn.playlistItems().list(playlistId=playlist_id, part='snippet',
                                            maxResults=num_videos, pageToken=next_page_token).execute()
    except HttpError as err:
        if err.resp.status in [404, 500, 503]:
            return None, None
    if results != None:
        for item in results['items']:
            if item != None:
                id_videos.append(item['snippet']['resourceId']['videoId'])
                titulos_videos.append(item['snippet']['title'])
                # Comprobar si hay más páginas de vídeos.
    return titulos_videos, id_videos

# Función para coger los comentarios de los videos asociados a id_videos y devolver cada comentario con la categoria del canal.
def coger_comentarios(conn, id_videos, categoria, nombre_canal, comentarios_canal):
    comentarios = []
    next_page_token = None
    results = None
    DetectorFactory.seed = 0
    #Se establece el 10% de  los comentarios requeridos por vídeo como máximo para no establecer los comentarios a un evento temporal.
    com_por_video = 0.10 * comentarios_canal
    com_video_actual = 0
    for id_video in id_videos:
        com_video_actual = 0
        try:
            results = conn.commentThreads().list(part='snippet', videoId=id_video,
                                                 textFormat='plainText').execute()
        except HttpError as err:
            if err.resp.status in [403, 404, 500, 503]:
                continue
        while results:
            for item in results['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                id_comentario = item['snippet']['topLevelComment']['id']
                ####FILTRADO DE COMENTARIOS########
                # Quitamos los emojis de los comentarios y eliminamos solo los comentarios con emojis además de quitar los comentarios que contengan urls
                comentario_limpio = deEmojify(comment)
                url = es_url(comentario_limpio)
                if comentario_limpio != "" and not url:
                    #Miramos tamaño del comentario sin espacios en blanco, el idioma (castellano, gallego, catalan) y si no es un comentario repetido.
                    tamaño_comentario = tamaño(comentario_limpio)
                    idioma = langid.classify(comentario_limpio)[0]
                    #Comentarios hasta el momento quitandole los espacios en blanco y poniendolos en minusculas.
                    comentarios_aux = [eliminarSignos(com) for com,id,nomC,cat in comentarios]
                    comentario_trim = eliminarSignos(comentario_limpio)
                    if (idioma == 'es' or idioma == 'gl' or idioma == 'ca') and tamaño_comentario >= 50 and not comentario_trim in comentarios_aux:
                        comentarios.append(
                            [comentario_limpio, id_comentario, nombre_canal, categoria])
                        # Poner límite de comentarios por canal
                        if (len(comentarios) == comentarios_canal):
                            return comentarios
                        #Aumentamos los comentarios por video y si tenemos ya los comentarios por video que queremos pasamos al siguiente video.
                        com_video_actual+=1
                        if (com_video_actual == com_por_video):
                            results = []
                            break
            if results == []:
                break
            # Comprobar si hay más páginas de comentarios
            if 'nextPageToken' in results:
                next_page_token = results['nextPageToken']
                results = conn.commentThreads().list(part='snippet', videoId=id_video,
                                                     textFormat='plainText', pageToken=next_page_token).execute()
            else:
                break

    return comentarios

#Funcion que quita signos de ortografia, espacio en blanco y pone en minúsculas.
def eliminarSignos(comentario):
    comentario = ''.join([i for i in comentario if i not in punctuation])
    comentario = comentario.replace(" ","").lower()
    return comentario

# Función que escribe el la lista de tripletas (titulo, id video, comentarios) en un fichero csv.
def write_to_csv(final, ):
    with io.open('./redNeuronal/data/comentarios.csv', "w", encoding="utf-8", newline="") as comments_file:
        comments_writer = csv.writer(comments_file)
        comments_writer.writerow(['Comentarios', 'Id_comentario', 'Nombre_canal', 'Categoria'])
        for row in final:
            comments_writer.writerow(row)

# Quedarse con la última palabra de una url
def ultima_palabra_url(url):
    index = url.rfind('/')
    return url[index+1:]

# Definir si una cadena contiene urls
def es_url(cadena):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    match = re.findall(regex, cadena)
    return len(match) > 0

# Concatenar las diferentes categorias de un canal
def concatenar_categorias(list):
    result = ''
    for element in list:
        result += str(element)+" "
    return result

# Coger urls de los ficheros
def leer_canales(fich_canales):
    f = open(fich_canales, "r")
    lineas = f.readlines()
    f.close()
    return lineas

# Aplanar listas de un solo nivel y listas de listas
def aplanar_lista(lst):
    return sum(lst, [])

# Separar url y categoria
def separar_info_canal(info):
    if(info[-1] == '\n'):
        url_categoria = info[:-1].split(sep=',')
    else:
        url_categoria = info.split(sep=',')
    url = url_categoria[0]
    categoria = url_categoria[1]
    return url, categoria

# Limpiar comentarios de emojis y otros simbolos
def deEmojify(text):
    return emoji.get_emoji_regexp().sub(u'', text)

#Tamaño de un comentario limpio.
def tamaño(comentario):
    com_sin_blancos = comentario.replace(" ","")
    return len(com_sin_blancos)

#Crear lista de un formato string definido.
def crearLista (string):
    string = string.replace("[", "")
    string = string.replace("]","")
    string = string.replace("'","")
    string = string.replace(",","")
    lista = string.split()
    return lista

def getIdsCanalesRestantes(topico):
    f = open(fichero_canales_restantes, "r")
    canales = {topico:crearLista(f.readline())}
    lista_id_canales = canales.get(topico)
    return lista_id_canales
   

def main():
    # Número de videos a mirar por canal.
    num_videos = 50
    # Número de canales objetivos.
    num_canales = [50]#, 15,20,30,40,50]
    # Número de comentarios a extraer por canal.
    comentarios_canal = [100]#[50]#,150]
    # Comentarios totales
    comentarios_total = []
    # Numero de canales busqueda (MAXIMO)
    num_busqueda = 50

    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    youtube = get_authenticated_service()
    global diccionario_topicos
    palabras_topicos = diccionario_topicos.values()
    topicos = diccionario_topicos.keys()
    # BUSCAMOS TODOS LOS CANALES POSIBLES CON EL DICCIONARIO QUE TENEMOS.
    canales = buscar_canales(youtube, palabras_topicos, num_busqueda,topicos )
    #Creamos el fichero donde se guardan los comentarios sino existe
    if not os.path.isfile(fichero_comentarios):
        with io.open(fichero_comentarios, "w", encoding="utf-8", newline="") as comments_file:
            comments_writer = csv.writer(comments_file)
            comments_writer.writerow(['Comentarios', 'Id_comentario', 'Nombre_canal', 'Categoria'])

    #Leer comentarios de un fichero de canales sobrantes de un tópico sólo. SÓLO NECESARIO SI SE PROCESAN LOS TÓPICOS INDIVIDUALMENTE.
    #lista_id_canales = getIdsCanalesRestantes("politica")

    canales_validos = 0
    # Empezar bucles de parrillas 
    for canales_objetivo in num_canales:
        for num_comentarios in comentarios_canal:
            # Miramos en la lista de canales por tópico cada lista de canales.
            for categoria in topicos:
                canales_topicos = canales[categoria]
                print("TOPICO ACTUAL" + categoria)
                # Para cada tópico reseteamos las variables de control.
                canales_validos = 0
                comentarios = []
                # Cogemos cada id de un tópico concreto.
                for id_canal in canales_topicos:

                    #NECESARIO PARA IR ACTUALIZANDO EL FICHERO DE CANALES RESTANTES DE UN TÓPICO. SÓLO NECESARIO SI SE PROCESAN LOS TÓPICOS INDIVIDUALMENTE.
                    #lista_id_canales.remove(id_canal)
                    #with open(fichero_canales_restantes, "w") as output:
                        #output.write(str(lista_id_canales))

                    # Cogemos el id de las subidas del canal.
                    _, id_uploads, nombre_canal = coger_id_upload(
                        youtube, id_canal)
                    # Si no tiene vídeos subidos al canal pasamos
                    print("Nombre del canal: " + nombre_canal)
                    # Cogemos los títulos de los num_videos últimos y los ids de los num_videos últimos.
                    _, id_videos = coger_videos(
                        youtube, id_uploads, num_videos)
                    # Cogemos los comentarios de los vídeos .
                    comentarios = coger_comentarios(youtube, id_videos, categoria, nombre_canal, num_comentarios)
                    # Miramos si el canal tiene suficiente comentarios.
                    cantidad_comentarios = len(comentarios)
                    print("Numero de comentarios: ", cantidad_comentarios)
                    if cantidad_comentarios == num_comentarios:
                        comentarios_total.append(comentarios)
                        canales_validos += 1
                        #Cada vez que encontramos un canal valido lo añadimos al csv
                        with io.open(fichero_comentarios, "a", encoding="utf-8", newline="") as comments_file:
                            comments_writer = csv.writer(comments_file)
                            for row in comentarios:
                                comments_writer.writerow(row)
                        print("Canales validos: ", canales_validos)
                        # Si obtenemos un número de canales válidos igual que el número de canales que buscamos al principio pasamos al siguiente tópico.
                        if canales_validos == canales_objetivo:
                            break


if __name__ == '__main__':
    main()

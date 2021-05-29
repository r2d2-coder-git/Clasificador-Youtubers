import re


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

def main():
    print(es_url("Hola mi nombre es juan https://empresas.blogthinkbig.com/redes-neuronales-artificiales/"))


if __name__ == '__main__':
    main()

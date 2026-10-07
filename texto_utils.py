"""Utilidades de limpieza de texto en español para licitaciones."""
import re
import unicodedata

STOPWORDS_ES = set("""
a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde durante e el ella ellas ellos en entre
era es esa esas ese eso esos esta estas este esto estos fue fueron ha han hasta la las le les lo los mas me mi mis mucho muy
ni no nos o otra otras otro otros para pero poco por porque que quien se sea segun ser si sin sobre su sus tambien tan te
tiene tienen todo todos tu tus un una unas uno unos y ya sus cada mediante traves vez via etc n nº n° nro num
""".split())

# Palabras muy frecuentes en licitaciones que no aportan al análisis
STOPWORDS_DOMINIO = set("""
adquisicion adquisiciones compra compras contratacion servicio servicios suministro licitacion licitaciones publica publico
segun bases convenio año ano anos años año2026 2026 2025 2027 unidad unidades varios varias tipo general generales
municipalidad municipal comuna region hospital departamento direccion adq serv san santa ppta solic para del
""".split())


def quitar_tildes(texto: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')


def limpiar(texto: str, quitar_dominio: bool = True) -> str:
    """Minúsculas, sin tildes, sin números ni símbolos, sin stopwords."""
    t = quitar_tildes(str(texto).lower())
    t = re.sub(r'[^a-zñ\s]', ' ', t)
    sw = STOPWORDS_ES | (STOPWORDS_DOMINIO if quitar_dominio else set())
    sw = {quitar_tildes(w) for w in sw}
    return ' '.join(w for w in t.split() if len(w) > 2 and w not in sw)

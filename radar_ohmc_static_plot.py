"""
Visualizador ESTATICO de radar OHMC sobre fondo personalizado
================================================================

Que hace este script
---------------------
1. Descarga una imagen ya procesada de radar (PNG con canal alpha /
   fondo transparente) desde la API publica de OHMC:

       https://webmet.ohmc.ar/api/v1/frames/{frame_id}/image.png?colormap={colormap}

2. La superpone sobre un mapa base propio, generado con matplotlib +
   cartopy:
       - fondo (tierra) color BLANCO
       - oceano color celeste marino clarito
       - limites provinciales y/o departamentales opcionales, importados
         desde archivos GeoJSON/Shapefile (ver CONFIG mas abajo)
       - colormap/leyenda del producto al costado de la imagen, con la
         paleta oficial de OHMC (colores + valores de referencia)
       - titulo dinamico con radar, fecha, producto y tilt (elevacion)
3. Guarda el resultado como una imagen PNG estatica (no es un mapa web,
   no requiere navegador ni servidor: es un archivo de imagen fijo).

Este script NO usa Leaflet ni ningun visor web: es un programa de
Python que se corre desde la terminal (o desde VSCode) y produce un
archivo de imagen.

Entorno recomendado (Miniconda3 + VSCode)
-------------------------------------------
    conda create -n radar-ohmc python=3.11 -y
    conda activate radar-ohmc
    conda install -c conda-forge cartopy matplotlib numpy requests pillow -y

    # Opcional, SOLO si vas a importar limites departamentales:
    conda install -c conda-forge geopandas -y

Nota: cartopy y geopandas tienen dependencias binarias (GEOS, PROJ,
GDAL); se recomienda instalarlas con conda (conda-forge), no con pip,
para evitar problemas de compilacion en Windows.

La primera vez que corras el script, cartopy va a descargar de internet
los shapefiles de costa/fronteras de Natural Earth (una sola vez, se
cachean localmente).

Uso basico (RECOMENDADO: por variable + radar)
------------------------------------------------
    python radar_ohmc_static_plot.py --radar-code RMA5 --variable DBZH

Esto descarga automaticamente el ULTIMO frame disponible de RMA5 para
"Z@0.5" (factor equivalente de reflectividad), usando SIEMPRE la
variante SIN FILTRAR ("DBZHo"), resuelve su colormap/bbox reales, y
genera el mapa con colorbar y titulo dinamico.

Variables disponibles (--variable, ver VARIABLE_CATALOG mas abajo):
    COLMAX, DBZH, VRAD, RHOHV, KDP, ZDR, WRAD

Por que SIEMPRE se usa la variante sin filtrar
-------------------------------------------------
OHMC ofrece, para varios productos, una variante "filtrada" (sin
sufijo, ej. "DBZH") que aplica un filtro polarimetrico (RHOHV > 0.87 y
DBZH > 30 dBZ) para limpiar ecos no meteorologicos en el visor web
interactivo. Ese filtro descarta directamente todo lo que este por
debajo de esos umbrales, lo que deja "huecos"/una imagen incompleta si
se usa para un mapa estatico. Por eso este script resuelve siempre la
variante SIN FILTRAR (con sufijo 'o', ej. "DBZHo"), incluso si se
especifica un --frame-id manual que resultara ser filtrado: en ese
caso se detecta y se reemplaza automaticamente por el equivalente sin
filtrar (ver resolve_frame()).

Uso avanzado (frame_id / colormap manuales)
-----------------------------------------------
Si necesitás un frame de una fecha/hora especifica en el pasado (no el
ultimo disponible), podés pasar su "frame_id" manualmente:

    python radar_ohmc_static_plot.py --frame-id 903042 --output rma14.png

El "frame_id" se consigue inspeccionando la pestaña "Network" del
navegador (F12) mientras se usa el visor web de OHMC: es el "id"
numerico del frame (ej: 901688, 903042), visible en las respuestas
JSON de endpoints como /api/v1/latest?radar_code=... o
/api/v1/cogs?product_key=.... El "colormap" (paleta) normalmente NO
hace falta especificarlo a mano: se resuelve solo desde la metadata
real del frame ("cog_cmap").

Como se resuelve el "bbox" (extension geografica del radar)
---------------------------------------------------------------
El bbox se resuelve en este orden de prioridad:

  1. Automatico por frame especifico (RECOMENDADO, por defecto): el
     bbox real del frame que se esta graficando, obtenido de
     /api/v1/cogs/{frame_id}. Esto es importante porque distintos
     productos pueden tener una cobertura distinta para el mismo radar
     (ej. VRAD suele cubrir un radio menor que COLMAX).

  2. Por radar_code (extent general): si por algun motivo no se pudo
     obtener el bbox del frame, se usa el extent general del radar
     (CONFIG["radar_code"]), consultando
     GET https://webmet.ohmc.ar/api/v1/radars?active_only=true

  3. Manual: como ultimo fallback, se usa el diccionario
     CONFIG["radar_bbox"] completado a mano.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
import requests
from PIL import Image

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(
        "Falta matplotlib. Instalá con:\n"
        "  conda install -c conda-forge matplotlib"
    )

# Zona horaria de Argentina (fija, sin horario de verano desde 2009).
ART_TZ = timezone(timedelta(hours=-3))

# Nombres de producto "en criollo" que reemplazan la descripcion cruda de
# la API de OHMC para ciertos product_key, segun lo pedido por el usuario.
# Para cualquier otro product_key no listado aca, se usa directamente el
# "product_description" que devuelve /api/v1/products.
PRODUCT_NAME_OVERRIDES = {
    "DBZH": "Factor equivalente de reflectividad",
    "DBZHo": "Factor equivalente de reflectividad",
    "VRAD": "Velocidad radial de dispersas lejanas al radar",
    "VRADo": "Velocidad radial de dispersas lejanas al radar",
}

# Catalogo de "variables" seleccionables por nombre amigable (CONFIG["variable"]
# / --variable), con su product_key SIN FILTRAR (el que se usa siempre, ver
# resolve_product_key) y, si existe, el product_key FILTRADO equivalente
# (usado solo para detectar y corregir automaticamente el caso en que un
# --frame-id manual apunte a la variante filtrada; ver resolve_frame).
#
# IMPORTANTE: RHOHV y VRAD no tienen variante filtrada en la API de OHMC
# (solo existen como "RHOHVo"/"VRADo"); para esos casos "filtered" es None.
VARIABLE_CATALOG = {
    "COLMAX": {"unfiltered": "COLMAXo", "filtered": "COLMAX"},
    "DBZH": {"unfiltered": "DBZHo", "filtered": "DBZH"},      # Z@0.5 / Factor equivalente de reflectividad
    "VRAD": {"unfiltered": "VRADo", "filtered": None},        # Velocidad radial (Doppler)
    "RHOHV": {"unfiltered": "RHOHVo", "filtered": None},      # Coeficiente de correlacion co-polar
    "KDP": {"unfiltered": "KDPo", "filtered": "KDP"},         # Diferencial de fase especifico
    "ZDR": {"unfiltered": "ZDRo", "filtered": "ZDR"},         # Reflectividad diferencial
    "WRAD": {"unfiltered": "WRADo", "filtered": None},        # Ancho espectral
}

# Mapeo inverso (filtrado -> sin filtrar), derivado de VARIABLE_CATALOG.
# Se usa como salvaguarda: si en algun momento se termina resolviendo un
# frame cuyo product_key real es una de estas claves FILTRADAS, el script
# descarta ese frame y busca automaticamente el equivalente sin filtrar.
FILTERED_PRODUCT_KEYS = {
    entry["filtered"]: entry["unfiltered"]
    for entry in VARIABLE_CATALOG.values()
    if entry["filtered"]
}

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    sys.exit(
        "Falta cartopy. Instalá con:\n"
        "  conda install -c conda-forge cartopy\n"
        "(cartopy tiene dependencias binarias: se recomienda conda, no pip)"
    )

# NOTA: este script NO dibuja una grilla de coordenadas (lineas/etiquetas
# de lat/lon). Se probaron varios enfoques (cerrar el anillo del borde
# del mapa a mano, evitar draw_labels=True usando set_xticks/set_yticks),
# pero el motor de gridliner de cartopy sigue disparando, en algunos
# entornos, el error:
#     shapely.errors.GEOSException: IllegalArgumentException:
#     Points of LinearRing do not form a closed linestring
# Como la grilla de coordenadas no es indispensable para el resultado
# final, se opto por quitarla directamente en vez de seguir peleando
# contra este problema de compatibilidad entre versiones de cartopy y
# shapely. El resto del mapa (fondo blanco, oceano celeste, limites
# departamentales opcionales, overlay del radar) no se ve afectado.


# =============================================================================
# CONFIG - editá estos valores segun el radar/producto que quieras graficar
# =============================================================================
CONFIG = {
    # --- Fuente del radar (API OHMC) ---
    "base_url": "https://webmet.ohmc.ar/api/v1",

    # --- Variable a graficar (RECOMENDADO usar esto en vez de frame_id/colormap) ---
    # Nombre amigable de la variable. Opciones validas (ver VARIABLE_CATALOG):
    #     "COLMAX" (reflectividad columna maxima), "DBZH" (Z@0.5, factor
    #     equivalente de reflectividad), "VRAD" (velocidad radial Doppler),
    #     "RHOHV" (coeficiente de correlacion co-polar), "KDP", "ZDR", "WRAD"
    # IMPORTANTE: el script SIEMPRE usa la variante SIN FILTRAR del producto
    # elegido (ej. "COLMAXo", "DBZHo", "VRADo"), nunca la filtrada. El
    # filtrado polarimetrico de OHMC descarta todo por debajo de ~30 dBZ
    # (ver RHOHV>0.87 y DBZH>30 en el visor web), lo que hace que la imagen
    # se vea "rara"/incompleta para varios usos. Ver resolve_product_key().
    "variable": "COLMAX",

    # --- Frame a graficar ---
    # Si "frame_id" es None (default), se usa automaticamente el ULTIMO
    # frame disponible para "radar_code" + la variable elegida (consultando
    # /api/v1/cogs?radar_code=...&product_key=...&limit=1). Si preferis un
    # frame especifico (ej. de una fecha pasada), poné su id manualmente
    # (ver Network tab del navegador, campo "id").
    "frame_id": None,

    # --- Colormap (paleta visual del PNG, ej. "grc_th", "grc_rho", "grc_vrad") ---
    # Si es None (default), se resuelve automaticamente a partir de la
    # metadata real del frame ("cog_cmap", que ya viene ligado al tipo de
    # producto). Fijalo a un valor manual solo si querés forzar una paleta
    # distinta a la que usa OHMC por defecto para ese producto.
    "colormap": None,

    # --- Bbox del radar: resolucion automatica (recomendado) ---
    # Poné el codigo del radar (ej. "RMA2", "RMA14") y el script busca
    # su "extent" automaticamente en /api/v1/radars?active_only=true.
    # Dejalo en None para usar "radar_bbox" (manual) en su lugar.
    "radar_code": "RMA2",

    # --- Bbox del radar (fallback manual, min/max lon/lat en grados) ---
    # Solo se usa si "radar_code" es None o no se encuentra en la lista
    # de radares activos. Completá esto a mano con el "extent"/"bbox"
    # que encuentres en el Network tab del navegador.
    "radar_bbox": {
        "min_lon": -61.14100202773579,
        "min_lat": -36.956669988771615,
        "max_lon": -55.890137972264206,
        "max_lat": -32.64497001122839,
    },

    # Opcional: URL de un endpoint JSON de OHMC (ej. /api/v1/cogs?...
    # o /api/v1/latest?...) donde buscar automaticamente el "bbox" del
    # frame_id, en vez de tipearlo a mano en "radar_bbox". Si es None,
    # se usa directamente "radar_bbox".
    "metadata_json_url": None,

    # --- Extensión del mapa (area visible) ---
    # Si "map_extent" es None, se calcula automaticamente a partir del
    # bbox del radar + un margen (en grados). Si querés un area fija
    # (ej. todo el pais), poné algo como [-75, -53, -56, -21].
    "map_extent": None,
    "extent_margin_deg": 1.5,

    # --- Colores del mapa base ---
    "land_color": "#FFFFFF",       # tierra: blanco
    "ocean_color": "#CFEAF5",      # oceano: celeste marino clarito
    "coastline_color": "#666666",
    "border_color": "#999999",

    # --- Limites provinciales (opcional) ---
    # Path a un archivo .geojson o .shp con límites de provincias
    # (ej. descargado de datos IGN/INDEC). Si es None, no se dibuja
    # esta capa. Se dibuja ANTES de los limites departamentales, con
    # una linea mas gruesa, para que ambas capas se distingan si se
    # usan juntas.
    "provinces_path": None,
    "provinces_color": "#555555",
    "provinces_linewidth": 1.0,

    # --- Limites departamentales (opcional) ---
    # Path a un archivo .geojson o .shp con límites departamentales
    # (ej. descargado de datos IGN/INDEC). Si es None, no se dibuja
    # esta capa.
    "departments_path": None,
    "departments_color": "#808080",
    "departments_linewidth": 0.5,

    # --- Overlay del radar ---
    "radar_opacity": 0.85,  # multiplicador extra sobre la transparencia ya presente en el PNG

    # --- Colorbar del producto (opcional, activada por defecto) ---
    # Dibuja al costado de la imagen la leyenda/escala de color oficial
    # de OHMC para el producto graficado (colores + valores de
    # referencia), consultando /api/v1/products. Se desactiva sola si
    # no hay datos suficientes (ej. producto sin 'references').
    "show_colorbar": True,

    # --- Titulo ---
    # Si "title" tiene un valor fijo (no None), se usa tal cual. Si es
    # None (default), el titulo se genera dinamicamente con el formato:
    #     {RADAR_CODE} {DD-MM-YYYY} {Nombre del producto} ({tilt})
    # a partir de la metadata real del frame (/api/v1/cogs/{frame_id})
    # y del producto (/api/v1/products). Ver build_title().
    "title": None,

    # --- Salida ---
    "output_path": "radar_output.png",
    "figsize": (10, 10),
    "dpi": 150,
    "show_plot": True,  # abre una ventana con el resultado ademas de guardarlo
}


def download_radar_image(cfg):
    """Descarga el PNG (con transparencia) del frame indicado y lo devuelve
    como un array numpy RGBA normalizado (valores 0-1), listo para imshow."""
    url = f"{cfg['base_url']}/frames/{cfg['frame_id']}/image.png?colormap={cfg['colormap']}"
    print(f"[info] Descargando imagen de radar: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGBA")
    arr = np.asarray(img).astype(np.float64) / 255.0
    return arr


def fetch_radar_extent(cfg):
    """Consulta /api/v1/radars?active_only=true y devuelve el bbox
    (convertido desde 'extent') del radar indicado en cfg['radar_code'].
    Devuelve None si no hay radar_code configurado, o si no se lo
    encuentra / falla la consulta."""
    radar_code = cfg.get("radar_code")
    if not radar_code:
        return None
    url = f"{cfg['base_url']}/radars?active_only=true"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        for radar in data.get("radars", []):
            if radar.get("code") == radar_code:
                ext = radar["extent"]
                print(f"[info] bbox de '{radar_code}' ({radar.get('title', '')}) "
                      f"obtenido automaticamente desde {url}")
                return {
                    "min_lon": ext["lon_min"],
                    "min_lat": ext["lat_min"],
                    "max_lon": ext["lon_max"],
                    "max_lat": ext["lat_max"],
                }
        print(f"[warn] radar_code '{radar_code}' no encontrado en {url}; "
              "se usa 'radar_bbox' del CONFIG.")
    except Exception as exc:
        print(f"[warn] No se pudo consultar {url} ({exc}); "
              "se usa 'radar_bbox' del CONFIG.")
    return None


def fetch_bbox_from_metadata(cfg):
    """Busca el bbox especifico del frame_id en cfg['metadata_json_url'],
    si esta configurado. Devuelve None si no aplica o falla."""
    url = cfg.get("metadata_json_url")
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        entries = data if isinstance(data, list) else data.get("results", [data])
        for entry in entries:
            if entry.get("id") == cfg["frame_id"] and entry.get("bbox"):
                print(f"[info] bbox obtenido desde metadata_json_url para frame {cfg['frame_id']}")
                return entry["bbox"]
        print("[warn] No se encontro un 'bbox' para ese frame_id en metadata_json_url.")
    except Exception as exc:
        print(f"[warn] No se pudo consultar metadata_json_url ({exc}).")
    return None


def resolve_bbox(cfg, frame_meta=None):
    """Resuelve el bbox a usar, en orden de prioridad:
    1) bbox real del frame especifico (frame_meta["bbox"], desde
       /api/v1/cogs/{frame_id} - RECOMENDADO: cada producto puede tener
       una cobertura distinta, ej. VRAD suele cubrir un radio menor que
       COLMAX para el mismo radar, asi que el extent general del radar
       puede quedar mal ajustado si se usa para cualquier producto)
    2) metadata_json_url (bbox de un frame puntual, si esta configurado
       manualmente; mantiene compatibilidad con configuraciones previas)
    3) radar_code (extent general del radar, desde /api/v1/radars)
    4) radar_bbox (fallback manual del CONFIG)
    """
    frame_meta = frame_meta or {}
    if frame_meta.get("bbox"):
        print(f"[info] bbox tomado de la metadata real del frame {cfg['frame_id']}.")
        return frame_meta["bbox"]
    bbox = fetch_bbox_from_metadata(cfg)
    if bbox:
        return bbox
    bbox = fetch_radar_extent(cfg)
    if bbox:
        return bbox
    print("[info] Usando 'radar_bbox' manual del CONFIG.")
    return cfg["radar_bbox"]


def resolve_product_key(cfg):
    """Resuelve el product_key SIN FILTRAR a usar, a partir del nombre
    amigable en cfg['variable'] (ver VARIABLE_CATALOG). Siempre devuelve
    la variante sin filtrar (ej. 'COLMAXo', 'DBZHo', 'VRADo', 'RHOHVo'),
    nunca la filtrada, para evitar la imagen "rara"/incompleta que deja
    el filtro polarimetrico de OHMC (corta todo por debajo de ~30 dBZ /
    RHOHV<0.87). Si 'variable' no esta en el catalogo, se avisa y se cae
    de vuelta a 'COLMAXo' como default seguro."""
    variable = (cfg.get("variable") or "COLMAX").upper()
    entry = VARIABLE_CATALOG.get(variable)
    if entry is None:
        print(f"[warn] variable '{variable}' no reconocida; opciones validas: "
              f"{', '.join(VARIABLE_CATALOG)}. Se usa 'COLMAX' (COLMAXo) por defecto.")
        entry = VARIABLE_CATALOG["COLMAX"]
    return entry["unfiltered"]


def resolve_frame_id(cfg, product_key):
    """Resuelve el frame_id a usar:
    - Si cfg['frame_id'] ya tiene un valor, se respeta tal cual (pero ver
      la salvaguarda en resolve_frame() para el caso en que apunte a una
      variante filtrada).
    - Si es None, se busca automaticamente el ULTIMO frame disponible para
      cfg['radar_code'] + product_key, consultando
      /api/v1/cogs?radar_code=...&product_key=...&limit=1."""
    if cfg.get("frame_id"):
        return cfg["frame_id"]

    radar_code = cfg.get("radar_code")
    if not radar_code:
        sys.exit(
            "[error] No se especifico 'frame_id' ni 'radar_code'; no hay forma "
            "de saber que frame descargar. Especifica al menos uno de los dos "
            "(--radar-code RMA5, por ejemplo)."
        )

    url = (f"{cfg['base_url']}/cogs?radar_code={radar_code}"
           f"&product_key={product_key}&limit=1")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        cogs = resp.json().get("cogs", [])
        if not cogs:
            sys.exit(
                f"[error] No se encontro ningun frame reciente para "
                f"radar_code='{radar_code}' product_key='{product_key}' en {url}. "
                "Verifica que el radar este activo y que la variable tenga datos."
            )
        frame_id = cogs[0]["id"]
        print(f"[info] Ultimo frame de '{radar_code}'/'{product_key}': "
              f"id={frame_id} ({cogs[0].get('observation_time')})")
        return frame_id
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit(f"[error] No se pudo consultar {url} ({exc}).")


def resolve_frame(cfg):
    """Punto de entrada unico para resolver que frame_id/colormap se van a
    descargar, aplicando la salvaguarda anti-filtrado pedida por el
    usuario. Devuelve (frame_id, colormap, frame_meta).

    Logica:
    1) Se resuelve el product_key SIN FILTRAR deseado (resolve_product_key).
    2) Se resuelve el frame_id (automatico o manual).
    3) Se consulta la metadata real de ese frame_id
       (/api/v1/cogs/{frame_id}). Si el product_key real del frame
       resultara ser una variante FILTRADA (ej. el usuario paso un
       --frame-id manual que en realidad corresponde a 'DBZH' en vez de
       'DBZHo'), se avisa y se reintenta automaticamente buscando el
       frame sin filtrar equivalente (mismo radar, misma variable) en vez
       de graficar la version filtrada."""
    product_key = resolve_product_key(cfg)
    frame_id = resolve_frame_id(cfg, product_key)
    cfg["frame_id"] = frame_id

    frame_meta = fetch_frame_metadata(cfg)
    real_key = frame_meta.get("product_key")

    is_filtered = (
        real_key and real_key in FILTERED_PRODUCT_KEYS
    )
    if is_filtered:
        print(f"[warn] El frame {frame_id} corresponde a '{real_key}', que es una "
              "variante FILTRADA de OHMC (descarta datos por debajo de ~30 dBZ / "
              "RHOHV<0.87). Se busca automaticamente el frame SIN FILTRAR "
              "equivalente en su lugar.")
        cfg["frame_id"] = None  # forzar re-resolucion automatica
        unfiltered_key = FILTERED_PRODUCT_KEYS.get(real_key, product_key)
        frame_id = resolve_frame_id(cfg, unfiltered_key)
        cfg["frame_id"] = frame_id
        frame_meta = fetch_frame_metadata(cfg)

    colormap = cfg.get("colormap") or frame_meta.get("cog_cmap") or "grc_th"
    if not cfg.get("colormap"):
        print(f"[info] colormap resuelto automaticamente desde la metadata: '{colormap}'")
    cfg["colormap"] = colormap

    return frame_id, colormap, frame_meta


def fetch_frame_metadata(cfg):
    """Consulta /api/v1/cogs/{frame_id} y devuelve el diccionario completo
    de metadata del frame (observation_time, elevation_angle, product_key,
    cog_vmin, cog_vmax, radar_code, etc.). Devuelve {} si falla, en cuyo
    caso el titulo dinamico y la colorbar usan valores por defecto."""
    url = f"{cfg['base_url']}/cogs/{cfg['frame_id']}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        print(f"[info] Metadata del frame {cfg['frame_id']} obtenida desde {url}")
        return data
    except Exception as exc:
        print(f"[warn] No se pudo obtener metadata de {url} ({exc}); "
              "el titulo y la colorbar usaran valores por defecto.")
        return {}


def fetch_product_info(cfg, product_key):
    """Consulta /api/v1/products y devuelve el diccionario del producto
    indicado (product_title, product_description, min_value, max_value,
    unit, references [lista de {value, color, title}]). Devuelve {} si
    no se encuentra o falla la consulta."""
    if not product_key:
        return {}
    url = f"{cfg['base_url']}/products"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        for product in data.get("products", []):
            if product.get("product_key") == product_key:
                print(f"[info] Info de producto '{product_key}' obtenida desde {url}")
                return product
        print(f"[warn] product_key '{product_key}' no encontrado en {url}; "
              "se omite la colorbar y se usa un nombre de producto generico.")
    except Exception as exc:
        print(f"[warn] No se pudo consultar {url} ({exc}).")
    return {}


def build_title(cfg, frame_meta, product_info):
    """Genera el titulo dinamico del mapa con el formato pedido:
        {RADAR_CODE} {DD-MM-YYYY} {Nombre del producto} ({tilt})
    Ejemplos:
        RMA5 11-07-2026 Factor equivalente de reflectividad (0.5)
        RMA5 11-07-2026 Velocidad radial de dispersas lejanas al radar (0.5)
    Si falta algun dato (metadata no disponible), se completa con lo que
    haya en CONFIG/argumentos como mejor esfuerzo, en vez de fallar."""
    radar_code = frame_meta.get("radar_code") or cfg.get("radar_code") or "RADAR"

    obs_time = frame_meta.get("observation_time")
    if obs_time:
        try:
            dt_utc = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
            date_str = dt_utc.astimezone(ART_TZ).strftime("%d-%m-%Y")
        except Exception:
            date_str = obs_time
    else:
        date_str = "fecha desconocida"

    product_key = frame_meta.get("product_key") or cfg.get("colormap", "")
    product_name = PRODUCT_NAME_OVERRIDES.get(product_key)
    if not product_name:
        product_name = (
            product_info.get("product_description")
            or product_info.get("product_title")
            or product_key
            or "Producto desconocido"
        )

    tilt = frame_meta.get("elevation_angle")
    if tilt is not None:
        return f"{radar_code} {date_str} {product_name} ({tilt:g})"
    return f"{radar_code} {date_str} {product_name}"


def build_colorbar(fig, ax, product_info):
    """Agrega al costado de la imagen la colorbar/leyenda oficial de OHMC
    para el producto graficado, usando las 'references' (valor + color)
    que devuelve /api/v1/products. Es una colorbar discreta (por bandas),
    fiel a como OHMC define sus umbrales de color, no un degradado
    continuo generico.

    Solo muestra el VALOR numerico (+ unidad) en cada marca; no se
    incluyen las descripciones textuales de OHMC (ej. "Lluvia muy
    intensa y granizo", "Banda brillante") a pedido del usuario, ya que
    esas descripciones estaban pensadas para el visor web interactivo,
    no para una leyenda de imagen estatica.

    No hace nada si no hay info suficiente (por ejemplo, si
    /api/v1/products no respondio o el producto no tiene 'references'
    definidas)."""
    references = product_info.get("references") or []
    vmax = product_info.get("max_value")

    # Descartar referencias sin valor/color, y valores duplicados.
    seen = set()
    entries = []
    for ref in references:
        value = ref.get("value")
        color = ref.get("color")
        if value is None or not color or value in seen:
            continue
        seen.add(value)
        entries.append((value, color))
    entries.sort(key=lambda e: e[0])

    if len(entries) < 2 or vmax is None:
        print("[info] No hay suficientes referencias de color para dibujar la colorbar; se omite.")
        return

    boundaries = [e[0] for e in entries] + [vmax]
    colors = [e[1] for e in entries]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    cbar = fig.colorbar(
        sm, ax=ax, orientation="vertical",
        fraction=0.045, pad=0.03, boundaries=boundaries, ticks=boundaries[:-1],
    )

    unit = product_info.get("unit") or ""
    labels = []
    for value, _color in entries:
        label = f"{value:g}"
        if unit and unit != "-":
            label += f" {unit}"
        labels.append(label)
    cbar.ax.set_yticklabels(labels, fontsize=7)

    legend_title = product_info.get("product_title") or product_info.get("product_description") or ""
    if legend_title:
        cbar.set_label(legend_title, fontsize=8)


def load_boundary_layer(path, layer_name="capa"):
    """Carga (opcionalmente) una capa de limites geograficos (provincias,
    departamentos, etc.) desde un archivo GeoJSON/Shapefile, usando
    geopandas. Devuelve None si no hay path configurado o si geopandas
    no esta instalado. 'layer_name' es solo para los mensajes en
    consola (ej. "limites departamentales", "limites provinciales").

    Repara automaticamente geometrias invalidas (ej. anillos que no
    cierran, topologia rota) antes de devolver la capa, para evitar
    errores tipo:
        shapely.errors.GEOSException: IllegalArgumentException:
        Points of LinearRing do not form a closed linestring
    Este problema puede aparecer con datasets de cualquier formato
    (GeoJSON, Shapefile, GPKG, etc.) que tengan geometrias mal formadas
    en el archivo de origen; no es especifico de ningun formato en
    particular ni de un tipo de limite (provincial/departamental) en
    especial."""
    if not path:
        return None
    try:
        import geopandas as gpd
    except ImportError:
        print(f"[warn] geopandas no esta instalado; se omiten los {layer_name}.")
        print("       Instalá con: conda install -c conda-forge geopandas")
        return None
    try:
        gdf = gpd.read_file(path)
        print(f"[info] {layer_name.capitalize()} cargados desde '{path}' ({len(gdf)} geometrias)")
    except Exception as exc:
        print(f"[warn] No se pudo leer '{path}': {exc}")
        return None

    # --- Descartar geometrias nulas/vacias antes de validar ---
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]

    # --- Reparar geometrias invalidas (anillos que no cierran, etc.) ---
    invalid_mask = ~gdf.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        print(f"[warn] {n_invalid} geometria(s) invalida(s) detectada(s) en '{path}'; "
              "se intenta reparar automaticamente con make_valid().")
        try:
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].make_valid()
        except AttributeError:
            # Fallback para versiones de geopandas/shapely sin make_valid():
            # el truco buffer(0) repara la mayoria de topologias rotas.
            gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)

        still_invalid = int((~gdf.is_valid).sum())
        if still_invalid > 0:
            print(f"[warn] {still_invalid} geometria(s) no se pudieron reparar; se descartan.")
            gdf = gdf[gdf.is_valid]
        else:
            print("[info] Geometrias reparadas correctamente.")

    if len(gdf) == 0:
        print(f"[warn] '{path}' no tiene geometrias validas para dibujar; se omite la capa.")
        return None

    return gdf


def compute_map_extent(cfg, bbox):
    if cfg.get("map_extent"):
        return cfg["map_extent"]
    m = cfg.get("extent_margin_deg", 1.5)
    return [
        bbox["min_lon"] - m,
        bbox["max_lon"] + m,
        bbox["min_lat"] - m,
        bbox["max_lat"] + m,
    ]


def build_plot(radar_rgba, bbox, cfg, frame_meta=None, product_info=None):
    frame_meta = frame_meta or {}
    product_info = product_info or {}
    extent = compute_map_extent(cfg, bbox)

    fig = plt.figure(figsize=cfg["figsize"], dpi=cfg["dpi"])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # --- Fondo del mapa: tierra blanca, oceano celeste marino clarito ---
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=cfg["ocean_color"], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=cfg["land_color"], zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=cfg["ocean_color"],
                    edgecolor=cfg["coastline_color"], linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=cfg["coastline_color"],
                    linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=cfg["border_color"],
                    linewidth=0.6, zorder=2)

    # --- Limites provinciales (opcional) ---
    provinces = load_boundary_layer(cfg.get("provinces_path"), "limites provinciales")
    if provinces is not None:
        provinces.boundary.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor=cfg["provinces_color"],
            linewidth=cfg["provinces_linewidth"],
            zorder=3,
        )

    # --- Limites departamentales (opcional) ---
    departments = load_boundary_layer(cfg.get("departments_path"), "limites departamentales")
    if departments is not None:
        departments.boundary.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor=cfg["departments_color"],
            linewidth=cfg["departments_linewidth"],
            zorder=4,
        )

    # --- Overlay del radar (con su propia transparencia), encima de todo ---
    radar_rgba = radar_rgba.copy()
    radar_rgba[..., 3] *= cfg["radar_opacity"]
    ax.imshow(
        radar_rgba,
        extent=[bbox["min_lon"], bbox["max_lon"], bbox["min_lat"], bbox["max_lat"]],
        transform=ccrs.PlateCarree(),
        origin="upper",
        interpolation="nearest",
        zorder=10,
    )

    # --- Colorbar del producto al costado (opcional) ---
    if cfg.get("show_colorbar", True):
        build_colorbar(fig, ax, product_info)

    # --- Titulo (fijo o dinamico, ver CONFIG["title"]) ---
    title = cfg.get("title") or build_title(cfg, frame_meta, product_info)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    return fig


def parse_args(cfg):
    parser = argparse.ArgumentParser(description="Visualizador estatico de radar OHMC.")
    parser.add_argument("--variable", type=str, default=cfg["variable"],
                         choices=list(VARIABLE_CATALOG),
                         help="Variable a graficar. El script SIEMPRE usa la variante "
                              "SIN FILTRAR (ej. --variable DBZH usa 'DBZHo'), para evitar "
                              "la imagen incompleta que deja el filtro polarimetrico de OHMC.")
    parser.add_argument("--frame-id", type=int, default=cfg["frame_id"],
                         help="ID de frame especifico. Si se omite, se usa automaticamente "
                              "el ultimo frame disponible para --radar-code + --variable.")
    parser.add_argument("--colormap", type=str, default=cfg["colormap"],
                         help="Colormap/paleta a forzar. Si se omite, se resuelve "
                              "automaticamente desde la metadata del frame.")
    parser.add_argument("--radar-code", type=str, default=cfg["radar_code"],
                         help="Codigo de radar (ej. RMA2, RMA14) para resolver el bbox "
                              "automaticamente desde /api/v1/radars?active_only=true.")
    parser.add_argument("--output", type=str, default=cfg["output_path"])
    parser.add_argument("--provinces", type=str, default=cfg["provinces_path"],
                         help="Path a un GeoJSON/Shapefile de limites provinciales.")
    parser.add_argument("--departments", type=str, default=cfg["departments_path"],
                         help="Path a un GeoJSON/Shapefile de limites departamentales.")
    parser.add_argument("--no-colorbar", action="store_true",
                         help="No dibujar la colorbar/leyenda del producto al costado.")
    parser.add_argument("--title", type=str, default=cfg["title"],
                         help="Titulo fijo del mapa. Si se omite, se genera "
                              "automaticamente a partir de la metadata del frame.")
    parser.add_argument("--no-show", action="store_true",
                         help="No abrir ventana de matplotlib; solo guardar el archivo.")
    args = parser.parse_args()

    cfg["variable"] = args.variable
    cfg["frame_id"] = args.frame_id
    cfg["colormap"] = args.colormap
    cfg["radar_code"] = args.radar_code
    cfg["output_path"] = args.output
    cfg["provinces_path"] = args.provinces
    cfg["departments_path"] = args.departments
    cfg["title"] = args.title
    if args.no_colorbar:
        cfg["show_colorbar"] = False
    if args.no_show:
        cfg["show_plot"] = False
    return cfg


def save_figure(fig, cfg):
    """Guarda la figura, intentando primero bbox_inches='tight' (recorte
    prolijo de margenes). Si esto dispara el bug conocido de Cartopy +
    Matplotlib + Shapely 2.x:

        shapely.errors.GEOSException: IllegalArgumentException:
        Points of LinearRing do not form a closed linestring

    (originado en cartopy/mpl/gridliner.py _draw_gridliner, al
    recalcular el "boundary" del mapa durante el redibujado extra que
    hace bbox_inches='tight'), se reintenta guardar SIN ese parametro,
    que evita ese redibujado y produce un resultado casi identico."""
    try:
        fig.savefig(cfg["output_path"], dpi=cfg["dpi"], bbox_inches="tight")
    except Exception as exc:
        print(f"[warn] Fallo el guardado con bbox_inches='tight' ({exc.__class__.__name__}: {exc}).")
        print("       Reintentando sin bbox_inches (bug conocido de Cartopy + Shapely 2.x).")
        fig.savefig(cfg["output_path"], dpi=cfg["dpi"])


def main():
    cfg = parse_args(CONFIG)

    # Resuelve frame_id/colormap (automatico por variable + radar_code, o
    # manual si se especifico --frame-id), aplicando la salvaguarda que
    # evita graficar variantes FILTRADAS de OHMC (ver resolve_frame()).
    frame_id, colormap, frame_meta = resolve_frame(cfg)

    radar_rgba = download_radar_image(cfg)
    bbox = resolve_bbox(cfg, frame_meta=frame_meta)

    product_info = fetch_product_info(cfg, frame_meta.get("product_key"))

    fig = build_plot(radar_rgba, bbox, cfg, frame_meta=frame_meta, product_info=product_info)

    save_figure(fig, cfg)
    print(f"[ok] Imagen guardada en: {cfg['output_path']}")

    if cfg.get("show_plot", True):
        try:
            plt.show()
        except Exception as exc:
            # Mismo bug de Cartopy/Shapely 2.x que puede afectar a savefig
            # (ver save_figure) puede repetirse al mostrar la ventana
            # interactiva, ya que tambien fuerza un redibujado. La imagen
            # ya se guardo en disco antes de llegar aca, asi que esto solo
            # afecta a la vista previa en pantalla, no al archivo de salida.
            print(f"[warn] No se pudo mostrar la ventana interactiva ({exc.__class__.__name__}: {exc}).")
            print(f"       La imagen ya fue guardada correctamente en '{cfg['output_path']}'.")


if __name__ == "__main__":
    main()

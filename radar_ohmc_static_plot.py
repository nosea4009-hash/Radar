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

Busqueda historica por fecha/hora
-------------------------------------
Para graficar un momento especifico del pasado (no el ultimo frame
disponible), usa --datetime en vez de buscar el frame_id a mano:

    python radar_ohmc_static_plot.py --radar-code RMA17 --variable DBZH \
        --datetime "2026-06-25 15:00"

Esto busca automaticamente el frame MAS CERCANO a esa fecha/hora
(interpretada en HORA ARGENTINA salvo que se indique otra zona
explicitamente, ej. "2026-06-25T18:00:00Z"), dentro de una ventana de
tolerancia de +/-30 min (configurable con --datetime-window).

IMPORTANTE: OHMC NO retiene los datos crudos indefinidamente. Se
confirmo probando la API real que la ventana de retencion es de
aproximadamente ~18-20 dias hacia atras desde la fecha actual (mas
alla de eso, la busqueda no encuentra nada porque los archivos ya
fueron purgados del sistema, no por un error de este script). Si
pedis una fecha mas vieja que eso, vas a ver un error explicando esto.

Uso avanzado (frame_id / colormap manuales)
-----------------------------------------------
Si ya sabes el "frame_id" exacto que necesitas, podés pasarlo
directamente (tiene prioridad sobre --datetime):

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
import os
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import matplotlib.colors as mcolors
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(
        "Falta matplotlib. Instalá con:\n"
        "  conda install -c conda-forge matplotlib"
    )

# Directorio donde vive este script (para resolver "fuenteSMN.ttf" y otros
# archivos relativos sin depender del directorio de trabajo actual).
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

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

# Mapeo sin filtrar -> filtrado (el sentido opuesto al de arriba). Se usa
# UNICAMENTE para resolver la leyenda/colorbar (ver ensure_colorbar_references()):
# algunos productos "sin filtrar" (ej. "COLMAXo", "DBZHo") vienen con
# 'references' VACIO en /api/v1/products (la paleta de colores oficial no
# esta definida ahi), mientras que su variante FILTRADA equivalente
# ("COLMAX", "DBZH") si trae la lista completa de colores/valores. Como
# ambas variantes usan la MISMA escala de color (mismo cog_cmap, mismo
# min_value/max_value), es seguro "prestarle" esas referencias a la
# variante sin filtrar solo para dibujar la colorbar, sin afectar en nada
# la imagen del radar en si (que sigue siendo siempre la sin filtrar).
UNFILTERED_TO_FILTERED_KEYS = {
    entry["unfiltered"]: entry["filtered"]
    for entry in VARIABLE_CATALOG.values()
    if entry["filtered"]
}

# =============================================================================
# Cobertura/radio del radar (240 km vs 450 km aprox.)
# =============================================================================
# OHMC genera, para el mismo radar y variable, distintos "volumenes"
# (vol_nr) segun la estrategia de barrido, cada uno con un radio de
# cobertura distinto. Se confirmo contra la API real (/api/v1/cogs?...)
# que:
#   vol_nr="01" -> radar_coverage_m ~236 km (etiquetado aqui como "240")
#   vol_nr="04" -> radar_coverage_m ~446 km (etiquetado aqui como "450")
# Los numeros "240"/"450" son etiquetas redondeadas para el usuario; el
# radio real exacto depende del radar y se toma siempre de la metadata
# real del frame (frame_meta["radar_coverage_m"]), no de esta tabla.
COVERAGE_VOL_NR = {
    240: "01",
    450: "04",
}

# =============================================================================
# Range rings (anillos de rango del radar, CONFIG["range_rings_km"])
# =============================================================================
# Dibuja circulos geodesicos (geograficamente precisos, calculados con
# cartopy.geodesic.Geodesic) centrados en la ubicacion real de la antena
# del radar (ver resolve_radar_center()), a las distancias indicadas en
# CONFIG["range_rings_km"]. Sirven de referencia visual rapida para saber
# hasta donde llega la cobertura del radar sin tener que estimarlo a ojo.
#
# Por defecto se dibujan 2 anillos: uno fijo de 120 km (radio tipico de
# "buena calidad" de deteccion de muchos productos de radar), y otro que
# coincide con la cobertura elegida en CONFIG["coverage_km"] (240 o
# 450 km) - si "coverage_km" no esta configurado, se usa 240 km como
# default razonable para el segundo anillo. Se puede pasar una lista
# propia de distancias (en km) para dibujar cualquier otro conjunto de
# anillos, o None/[] para no dibujar ninguno.
DEFAULT_RANGE_RING_KM = 120

# =============================================================================
# Tema del mapa: claro ("light") vs oscuro ("dark")
# =============================================================================
# Cada tema es un conjunto de colores coordinado para tierra, oceano, y
# TODOS los limites (nacional/pais, provincial, departamental), aplicado
# de una sola vez con CONFIG["theme"] / --theme, en vez de tener que
# tocar cada color por separado.
#
#   "light" (default, look actual del script): tierra blanca, oceano
#   celeste marino clarito, limites en tonos de gris oscuro/medio.
#
#   "dark": tierra y oceano NEGROS, con limites nacionales, provinciales
#   y departamentales en GRIS CLARO (para que se distingan sobre el
#   fondo negro). Pensado para resaltar el overlay de color del radar.
#
# "custom" (o cualquier valor no listado en THEME_PRESETS, ej. None) NO
# aplica ningun preset: se respetan los colores que ya esten seteados a
# mano en CONFIG (land_color, ocean_color, coastline_color, etc.), por
# si se quiere una combinacion propia distinta a "light"/"dark".
THEME_PRESETS = {
    "light": {
        "land_color": "#FFFFFF",
        "ocean_color": "#CFEAF5",
        "coastline_color": "#666666",
        "border_color": "#999999",
        "provinces_color": "#555555",
        "departments_color": "#808080",
    },
    "dark": {
        "land_color": "#000000",
        "ocean_color": "#000000",
        "coastline_color": "#D3D3D3",
        "border_color": "#D3D3D3",
        "provinces_color": "#D3D3D3",
        "departments_color": "#D3D3D3",
    },
}


def apply_theme(cfg):
    """Si cfg['theme'] coincide con una clave de THEME_PRESETS ("light" o
    "dark"), sobreescribe en 'cfg' los colores de tierra/oceano/limites
    con los del preset elegido. Si cfg['theme'] es None o no coincide con
    ningun preset (ej. "custom"), no hace nada: se respetan los colores
    que ya esten definidos en CONFIG/'cfg' tal cual."""
    theme = cfg.get("theme")
    preset = THEME_PRESETS.get(theme)
    if preset is None:
        return
    cfg.update(preset)
    print(f"[info] Tema '{theme}' aplicado (tierra/oceano/limites).")


# =============================================================================
# Fuente de texto principal del plot (CONFIG["font_path"] / --font)
# =============================================================================
# Todo el texto que dibuja este script (titulo, colorbar, y las etiquetas
# de localidades del tema "dark", ver LOCALITIES) usa una sola fuente
# configurable, en vez de la fuente default de matplotlib (DejaVu Sans).
#
# Por defecto apunta a "fuenteSMN.ttf", incluida en este mismo repositorio
# junto al script. Esa fuente es en realidad una copia renombrada de
# "DejaVu Sans Mono" (variante monoespaciada de la tipografia default de
# matplotlib, de uso libre/redistribuible), pedida asi por el usuario;
# se puede reemplazar por cualquier otro archivo .ttf/.otf propio sin
# cambiar el codigo, solo actualizando CONFIG["font_path"] / --font.
DEFAULT_FONT_FILENAME = "fuenteSMN.ttf"


def resolve_font_path(cfg):
    """Resuelve el path al archivo de fuente a usar, a partir de
    cfg['font_path']:
    - None (default) -> busca DEFAULT_FONT_FILENAME ("fuenteSMN.ttf") en
      el mismo directorio que este script (SCRIPT_DIR).
    - Un path (str) -> se usa tal cual, si el archivo existe.
    Devuelve un Path si el archivo existe, o None si no se encontro
    (en cuyo caso se usa la fuente default de matplotlib, sin fallar)."""
    font_path = cfg.get("font_path")
    candidate = Path(font_path) if font_path else (SCRIPT_DIR / DEFAULT_FONT_FILENAME)
    if candidate.is_file():
        return candidate
    if font_path:
        print(f"[warn] No se encontro el archivo de fuente '{candidate}'; "
              "se usa la fuente default de matplotlib.")
    return None


def apply_font(cfg):
    """Registra el archivo de fuente resuelto (ver resolve_font_path) en
    matplotlib.font_manager, y lo fija como fuente ('font.family') para
    TODO el texto dibujado por el script (titulo, colorbar, etiquetas de
    localidades). Si no se encuentra ningun archivo de fuente valido, no
    hace nada y se sigue usando la fuente default de matplotlib."""
    font_file = resolve_font_path(cfg)
    if font_file is None:
        return
    try:
        fm.fontManager.addfont(str(font_file))
        font_name = fm.FontProperties(fname=str(font_file)).get_name()
        plt.rcParams["font.family"] = font_name
        print(f"[info] Fuente '{font_file.name}' aplicada como fuente principal del plot "
              f"(font.family='{font_name}').")
    except Exception as exc:
        print(f"[warn] No se pudo cargar la fuente '{font_file}' ({exc}); "
              "se usa la fuente default de matplotlib.")


# =============================================================================
# Etiquetas de localidades principales (solo tema "dark")
# =============================================================================
# A pedido del usuario, en el tema "dark" (fondo negro) se agregan
# etiquetas de texto BLANCO con el nombre de algunas localidades
# principales, para dar referencia geografica sin necesitar limites
# provinciales/departamentales cargados. Coordenadas aproximadas del
# centro de cada localidad (WGS84, grados decimales).
LOCALITIES = {
    "Buenos Aires": (-34.6037, -58.3816),
    "Pergamino": (-33.8961, -60.5764),
    "Parana": (-31.7333, -60.5167),
    "Bahia Blanca": (-38.7196, -62.2724),
    "Olavarria": (-36.8927, -60.3225),
    "San Francisco (Cordoba)": (-31.4241, -62.0836),
}


def add_locality_labels(ax, extent, cfg):
    """Dibuja, solo si cfg['theme'] es 'dark', un punto + etiqueta de
    texto en color BLANCO para cada localidad de LOCALITIES cuya
    coordenada caiga dentro del 'extent' visible del mapa (las que
    quedan fuera del area graficada se omiten, para no ensuciar el plot
    con etiquetas invisibles/cortadas en el borde)."""
    if cfg.get("theme") != "dark":
        return
    lon_min, lon_max, lat_min, lat_max = extent
    for name, (lat, lon) in LOCALITIES.items():
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        ax.plot(lon, lat, marker="o", markersize=3, color="white",
                transform=ccrs.PlateCarree(), zorder=20)
        ax.text(lon, lat, f"  {name}", color="white", fontsize=7,
                ha="left", va="center", transform=ccrs.PlateCarree(), zorder=20)


def resolve_range_rings_km(cfg):
    """Resuelve la lista de distancias (en km) a las que dibujar range
    rings, a partir de cfg['range_rings_km']:
    - None (default): se resuelve automaticamente a [120, coverage_km],
      usando cfg['coverage_km'] si esta configurado (240 o 450), o 240
      como default razonable si no lo esta. DEFAULT_RANGE_RING_KM (120)
      siempre se incluye como referencia de "buena cobertura".
    - Lista/tupla de numeros: se usa tal cual (permite personalizar
      completamente los anillos, ej. [100, 200, 300]).
    - Lista vacia o cualquier valor "falsy": no se dibuja ningun anillo.
    Devuelve una lista de floats (sin duplicados, ordenada)."""
    rings = cfg.get("range_rings_km")
    if rings is None:
        coverage_km = cfg.get("coverage_km") or 240
        rings = [DEFAULT_RANGE_RING_KM, coverage_km]
    if not rings:
        return []
    return sorted({float(r) for r in rings if r})


def add_range_rings(ax, cfg, frame_meta=None, bbox=None):
    """Dibuja circulos geodesicos (calculados con cartopy.geodesic, para
    que sean geograficamente precisos y no se vean "achatados" lejos del
    ecuador) centrados en la ubicacion real de la antena del radar (ver
    resolve_radar_center()), a las distancias resueltas por
    resolve_range_rings_km(). Cada anillo se etiqueta con su distancia
    (ej. "120 km") junto al borde. No dibuja nada si no se pudo resolver
    el centro del radar, o si la lista de distancias esta vacia."""
    ring_km_list = resolve_range_rings_km(cfg)
    if not ring_km_list:
        return

    center = resolve_radar_center(cfg, frame_meta=frame_meta, bbox=bbox)
    if center is None:
        print("[warn] No se pudo resolver el centro del radar; se omiten los range rings.")
        return
    center_lat, center_lon = center

    from cartopy.geodesic import Geodesic
    geod = Geodesic()
    ring_color = cfg.get("range_rings_color") or (
        "#FFFFFF" if cfg.get("theme") == "dark" else "#444444"
    )

    for ring_km in ring_km_list:
        circle_pts = geod.circle(lon=center_lon, lat=center_lat,
                                  radius=ring_km * 1000.0, n_samples=180)
        ax.plot(circle_pts[:, 0], circle_pts[:, 1], color=ring_color,
                linewidth=0.7, linestyle="--", alpha=0.8,
                transform=ccrs.PlateCarree(), zorder=15)
        # Etiqueta de distancia, ubicada en el punto mas al este del anillo
        # (angulo 90=Este en la convencion de Geodesic.circle) para que no
        # se solapen entre si los distintos anillos.
        label_pt = geod.direct(points=[[center_lon, center_lat]],
                                azimuths=[90], distances=[ring_km * 1000.0])[0]
        ax.text(label_pt[0], label_pt[1], f" {ring_km:g} km", color=ring_color,
                fontsize=6, ha="left", va="center",
                transform=ccrs.PlateCarree(), zorder=15)

# =============================================================================
# Busqueda historica por fecha/hora (CONFIG["target_datetime"] / --datetime)
# =============================================================================
# OHMC permite consultar /api/v1/cogs con un rango de tiempo (start_time/
# end_time, en UTC), lo que permite buscar frames de fechas pasadas, no
# solo el "ultimo" disponible. Se confirmo probando la API real que la
# retencion de datos crudos NO es indefinida: hay un corte encontrado
# entre el 22/06/2026 (sin datos) y el 25/06/2026 (con datos), con "hoy"
# en 11-12/07/2026. Esto da una ventana de retencion de
# aproximadamente 17 a 20 dias hacia atras desde la fecha actual. Pedir
# una fecha mas vieja que eso (ej. 5 meses atras) devuelve una lista
# vacia porque esos archivos ya fueron purgados del sistema, no porque
# el mecanismo de busqueda este mal.
#
# RETENTION_DAYS_ESTIMATE se usa solo para dar un mensaje de error mas
# util (aclarando que probablemente el dato ya no exista) cuando la
# busqueda no encuentra nada y la fecha pedida es mas vieja que esta
# estimacion; NO se usa para bloquear la busqueda en si (siempre se
# intenta igual, por si la retencion real es mayor a la estimada aqui).
RETENTION_DAYS_ESTIMATE = 18

# =============================================================================
# Soporte de paletas de colores PROPIAS para la colorbar (ademas de la
# paleta oficial de OHMC, que sigue siendo la que se usa por defecto).
# =============================================================================
#
# IMPORTANTE - que SI y que NO cambia esta paleta:
#   La imagen del radar en si (el PNG que descarga download_radar_image())
#   viene YA COLOREADA por el servidor de OHMC segun su propio 'colormap'
#   (grc_th, grc_rho, grc_vrad, etc.) - eso no se puede recolorear desde
#   afuera, porque OHMC no expone los valores crudos de dBZ/m/s, solo el
#   PNG final. Lo que SI controla este script es la COLORBAR/leyenda que
#   se dibuja al costado (build_colorbar); "cfg['palette']" reemplaza los
#   colores de esa leyenda por una paleta propia (ej. una de MetPy), en
#   vez de los colores oficiales de OHMC. Sirve para tener una leyenda
#   con una estetica distinta, pero NO cambia los colores de la imagen
#   del radar en si.
#
# Fuentes de paleta soportadas en CONFIG["palette"] / --palette:
#   - None (default): usa la paleta oficial de OHMC (colores reales de
#     /api/v1/products -> "references"), tal como antes.
#   - Nombre de una paleta de MetPy (ej. "NWSStormClearReflectivity",
#     "NWS8bitVel"): requiere `pip install metpy`. Ver lista completa en
#     https://unidata.github.io/MetPy/latest/api/generated/metpy.plots.ctables.html
#   - Path a un archivo ".pal"/".tbl" (formato GEMPAK/MetPy: un color por
#     linea, como tupla RGB "(r, g, b)" en 0-1, o nombre HTML, o hex).
try:
    import matplotlib.colors as _mcolors_for_palette
except ImportError:
    _mcolors_for_palette = None


def load_metpy_palette(name):
    """Intenta cargar una paleta por nombre desde metpy.plots.ctables.
    Devuelve una lista de colores hex, o None si metpy no esta instalado
    o el nombre no existe en su registry."""
    try:
        from metpy.plots import ctables
    except ImportError:
        print("[warn] No se pudo importar 'metpy' para cargar la paleta "
              f"'{name}'. Instalá con: conda install -c conda-forge metpy "
              "(o: pip install metpy)")
        return None
    if name not in ctables.registry:
        print(f"[warn] '{name}' no es una paleta reconocida de MetPy.")
        return None
    raw_colors = ctables.registry[name]
    return [_mcolors_for_palette.to_hex(c) for c in raw_colors]


def load_pal_file_palette(path):
    """Carga una paleta desde un archivo .pal/.tbl (formato GEMPAK/MetPy:
    un color por linea, como tupla RGB "(r, g, b)" en 0-1, nombre HTML, o
    hex). Reutiliza el parser de metpy si esta disponible; si no, hace un
    parseo basico propio (solo tuplas RGB y colores hex/nombre simples)."""
    try:
        from metpy.plots.ctables import read_colortable
        with open(path) as fobj:
            raw_colors = read_colortable(fobj)
        return [_mcolors_for_palette.to_hex(c) for c in raw_colors]
    except ImportError:
        pass
    except Exception as exc:
        print(f"[warn] No se pudo leer '{path}' como tabla de colores: {exc}")
        return None

    # Fallback sin metpy: parseo linea por linea, ignorando comentarios ('#').
    if _mcolors_for_palette is None:
        print("[warn] Falta matplotlib para interpretar colores del archivo .pal.")
        return None
    colors = []
    try:
        with open(path) as fobj:
            for line in fobj:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    colors.append(_mcolors_for_palette.to_hex(eval(line, {"__builtins__": {}})))
                except Exception:
                    continue
    except OSError as exc:
        print(f"[warn] No se pudo abrir '{path}': {exc}")
        return None
    return colors or None


def resolve_palette_colors(cfg):
    """Resuelve la lista de colores hex a usar en la colorbar, a partir de
    cfg['palette']:
    - None -> devuelve None (se usa la paleta oficial de OHMC, sin cambios)
    - termina en .pal/.tbl, o es un path existente -> load_pal_file_palette
    - en cualquier otro caso -> se interpreta como nombre de paleta MetPy
    Devuelve None si no se pudo resolver (se cae de vuelta a OHMC)."""
    palette = cfg.get("palette")
    if not palette:
        return None
    p = Path(palette)
    if p.suffix.lower() in (".pal", ".tbl") or p.is_file():
        colors = load_pal_file_palette(palette)
    else:
        colors = load_metpy_palette(palette)
    if colors:
        print(f"[info] Paleta '{palette}' cargada ({len(colors)} colores); "
              "se usa para la colorbar en vez de la paleta oficial de OHMC.")
    else:
        print(f"[warn] No se pudo resolver la paleta '{palette}'; "
              "se usa la paleta oficial de OHMC para la colorbar.")
    return colors

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

    # --- Cobertura del radar: 240 km vs 450 km (opcional) ---
    # Varios radares (ej. RMA5) tienen dos "volumenes" (vol_nr) con radio
    # de cobertura distinto para el mismo producto: uno de ~240 km y otro
    # de ~450 km (ver COVERAGE_VOL_NR). Si "coverage_km" es None (default),
    # se usa el comportamiento por defecto de OHMC (el primer resultado
    # que devuelva /api/v1/cogs, sin forzar ningun vol_nr en particular).
    # Poné 240 o 450 para forzar una cobertura especifica. No tiene efecto
    # si se especifica un "frame_id" manual (ese frame ya tiene su propio
    # radio fijo, visible en frame_meta["radar_coverage_m"]).
    "coverage_km": None,

    # --- Busqueda historica por fecha/hora (opcional) ---
    # Si "target_datetime" tiene un valor, se busca automaticamente el
    # frame MAS CERCANO a esa fecha/hora (en vez del ultimo disponible).
    # Acepta:
    #   - Un string ISO 8601 SIN zona horaria (ej. "2026-02-18 20:30" o
    #     "2026-02-18T20:30:00"): se interpreta en HORA ARGENTINA (ART,
    #     UTC-3) y se convierte a UTC internamente para consultar la API.
    #   - Un string ISO 8601 CON zona horaria explicita (ej.
    #     "2026-02-18T23:30:00Z" o "...-03:00"): se respeta la zona
    #     indicada, sin asumir hora argentina.
    #   - Un objeto datetime de Python (naive = hora argentina, aware =
    #     se respeta su tzinfo).
    # "datetime_window_minutes" define cuanto se abre la busqueda hacia
    # adelante/atras del momento pedido (los frames de OHMC no caen
    # siempre en un timestamp exacto). Si no se encuentra nada dentro de
    # esa ventana, se avisa (y si la fecha es muy vieja, se aclara que
    # probablemente el dato ya fue purgado del sistema - ver
    # RETENTION_DAYS_ESTIMATE). No tiene efecto si se especifica un
    # "frame_id" manual (que tiene maxima prioridad).
    "target_datetime": None,
    "datetime_window_minutes": 30,

    # --- Frame a graficar ---
    # Si "frame_id" es None (default) y "target_datetime" tampoco esta
    # configurado, se usa automaticamente el ULTIMO frame disponible para
    # "radar_code" + la variable elegida (consultando
    # /api/v1/cogs?radar_code=...&product_key=...&limit=1). Si preferis un
    # frame especifico (ej. de una fecha pasada), poné su id manualmente
    # (ver Network tab del navegador, campo "id"), o usa "target_datetime"
    # para buscarlo por fecha/hora en vez de por id.
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
    # bbox del radar + un margen. Si querés un area fija (ej. todo el
    # pais), poné algo como [-75, -53, -56, -21].
    "map_extent": None,

    # El margen se calcula como PORCENTAJE del ancho/alto del bbox del
    # frame (en vez de un valor fijo en grados), para que el zoom quede
    # proporcional sin importar si el radar cubre 240 km o 450 km: con un
    # margen fijo en grados, la cobertura de 240 km (bbox mas chico)
    # terminaba viendose con MAS margen relativo (mas "alejada") que la
    # de 450 km, al reves de lo esperado. "extent_margin_pct" (0.08 =
    # 8% del ancho/alto del bbox) resuelve esto: cada cobertura queda
    # encuadrada con el mismo nivel de zoom relativo. "extent_margin_deg"
    # se mantiene como fallback fijo por si se prefiere el comportamiento
    # anterior (poné extent_margin_pct=None para usarlo).
    "extent_margin_pct": 0.08,
    "extent_margin_deg": 1.5,

    # --- Tema del mapa: "light" o "dark" (ver THEME_PRESETS) ---
    # Aplica de una sola vez un conjunto coordinado de colores para
    # tierra, oceano y TODOS los limites (nacional/provincial/
    # departamental). "light" = tierra blanca / oceano celeste (look
    # anterior del script). "dark" = tierra y oceano negros, limites en
    # gris claro. Poné None (o cualquier valor no listado en
    # THEME_PRESETS, ej. "custom") para NO aplicar ningun preset y usar
    # los colores individuales de mas abajo (land_color, ocean_color,
    # etc.) tal cual esten configurados. Ver apply_theme().
    "theme": "light",

    # --- Fuente de texto principal del plot (opcional) ---
    # Si "font_path" es None (default), se usa "fuenteSMN.ttf" (incluida
    # junto a este script; es una copia de DejaVu Sans Mono renombrada a
    # pedido del usuario). Poné un path a tu propio archivo .ttf/.otf
    # para usar otra fuente. Si el archivo no se encuentra, se cae de
    # vuelta a la fuente default de matplotlib sin fallar. Ver
    # apply_font().
    "font_path": None,

    # --- Range rings: anillos de rango del radar (opcional) ---
    # Dibuja circulos geodesicos de referencia centrados en la antena
    # real del radar, para ver rapido hasta donde llega la cobertura.
    # Si "range_rings_km" es None (default), se dibujan automaticamente
    # 2 anillos: uno fijo de 120 km (ver DEFAULT_RANGE_RING_KM) y otro
    # que coincide con "coverage_km" (240 o 450; si "coverage_km" no
    # esta configurado, se usa 240 km). Poné una lista propia (ej.
    # [100, 200, 300]) para elegir tus propias distancias, o [] para no
    # dibujar ningun anillo. "range_rings_color" es None por defecto:
    # se usa blanco en tema "dark" y gris oscuro en tema "light"/otros
    # (ver add_range_rings()); poné un color fijo para forzarlo.
    "range_rings_km": None,
    "range_rings_color": None,

    # --- Colores del mapa base ---
    # NOTA: si "theme" arriba coincide con un preset ("light"/"dark"),
    # estos valores se SOBREESCRIBEN automaticamente por los del preset
    # (ver apply_theme()); solo tienen efecto real si "theme" es None o
    # un valor no reconocido (ej. "custom").
    "land_color": "#FFFFFF",       # tierra: blanco
    "ocean_color": "#CFEAF5",      # oceano: celeste marino clarito
    "coastline_color": "#666666",
    "border_color": "#999999",

    # --- Limites provinciales (opcional) ---
    # Path a un archivo .geojson o .shp con límites de provincias
    # (ej. descargado de datos IGN/INDEC). Si es None, no se dibuja
    # esta capa. Se dibuja ANTES de los limites departamentales, con
    # una linea mas gruesa, para que ambas capas se distingan si se
    # usan juntas. El color (provinces_color) tambien se sobreescribe
    # por "theme" si corresponde, igual que los colores de arriba.
    "provinces_path": None,
    "provinces_color": "#555555",
    "provinces_linewidth": 1.0,

    # --- Limites departamentales (opcional) ---
    # Path a un archivo .geojson o .shp con límites departamentales
    # (ej. descargado de datos IGN/INDEC). Si es None, no se dibuja
    # esta capa. El color (departments_color) tambien se sobreescribe
    # por "theme" si corresponde, igual que los colores de arriba.
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

    # --- Paleta CUSTOM para la colorbar (opcional) ---
    # Si es None (default), la colorbar usa los colores OFICIALES de OHMC.
    # Si tiene un valor, puede ser:
    #   - Nombre de una paleta de MetPy (ej. "NWSStormClearReflectivity")
    #   - Path a un archivo .pal/.tbl (formato GEMPAK/MetPy)
    # IMPORTANTE: esto solo cambia la leyenda dibujada por este script, NO
    # los colores de la imagen del radar en si (ver nota en
    # resolve_palette_colors()). Requiere 'pip install metpy' salvo que se
    # use un archivo .pal ya en formato compatible sin metpy instalado.
    "palette": None,

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


def fetch_radar_info(cfg):
    """Consulta /api/v1/radars?active_only=true y devuelve el diccionario
    COMPLETO del radar indicado en cfg['radar_code'] (incluye 'extent',
    'center_lat', 'center_long', 'title', etc.). Devuelve None si no hay
    radar_code configurado, o si no se lo encuentra / falla la consulta.
    Se cachea en cfg['_radar_info_cache'] para no repetir la consulta si
    se llama mas de una vez con el mismo radar_code (ver
    fetch_radar_extent() y resolve_radar_center())."""
    radar_code = cfg.get("radar_code")
    if not radar_code:
        return None
    cache = cfg.setdefault("_radar_info_cache", {})
    if radar_code in cache:
        return cache[radar_code]
    url = f"{cfg['base_url']}/radars?active_only=true"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        for radar in data.get("radars", []):
            cache[radar.get("code")] = radar
        if radar_code in cache:
            return cache[radar_code]
        print(f"[warn] radar_code '{radar_code}' no encontrado en {url}.")
    except Exception as exc:
        print(f"[warn] No se pudo consultar {url} ({exc}).")
    cache[radar_code] = None
    return None


def fetch_radar_extent(cfg):
    """Devuelve el bbox (convertido desde 'extent') del radar indicado en
    cfg['radar_code'], usando fetch_radar_info(). Devuelve None si no se
    encuentra el radar o falla la consulta (en cuyo caso se usa
    'radar_bbox' del CONFIG como fallback, ver resolve_bbox())."""
    radar = fetch_radar_info(cfg)
    if radar is None:
        return None
    ext = radar["extent"]
    print(f"[info] bbox de '{cfg.get('radar_code')}' ({radar.get('title', '')}) "
          f"obtenido automaticamente desde /api/v1/radars.")
    return {
        "min_lon": ext["lon_min"],
        "min_lat": ext["lat_min"],
        "max_lon": ext["lon_max"],
        "max_lat": ext["lat_max"],
    }


def resolve_radar_center(cfg, frame_meta=None, bbox=None):
    """Resuelve las coordenadas (lat, lon) del CENTRO del radar (donde
    esta ubicada la antena), usadas como centro de los range rings (ver
    add_range_rings()). Prioridad:
    1) 'center_lat'/'center_long' reales del radar, desde
       /api/v1/radars?active_only=true (RECOMENDADO: es la ubicacion
       exacta de la antena, no un punto aproximado).
    2) Si no se pudo obtener (radar_code no configurado/no encontrado),
       se usa el centro geometrico del bbox como aproximacion.
    Devuelve (lat, lon), o None si no hay ninguna fuente disponible."""
    radar = fetch_radar_info(cfg)
    if radar and radar.get("center_lat") is not None and radar.get("center_long") is not None:
        return radar["center_lat"], radar["center_long"]
    if bbox:
        print("[warn] No se pudo obtener el centro real del radar; se usa el "
              "centro geometrico del bbox como aproximacion para los range rings.")
        return (
            (bbox["min_lat"] + bbox["max_lat"]) / 2,
            (bbox["min_lon"] + bbox["max_lon"]) / 2,
        )
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


def resolve_vol_nr(cfg):
    """Resuelve el 'vol_nr' a pasarle a /api/v1/cogs a partir de
    cfg['coverage_km'] (240 o 450), usando COVERAGE_VOL_NR. Devuelve None
    si cfg['coverage_km'] no esta configurado (comportamiento por defecto
    de OHMC, sin forzar ninguna cobertura en particular) o si el valor no
    es reconocido (240/450)."""
    coverage_km = cfg.get("coverage_km")
    if not coverage_km:
        return None
    vol_nr = COVERAGE_VOL_NR.get(coverage_km)
    if vol_nr is None:
        print(f"[warn] coverage_km={coverage_km} no reconocido; opciones validas: "
              f"{', '.join(str(k) for k in COVERAGE_VOL_NR)}. Se ignora.")
        return None
    return vol_nr


def parse_target_datetime(value):
    """Convierte cfg['target_datetime'] (string ISO 8601, con o sin zona
    horaria, o un objeto datetime) a un datetime AWARE en UTC, listo para
    consultar la API. Reglas:
    - str SIN zona horaria (ej. "2026-02-18 20:30", "2026-02-18T20:30:00")
      -> se interpreta en HORA ARGENTINA (ART, UTC-3).
    - str CON zona horaria explicita (ej. "...Z", "...-03:00") -> se
      respeta esa zona tal cual.
    - datetime naive (sin tzinfo) -> se interpreta en hora argentina.
    - datetime aware (con tzinfo) -> se respeta su zona.
    Devuelve None si 'value' es None. Lanza ValueError si el string no se
    puede parsear (mensaje pensado para mostrarse directo al usuario)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(
                f"No se pudo interpretar la fecha/hora '{value}'. Usa un formato "
                "ISO 8601, ej. '2026-02-18 20:30' (se asume hora argentina) o "
                "'2026-02-18T23:30:00Z' (UTC explicito)."
            )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ART_TZ)
    return dt.astimezone(timezone.utc)


def resolve_frame_id_by_datetime(cfg, product_key, vol_nr=None):
    """Busca el frame MAS CERCANO a cfg['target_datetime'] (ver
    parse_target_datetime) para cfg['radar_code'] + product_key (+ vol_nr
    si se especifica), abriendo una ventana de +/- cfg['datetime_window_minutes']
    alrededor del momento pedido y consultando
    /api/v1/cogs?radar_code=...&product_key=...&start_time=...&end_time=...

    Si no se encuentra nada, corta la ejecucion con un mensaje claro,
    aclarando la posible causa (retencion de datos limitada en OHMC, ver
    RETENTION_DAYS_ESTIMATE) cuando la fecha pedida es vieja."""
    radar_code = cfg.get("radar_code")
    if not radar_code:
        sys.exit(
            "[error] 'target_datetime' requiere tambien 'radar_code' (--radar-code), "
            "para saber en que radar buscar."
        )

    try:
        target_utc = parse_target_datetime(cfg["target_datetime"])
    except ValueError as exc:
        sys.exit(f"[error] {exc}")

    window = timedelta(minutes=cfg.get("datetime_window_minutes", 30))
    start_time = (target_utc - window).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time = (target_utc + window).strftime("%Y-%m-%dT%H:%M:%SZ")

    url = (f"{cfg['base_url']}/cogs?radar_code={radar_code}"
           f"&product_key={product_key}&start_time={start_time}&end_time={end_time}"
           f"&limit=200")
    if vol_nr:
        url += f"&vol_nr={vol_nr}"

    target_art_str = target_utc.astimezone(ART_TZ).strftime("%d-%m-%Y %H:%M")
    print(f"[info] Buscando frame mas cercano a {target_art_str} (hora ARG) "
          f"para '{radar_code}'/'{product_key}'...")
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        cogs = resp.json().get("cogs", [])
    except Exception as exc:
        sys.exit(f"[error] No se pudo consultar {url} ({exc}).")

    if not cogs:
        age_days = (datetime.now(timezone.utc) - target_utc).days
        retention_hint = ""
        if age_days > RETENTION_DAYS_ESTIMATE:
            retention_hint = (
                f"\n       La fecha pedida es de hace ~{age_days} dias. Se confirmo "
                f"probando la API que OHMC retiene los datos crudos solo unos "
                f"~{RETENTION_DAYS_ESTIMATE} dias hacia atras desde la fecha actual "
                "(mas alla de eso, ya no hay resultados). Es probable que el archivo "
                "original ya haya sido purgado del sistema, sin forma de recuperarlo "
                "via esta API."
            )
        sys.exit(
            f"[error] No se encontro ningun frame para radar_code='{radar_code}' "
            f"product_key='{product_key}' dentro de +/-{window} de {target_art_str} "
            f"(hora ARG) en {url}.{retention_hint}"
        )

    def obs_dt(entry):
        return datetime.fromisoformat(entry["observation_time"].replace("Z", "+00:00"))

    closest = min(cogs, key=lambda c: abs((obs_dt(c) - target_utc).total_seconds()))
    diff = obs_dt(closest) - target_utc
    frame_id = closest["id"]
    print(f"[info] Frame mas cercano encontrado: id={frame_id} "
          f"({closest.get('observation_time')}, diferencia de {diff}).")
    return frame_id


def resolve_frame_id(cfg, product_key, vol_nr=None):
    """Resuelve el frame_id a usar, en orden de prioridad:
    1) cfg['frame_id'] fijo, si ya tiene un valor (pero ver la
       salvaguarda en resolve_frame() para el caso en que apunte a una
       variante filtrada).
    2) cfg['target_datetime'], si esta configurado: busca el frame MAS
       CERCANO a esa fecha/hora (ver resolve_frame_id_by_datetime()).
    3) Si ninguno de los dos esta configurado, se busca automaticamente
       el ULTIMO frame disponible para cfg['radar_code'] + product_key
       (+ vol_nr si se especifica, para elegir entre la cobertura de
       240 km o 450 km; ver COVERAGE_VOL_NR/cfg['coverage_km']),
       consultando /api/v1/cogs?radar_code=...&product_key=...&limit=1[&vol_nr=...]."""
    if cfg.get("frame_id"):
        return cfg["frame_id"]

    if cfg.get("target_datetime"):
        return resolve_frame_id_by_datetime(cfg, product_key, vol_nr=vol_nr)

    radar_code = cfg.get("radar_code")
    if not radar_code:
        sys.exit(
            "[error] No se especifico 'frame_id' ni 'radar_code'; no hay forma "
            "de saber que frame descargar. Especifica al menos uno de los dos "
            "(--radar-code RMA5, por ejemplo)."
        )

    url = (f"{cfg['base_url']}/cogs?radar_code={radar_code}"
           f"&product_key={product_key}&limit=1")
    if vol_nr:
        url += f"&vol_nr={vol_nr}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        cogs = resp.json().get("cogs", [])
        if not cogs:
            extra = f" con vol_nr='{vol_nr}' (cobertura {cfg.get('coverage_km')} km)" if vol_nr else ""
            sys.exit(
                f"[error] No se encontro ningun frame reciente para "
                f"radar_code='{radar_code}' product_key='{product_key}'{extra} en {url}. "
                "Verifica que el radar este activo, que la variable tenga datos, y que "
                "ese radar realmente tenga una estrategia de cobertura extendida (450 km) "
                "si se pidio 'coverage_km=450'."
            )
        frame_id = cogs[0]["id"]
        print(f"[info] Ultimo frame de '{radar_code}'/'{product_key}'"
              f"{f' (vol_nr={vol_nr})' if vol_nr else ''}: "
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
       de graficar la version filtrada. Tambien resuelve 'vol_nr' a partir
       de cfg['coverage_km'] (240 o 450), para elegir la cobertura de
       240 km o 450 km cuando la resolucion es automatica (ver
       COVERAGE_VOL_NR)."""
    product_key = resolve_product_key(cfg)
    vol_nr = resolve_vol_nr(cfg)
    frame_id = resolve_frame_id(cfg, product_key, vol_nr=vol_nr)
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
        frame_id = resolve_frame_id(cfg, unfiltered_key, vol_nr=vol_nr)
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


def fetch_all_products(cfg):
    """Consulta /api/v1/products una sola vez y devuelve la lista completa
    de productos. Devuelve [] si falla la consulta."""
    url = f"{cfg['base_url']}/products"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json().get("products", [])
    except Exception as exc:
        print(f"[warn] No se pudo consultar {url} ({exc}).")
        return []


def fetch_product_info(cfg, product_key):
    """Devuelve el diccionario de producto (product_title,
    product_description, min_value, max_value, unit, references [lista de
    {value, color, title}]) correspondiente a 'product_key', consultando
    /api/v1/products. Devuelve {} si no se encuentra o falla la consulta.

    Si el producto encontrado tiene 'references' VACIO (esto ocurre con
    las variantes SIN FILTRAR de OHMC, ej. 'COLMAXo'/'DBZHo', que este
    script usa siempre por la salvaguarda anti-filtrado, ver
    resolve_product_key()), se completan las 'references' "prestandolas"
    de la variante FILTRADA equivalente (ej. 'COLMAX'/'DBZH'), que si las
    trae. Esto es seguro porque ambas variantes comparten la misma escala
    de color (mismo cog_cmap, mismo min_value/max_value en OHMC); lo unico
    que falta en la variante sin filtrar es la lista de colores/valores
    de referencia para dibujar la leyenda, no un rango distinto. Sin este
    fallback, la colorbar queda vacia con el aviso:
        [info] No hay suficientes referencias de color para dibujar la
        colorbar; se omite."""
    if not product_key:
        return {}
    products = fetch_all_products(cfg)
    if not products:
        return {}

    by_key = {p.get("product_key"): p for p in products}
    product = by_key.get(product_key)
    if product is None:
        print(f"[warn] product_key '{product_key}' no encontrado en /api/v1/products; "
              "se omite la colorbar y se usa un nombre de producto generico.")
        return {}

    print(f"[info] Info de producto '{product_key}' obtenida desde /api/v1/products")

    if not product.get("references"):
        filtered_key = UNFILTERED_TO_FILTERED_KEYS.get(product_key)
        filtered_product = by_key.get(filtered_key) if filtered_key else None
        if filtered_product and filtered_product.get("references"):
            print(f"[info] '{product_key}' no trae 'references' propias; se usan "
                  f"las de su variante filtrada '{filtered_key}' solo para la "
                  "colorbar (misma escala de color; la imagen del radar sigue "
                  f"siendo la de '{product_key}', sin filtrar).")
            product = dict(product)  # no mutar el dict cacheado
            product["references"] = filtered_product["references"]

    return product


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


def build_colorbar(fig, ax, cfg, product_info, frame_meta=None):
    """Agrega al costado de la imagen la colorbar/leyenda del producto
    graficado. Por defecto usa la paleta OFICIAL de OHMC (discreta, por
    bandas, con las 'references' de /api/v1/products). Si cfg['palette']
    resuelve a una paleta propia (MetPy o archivo .pal/.tbl, ver
    resolve_palette_colors()), se dibuja en su lugar una colorbar
    CONTINUA con esa paleta, sobre el rango [min_value, max_value] del
    producto (o el vmin/vmax del frame como fallback).

    IMPORTANTE: esto solo cambia la leyenda dibujada por este script; NO
    afecta a los colores de la imagen del radar en si (esa la colorea
    OHMC en el servidor, ver nota junto a resolve_palette_colors()).

    Si se usa la paleta oficial de OHMC, solo se muestra el VALOR
    numerico (+ unidad) en cada marca; no se incluyen las descripciones
    textuales de OHMC (ej. "Lluvia muy intensa y granizo"), a pedido del
    usuario.

    No hace nada si no hay info suficiente en ningun caso."""
    frame_meta = frame_meta or {}
    custom_colors = resolve_palette_colors(cfg)

    if custom_colors:
        vmin = product_info.get("min_value")
        vmax = product_info.get("max_value")
        if vmin is None:
            vmin = frame_meta.get("cog_vmin")
        if vmax is None:
            vmax = frame_meta.get("cog_vmax")
        if vmin is None or vmax is None:
            print("[warn] No hay rango [min_value, max_value] para la paleta custom; se omite la colorbar.")
            return
        cmap = mcolors.ListedColormap(custom_colors, name=cfg.get("palette"))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.045, pad=0.03)
        unit = product_info.get("unit") or ""
        label = cfg.get("palette", "")
        if unit and unit != "-":
            label += f" ({unit})" if label else unit
        if label:
            cbar.set_label(label, fontsize=8)
        return

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

    # GDAL/OGR (usado por geopandas/fiona/pyogrio para leer GeoJSON) tiene
    # un limite de tamaño por objeto/feature (OGR_GEOJSON_MAX_OBJ_SIZE,
    # default 200 MB), pensado para evitar cargar accidentalmente
    # archivos corruptos. Con GeoJSON de municipios/departamentos de
    # Argentina (muchos vertices por poligono) esto puede dispararse
    # incluso con archivos legitimos, con un error tipo:
    #     GeoJSON object too complex/large. You may define the
    #     OGR_GEOJSON_MAX_OBJ_SIZE configuration option ...
    # Por eso se desactiva el limite (valor "0") antes de leer, y se
    # restaura el valor previo despues, para no afectar otras lecturas
    # que puedan depender de el en el mismo proceso.
    prev_max_obj_size = os.environ.get("OGR_GEOJSON_MAX_OBJ_SIZE")
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
    try:
        gdf = gpd.read_file(path)
        print(f"[info] {layer_name.capitalize()} cargados desde '{path}' ({len(gdf)} geometrias)")
    except Exception as exc:
        print(f"[warn] No se pudo leer '{path}': {exc}")
        return None
    finally:
        if prev_max_obj_size is None:
            os.environ.pop("OGR_GEOJSON_MAX_OBJ_SIZE", None)
        else:
            os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = prev_max_obj_size

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
    """Calcula el extent visible del mapa a partir del bbox del frame.
    Prioridad del margen:
    1) cfg['map_extent'] fijo, si esta configurado.
    2) cfg['extent_margin_pct'] (RECOMENDADO): margen proporcional al
       ancho/alto del bbox (ej. 0.08 = 8%). Mantiene el mismo nivel de
       zoom relativo sin importar si el radar cubre 240 km o 450 km.
    3) cfg['extent_margin_deg']: margen fijo en grados (comportamiento
       anterior), usado solo si 'extent_margin_pct' es None/0.
    """
    if cfg.get("map_extent"):
        return cfg["map_extent"]

    lon_span = bbox["max_lon"] - bbox["min_lon"]
    lat_span = bbox["max_lat"] - bbox["min_lat"]
    pct = cfg.get("extent_margin_pct")
    if pct:
        m_lon = lon_span * pct
        m_lat = lat_span * pct
    else:
        m_lon = m_lat = cfg.get("extent_margin_deg", 1.5)

    return [
        bbox["min_lon"] - m_lon,
        bbox["max_lon"] + m_lon,
        bbox["min_lat"] - m_lat,
        bbox["max_lat"] + m_lat,
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

    # --- Etiquetas de localidades principales (solo tema "dark") ---
    # zorder mayor al del overlay del radar (10), para que el nombre de
    # la localidad se siga viendo bien aunque quede "debajo" del eco.
    add_locality_labels(ax, extent, cfg)

    # --- Range rings (anillos de rango del radar, opcional) ---
    add_range_rings(ax, cfg, frame_meta=frame_meta, bbox=bbox)

    # --- Colorbar del producto al costado (opcional) ---
    if cfg.get("show_colorbar", True):
        build_colorbar(fig, ax, cfg, product_info, frame_meta=frame_meta)

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
                              "el ultimo frame disponible para --radar-code + --variable "
                              "(o el mas cercano a --datetime, si se especifica).")
    parser.add_argument("--datetime", type=str, default=cfg["target_datetime"],
                         help="Busca el frame mas cercano a esta fecha/hora en vez del "
                              "ultimo disponible. Formato ISO 8601, ej. '2026-02-18 20:30' "
                              "(se asume hora ARGENTINA) o '2026-02-18T23:30:00Z' (UTC "
                              "explicito). OHMC retiene los datos crudos solo por un "
                              f"tiempo limitado (~{RETENTION_DAYS_ESTIMATE} dias hacia atras "
                              "al momento de escribir esto); fechas mas viejas probablemente "
                              "no tengan datos disponibles.")
    parser.add_argument("--datetime-window", type=int, default=cfg["datetime_window_minutes"],
                         help="Ventana de tolerancia en minutos (+/-) alrededor de --datetime "
                              "para buscar el frame mas cercano. Default: "
                              f"{cfg['datetime_window_minutes']} minutos.")
    parser.add_argument("--colormap", type=str, default=cfg["colormap"],
                         help="Colormap/paleta a forzar. Si se omite, se resuelve "
                              "automaticamente desde la metadata del frame.")
    parser.add_argument("--radar-code", type=str, default=cfg["radar_code"],
                         help="Codigo de radar (ej. RMA2, RMA14) para resolver el bbox "
                              "automaticamente desde /api/v1/radars?active_only=true.")
    parser.add_argument("--coverage-km", type=int, default=cfg["coverage_km"],
                         choices=list(COVERAGE_VOL_NR),
                         help="Radio de cobertura a usar cuando la resolucion del frame "
                              "es automatica (sin --frame-id): 240 o 450 km. Si se omite, "
                              "se usa el comportamiento por defecto de OHMC.")
    parser.add_argument("--output", type=str, default=cfg["output_path"])
    parser.add_argument("--provinces", type=str, default=cfg["provinces_path"],
                         help="Path a un GeoJSON/Shapefile de limites provinciales.")
    parser.add_argument("--departments", type=str, default=cfg["departments_path"],
                         help="Path a un GeoJSON/Shapefile de limites departamentales.")
    parser.add_argument("--no-colorbar", action="store_true",
                         help="No dibujar la colorbar/leyenda del producto al costado.")
    parser.add_argument("--palette", type=str, default=cfg["palette"],
                         help="Paleta CUSTOM para la colorbar: nombre de una paleta de "
                              "MetPy (ej. NWSStormClearReflectivity) o path a un archivo "
                              ".pal/.tbl. Si se omite, se usa la paleta oficial de OHMC. "
                              "NO cambia los colores de la imagen del radar en si.")
    parser.add_argument("--title", type=str, default=cfg["title"],
                         help="Titulo fijo del mapa. Si se omite, se genera "
                              "automaticamente a partir de la metadata del frame.")
    parser.add_argument("--theme", type=str, default=cfg["theme"],
                         choices=list(THEME_PRESETS),
                         help="Tema de color del mapa base: 'light' (tierra blanca/oceano "
                              "celeste, limites en gris oscuro) o 'dark' (tierra y oceano "
                              "negros, limites en gris claro). Aplica de una sola vez a "
                              "tierra, oceano, y limites nacionales/provinciales/"
                              "departamentales.")
    parser.add_argument("--font", type=str, default=cfg["font_path"],
                         help="Path a un archivo .ttf/.otf a usar como fuente principal "
                              "del plot (titulo, colorbar, etiquetas de localidades). Si "
                              f"se omite, se usa '{DEFAULT_FONT_FILENAME}' (incluida junto "
                              "al script).")
    parser.add_argument("--range-rings", type=str, default=None,
                         help="Distancias (km) para los range rings, separadas por coma "
                              "(ej. '120,240' o '100,200,300'). Si se omite, se dibujan "
                              f"automaticamente {DEFAULT_RANGE_RING_KM} km + el valor de "
                              "--coverage-km (240 por defecto). Pasa '' (vacio) para no "
                              "dibujar ningun anillo.")
    parser.add_argument("--range-rings-color", type=str, default=cfg["range_rings_color"],
                         help="Color de los range rings. Si se omite, se usa blanco con "
                              "--theme dark, o gris oscuro en caso contrario.")
    parser.add_argument("--no-show", action="store_true",
                         help="No abrir ventana de matplotlib; solo guardar el archivo.")
    args = parser.parse_args()

    cfg["variable"] = args.variable
    cfg["frame_id"] = args.frame_id
    cfg["target_datetime"] = args.datetime
    cfg["datetime_window_minutes"] = args.datetime_window
    cfg["colormap"] = args.colormap
    cfg["radar_code"] = args.radar_code
    cfg["coverage_km"] = args.coverage_km
    cfg["output_path"] = args.output
    cfg["provinces_path"] = args.provinces
    cfg["departments_path"] = args.departments
    cfg["title"] = args.title
    cfg["palette"] = args.palette
    cfg["theme"] = args.theme
    cfg["font_path"] = args.font
    if args.range_rings is not None:
        text = args.range_rings.strip()
        cfg["range_rings_km"] = (
            [float(v) for v in text.split(",") if v.strip()] if text else []
        )
    cfg["range_rings_color"] = args.range_rings_color
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
    apply_theme(cfg)
    apply_font(cfg)

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

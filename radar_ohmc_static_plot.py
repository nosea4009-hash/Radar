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
       - grilla de coordenadas (lat/lon) con etiquetas
       - limites departamentales opcionales, importados desde un
         archivo GeoJSON/Shapefile (ver CONFIG mas abajo)
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

Uso basico
----------
    python radar_ohmc_static_plot.py

Para cambiar el radar/producto/colormap, o el path a los limites
departamentales, editá la seccion CONFIG mas abajo. Tambien podés
pasar algunos parametros por linea de comandos, por ejemplo:

    python radar_ohmc_static_plot.py --frame-id 903042 --colormap grc_th --output rma14.png

Como conseguir el "frame_id", el "colormap" y el "bbox"
---------------------------------------------------------
Estos valores se consiguen inspeccionando la pestaña "Network" del
navegador (F12) mientras se usa el visor web de OHMC:
  - "frame_id": es el "id" numerico del frame (ej: 901688, 903042),
    visible en las respuestas JSON de endpoints como
    /api/v1/latest?radar_code=... o /api/v1/cogs?product_key=...
  - "colormap": el nombre de la paleta usada en la URL de la imagen
    (ej: "grc_th", "grc_rain"), visible en las requests a
    /api/v1/frames/{id}/image.png?colormap=...
  - "bbox": las coordenadas geograficas (min_lon, min_lat, max_lon,
    max_lat) del radar, tambien presentes en esos mismos JSON.

Como el endpoint de la imagen (.../image.png) devuelve solo el PNG (sin
metadata), este script NO adivina el bbox: hay que completarlo a mano
en CONFIG["radar_bbox"], o bien apuntar CONFIG["metadata_json_url"] a
la URL de un endpoint JSON que ya hayas confirmado que contiene ese
"bbox" para el frame elegido (el script lo busca automaticamente ahi).
"""

import argparse
import sys
from io import BytesIO

import numpy as np
import requests
from PIL import Image

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit(
        "Falta matplotlib. Instalá con:\n"
        "  conda install -c conda-forge matplotlib"
    )

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    sys.exit(
        "Falta cartopy. Instalá con:\n"
        "  conda install -c conda-forge cartopy\n"
        "(cartopy tiene dependencias binarias: se recomienda conda, no pip)"
    )


# =============================================================================
# CONFIG - editá estos valores segun el radar/producto que quieras graficar
# =============================================================================
CONFIG = {
    # --- Fuente del radar (API OHMC) ---
    "base_url": "https://webmet.ohmc.ar/api/v1",
    "frame_id": 901688,
    "colormap": "grc_rain",

    # --- Bbox del radar (min/max lon/lat), en grados decimales ---
    # Completá esto a mano con el "bbox" que encontraste en el JSON de
    # metadata de OHMC para el frame_id elegido (ver docstring arriba).
    "radar_bbox": {
        "min_lon": -63.76781513732091,
        "min_lat": -38.285569273926185,
        "max_lon": -58.37274486267906,
        "max_lat": -34.03385620041487,
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

    # --- Grilla de coordenadas ---
    "grid_step_deg": 2.0,

    # --- Colores del mapa base ---
    "land_color": "#FFFFFF",       # tierra: blanco
    "ocean_color": "#CFEAF5",      # oceano: celeste marino clarito
    "coastline_color": "#666666",
    "border_color": "#999999",

    # --- Limites departamentales (opcional) ---
    # Path a un archivo .geojson o .shp con límites departamentales
    # (ej. descargado de datos IGN/INDEC). Si es None, no se dibuja
    # esta capa.
    "departments_path": None,
    "departments_color": "#808080",
    "departments_linewidth": 0.5,

    # --- Overlay del radar ---
    "radar_opacity": 0.85,  # multiplicador extra sobre la transparencia ya presente en el PNG

    # --- Salida ---
    "title": "Radar OHMC",
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


def resolve_bbox(cfg):
    """Devuelve el bbox a usar: si hay 'metadata_json_url' configurado,
    intenta buscar el bbox del frame_id ahi; si no, usa 'radar_bbox'."""
    url = cfg.get("metadata_json_url")
    if url:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            entries = data if isinstance(data, list) else data.get("results", [data])
            for entry in entries:
                if entry.get("id") == cfg["frame_id"] and entry.get("bbox"):
                    print(f"[info] bbox obtenido desde metadata_json_url para frame {cfg['frame_id']}")
                    return entry["bbox"]
            print("[warn] No se encontro un 'bbox' para ese frame_id en metadata_json_url; "
                  "se usa 'radar_bbox' del CONFIG.")
        except Exception as exc:
            print(f"[warn] No se pudo consultar metadata_json_url ({exc}); "
                  "se usa 'radar_bbox' del CONFIG.")
    return cfg["radar_bbox"]


def load_departments(cfg):
    """Carga (opcionalmente) una capa de limites departamentales desde un
    archivo GeoJSON/Shapefile, usando geopandas. Devuelve None si no hay
    path configurado o si geopandas no esta instalado."""
    path = cfg.get("departments_path")
    if not path:
        return None
    try:
        import geopandas as gpd
    except ImportError:
        print("[warn] geopandas no esta instalado; se omiten los limites departamentales.")
        print("       Instalá con: conda install -c conda-forge geopandas")
        return None
    try:
        gdf = gpd.read_file(path)
        print(f"[info] Limites departamentales cargados desde '{path}' ({len(gdf)} geometrias)")
        return gdf
    except Exception as exc:
        print(f"[warn] No se pudo leer '{path}': {exc}")
        return None


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


def build_plot(radar_rgba, bbox, cfg):
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

    # --- Limites departamentales (opcional) ---
    departments = load_departments(cfg)
    if departments is not None:
        departments.boundary.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor=cfg["departments_color"],
            linewidth=cfg["departments_linewidth"],
            zorder=3,
        )

    # --- Grilla de coordenadas ---
    gl = ax.gridlines(
        draw_labels=True, linestyle="--", linewidth=0.5,
        color="gray", alpha=0.6, zorder=4,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(cfg["grid_step_deg"])
    gl.ylocator = mticker.MultipleLocator(cfg["grid_step_deg"])

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

    ax.set_title(cfg.get("title", "Radar OHMC"), fontsize=12)
    fig.tight_layout()
    return fig


def parse_args(cfg):
    parser = argparse.ArgumentParser(description="Visualizador estatico de radar OHMC.")
    parser.add_argument("--frame-id", type=int, default=cfg["frame_id"])
    parser.add_argument("--colormap", type=str, default=cfg["colormap"])
    parser.add_argument("--output", type=str, default=cfg["output_path"])
    parser.add_argument("--departments", type=str, default=cfg["departments_path"],
                         help="Path a un GeoJSON/Shapefile de limites departamentales.")
    parser.add_argument("--no-show", action="store_true",
                         help="No abrir ventana de matplotlib; solo guardar el archivo.")
    args = parser.parse_args()

    cfg["frame_id"] = args.frame_id
    cfg["colormap"] = args.colormap
    cfg["output_path"] = args.output
    cfg["departments_path"] = args.departments
    if args.no_show:
        cfg["show_plot"] = False
    return cfg


def main():
    cfg = parse_args(CONFIG)

    radar_rgba = download_radar_image(cfg)
    bbox = resolve_bbox(cfg)
    fig = build_plot(radar_rgba, bbox, cfg)

    fig.savefig(cfg["output_path"], dpi=cfg["dpi"], bbox_inches="tight")
    print(f"[ok] Imagen guardada en: {cfg['output_path']}")

    if cfg.get("show_plot", True):
        plt.show()


if __name__ == "__main__":
    main()

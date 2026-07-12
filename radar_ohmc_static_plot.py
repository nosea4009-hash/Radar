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

Como conseguir el "frame_id" y el "colormap"
-----------------------------------------------
Estos valores se consiguen inspeccionando la pestaña "Network" del
navegador (F12) mientras se usa el visor web de OHMC:
  - "frame_id": es el "id" numerico del frame (ej: 901688, 903042),
    visible en las respuestas JSON de endpoints como
    /api/v1/latest?radar_code=... o /api/v1/cogs?product_key=...
  - "colormap": el nombre de la paleta usada en la URL de la imagen
    (ej: "grc_th", "grc_rain"), visible en las requests a
    /api/v1/frames/{id}/image.png?colormap=...

Como se resuelve el "bbox" (extension geografica del radar)
---------------------------------------------------------------
El endpoint de la imagen (.../image.png) devuelve solo el PNG, sin
metadata. Por eso el bbox se resuelve en este orden de prioridad:

  1. Automatico por "radar_code" (RECOMENDADO): si CONFIG["radar_code"]
     tiene un valor (ej. "RMA2", "RMA14"), el script consulta:

         GET https://webmet.ohmc.ar/api/v1/radars?active_only=true

     que devuelve, para cada radar, un campo "extent" con
     lat_min/lat_max/lon_min/lon_max. El bbox se toma de ahi
     automaticamente: no hace falta copiar numeros a mano.

  2. Manual: si CONFIG["radar_code"] es None (o no se encuentra en la
     lista de radares activos), se usa directamente el diccionario
     CONFIG["radar_bbox"] que hayas completado a mano (por ejemplo,
     copiando el "bbox"/"extent" que veas en el Network tab del
     navegador).

  3. Por frame especifico (avanzado): si ademas configurás
     CONFIG["metadata_json_url"] apuntando a un endpoint JSON que
     incluya un campo "bbox" por frame (ej. /api/v1/cogs?product_key=...),
     el script busca ahi el bbox exacto de ese frame_id puntual, en vez
     del extent general del radar. Es opcional y solo tiene sentido si
     necesitás el bbox de una observacion particular, no el del radar
     en general (que no cambia entre observaciones).
"""

import argparse
import sys
from io import BytesIO

import numpy as np
import requests
from PIL import Image

try:
    import matplotlib.pyplot as plt
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
    "frame_id": 901688,
    "colormap": "grc_rain",

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


def resolve_bbox(cfg):
    """Resuelve el bbox a usar, en orden de prioridad:
    1) metadata_json_url (bbox exacto de un frame puntual, si esta configurado)
    2) radar_code (extent automatico desde /api/v1/radars)
    3) radar_bbox (fallback manual del CONFIG)
    """
    bbox = fetch_bbox_from_metadata(cfg)
    if bbox:
        return bbox
    bbox = fetch_radar_extent(cfg)
    if bbox:
        return bbox
    print("[info] Usando 'radar_bbox' manual del CONFIG.")
    return cfg["radar_bbox"]


def load_departments(cfg):
    """Carga (opcionalmente) una capa de limites departamentales desde un
    archivo GeoJSON/Shapefile, usando geopandas. Devuelve None si no hay
    path configurado o si geopandas no esta instalado.

    Repara automaticamente geometrias invalidas (ej. anillos que no
    cierran, topologia rota) antes de devolver la capa, para evitar
    errores tipo:
        shapely.errors.GEOSException: IllegalArgumentException:
        Points of LinearRing do not form a closed linestring
    Este problema puede aparecer con datasets de límites departamentales
    de cualquier formato (GeoJSON, Shapefile, GPKG, etc.) que tengan
    geometrias mal formadas en el archivo de origen; no es especifico
    de ningun formato en particular."""
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
    parser.add_argument("--radar-code", type=str, default=cfg["radar_code"],
                         help="Codigo de radar (ej. RMA2, RMA14) para resolver el bbox "
                              "automaticamente desde /api/v1/radars?active_only=true.")
    parser.add_argument("--output", type=str, default=cfg["output_path"])
    parser.add_argument("--departments", type=str, default=cfg["departments_path"],
                         help="Path a un GeoJSON/Shapefile de limites departamentales.")
    parser.add_argument("--no-show", action="store_true",
                         help="No abrir ventana de matplotlib; solo guardar el archivo.")
    args = parser.parse_args()

    cfg["frame_id"] = args.frame_id
    cfg["colormap"] = args.colormap
    cfg["radar_code"] = args.radar_code
    cfg["output_path"] = args.output
    cfg["departments_path"] = args.departments
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

    radar_rgba = download_radar_image(cfg)
    bbox = resolve_bbox(cfg)
    fig = build_plot(radar_rgba, bbox, cfg)

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

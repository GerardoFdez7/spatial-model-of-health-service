"""Funciones compartidas por todos los tasks.

Convención de notación:
  - Cada píxel del ráster WorldPop es una celda (i, j) de la grilla regular G; su población
    es el peso de demanda d_i del problema de cobertura.
  - Las instalaciones son las ubicaciones candidatas/instaladas j con x_j = 1.
  - N_i = { j : dist(i, j) <= d } es el conjunto de instalaciones que cubren la demanda i.
"""
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree

from config import CRS_GEO, CRS_PROJ, ISO3, PROC, RAW

# --------------------------------------------------------------------------- capas
def _read(path, crs=CRS_PROJ):
    gdf = gpd.read_file(path)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf.to_crs(crs)


@lru_cache(maxsize=None)
def load_adm(level: int, crs: str = CRS_PROJ) -> gpd.GeoDataFrame:
    """Polígonos GADM del nivel dado, reproyectados (por defecto a metros)."""
    return _read(RAW / f"gadm41_{ISO3}_{level}.json", crs)


@lru_cache(maxsize=None)
def load_facilities(crs: str = CRS_PROJ) -> gpd.GeoDataFrame:
    """Instalaciones de salud ya filtradas al territorio (salida de 01_download.py)."""
    return gpd.read_file(PROC / "facilities.gpkg").to_crs(crs)


@lru_cache(maxsize=None)
def pop_raster_path():
    return next(RAW.glob("*.tif"))


@lru_cache(maxsize=None)
def load_pop_points() -> gpd.GeoDataFrame:
    """Convierte el ráster de población en una nube de 'celdas demanda'.

    Cada píxel con pop > 0 pasa a ser un punto (centro del píxel) con su población `pop`. Se proyecta a metros y se asigna a su municipio (nivel 2).
    Los píxeles costeros/fronterizos cuyo centro cae fuera de GADM se asignan al municipio más
    cercano si está a <= 1.5 km, para no perder población por el efecto de borde de la grilla.
    """
    with rasterio.open(pop_raster_path()) as src:
        band = src.read(1, masked=True).astype("float64")
        rows, cols = np.where((~band.mask) & (band.filled(0) > 0))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        pts = gpd.GeoDataFrame({"pop": band[rows, cols].data, "row": rows, "col": cols},
                               geometry=gpd.points_from_xy(xs, ys), crs=src.crs).to_crs(CRS_PROJ)
    adm2 = load_adm(2)[["GID_2", "GID_1", "geometry"]]
    inside = gpd.sjoin(pts, adm2, how="left", predicate="within").drop(columns="index_right")
    inside = inside[~inside.index.duplicated()]
    miss = inside["GID_2"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts.loc[miss[miss].index], adm2, how="left", max_distance=1500)
        near = near[~near.index.duplicated()]
        inside.loc[near.index, ["GID_2", "GID_1"]] = near[["GID_2", "GID_1"]].values
    return inside.dropna(subset=["GID_2"]).reset_index(drop=True)


def xy(gdf) -> np.ndarray:
    """Coordenadas (N,2) de una capa de puntos."""
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


def nearest_facility(points_xy: np.ndarray, fac: gpd.GeoDataFrame):
    """Distancia (m) e índice a la instalación más cercana para cada punto.

    min_j dist(i, j) determina si y_i = 1 para un radio d: y_i = 1  <=>  min_j dist(i,j) <= d."""
    return cKDTree(xy(fac)).query(points_xy)


def covered_population(pop: gpd.GeoDataFrame, dist_m: np.ndarray, radius_km: float) -> float:
    """sum_i d_i * y_i  con y_i = [dist_i <= radio]  (función objetivo del MCLP, diap. 9)."""
    return float(pop["pop"].values[dist_m <= radius_km * 1000].sum())


# --------------------------------------------------------------------------- tipos
TYPE_ORDER = ["hospital", "clinic", "doctors", "pharmacy", "dentist", "other"]
TYPE_COLORS = {"hospital": "#d62728", "clinic": "#1f77b4", "doctors": "#2ca02c",
               "pharmacy": "#9467bd", "dentist": "#ff7f0e", "other": "#7f7f7f"}


def normalize_type(amenity, healthcare) -> str:
    """Unifica etiquetas OSM/healthsites en las categorías del análisis."""
    for v in (amenity, healthcare):
        v = str(v).lower() if v is not None and not (isinstance(v, float) and np.isnan(v)) else ""
        if v in ("hospital", "clinic", "doctors", "pharmacy", "dentist"):
            return v
        if v in ("doctor",):
            return "doctors"
    return "other"


# --------------------------------------------------------------------------- mapas
def add_scalebar(ax, length_km=100, loc=(0.05, 0.05)):
    """Barra de escala gráfica; el eje debe estar en metros (CRS proyectado)."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * loc[0]; y = y0 + (y1 - y0) * loc[1]
    L = length_km * 1000
    ax.plot([x, x + L], [y, y], color="k", lw=4, solid_capstyle="butt", zorder=10)
    ax.plot([x, x + L / 2], [y, y], color="w", lw=2, solid_capstyle="butt", zorder=11)
    ax.text(x + L / 2, y + (y1 - y0) * 0.012, f"{length_km} km", ha="center", va="bottom",
            fontsize=8, zorder=12)


def add_north(ax, loc=(0.92, 0.85)):
    """Flecha de orientación norte (UTM: el norte de la grilla apunta hacia arriba)."""
    ax.annotate("N", xy=loc, xytext=(loc[0], loc[1] - 0.09), xycoords="axes fraction",
                ha="center", fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2))


def base_axes(figsize=(9, 9), title=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()
    return fig, ax


def municipalities(crs: str = CRS_PROJ) -> gpd.GeoDataFrame:
    """Nivel 2 sin los 2 cuerpos de agua de GADM (lagos de Izabal y Atitlán, NAME_2 = '?')."""
    a = load_adm(2, crs)
    a = a[a["ENGTYPE_2"] == "Municipality"].copy()
    a["label"] = a["NAME_2"] + " (" + a["NAME_1"] + ")"   # NAME_2 se repite entre departamentos
    return a.reset_index(drop=True)

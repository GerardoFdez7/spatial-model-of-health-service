# Funciones compartidas: carga de capas procesadas, buffers, cobertura poblacional y elementos de mapas.
# Notación MCLP: demanda i con peso d_i (población), hospitales j, cobertura y_i = 1 si dist(i, j) <= S.
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from config import CRS_PROJ, PROC


# Carga los condados del estado con su población (salida de t1_1_carga.py).
@lru_cache(maxsize=None)
def load_counties() -> gpd.GeoDataFrame:
    return gpd.read_file(PROC / "condados_estado.gpkg").to_crs(CRS_PROJ)


# Carga los hospitales del estado que tienen dato de camas.
@lru_cache(maxsize=None)
def load_hospitals() -> gpd.GeoDataFrame:
    return gpd.read_file(PROC / "hospitales_estado.gpkg").to_crs(CRS_PROJ)


# Carga el polígono del estado.
@lru_cache(maxsize=None)
def load_state() -> gpd.GeoDataFrame:
    return gpd.read_file(PROC / "estado.gpkg").to_crs(CRS_PROJ)


# Devuelve el límite estatal como una sola geometría.
def state_polygon():
    return load_state().geometry.union_all()


# Devuelve las coordenadas (N, 2) de una capa de puntos.
def xy(gdf) -> np.ndarray:
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


# Crea un buffer de radio S en cada hospital y los une (unary_union): región donde y_i = 1.
def coverage_polygon(points: gpd.GeoDataFrame, radius_km: float):
    buffers = points.geometry.buffer(radius_km * 1000, quad_segs=32)
    return shapely.union_all(buffers.values)


# Fracción del área de cada condado dentro de la cobertura: área(c ∩ U) / área(c).
def covered_fraction(counties: gpd.GeoDataFrame, poly) -> pd.Series:
    return counties.geometry.intersection(poly).area / counties.geometry.area


# Población cubierta con densidad uniforme: sum_c P_c * área(c ∩ U) / área(c).
def covered_population(counties: gpd.GeoDataFrame, poly) -> float:
    return float((counties["poblacion"] * covered_fraction(counties, poly)).sum())


# Distancia (m) e índice del punto objetivo más cercano: min_j dist(i, j).
def nearest(points_xy: np.ndarray, targets: gpd.GeoDataFrame):
    return cKDTree(xy(targets)).query(points_xy)


# Normalización min-max: x' = (x - min) / (max - min).
def minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


TYPE_COLORS = {"GENERAL ACUTE CARE": "#1f77b4", "CRITICAL ACCESS": "#d62728", "PSYCHIATRIC": "#9467bd",
               "LONG TERM CARE": "#8c564b", "REHABILITATION": "#2ca02c", "MILITARY": "#7f7f7f",
               "CHILDREN": "#ff7f0e", "SPECIAL": "#e377c2", "WOMEN": "#17becf",
               "CHRONIC DISEASE": "#bcbd22", "OTHER": "#000000"}


# Dibuja una barra de escala gráfica (el eje debe estar en metros).
def add_scalebar(ax, length_km=100, loc=(0.05, 0.04)):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * loc[0]; y = y0 + (y1 - y0) * loc[1]
    L = length_km * 1000
    for k in range(4):
        ax.plot([x + k * L / 4, x + (k + 1) * L / 4], [y, y], color="k" if k % 2 == 0 else "w",
                lw=5, solid_capstyle="butt", zorder=11)
    ax.plot([x, x + L], [y, y], color="k", lw=7, solid_capstyle="butt", zorder=10)
    for k, lab in [(0, "0"), (2, f"{length_km // 2}"), (4, f"{length_km} km")]:
        ax.text(x + k * L / 4, y + (y1 - y0) * 0.012, lab, ha="center", va="bottom", fontsize=8, zorder=12)


# Dibuja la flecha del norte.
def add_north(ax, loc=(0.93, 0.93)):
    ax.annotate("N", xy=loc, xytext=(loc[0], loc[1] - 0.08), xycoords="axes fraction", ha="center",
                fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="k", lw=2))


# Crea una figura de mapa sin ejes con su título.
def base_axes(figsize=(8, 10), title=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()
    return fig, ax


# Ajusta los límites del mapa al estado con un margen.
def frame(ax, pad=15_000):
    minx, miny, maxx, maxy = load_state().total_bounds
    ax.set_xlim(minx - pad, maxx + pad); ax.set_ylim(miny - pad, maxy + pad)

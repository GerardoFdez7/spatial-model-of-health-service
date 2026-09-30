"""Task 1.2 - Análisis exploratorio espacial (reproyección, mapa base y estadísticas por departamento)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from common import (TYPE_COLORS, TYPE_ORDER, add_north, add_scalebar, base_axes, load_adm,
                    load_facilities, load_pop_points, municipalities, xy)
from config import CRS_PROJ, OUT

adm1, adm2, fac, pop = load_adm(1), municipalities(), load_facilities(), load_pop_points()

# ---- 1.2a: todas las capas ya salen en EPSG:32615 (metros) desde los loaders --------------------
print("== 1.2a CRS de trabajo ==")
for name, g in [("nivel1", adm1), ("nivel2", adm2), ("instalaciones", fac), ("población", pop)]:
    print(f"  {name}: {g.crs.to_string()}  (unidad: {g.crs.axis_info[0].unit_name})")
for lon, lat, lugar in [(-90.5, 15.5, "centro del país"), (-88.2, 15.7, "extremo este (Izabal)")]:
    k = 0.9996 * (1 + (np.radians(lon + 93) * np.cos(np.radians(lat))) ** 2 / 2)   # factor de escala UTM
    print(f"  Factor de escala UTM 15N en {lugar}: k = {k:.5f}  (error lineal {abs(k-1)*100:.2f} %)")

# ---- 1.2b: mapa base de tres capas ---------------------------------------------------------------
fig, ax = base_axes((10, 10), "Línea base: Guatemala – instalaciones de salud (n = %d)" % len(fac))
adm2.plot(ax=ax, color="#e6e6e6", edgecolor="#bdbdbd", linewidth=0.2)
for t in TYPE_ORDER:                                    # color según tipo
    s = fac[fac["type"] == t]
    ax.scatter(s.geometry.x, s.geometry.y, s=7, color=TYPE_COLORS[t], alpha=.75, linewidths=0, zorder=5)
adm1.boundary.plot(ax=ax, color="black", linewidth=0.9, zorder=6)   # nivel 1: contorno sin relleno
minx, miny, maxx, maxy = adm1.total_bounds
ax.set_xlim(minx - 2e4, maxx + 2e4); ax.set_ylim(miny - 2e4, maxy + 2e4)
add_scalebar(ax, 100); add_north(ax)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=TYPE_COLORS[t],
                          label=f"{t} ({(fac['type']==t).sum()})") for t in TYPE_ORDER] +
          [Line2D([], [], color="k", lw=1, label="Departamentos (nivel 1)"),
           Line2D([], [], marker="s", ls="", color="#e6e6e6", label="Municipios (nivel 2)")],
          loc="lower right", fontsize=8, frameon=True)
fig.savefig(OUT / "t1_2_mapa_base.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---- 1.2c: estadísticas por departamento -----------------------------------------------------------
# población del departamento = suma de los píxeles WorldPop asignados a él (máscara zonal)
pop1 = pop.groupby("GID_1")["pop"].sum().rename("poblacion")
f1 = gpd.sjoin(fac.drop(columns=["GID_1", "NAME_1"], errors="ignore"), adm1[["GID_1", "geometry"]], how="left", predicate="within")
f1 = f1[~f1.index.duplicated()]
n_fac = f1.groupby("GID_1").size().rename("n_instalaciones")

def mean_nn_km(gid):
    """Distancia promedio al vecino más cercano entre instalaciones del mismo departamento
    (k=2 porque el vecino 1 de cada punto es él mismo)."""
    pts = xy(f1[f1["GID_1"] == gid])
    if len(pts) < 2:
        return np.nan
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1].mean() / 1000

tab = adm1[["GID_1", "NAME_1"]].copy()
tab["area_km2"] = adm1.geometry.area.values / 1e6
tab = tab.join(n_fac, on="GID_1").join(pop1, on="GID_1")
tab["n_instalaciones"] = tab["n_instalaciones"].fillna(0).astype(int)
tab["inst_por_100k_hab"] = tab["n_instalaciones"] / tab["poblacion"] * 1e5
tab["dist_media_vecino_km"] = tab["GID_1"].map(mean_nn_km)
tab = tab.sort_values("inst_por_100k_hab")
tab.round(2).to_csv(OUT / "t1_2_estadisticas_departamentos.csv", index=False)
pd.set_option("display.width", 200)
print("== 1.2c Estadísticas por departamento (ordenadas por densidad ascendente) ==")
print(tab.drop(columns="GID_1").round(2).to_string(index=False))
print("\nTres departamentos con menor densidad de cobertura:")
print(tab.head(3)[["NAME_1", "inst_por_100k_hab"]].round(2).to_string(index=False))
print(f"\nPoblación total asignada: {tab.poblacion.sum():,.0f}; área total: {tab.area_km2.sum():,.0f} km²")

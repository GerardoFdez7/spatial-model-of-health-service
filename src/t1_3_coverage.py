"""Task 1.3 - Buffers de cobertura y población cubierta.

Relación con el documento: el buffer de radio d alrededor de cada instalación j
materializa el conjunto N_i = {j : dist(i,j) <= d}; la unión de buffers es el área donde y_i = 1;
la población cubierta es  sum_i d_i * y_i  (función objetivo del MCLP) con x_j = 1 para todo j existente.
"""
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from common import (add_north, add_scalebar, base_axes, covered_population, load_adm,
                    load_facilities, load_pop_points, municipalities, nearest_facility,
                    pop_raster_path, xy)
from config import CRS_GEO, CRS_PROJ, OUT, RADII_KM, REF_RADIUS_KM

adm1, adm2, fac, pop = load_adm(1), municipalities(), load_facilities(), load_pop_points()
territory = adm1.geometry.union_all()


def coverage_polygon(facilities: gpd.GeoDataFrame, radius_km: float):
    """1.3a: buffer circular (en metros, CRS proyectado) por instalación + unary_union."""
    buffers = facilities.geometry.buffer(radius_km * 1000, quad_segs=32)
    return shapely.union_all(buffers.values)


def population_in_polygon_raster(poly_proj) -> float:
    """1.3b: enmascara el ráster WorldPop con el polígono de cobertura y suma sus píxeles.
    El polígono (metros) se lleva al CRS del ráster (grados) para usar rasterio.features."""
    poly_geo = gpd.GeoSeries([poly_proj], crs=CRS_PROJ).to_crs(CRS_GEO).iloc[0]
    with rasterio.open(pop_raster_path()) as src:
        band = src.read(1, masked=True)
        inside = rasterio.features.geometry_mask([poly_geo], out_shape=band.shape,
                                                 transform=src.transform, invert=True)
        return float(band.filled(0)[inside].sum())


def run(name, fac_subset):
    total = pop["pop"].sum()
    rows, polys = [], {}
    dist, _ = nearest_facility(xy(pop), fac_subset)
    for r in RADII_KM:
        poly = coverage_polygon(fac_subset, r); polys[r] = poly
        cov_mask = population_in_polygon_raster(poly)          # método ráster (rasterio.mask)
        cov_pts = covered_population(pop, dist, r)             # método analítico (distancia <= r)
        rows.append({"radio_km": r, "poblacion_cubierta": round(cov_mask), "poblacion_total": round(total),
                     "porcentaje_cobertura": round(100 * cov_mask / total, 2),
                     "chequeo_por_distancia": round(cov_pts),
                     "area_cobertura_km2": round(poly.area / 1e6)})
    t = pd.DataFrame(rows)
    print(f"\n== 1.3b Cobertura ({name}, n={len(fac_subset)}) ==")
    print(t.to_string(index=False))
    return t, polys


if __name__ == "__main__":
    print("== 1.3a Buffers: instalaciones en CRS", fac.crs.to_string())
    t_all, polys = run("todas las instalaciones", fac)
    t_all.to_csv(OUT / "t1_3_cobertura.csv", index=False)
    # Análisis de sensibilidad: las farmacias no prestan atención clínica; ¿cuánto cambia la cobertura sin ellas?
    t_np, _ = run("sin farmacias", fac[fac["type"] != "pharmacy"])
    t_np.to_csv(OUT / "t1_3_cobertura_sin_farmacias.csv", index=False)

    # ---- 1.3c: mapa a 10 km por municipio -----------------------------------------------------------
    cov = polys[REF_RADIUS_KM]
    cover_frac = adm2.geometry.intersection(cov).area / adm2.geometry.area
    adm2["cov_frac"] = cover_frac.values
    adm2["cat"] = np.select([adm2.cov_frac >= 0.999, adm2.cov_frac <= 0.001], [2, 0], default=1)
    counts = adm2["cat"].value_counts().reindex([0, 1, 2]).fillna(0).astype(int)
    print(f"\n== 1.3c Municipios a {REF_RADIUS_KM} km: sin cobertura={counts[0]}, parcial={counts[1]}, completa={counts[2]}")
    colors = ["#b2182b", "#fee090", "#2166ac"]                 # divergente: rojo (sin cobertura) - blanco - azul
    fig, ax = base_axes((10, 10), f"Cobertura a {REF_RADIUS_KM} km por municipio (nivel 2)")
    adm2.plot(ax=ax, column="cat", cmap=ListedColormap(colors), edgecolor="#555", linewidth=0.25)
    adm1.boundary.plot(ax=ax, color="k", linewidth=0.7)
    ax.scatter(fac.geometry.x, fac.geometry.y, s=3, color="k", alpha=.6, zorder=5, linewidths=0)
    minx, miny, maxx, maxy = adm1.total_bounds
    ax.set_xlim(minx - 2e4, maxx + 2e4); ax.set_ylim(miny - 2e4, maxy + 2e4)
    add_scalebar(ax, 100); add_north(ax)
    labels = ["Sin cobertura", "Parcialmente cubierto", "Completamente cubierto"]
    ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=10, markerfacecolor=c, markeredgecolor="#555",
                              label=f"{l} ({counts[i]})") for i, (c, l) in enumerate(zip(colors, labels))] +
              [Line2D([], [], marker="o", ls="", ms=4, color="k", label="Instalación de salud")],
              loc="lower right", fontsize=9)
    fig.savefig(OUT / "t1_3_mapa_cobertura_10km.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    adm2[["GID_2", "label", "cov_frac", "cat"]].to_csv(OUT / "t1_3_municipios_cobertura_10km.csv", index=False)

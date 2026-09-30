"""Task 2.1 - Sensibilidad del análisis de buffer.

Idea central: el buffer supone que TODA la población a distancia <= d tiene acceso igual (y_i binario).
Aquí se mide qué tan lejos queda realmente la demanda no cubierta y cómo crece y_i al aumentar d.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree
import shapely

from common import (covered_population, load_facilities, load_pop_points, municipalities,
                    nearest_facility, xy)
from config import CURVE_RADII_KM, OUT, REF_RADIUS_KM, SPEED_KMH

adm2, fac, pop = municipalities(), load_facilities(), load_pop_points()
fac_xy = xy(fac)
pop_xy = xy(pop)
dist, nn_idx = nearest_facility(pop_xy, fac)              # dist_i = min_j dist(i, j)
pop["dist_m"] = dist
pop["uncovered"] = np.where(dist > REF_RADIUS_KM * 1000, pop["pop"], 0.0)   # d_i * (1 - y_i)

# ------------------------------------------------------------------------------------------- 2.1a
g = pop.groupby("GID_2")
mun = adm2.set_index("GID_2")[["label", "geometry"]].join(pd.DataFrame({
    "poblacion": g["pop"].sum(),
    "pob_sin_cobertura": g["uncovered"].sum(),
}))
mun = mun.dropna(subset=["poblacion"])
# centroide poblacional: punto donde se concentra la demanda del municipio (peso d_i)
cx = (pop["pop"] * pop_xy[:, 0]).groupby(pop["GID_2"]).sum() / mun["poblacion"]
cy = (pop["pop"] * pop_xy[:, 1]).groupby(pop["GID_2"]).sum() / mun["poblacion"]
d_c, i_c = cKDTree(fac_xy).query(np.column_stack([cx.reindex(mun.index), cy.reindex(mun.index)]))
mun["dist_centroide_km"] = d_c / 1000
mun["tipo_mas_cercana"] = fac["type"].values[i_c]
# ¿y el hospital/clínica más cercano? (una farmacia no sustituye atención clínica)
clin = fac[fac["type"].isin(["hospital", "clinic"])]
c_tree = cKDTree(xy(clin))
d_cl, i_cl = c_tree.query(np.column_stack([cx.reindex(mun.index), cy.reindex(mun.index)]))
mun["dist_hosp_clinica_km"] = d_cl / 1000
mun["tipo_hosp_clinica"] = clin["type"].values[i_cl]
mun["dist_media_pond_km"] = (pop["pop"] * pop["dist_m"]).groupby(pop["GID_2"]).sum() / mun["poblacion"] / 1000

top5 = mun.sort_values("pob_sin_cobertura", ascending=False).head(5).copy()
print(f"== 2.1a Top 5 municipios por población sin cobertura a {REF_RADIUS_KM} km ==")

# ------------------------------------------------------------------------------------------- 2.1b
def min_radius_km(geom) -> float:
    """Radio mínimo R tal que el polígono queda contenido en la unión de discos de radio R:
        R*(P) = max_{p in P} min_j dist(p, j).
    El máximo de la distancia al punto más cercano se alcanza en (i) un vértice del borde o
    (ii) un vértice del diagrama de Voronoi de las instalaciones que caiga dentro del polígono."""
    boundary = shapely.segmentize(geom.boundary, 500)          # borde densificado cada 500 m
    pts = shapely.get_coordinates(boundary)
    cand = [pts]
    inside = vor.vertices[shapely.contains_xy(geom, vor.vertices[:, 0], vor.vertices[:, 1])]
    if len(inside):
        cand.append(inside)
    cand = np.vstack(cand)
    return cKDTree(fac_xy).query(cand)[0].max() / 1000


vor = Voronoi(fac_xy)
# Radio poblacional: cubre TODA la población del municipio (píxeles con pop>0); es más relevante que el geométrico
pop_r = pop.groupby("GID_2")["dist_m"].max() / 1000
top5["radio_min_geom_km"] = [min_radius_km(mun.loc[i, "geometry"]) for i in top5.index]
top5["radio_min_pob_km"] = pop_r.reindex(top5.index)
top5["tiempo_geom_min"] = top5["radio_min_geom_km"] / SPEED_KMH * 60
top5["tiempo_pob_min"] = top5["radio_min_pob_km"] / SPEED_KMH * 60
cols = ["label", "poblacion", "pob_sin_cobertura", "dist_centroide_km", "dist_media_pond_km",
        "tipo_mas_cercana", "dist_hosp_clinica_km", "tipo_hosp_clinica", "radio_min_pob_km", "radio_min_geom_km", "tiempo_pob_min", "tiempo_geom_min"]
out = top5[cols].round(1)
out.to_csv(OUT / "t2_1_top5_municipios.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.to_string())
print(f"\n== 2.1b Tiempo aproximado = radio / {SPEED_KMH} km/h (distancia recta; en carretera real x1.3 aprox.) ==")

# ------------------------------------------------------------------------------------------- 2.1c
total = pop["pop"].sum()
curve = pd.DataFrame({"radio_km": CURVE_RADII_KM})
# y_i(d) = [dist_i <= d]; cobertura(d) = sum_i d_i y_i(d) / sum_i d_i
curve["pct_cubierta"] = [100 * covered_population(pop, dist, r) / total for r in CURVE_RADII_KM]
curve["ganancia_marginal_pp_por_km"] = curve["pct_cubierta"].diff().fillna(curve["pct_cubierta"])
# Punto de inflexión 1 (rodilla): máxima distancia a la cuerda que une los extremos (criterio Kneedle)
x = (curve["radio_km"] - curve["radio_km"].min()) / (curve["radio_km"].max() - curve["radio_km"].min())
y = (curve["pct_cubierta"] - curve["pct_cubierta"].min()) / (curve["pct_cubierta"].max() - curve["pct_cubierta"].min())
knee = int(curve["radio_km"].iloc[int(np.argmax(y - x))])
# Criterio 2: primer radio donde ganar 1 km más aporta < 1 punto porcentual
r_1pp = int(curve[curve["ganancia_marginal_pp_por_km"] < 1.0]["radio_km"].iloc[0])
curve.round(3).to_csv(OUT / "t2_1_curva_cobertura.csv", index=False)
pk = lambda r: float(curve.set_index("radio_km").loc[r, "pct_cubierta"])
print(f"== 2.1c Curva: rodilla (Kneedle) r={knee} km "
      f"({pk(knee):.1f} %); ganancia <1 pp/km desde r={r_1pp} km ({pk(r_1pp):.1f} %)")
print(f"   Cobertura a 5/10/20/30/50 km: " + ", ".join(f"{r}km={pk(r):.1f}%" for r in (5, 10, 20, 30, 50)))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(curve.radio_km, curve.pct_cubierta, color="#2166ac", lw=2.2)
ax[0].axvline(knee, color="#b2182b", ls="--"); ax[0].axvline(r_1pp, color="#4d4d4d", ls=":")
ax[0].annotate(f"Rodilla: {knee} km ({pk(knee):.0f} %)", (knee, pk(knee)), (knee + 6, pk(knee) - 35),
               arrowprops=dict(arrowstyle="->"), color="#b2182b")
ax[0].annotate(f"Ganancia < 1 pp/km desde {r_1pp} km", (r_1pp, pk(r_1pp)), (r_1pp + 3, pk(r_1pp) - 55),
               arrowprops=dict(arrowstyle="->"), color="#4d4d4d")
ax[0].set(xlabel="Radio de buffer (km)", ylabel="Población cubierta (%)", title="Curva de cobertura acumulada",
          ylim=(0, 102), xlim=(0, 50)); ax[0].grid(alpha=.3)
ax[1].bar(curve.radio_km, curve.ganancia_marginal_pp_por_km, color="#2166ac")
ax[1].axhline(1, color="#4d4d4d", ls=":")
ax[1].set(xlabel="Radio de buffer (km)", ylabel="Ganancia marginal (pp por km)", title="Rendimiento marginal decreciente")
ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "t2_1_curva_cobertura.png", dpi=160); plt.close(fig)

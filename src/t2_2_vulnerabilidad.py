# Task 2.2: índice compuesto de vulnerabilidad de acceso hospitalario por condado y mapa por quintiles.
# IVA_c = w1*C1_c + w2*C2_c + w3*C3_c, con w1 + w2 + w3 = 1 y cada C_k normalizado a [0, 1].
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from common import (add_north, add_scalebar, base_axes, frame, load_counties, load_hospitals, load_state,
                    minmax, nearest, xy)
from config import OCC_RADIUS_KM, OUT, PROC, STATE_NAME, WEIGHTS

pd.set_option("display.width", 220)
cond, hosp, est = load_counties(), load_hospitals(), load_state()
stats = pd.read_csv(OUT / "t1_2_estadisticas_condados.csv", dtype={"fips": str})
cond = cond.merge(stats[["fips", "hospitales", "camas_total", "camas_10k"]], on="fips")
cent = cond.geometry.centroid

# Componente 1: distancia del centroide al hospital más cercano (km)
cond["dist_km"] = nearest(xy(cent), hosp)[0] / 1000

# Componente 2: inverso de camas per cápita, P_c / camas_c; sin hospitales -> máximo observado
inv = cond["poblacion"] / cond["camas_total"].replace(0, np.nan)
cond["hab_por_cama"] = inv.fillna(inv.max())

# Componente 3: ocupación promedio de hospitales a <= 50 km; sin hospitales cerca -> máximo observado
occ = hosp["ocupacion_total"].values
vecinos = cKDTree(xy(hosp)).query_ball_point(xy(cent), r=OCC_RADIUS_KM * 1000)
cond["ocup_50km"] = [np.nanmean(occ[v]) if len(v) and np.isfinite(occ[v]).any() else np.nan for v in vecinos]
sin50 = cond["ocup_50km"].isna()
cond["ocup_50km"] = cond["ocup_50km"].fillna(cond["ocup_50km"].max())
print(f"- 2.2 Hospitales sin dato de ocupación: {np.isnan(occ).sum()}")
print(f"- 2.2 Condados sin hospital a <= {OCC_RADIUS_KM} km: {sin50.sum()} -> {', '.join(cond.loc[sin50, 'nombre'])}")

# 2.2a Normalización min-max: C' = (C - min) / (max - min)
cond["C1"] = minmax(cond["dist_km"])
cond["C2"] = minmax(cond["hab_por_cama"])
cond["C3"] = minmax(cond["ocup_50km"])
print("\n- 2.2a Componentes crudos y normalizados")
print(cond[["dist_km", "hab_por_cama", "ocup_50km", "C1", "C2", "C3"]].describe().round(3).to_string())
print("- 2.2a Correlación de Spearman entre componentes")
print(cond[["C1", "C2", "C3"]].corr(method="spearman").round(2).to_string())

# 2.2b Índice ponderado
w = WEIGHTS
cond["IVA"] = w["C1_dist"] * cond["C1"] + w["C2_camas"] * cond["C2"] + w["C3_ocup"] * cond["C3"]
print(f"\n- 2.2b Pesos: {w}")

# 2.2c Quintiles y ranking
labels = ["Muy baja", "Baja", "Media", "Alta", "Muy alta"]
cond["quintil"] = pd.qcut(cond["IVA"], 5, labels=labels)
cond["rank"] = cond["IVA"].rank(ascending=False, method="first").astype(int)
cols = ["rank", "fips", "nombre", "poblacion", "dist_km", "hab_por_cama", "ocup_50km", "C1", "C2", "C3", "IVA",
        "quintil", "hospitales"]
tabla = cond[cols].sort_values("rank")
tabla.round(3).to_csv(OUT / "t2_2_indice_vulnerabilidad.csv", index=False)
print("\n- 2.2c Índice de vulnerabilidad (ordenado)")
print(tabla.round(3).to_string(index=False))
print("- 2.2c Límites de los quintiles:", np.round(cond["IVA"].quantile([0, .2, .4, .6, .8, 1]).values, 3))

# 2.2c Mapa coroplético con los diez condados más vulnerables etiquetados
colors = ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"]
fig, ax = base_axes((8.5, 11), f"{STATE_NAME}: índice de vulnerabilidad de acceso hospitalario\n"
                               f"(quintiles; pesos C1={w['C1_dist']}, C2={w['C2_camas']}, C3={w['C3_ocup']})")
cond["q_code"] = cond["quintil"].cat.codes
cond.plot(ax=ax, column="q_code", cmap=ListedColormap(colors), vmin=0, vmax=4, edgecolor="#555", linewidth=0.5)
est.boundary.plot(ax=ax, color="k", linewidth=1.2)
ax.scatter(hosp.geometry.x, hosp.geometry.y, s=10, color="#1f3b73", zorder=5, linewidths=0)
texts = []
for _, r in cond[cond["rank"] <= 10].iterrows():
    p = r.geometry.representative_point()
    texts.append(ax.text(p.x, p.y, f"{r['rank']}. {r['nombre']}", fontsize=8, fontweight="bold", ha="center",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.8), zorder=6))
adjust_text(texts, ax=ax)
frame(ax); add_scalebar(ax, 100, loc=(0.06, 0.03)); add_north(ax)
cnt = cond["quintil"].value_counts().reindex(labels)
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=11, markerfacecolor=c, markeredgecolor="#555",
                          label=f"{l} ({cnt[l]})") for c, l in zip(colors, labels)] +
          [Line2D([], [], marker="o", ls="", ms=4, color="#1f3b73", label="Hospital")],
          loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=8.5, title="Vulnerabilidad (quintil)")
ax.text(0.01, -0.01, "Etiquetas: los 10 condados con índice más alto (1 = más vulnerable)",
        transform=ax.transAxes, fontsize=8)
fig.savefig(OUT / "t2_2_mapa_vulnerabilidad.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# Capa con el índice, usada en t3_1 y t3_3
cond.drop(columns=["q_code"]).to_file(PROC / "condados_indice.gpkg", driver="GPKG")

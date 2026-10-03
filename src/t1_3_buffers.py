# Task 1.3: buffers de 10, 25 y 50 km, población cubierta por radio y mapa de cobertura a 25 km.
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from common import (add_north, add_scalebar, base_axes, coverage_polygon, covered_fraction, frame,
                    load_counties, load_hospitals, load_state, state_polygon)
from config import FULL_TOL, OUT, RADII_KM, REF_RADIUS_KM, STATE_NAME

cond, hosp, est = load_counties(), load_hospitals(), load_state()
P = cond["poblacion"].sum()
state = state_polygon()

# 1.3a y 1.3b Buffers unidos por radio y población cubierta
# Pcub(S) = sum_c P_c * área(c ∩ U_S) / área(c),   cobertura % = 100 * Pcub(S) / sum_c P_c
rows, polys = [], {}
for r in RADII_KM:
    U = coverage_polygon(hosp, r)
    polys[r] = U
    frac = covered_fraction(cond, U)
    pcub = float((cond["poblacion"] * frac).sum())
    rows.append({"radio_km": r, "poblacion_cubierta": round(pcub), "poblacion_total": int(P),
                 "porcentaje_cobertura": round(100 * pcub / P, 2),
                 "condados_completos": int((frac >= FULL_TOL).sum()),
                 "condados_sin_cobertura": int((frac <= 1 - FULL_TOL).sum())})
    cond[f"frac_{r}"] = frac.values
tabla = pd.DataFrame(rows)
tabla.to_csv(OUT / "t1_3_cobertura.csv", index=False)
print("- 1.3b Cobertura poblacional por radio")
print(tabla.to_string(index=False))

# 1.3c Clasificación de condados a 25 km
f = cond[f"frac_{REF_RADIUS_KM}"]
cond["cat"] = np.select([f >= FULL_TOL, f <= 1 - FULL_TOL], [2, 0], default=1)
counts = cond["cat"].value_counts().reindex([0, 1, 2]).fillna(0).astype(int)
print(f"\n- 1.3c Condados a {REF_RADIUS_KM} km: sin cobertura = {counts[0]}, parcial = {counts[1]}, "
      f"completa = {counts[2]}")
out = cond[["fips", "nombre", "poblacion"] + [f"frac_{r}" for r in RADII_KM] + ["cat"]].copy()
out = out.sort_values(f"frac_{REF_RADIUS_KM}")
out.round(3).to_csv(OUT / "t1_3_condados_cobertura.csv", index=False)
print(out.round(3).to_string(index=False))

# 1.3c Mapa con escala divergente; la clase "parcial" se subdivide según la fracción cubierta
bins = [-0.01, 1 - FULL_TOL, 0.25, 0.50, 0.75, FULL_TOL, 1.01]
colors = ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"]
labels = ["No cubierto (< 0.1 %)", "Parcial: 0.1-25 %", "Parcial: 25-50 %", "Parcial: 50-75 %",
          "Parcial: 75-99.9 %", "Completamente cubierto (>= 99.9 %)"]
cond["clase"] = pd.cut(f, bins, labels=False, right=False)
nclase = cond["clase"].value_counts().reindex(range(6)).fillna(0).astype(int)
fig, ax = base_axes((8.5, 11), f"{STATE_NAME}: cobertura hospitalaria con buffer de {REF_RADIUS_KM} km\n"
                               f"(fracción del área de cada condado dentro de la cobertura)")
cond.plot(ax=ax, column="clase", cmap=ListedColormap(colors), vmin=0, vmax=5, edgecolor="#555", linewidth=0.5)
gpd.GeoSeries([polys[REF_RADIUS_KM].intersection(state)], crs=cond.crs).boundary.plot(
    ax=ax, color="#08306b", linewidth=0.8, linestyle="--")
est.boundary.plot(ax=ax, color="k", linewidth=1.3)
ax.scatter(hosp.geometry.x, hosp.geometry.y, s=18, color="k", edgecolor="w", linewidth=0.4, zorder=5)
frame(ax); add_scalebar(ax, 100, loc=(0.06, 0.03)); add_north(ax)
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=11, markerfacecolor=c, markeredgecolor="#555",
                          label=f"{l} ({nclase[i]})") for i, (c, l) in enumerate(zip(colors, labels))] +
          [Line2D([], [], color="#08306b", ls="--", label=f"Borde del área a {REF_RADIUS_KM} km"),
           Line2D([], [], marker="o", ls="", ms=5, markerfacecolor="k", markeredgecolor="w", label="Hospital")],
          loc="upper right", bbox_to_anchor=(1.0, 0.84), fontsize=8, title="Condados (n)", title_fontsize=8.5)
print("- 1.3c Condados por clase de cobertura:", dict(zip(labels, nclase)))
pct = tabla.loc[tabla.radio_km == REF_RADIUS_KM, "porcentaje_cobertura"].iloc[0]
ax.text(0.01, -0.01, f"Población cubierta a {REF_RADIUS_KM} km: {pct:.1f} %", transform=ax.transAxes, fontsize=9)
fig.savefig(OUT / "t1_3_mapa_cobertura_25km.png", dpi=170, bbox_inches="tight"); plt.close(fig)

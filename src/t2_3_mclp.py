# Task 2.3: MCLP simplificado; elige 3 sitios para hospitales nuevos con un algoritmo voraz (radio S = 25 km).
# MCLP: max sum_i d_i y_i  s.a.  y_i <= sum_{j in N_i} x_j,  sum_j x_j = p,  x, y binarias.
import itertools

import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from common import (add_north, add_scalebar, base_axes, coverage_polygon, covered_population, frame,
                    load_counties, load_hospitals, load_state, state_polygon)
from config import FULL_TOL, GRID_STEP_M, N_NEW, OUT, REF_RADIUS_KM, STATE_NAME

pd.set_option("display.width", 220)
cond, hosp, est = load_counties(), load_hospitals(), load_state()
state = state_polygon()
P = cond["poblacion"].sum()
R = REF_RADIUS_KM * 1000
U0 = coverage_polygon(hosp, REF_RADIUS_KM)
dens = (cond["poblacion"] / cond.geometry.area).values   # densidad uniforme por condado (hab/m²)


# Población adicional del candidato: g(k | U) = sum_c P_c * área(B_k ∩ (c \ U)) / área(c).
def gain(buf, U):
    unc = cond.geometry.difference(U)
    return float((unc.intersection(buf).area.values * dens).sum())


# 2.3a Grilla regular de 50 km recortada al estado
minx, miny, maxx, maxy = est.total_bounds
xs = np.arange(minx + GRID_STEP_M / 2, maxx, GRID_STEP_M)
ys = np.arange(miny + GRID_STEP_M / 2, maxy, GRID_STEP_M)
grid = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*[a.ravel() for a in np.meshgrid(xs, ys)]), crs=cond.crs)
cand = gpd.sjoin(grid, est[["geometry"]], how="inner", predicate="intersects").drop(columns="index_right")
cand = cand.reset_index(drop=True)
print(f"- 2.3a Grilla de {GRID_STEP_M / 1000:.0f} km: {len(grid)} puntos en el rectángulo; {len(cand)} dentro del estado")

# 2.3b Población adicional de cada candidato con la cobertura actual
cand["buffer"] = cand.geometry.buffer(R, quad_segs=32)
pcov0 = covered_population(cond, U0)
print(f"\n- 2.3b Cobertura actual a {REF_RADIUS_KM} km: {pcov0:,.0f} hab ({100 * pcov0 / P:.2f} %); "
      f"sin cobertura: {P - pcov0:,.0f} hab")
cand["ganancia_inicial"] = [gain(b, U0) for b in cand["buffer"]]
cj = gpd.sjoin(cand[["geometry"]], cond[["nombre", "geometry"]], how="left", predicate="within")
cand["condado"] = cj.groupby(level=0)["nombre"].first().reindex(cand.index).fillna("(borde)")
print(cand["ganancia_inicial"].describe().round(0).to_string())

# 2.3c Algoritmo voraz: k* = argmax g(k | U); U <- U ∪ B_k*; se repite p = 3 veces
U, sel, rows = U0, [], []
for it in range(1, N_NEW + 1):
    g = np.array([gain(b, U) if k not in sel else -1 for k, b in enumerate(cand["buffer"])])
    k = int(g.argmax()); sel.append(k)
    U = shapely.union(U, cand.loc[k, "buffer"])
    pc = covered_population(cond, U)
    rows.append({"iteracion": it, "condado": cand.loc[k, "condado"], "pob_adicional": round(g[k]),
                 "pob_cubierta_total": round(pc), "pct_cobertura": round(100 * pc / P, 2)})
res = pd.DataFrame(rows)
ll = gpd.GeoSeries(cand.geometry[sel].values, crs=cond.crs).to_crs(4326)
res.insert(2, "lon", ll.x.round(4).values); res.insert(3, "lat", ll.y.round(4).values)
res.to_csv(OUT / "t2_3_greedy.csv", index=False)
print("\n- 2.3c Selección voraz")
print(res.to_string(index=False))
print(f"- 2.3c Cobertura: {100 * pcov0 / P:.2f} % -> {res.pct_cobertura.iloc[-1]:.2f} % "
      f"(+{res.pob_adicional.sum():,.0f} hab)")

# Condados de donde sale la población adicional de cada punto
Uk = U0
for i, k in enumerate(sel, 1):
    part = cond.geometry.difference(Uk).intersection(cand.loc[k, "buffer"]).area.values * dens
    d = pd.Series(part, index=cond["nombre"]).sort_values(ascending=False)
    print(f"- 2.3c Punto {i}: " + ", ".join(f"{n} {v:,.0f}" for n, v in d[d > 100].items()))
    Uk = shapely.union(Uk, cand.loc[k, "buffer"])

# Con separación 50 km = 2S los buffers no se traslapan, así que el óptimo es la suma de las 3 mayores ganancias
b = cand["buffer"].values
ov = max(shapely.intersection(b[i], b[j]).area for i, j in itertools.combinations(range(len(b)), 2)
         if cand.geometry[i].distance(cand.geometry[j]) < 2 * R + 1)
top3 = cand["ganancia_inicial"].nlargest(3).sum()
print(f"- 2.3c Traslape máximo entre candidatos: {ov:.3f} m²; óptimo = {top3:,.0f} hab; "
      f"voraz = {res.pob_adicional.sum():,.0f} hab")

# 2.3c Mapa de cobertura final con los tres hospitales propuestos
frac = cond.geometry.intersection(U).area / cond.geometry.area
cat = np.select([frac >= FULL_TOL, frac <= 1 - FULL_TOL], [2, 0], default=1)
colors = ["#b2182b", "#f7f7f7", "#2166ac"]
fig, ax = base_axes((8.5, 11), f"{STATE_NAME}: MCLP voraz, {N_NEW} hospitales nuevos (S = {REF_RADIUS_KM} km)")
cond.assign(cat=cat).plot(ax=ax, column="cat", cmap=ListedColormap(colors), vmin=0, vmax=2, edgecolor="#777",
                          linewidth=0.4)
gpd.GeoSeries([U0.intersection(state)], crs=cond.crs).plot(ax=ax, color="#2166ac", alpha=.25, linewidth=0)
gpd.GeoSeries([cand.loc[k, "buffer"] for k in sel], crs=cond.crs).plot(ax=ax, facecolor="#1a9850", alpha=.35,
                                                                        edgecolor="#1a9850", linewidth=1.5)
est.boundary.plot(ax=ax, color="k", linewidth=1.2)
ax.scatter(cand.geometry.x, cand.geometry.y, s=6, color="#888", marker="+", zorder=4)
ax.scatter(hosp.geometry.x, hosp.geometry.y, s=14, color="k", zorder=5)
for i, k in enumerate(sel, 1):
    x, y = cand.geometry.x[k], cand.geometry.y[k]
    ax.scatter(x, y, s=220, marker="*", color="#1a9850", edgecolor="k", zorder=7)
    ax.annotate(f"{i}. {cand.loc[k, 'condado']}\n+{res.pob_adicional[i - 1]:,.0f} hab", (x, y), xytext=(10, 8),
                textcoords="offset points", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1a9850", alpha=.9), zorder=8)
frame(ax); add_scalebar(ax, 100, loc=(0.06, 0.03)); add_north(ax)
ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=10, markerfacecolor=c, markeredgecolor="#777", label=l)
                   for c, l in zip(colors, ["Condado no cubierto", "Parcialmente cubierto", "Completamente cubierto"])] +
          [Line2D([], [], marker="s", ls="", ms=10, markerfacecolor="#2166ac", alpha=.35, label="Cobertura actual 25 km"),
           Line2D([], [], marker="s", ls="", ms=10, markerfacecolor="#1a9850", alpha=.5, label="Cobertura nueva"),
           Line2D([], [], marker="*", ls="", ms=13, markerfacecolor="#1a9850", markeredgecolor="k", label="Hospital propuesto"),
           Line2D([], [], marker="o", ls="", ms=4, color="k", label="Hospital existente"),
           Line2D([], [], marker="+", ls="", ms=6, color="#888", label=f"Candidato (grilla {GRID_STEP_M // 1000} km)")],
          loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=8)
ax.text(0.01, -0.01, f"Cobertura: {100 * pcov0 / P:.1f} % -> {res.pct_cobertura.iloc[-1]:.1f} %",
        transform=ax.transAxes, fontsize=9)
fig.savefig(OUT / "t2_3_mapa_mclp.png", dpi=170, bbox_inches="tight"); plt.close(fig)

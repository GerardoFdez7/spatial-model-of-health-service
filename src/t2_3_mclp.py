"""Task 2.3 - MCLP simplificado (Maximum Coverage Location Problem) con heurística voraz.

    max  sum_i d_i y_i     s.a.   y_i <= sum_{j in N_i} x_j   para todo i,     sum_j x_j <= p,   x_j, y_i in {0,1}

  i  : celda de demanda = píxel WorldPop sin cobertura actual (d_i = su población)
  j  : punto candidato de la grilla de 20 km (x_j = 1 si se instala una instalación nueva)
  N_i: candidatos a distancia <= 10 km de la celda i
  p  : 5 instalaciones nuevas
Las instalaciones existentes ya están 'instaladas' (x_j = 1 fijo): por eso la demanda i sólo entra al
problema si NO está cubierta hoy, y el objetivo mide la población ADICIONAL cubierta.
"""
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, vstack
from scipy.spatial import cKDTree

from common import (add_north, add_scalebar, base_axes, load_adm, load_facilities, load_pop_points,
                    nearest_facility, xy)
from config import CRS_PROJ, GRID_STEP_M, N_NEW, OUT, REF_RADIUS_KM

adm1, adm2, fac, pop = load_adm(1), load_adm(2), load_facilities(), load_pop_points()
R = REF_RADIUS_KM * 1000
total = pop["pop"].sum()

# ------------------------------------------------------------------------------------------- 2.3a
minx, miny, maxx, maxy = adm1.total_bounds
gx, gy = np.meshgrid(np.arange(minx, maxx + GRID_STEP_M, GRID_STEP_M),
                     np.arange(miny, maxy + GRID_STEP_M, GRID_STEP_M))
cand = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=CRS_PROJ)
n_grid = len(cand)
land = adm1.geometry.union_all()                                          # intersección con el nivel 1
water = adm2[adm2["ENGTYPE_2"] != "Municipality"].geometry.union_all()    # lagos de Izabal y Atitlán
keep = shapely.contains_xy(land, cand.geometry.x.values, cand.geometry.y.values) & \
    ~shapely.contains_xy(water, cand.geometry.x.values, cand.geometry.y.values)
cand = cand[keep].reset_index(drop=True)
print(f"== 2.3a Grilla de {GRID_STEP_M/1000:.0f} km: {n_grid} puntos -> {len(cand)} dentro del territorio y fuera de agua")

# ------------------------------------------------------------------------------------------- 2.3b
dist, _ = nearest_facility(xy(pop), fac)
unc = pop[dist > R].reset_index(drop=True)           # demanda i sin cobertura: y_i = 0 hoy
covered_now = total - unc["pop"].sum()
print(f"   Cobertura actual a {REF_RADIUS_KM} km: {100*covered_now/total:.2f} %  "
      f"(sin cobertura: {unc['pop'].sum():,.0f} hab en {len(unc)} celdas)")
d = unc["pop"].values
neigh = cKDTree(xy(unc)).query_ball_point(xy(cand), r=R)      # celdas i que cubriría el candidato j
gain0 = np.array([d[ix].sum() for ix in neigh])
cand["pob_adicional"] = gain0
print(f"   Población adicional por candidato: máx {gain0.max():,.0f}; mediana {np.median(gain0):,.0f}; "
      f"candidatos con ganancia 0: {(gain0 == 0).sum()}")

# ------------------------------------------------------------------------------------------- 2.3c
uncovered = np.ones(len(unc), dtype=bool)               # estado del mapa de cobertura (y_i = 0)
selected, rows, cum = [], [], 0.0
for it in range(1, N_NEW + 1):
    gains = np.array([d[ix][uncovered[ix]].sum() for ix in neigh])   # ganancia marginal actual
    gains[selected] = -1
    j = int(gains.argmax())                                          # paso voraz: mayor población nueva
    selected.append(j)
    uncovered[neigh[j]] = False                                      # actualizar mapa de cobertura
    cum += gains[j]
    pt = cand.geometry.iloc[j]
    rows.append({"iter": it, "x_utm": round(pt.x), "y_utm": round(pt.y),
                 "pob_adicional": round(gains[j]), "pob_adicional_acum": round(cum),
                 "cobertura_total_pct": round(100 * (covered_now + cum) / total, 2)})
res = pd.DataFrame(rows)
sel = cand.iloc[selected].copy()
sj = gpd.sjoin(sel, adm2[["NAME_1", "NAME_2", "geometry"]], how="left", predicate="within")
res["departamento"] = sj["NAME_1"].values
res["municipio"] = sj["NAME_2"].values
res.to_csv(OUT / "t2_3_greedy.csv", index=False)
pd.set_option("display.width", 220)
print("\n== 2.3c Selección voraz de 5 puntos ==")
print(res.to_string(index=False))
print(f"\nCobertura: {100*covered_now/total:.2f} % -> {res.cobertura_total_pct.iloc[-1]:.2f} % "
      f"(+{res.pob_adicional_acum.iloc[-1]:,.0f} hab)")

# ---- Verificación: solución óptima del MCLP con MILP (HiGHS) --------------------------------------
# Solo entran las celdas cubribles por algún candidato; y_i se relaja a [0,1] (con x binario, y es entero en el óptimo).
active = np.unique(np.concatenate([np.asarray(n, dtype=int) for n in neigh]))
pos = {int(i): k for k, i in enumerate(active)}
ri, ci = [], []
for j, ix in enumerate(neigh):
    for i in ix:
        ri.append(pos[i])
        ci.append(j)
A_x = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(active), len(cand)))   # 1 si j in N_i
n_y = len(active)
A = vstack([hstack([identity(n_y), -A_x]),                                       # y_i - sum_{j in N_i} x_j <= 0
            hstack([csr_matrix((1, n_y)), csr_matrix(np.ones((1, len(cand))))])])  # sum_j x_j <= p
ub = np.r_[np.zeros(n_y), N_NEW]
c = -np.r_[d[active], np.zeros(len(cand))]                                       # maximizar sum d_i y_i
integrality = np.r_[np.zeros(n_y), np.ones(len(cand))]
sol = milp(c, constraints=LinearConstraint(A.tocsr(), ub=ub), integrality=integrality,
           bounds=Bounds(0, 1), options={"time_limit": 300, "mip_rel_gap": 1e-4})
if sol.x is not None:
    opt_gain = -sol.fun
    print(f"\nVerificación MILP (HiGHS): '{sol.message}'  óptimo = {opt_gain:,.0f} hab adicionales; "
          f"voraz = {cum:,.0f} ({100*cum/opt_gain:.1f} % del óptimo; garantía teórica >= 63.2 % = 1-1/e)")
    print(f"   Cobertura total óptima: {100*(covered_now+opt_gain)/total:.2f} %")
    pd.Series({"optimo": opt_gain, "voraz": cum, "razon": cum / opt_gain, "status": sol.message}
              ).to_csv(OUT / "t2_3_milp_check.csv")

# ---- mapa -------------------------------------------------------------------------------------------
fig, ax = base_axes((10, 10), f"MCLP voraz: 5 nuevas instalaciones (radio {REF_RADIUS_KM} km)")
adm2.plot(ax=ax, color="#eeeeee", edgecolor="#cccccc", linewidth=.2)
cov_now = shapely.union_all(fac.geometry.buffer(R, quad_segs=24).values)
gpd.GeoSeries([cov_now], crs=CRS_PROJ).plot(ax=ax, color="#9ecae1", alpha=.7)
adm1.boundary.plot(ax=ax, color="k", linewidth=.6)
gpd.GeoSeries(sel.geometry.buffer(R, quad_segs=24).values, crs=CRS_PROJ).plot(
    ax=ax, color="#fdae61", alpha=.65, edgecolor="#b35806")
ax.scatter(cand.geometry.x, cand.geometry.y, s=4, color="#636363", zorder=4)
ax.scatter(fac.geometry.x, fac.geometry.y, s=2, color="#08519c", zorder=5)
for k, (_, r) in enumerate(sel.iterrows(), 1):
    ax.scatter(r.geometry.x, r.geometry.y, s=150, marker="*", color="#d7191c", edgecolor="k", zorder=8)
    ax.text(r.geometry.x + 6000, r.geometry.y + 6000, str(k), fontsize=12, fontweight="bold", zorder=9)
ax.set_xlim(minx - 2e4, maxx + 2e4)
ax.set_ylim(miny - 2e4, maxy + 2e4)
add_scalebar(ax, 100)
add_north(ax)
ax.legend(handles=[Patch(fc="#9ecae1", label=f"Cobertura actual ({100*covered_now/total:.1f} %)"),
                   Patch(fc="#fdae61", ec="#b35806", label="Cobertura adicional (5 nuevas)"),
                   Line2D([], [], marker="*", ls="", ms=13, mfc="#d7191c", mec="k", label="Sitio seleccionado"),
                   Line2D([], [], marker="o", ls="", ms=3, color="#636363", label="Candidatos (grilla 20 km)")],
          loc="lower right", fontsize=9)
fig.savefig(OUT / "t2_3_mapa_mclp.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Task 3.3: sensibilidad del índice de vulnerabilidad a los pesos y diagrama del caso adverso del algoritmo voraz.
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from config import OUT, PROC, WEIGHTS

pd.set_option("display.width", 220)
idx = gpd.read_file(PROC / "condados_indice.gpkg")
C = idx[["C1", "C2", "C3"]].values

# 3.3a Índice IVA = w1*C1 + w2*C2 + w3*C3 recalculado con cinco configuraciones de pesos
configs = {"Base (0.45/0.35/0.20)": (WEIGHTS["C1_dist"], WEIGHTS["C2_camas"], WEIGHTS["C3_ocup"]),
           "Iguales (1/3 c/u)": (1 / 3, 1 / 3, 1 / 3),
           "Distancia (0.70/0.15/0.15)": (0.70, 0.15, 0.15),
           "Camas (0.20/0.60/0.20)": (0.20, 0.60, 0.20),
           "Ocupación (0.20/0.20/0.60)": (0.20, 0.20, 0.60)}
ranks = pd.DataFrame(index=idx["nombre"])
for name, w in configs.items():
    s = pd.Series(C @ np.array(w), index=idx["nombre"])
    ranks[name] = s.rank(ascending=False, method="first").astype(int)

# 3.3a Estabilidad frente a la base: top-10 y top-5 en común, Spearman (rho) y Kendall (tau)
base = list(configs)[0]
top_base = set(ranks.index[ranks[base] <= 10])
rows = []
for name in configs:
    top = set(ranks.index[ranks[name] <= 10])
    rows.append({"configuracion": name, "top10_comunes_con_base": len(top & top_base),
                 "top5_comunes_con_base": len(set(ranks.index[ranks[name] <= 5]) &
                                              set(ranks.index[ranks[base] <= 5])),
                 "spearman_rho": round(spearmanr(ranks[base], ranks[name])[0], 3),
                 "kendall_tau": round(kendalltau(ranks[base], ranks[name])[0], 3),
                 "entran": ", ".join(sorted(top - top_base)) or "-",
                 "salen": ", ".join(sorted(top_base - top)) or "-"})
stab = pd.DataFrame(rows)
stab.to_csv(OUT / "t3_3a_estabilidad.csv", index=False)
union_top = ranks[(ranks <= 10).any(axis=1)].sort_values(base)
union_top.to_csv(OUT / "t3_3a_rankings.csv")
print("- 3.3a Rankings (condados que aparecen en algún top-10)")
print(union_top.to_string())
print("\n- 3.3a Estabilidad respecto de la configuración base")
print(stab.to_string(index=False))

# 3.3b Diagrama del caso adverso: con p = 2, el sitio de compromiso M (80 hab) atrae al voraz,
# que llega a 100 hab, mientras que el óptimo A' + B' cubre 120 hab
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_aspect("equal"); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-2.3, 1.75); ax.axis("off")
r = 1.3
dots = {"A": [(-2.6, .3), (-2.6, -.3), (-1.2, .35), (-1.2, -.35), (-.9, .35), (-.9, -.35)]}
dots["B"] = [(-x, y) for x, y in dots["A"]]
for k, pts in dots.items():
    xs, ys = zip(*pts)
    ax.add_patch(mpatches.FancyBboxPatch((min(xs) - .25, -.6), max(xs) - min(xs) + .5, 1.2,
                                         boxstyle="round,pad=0.05", fc="#fdae61", ec="none", alpha=.35, zorder=0))
    ax.scatter(xs, ys, s=70, color="#e66101", edgecolor="k", zorder=4)
    cx = np.mean([min(xs), max(xs)])
    ax.add_patch(plt.Circle((cx, 0), r, fill=False, ec="#1a9850", lw=2, ls="--", zorder=3))
    ax.plot(cx, 0, marker="P", ms=13, color="#1a9850", mec="k", zorder=5)
    ax.text(cx, -1.45, f"{k}'  (centro del pueblo {k})\ncubre 6 puntos = 60 hab", ha="center", va="top",
            fontsize=9.5, color="#1a9850", fontweight="bold")
    ax.text(cx, .78, f"Pueblo {k}", ha="center", va="bottom", fontsize=10)
ax.add_patch(plt.Circle((0, 0), r, fill=False, ec="#d62728", lw=2.6, zorder=3))
ax.plot(0, 0, marker="*", ms=20, color="#d62728", mec="k", zorder=5)
ax.text(0, -1.45, "M (punto medio)\ncubre 4 + 4 = 80 hab\n→ el voraz lo elige 1.º", ha="center", va="top",
        fontsize=9.5, color="#d62728", fontweight="bold")
ax.text(0, 1.72, "p = 2, mismo radio S para todos los candidatos, cada punto = 10 hab\n"
                 "Voraz: M (80) + A' (+20) = 100 hab     Óptimo: A' + B' = 120 hab     Voraz/óptimo = 83 %",
        ha="center", va="top", fontsize=10, bbox=dict(fc="white", ec="#555"))
ax.set_title("Task 3.3b: configuración adversa para el algoritmo voraz del MCLP", fontsize=12, fontweight="bold")
fig.savefig(OUT / "t3_3b_caso_adverso.png", dpi=170, bbox_inches="tight"); plt.close(fig)

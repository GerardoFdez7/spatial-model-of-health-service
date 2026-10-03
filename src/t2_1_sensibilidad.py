# Task 2.1: distancia de cada condado al hospital más cercano, curva de cobertura acumulada y tipo de hospital más cercano.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import coverage_polygon, covered_population, load_counties, load_hospitals, nearest, xy
from config import CURVE_RADII_KM, OUT, STATE_NAME

pd.set_option("display.width", 220)
cond, hosp = load_counties(), load_hospitals()
P = cond["poblacion"].sum()

# 2.1a Distancia del centroide al hospital más cercano: d_c = min_j ||centroide(c) - h_j||
cent = cond.geometry.centroid
d, idx = nearest(xy(cent), hosp)
cond["dist_km"] = d / 1000
cond["hosp_cercano"] = hosp["nombre"].values[idx]
cond["ciudad_cercano"] = hosp["ciudad"].values[idx]
cond["tipo_cercano"] = hosp["tipo"].values[idx]
cond["camas_cercano"] = hosp["camas_total"].values[idx].astype(int)
cond["uci_cercano"] = hosp["camas_icu"].values[idx]
cols = ["fips", "nombre", "poblacion", "dist_km", "hosp_cercano", "ciudad_cercano", "tipo_cercano",
        "camas_cercano", "uci_cercano"]
tabla = cond[cols].sort_values("dist_km", ascending=False).reset_index(drop=True)
tabla.round(2).to_csv(OUT / "t2_1a_distancias.csv", index=False)
print("- 2.1a Diez condados con mayor distancia al hospital más cercano")
print(tabla.head(10).round(1).to_string(index=False))
print(f"- 2.1a Mediana = {cond.dist_km.median():.1f} km; condados a > 25 km: {(cond.dist_km > 25).sum()}, "
      f"a > 50 km: {(cond.dist_km > 50).sum()}")

# 2.1b Curva de cobertura acumulada: C(S) = 100 * Pcub(S) / P, con S = 5, 10, ..., 100 km
c = pd.DataFrame([{"radio_km": r, "pct_cubierto": 100 * covered_population(cond, coverage_polygon(hosp, r)) / P}
                  for r in CURVE_RADII_KM])
c["ganancia_pp"] = c["pct_cubierto"].diff().fillna(c["pct_cubierto"])
r_infl = int(c.loc[c["ganancia_pp"].idxmax(), "radio_km"])          # pendiente máxima
# Kneedle: máxima distancia entre la curva normalizada y la recta que une sus extremos
x = (c.radio_km - c.radio_km.min()) / (c.radio_km.max() - c.radio_km.min())
y = (c.pct_cubierto - c.pct_cubierto.min()) / (c.pct_cubierto.max() - c.pct_cubierto.min())
c["kneedle"] = y - x
r_knee = int(c.loc[c["kneedle"].idxmax(), "radio_km"])
r_marg = int(c.loc[(c["ganancia_pp"] < 2) & (c.radio_km > r_infl), "radio_km"].min())
c.round(3).to_csv(OUT / "t2_1b_curva_cobertura.csv", index=False)
print("\n- 2.1b Curva de cobertura")
print(c.round(2).to_string(index=False))
print(f"- 2.1b Pendiente máxima en S = {r_infl} km; rodilla (Kneedle) en S = {r_knee} km; "
      f"ganancia < 2 pp por 5 km desde S = {r_marg} km")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(c.radio_km, c.pct_cubierto, "-o", color="#2a6fb0", ms=4, lw=2)
ax.axvline(r_knee, color="#d62728", ls="--", lw=1.2)
yk = c.loc[c.radio_km == r_knee, "pct_cubierto"].iloc[0]
ax.annotate(f"Rodilla: {r_knee} km ({yk:.1f} %)", xy=(r_knee, yk), xytext=(r_knee + 8, yk - 18),
            arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=10)
yi = c.loc[c.radio_km == r_infl, "pct_cubierto"].iloc[0]
ax.plot([r_infl], [yi], "s", color="#ff7f0e", ms=8, zorder=5)
ax.annotate(f"Pendiente máxima: {r_infl} km", xy=(r_infl, yi), xytext=(r_infl + 6, yi - 12),
            arrowprops=dict(arrowstyle="->", color="#ff7f0e"), color="#ff7f0e", fontsize=9)
for r in (10, 25, 50):
    v = c.loc[c.radio_km == r, "pct_cubierto"].iloc[0]
    ax.annotate(f"{v:.1f} %", (r, v), textcoords="offset points", xytext=(-8, 8), fontsize=8, color="#333")
ax2 = ax.twinx()
ax2.bar(c.radio_km, c.ganancia_pp, width=3, color="#999", alpha=.35)
ax2.set_ylabel("Ganancia marginal (pp por +5 km)", color="#666")
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.set(xlabel="Radio del buffer (km)", ylabel="% de la población cubierta", ylim=(0, 102), xticks=range(0, 105, 10),
       title=f"{STATE_NAME}: curva de cobertura acumulada (5-100 km)")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "t2_1b_curva_cobertura.png", dpi=170); plt.close(fig)

# 2.1c Tipo y camas del hospital más cercano a los cinco condados más alejados
top5 = tabla.head(5)
print("\n- 2.1c Hospital más cercano a los cinco condados más alejados")
print(top5[["nombre", "dist_km", "hosp_cercano", "tipo_cercano", "camas_cercano", "uci_cercano"]].round(1)
      .to_string(index=False))
top5.round(2).to_csv(OUT / "t2_1c_top5_tipo.csv", index=False)
med_resto = cond.loc[~cond.fips.isin(top5.fips), "camas_cercano"].median()
rho = cond[["dist_km", "camas_cercano"]].corr(method="spearman").iloc[0, 1]
print(f"- 2.1c Mediana de camas del hospital más cercano: top-5 = {top5.camas_cercano.median():.0f}, "
      f"resto = {med_resto:.0f}; Spearman distancia vs camas = {rho:.2f}")

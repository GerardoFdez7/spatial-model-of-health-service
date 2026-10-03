# Task 1.2: estadísticas por condado (hospitales, camas, camas UCI, población y tasas por 10,000 hab.) e histogramas.
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

from common import load_counties, load_hospitals
from config import OUT, STATE_NAME

pd.set_option("display.width", 220)
cond, hosp = load_counties(), load_hospitals()

# 1.2 Tabla por condado; cada hospital se asigna a su condado por punto dentro de polígono
hj = gpd.sjoin(hosp, cond[["fips", "geometry"]], how="left", predicate="within")
print(f"- 1.2 camas_icu nulo en {hosp['camas_icu'].isna().sum()} de {len(hosp)} hospitales (se suman como 0)")
agg = hj.groupby("fips").agg(hospitales=("nombre", "size"), camas_total=("camas_total", "sum"),
                             camas_uci=("camas_icu", lambda s: s.fillna(0).sum()))
t = cond[["fips", "nombre", "poblacion"]].merge(agg, on="fips", how="left")
t[["hospitales", "camas_total", "camas_uci"]] = t[["hospitales", "camas_total", "camas_uci"]].fillna(0)
# Tasas: camas_10k = 10,000 * camas / P_c
t["camas_10k"] = 1e4 * t["camas_total"] / t["poblacion"]
t["camas_uci_10k"] = 1e4 * t["camas_uci"] / t["poblacion"]
t = t.astype({"hospitales": int, "camas_total": int, "camas_uci": int, "poblacion": int})
t = t.sort_values("camas_10k", ascending=False).reset_index(drop=True)
t.round(2).to_csv(OUT / "t1_2_estadisticas_condados.csv", index=False)
print("\n- 1.2 Tabla por condado (ordenada por camas/10k)")
print(t.round(2).to_string(index=False))
tot = t[["hospitales", "camas_total", "camas_uci", "poblacion"]].sum()
print(f"\n- 1.2 Total estado: {tot['hospitales']} hospitales, {tot['camas_total']:,} camas, {tot['camas_uci']:,} UCI, "
      f"{tot['poblacion']:,} hab -> {1e4 * tot['camas_total'] / tot['poblacion']:.2f} camas/10k, "
      f"{1e4 * tot['camas_uci'] / tot['poblacion']:.2f} UCI/10k")
print(f"- 1.2 Condados sin hospital: {(t['hospitales'] == 0).sum()} de {len(t)}")

# 1.2a Cinco condados con menos camas por habitante (> 10,000 hab.); empates se resuelven por población
a = t[t["poblacion"] > 10_000].sort_values(["camas_10k", "poblacion"], ascending=[True, False])
print(f"\n- 1.2a Condados con > 10,000 hab: {len(a)}; con 0 camas: {(a['camas_10k'] == 0).sum()}")
print(a.head(5).round(2).to_string(index=False))
a.head(5).round(2).to_csv(OUT / "t1_2a_menos_camas.csv", index=False)

# 1.2b Cinco condados con mayor concentración de camas y sus hospitales
b = t.head(5)
print("\n- 1.2b Mayor concentración de camas por habitante")
print(b.round(2).to_string(index=False))
b.round(2).to_csv(OUT / "t1_2b_mas_camas.csv", index=False)
detalle = hj[hj["fips"].isin(b["fips"])][["fips", "nombre", "tipo", "ciudad", "camas_total", "camas_icu",
                                           "ocupacion_total"]].sort_values(["fips", "camas_total"])
print(detalle.to_string(index=False))

# 1.2c Histogramas de camas por hospital y de camas por habitante
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
x = hosp["camas_total"]
axs[0].hist(x, bins=np.arange(0, x.max() + 25, 25), color="#2a6fb0", edgecolor="white")
axs[0].axvline(x.mean(), color="#d62728", ls="--", label=f"media = {x.mean():.0f}")
axs[0].axvline(x.median(), color="k", ls=":", label=f"mediana = {x.median():.0f}")
axs[0].set(title=f"Camas totales por hospital ({STATE_NAME}, n = {len(x)})", xlabel="Camas totales",
           ylabel="Número de hospitales"); axs[0].legend()
y = t["camas_10k"]
axs[1].hist(y, bins=np.arange(0, y.max() + 5, 5), color="#2a9d8f", edgecolor="white")
axs[1].axvline(y.mean(), color="#d62728", ls="--", label=f"media = {y.mean():.1f}")
axs[1].axvline(y.median(), color="k", ls=":", label=f"mediana = {y.median():.1f}")
axs[1].set(title=f"Camas por 10,000 hab. por condado (n = {len(y)})", xlabel="Camas por 10,000 habitantes",
           ylabel="Número de condados"); axs[1].legend()
for ax in axs: ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "t1_2c_histogramas.png", dpi=170); plt.close(fig)

# Atípicos: valores mayores a Q3 + 1.5·IQR
for name, s in [("camas por hospital", x), ("camas/10k por condado", y)]:
    q1, q3 = s.quantile([.25, .75]); lim = q3 + 1.5 * (q3 - q1)
    print(f"- 1.2c {name}: media={s.mean():.2f}, mediana={s.median():.2f}, asimetría={skew(s):.2f}, "
          f"atípicos (> {lim:.1f}): {(s > lim).sum()}, ceros: {(s == 0).sum()}")
print(f"- 1.2c Participación de los 5 hospitales más grandes en las camas del estado: "
      f"{100 * hosp.nlargest(5, 'camas_total')['camas_total'].sum() / x.sum():.1f} %")

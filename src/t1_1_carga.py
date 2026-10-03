# Task 1.1: carga los cuatro archivos, los limpia, filtra Idaho, reproyecta a metros y genera el mapa base.
import matplotlib
matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import shapely
from matplotlib.lines import Line2D

from common import TYPE_COLORS, add_north, add_scalebar, base_axes
from config import (CRS_PROJ, F_CONDADOS, F_ESTADOS, F_HOSP, F_POB, OUT, PROC, STATE_ABBR, STATE_FIPS,
                    STATE_NAME)

pd.set_option("display.width", 200)
print(f"geopandas {gpd.__version__} | shapely {shapely.__version__} | pyproj {pyproj.__version__} | pandas {pd.__version__}")

# 1.1a Carga de los cuatro archivos
hosp = gpd.read_file(F_HOSP)
cond = gpd.read_file(F_CONDADOS)
est = gpd.read_file(F_ESTADOS)
pob = pd.read_csv(F_POB)

rows = []
for name, df in [("hospitales_eeuu.geojson", hosp), ("condados_eeuu.geojson", cond),
                 ("estados_eeuu.geojson", est), ("poblacion_condados.csv", pob)]:
    crs = df.crs.to_string() if isinstance(df, gpd.GeoDataFrame) else "no aplica (tabla sin geometría)"
    rows.append({"archivo": name, "registros": len(df), "crs": crs, "columnas": ", ".join(df.columns)})
    print(f"\n- {name}: {len(df):,} registros | CRS = {crs}")
    print(df.dtypes.to_string())
pd.DataFrame(rows).to_csv(OUT / "t1_1_resumen_archivos.csv", index=False)

# Hospitales por estado, para justificar la elección del estado
por_estado = (hosp.groupby("estado").agg(hospitales=("nombre", "size"),
                                         con_camas=("camas_total", lambda s: int(s.notna().sum())))
              .sort_values("hospitales"))
por_estado.to_csv(OUT / "t1_1_hospitales_por_estado.csv")
print("\nEstados alrededor de 50 hospitales:")
print(por_estado[(por_estado.hospitales >= 40) & (por_estado.hospitales <= 75)].to_string())

# 1.1b Limpieza de población: FIPS >= 80000 no son condados reales
n0 = len(pob)
n_esp = (pob["fips"] >= 80000).sum()
pob = pob[pob["fips"] < 80000].copy()
pob["fips"] = pob["fips"].astype(int).astype(str).str.zfill(5)   # 1001 -> "01001"
print(f"\n- 1.1b Población: {n0} registros -> eliminados {n_esp} con FIPS >= 80000 -> {len(pob)}")
print(f"- 1.1b Población nula tras la limpieza: {pob['poblacion'].isna().sum()} registros")

# 1.1b Limpieza de hospitales: porcentaje de camas_total nulo
pct_null = 100 * hosp["camas_total"].isna().mean()
print(f"- 1.1b Hospitales con camas_total nulo: {hosp['camas_total'].isna().sum():,} de {len(hosp):,} "
      f"({pct_null:.2f} %)")

# 1.1c Filtro por estado y unión de población con la geometría
hosp_st_all = hosp[hosp["estado"] == STATE_ABBR]
hosp_st = hosp_st_all[hosp_st_all["camas_total"].notna()].copy()
sin_camas = hosp_st_all[hosp_st_all["camas_total"].isna()]
print(f"\n- 1.1c {STATE_NAME}: {len(hosp_st_all)} hospitales; {len(sin_camas)} sin camas_total; "
      f"quedan {len(hosp_st)} para el análisis")
print(sin_camas[["nombre", "tipo", "ciudad", "condado"]].to_string(index=False))

cond_st = cond[cond["cod_estado"] == STATE_FIPS].copy()
pob_st = pob[pob["fips"].str[:2] == STATE_FIPS]
cond_st = cond_st.merge(pob_st[["fips", "condado", "poblacion"]], on="fips", how="left", validate="1:1")
print(f"- 1.1c Condados con geometría: {len(cond_st)} | registros de población: {len(pob_st)} | "
      f"condados con población tras la unión: {cond_st['poblacion'].notna().sum()}")
assert len(cond_st) == len(pob_st) == cond_st["poblacion"].notna().sum(), "No coinciden condados y población"
est_st = est[est["codigo"] == STATE_ABBR].copy()
print(f"- 1.1c Población total del estado: {cond_st['poblacion'].sum():,.0f}")

# 1.1d Reproyección a metros
cond_st, hosp_st, est_st = (g.to_crs(CRS_PROJ) for g in (cond_st, hosp_st, est_st))
print(f"\n- 1.1d CRS proyectado: {CRS_PROJ} = {pyproj.CRS(CRS_PROJ).name}")

# Distorsión de la Transversal de Mercator: k ≈ k0 [1 + (Δλ cos φ)^2 / 2], evaluada dentro del estado
lon, lat = np.meshgrid(np.linspace(-117.25, -111.04, 200), np.linspace(41.99, 49.0, 200))
inside = shapely.contains_xy(est[est.codigo == STATE_ABBR].geometry.iloc[0], lon, lat)
for crs in [CRS_PROJ, "EPSG:26911"]:
    k = pyproj.Proj(crs).get_factors(lon[inside], lat[inside]).meridional_scale
    print(f"- 1.1d {crs} ({pyproj.CRS(crs).name}): k en [{k.min():.5f}, {k.max():.5f}] "
          f"-> error máximo {100 * abs(k - 1).max():.3f} %")
area_km2 = est_st.geometry.area.iloc[0] / 1e6
print(f"- 1.1d Área del estado: {area_km2:,.0f} km² (oficial: 216,443 km², diferencia "
      f"{100 * (area_km2 / 216_443 - 1):+.2f} %)")

for g, name in [(cond_st, "condados_estado"), (hosp_st, "hospitales_estado"), (est_st, "estado")]:
    g.to_file(PROC / f"{name}.gpkg", driver="GPKG")

# 1.1e Mapa base: condados, hospitales por tipo y límite estatal
fig, ax = base_axes((8.5, 11), f"{STATE_NAME}: condados, hospitales por tipo y límite estatal\n"
                               f"(línea base, n = {len(hosp_st)} hospitales con dato de camas)")
cond_st.plot(ax=ax, color="#e6e6e6", edgecolor="#9a9a9a", linewidth=0.5)
est_st.boundary.plot(ax=ax, color="k", linewidth=1.4)
handles = []
for t, g in hosp_st.groupby("tipo"):
    ax.scatter(g.geometry.x, g.geometry.y, s=38, color=TYPE_COLORS.get(t, "k"), edgecolor="k",
               linewidth=0.4, zorder=5)
    handles.append(Line2D([], [], marker="o", ls="", ms=7, markerfacecolor=TYPE_COLORS.get(t, "k"),
                          markeredgecolor="k", label=f"{t.title()} ({len(g)})"))
sc = sin_camas.to_crs(CRS_PROJ)
ax.scatter(sc.geometry.x, sc.geometry.y, s=30, marker="x", color="#555", zorder=4)
handles.append(Line2D([], [], marker="x", ls="", ms=7, color="#555",
                      label=f"Sin dato de camas, excluido ({len(sin_camas)})"))
handles.append(Line2D([], [], color="k", lw=1.4, label="Límite estatal"))
handles.append(Line2D([], [], marker="s", ls="", ms=10, markerfacecolor="#e6e6e6", markeredgecolor="#9a9a9a",
                      label=f"Condados ({len(cond_st)})"))
minx, miny, maxx, maxy = est_st.total_bounds
ax.set_xlim(minx - 15e3, maxx + 15e3); ax.set_ylim(miny - 15e3, maxy + 15e3)
add_scalebar(ax, 100, loc=(0.06, 0.03)); add_north(ax)
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=8, title="Tipo de hospital",
          title_fontsize=9, frameon=True)
ax.text(0.01, -0.01, f"CRS: {CRS_PROJ} (NAD83 / Idaho Transverse Mercator)", transform=ax.transAxes,
        fontsize=7, color="#555")
fig.savefig(OUT / "t1_1_mapa_base.png", dpi=170, bbox_inches="tight"); plt.close(fig)

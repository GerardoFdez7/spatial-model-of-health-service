"""Configuración central del laboratorio: territorio, rutas, CRS y parámetros del modelo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
for _p in (RAW, PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# --- Territorio: Guatemala -------------------------------------------------
ISO3 = "GTM"
COUNTRY = "Guatemala"

# --- Sistemas de coordenadas ----------------------------------------------
CRS_GEO = "EPSG:4326"    # WGS84 (grados): formato nativo de GADM, healthsites y WorldPop
CRS_PROJ = "EPSG:32615"  # WGS84 / UTM zona 15N (metros): ver justificación en RESPUESTAS.md

# --- Fuentes ----------------------------------------------------------------
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso}_{lvl}.json"
# WorldPop Global2 R2025A: 2025 es el año más reciente ESTIMADO (2026-2030 son proyecciones).
WORLDPOP_URL = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/"
                "{iso}/v1/1km_ua/constrained/{iso_l}_pop_2025_CN_1km_R2025A_UA_v1.tif")
# Respaldo (serie histórica 2000-2020) por si R2025A no estuviera disponible.
WORLDPOP_URL_FALLBACK = ("https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/"
                         "{iso}/{iso_l}_ppp_2020_1km_Aggregated.tif")
HEALTHSITES_URL = "https://healthsites.io/api/v2/facilities/"      # endpoint pedido en el enunciado
HEALTHSITES_URL_V3 = "https://healthsites.io/api/v3/facilities/"   # solo si el servidor rechaza la v2 (deprecada)
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]

# --- Parámetros del modelo ---------------------------------------------------
RADII_KM = [5, 10, 20]        # Task 1.3
REF_RADIUS_KM = 10            # radio de referencia (Task 1.3c, 2.1, 2.2, 2.3)
CURVE_RADII_KM = list(range(1, 51))   # Task 2.1c
SPEED_KMH = 40                # Task 2.1b
GRID_STEP_M = 20_000          # Task 2.3a
N_NEW = 5                     # Task 2.3c
# Pesos del índice de vulnerabilidad (Task 2.2b), suman 1
WEIGHTS = {"dist": 0.40, "unc_density": 0.25, "capacity": 0.35}

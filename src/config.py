# Configuración central del laboratorio: estado elegido, rutas, CRS y parámetros del modelo.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # los 4 archivos de Canvas
PROC = ROOT / "data" / "processed"   # capas intermedias
OUT = ROOT / "outputs"               # tablas, mapas y logs del informe
for _p in (RAW, PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

F_ESTADOS = RAW / "estados_eeuu.geojson"
F_HOSP = RAW / "hospitales_eeuu.geojson"
F_CONDADOS = RAW / "condados_eeuu.geojson"
F_POB = RAW / "poblacion_condados.csv"

# Estado elegido: el de 50 hospitales o el valor inmediato mayor (Idaho, 55)
STATE_ABBR = "ID"
STATE_FIPS = "16"
STATE_NAME = "Idaho"

# CRS proyectado en metros: NAD83 / Idaho Transverse Mercator
CRS_PROJ = "EPSG:8826"

# Parámetros del modelo
RADII_KM = [10, 25, 50]                    # Task 1.3a
REF_RADIUS_KM = 25                         # Task 1.3c y 2.3
CURVE_RADII_KM = list(range(5, 105, 5))    # Task 2.1b
OCC_RADIUS_KM = 50                         # Task 2.2, componente 3
GRID_STEP_M = 50_000                       # Task 2.3a
N_NEW = 3                                  # Task 2.3c
ISO_MIN = 30                               # Task 3.1
WEIGHTS = {"C1_dist": 0.45, "C2_camas": 0.35, "C3_ocup": 0.20}   # Task 2.2b, suman 1
FULL_TOL = 0.999                           # fracción de área para "completamente cubierto"

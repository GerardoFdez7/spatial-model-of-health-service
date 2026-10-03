# Modelo espacial de cobertura hospitalaria — Idaho (Laboratorio 5, CC2017)

¿Qué fracción de la población de Idaho tiene acceso a servicios hospitalarios dentro de una distancia razonable y dónde están las brechas más críticas? Se combinan los cuatro archivos provistos en Canvas: hospitales (`hospitales_eeuu.geojson`), condados (`condados_eeuu.geojson`), población por condado (`poblacion_condados.csv`) y límites estatales (`estados_eeuu.geojson`).

**Estado elegido:** Idaho (FIPS 16). Ningún estado tiene exactamente 50 hospitales en los datos; Idaho es el de valor inmediatamente mayor (55; 48 con dato de camas).
**CRS de trabajo:** EPSG:8826, NAD83 / Idaho Transverse Mercator (metros).

## Uso
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python src/run_all.py             # corre todo; tablas, mapas y logs en outputs/
python src/run_all.py t3_1        # solo un task (por prefijo)
```
`t3_1_isocronas.py` obtiene la red vial de OpenStreetMap con OSMnx. Primero prueba varios servidores Overpass (el principal suele estar saturado). Si ninguno responde, descargue el extracto de Idaho de Geofabrik (https://download.geofabrik.de/north-america/us/idaho-latest.osm.pbf), guárdelo en `data/raw/idaho-latest.osm.pbf` y vuelva a correr: el script filtra las vías con `osmium` y arma el grafo con `ox.graph_from_xml`. Los grafos quedan en caché en `data/processed/osm/`.

## Estructura
| Ruta | Contenido |
|---|---|
| `src/config.py`, `src/common.py` | Estado, CRS, parámetros y funciones compartidas (buffers, cobertura, mapas) |
| `src/t1_1_carga.py` | 1.1 carga, limpieza, filtro por estado, reproyección, mapa base |
| `src/t1_2_estadisticas.py` | 1.2 estadísticas por condado e histogramas |
| `src/t1_3_buffers.py` | 1.3 buffers 10/25/50 km y cobertura poblacional |
| `src/t2_1_sensibilidad.py` | 2.1 distancia al hospital más cercano y curva de cobertura |
| `src/t2_2_vulnerabilidad.py` | 2.2 índice compuesto de vulnerabilidad |
| `src/t2_3_mclp.py` | 2.3 MCLP con algoritmo voraz |
| `src/t3_1_isocronas.py` | 3.1 buffer de 25 km vs isocrona de 30 min (OSMnx) |
| `src/t3_3_sensibilidad_pesos.py` | 3.3 sensibilidad de pesos y caso adverso del voraz |
| `src/run_all.py` | Ejecuta todo en orden y guarda `outputs/<script>_log.txt` |
| `data/raw/` | Los cuatro archivos de Canvas |
| `outputs/` | Tablas CSV, mapas PNG y logs citados en el documento |

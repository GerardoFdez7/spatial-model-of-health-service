# Modelo espacial de cobertura de servicios de salud (Guatemala)

¿Qué fracción de la población de Guatemala vive cerca de un servicio de salud y dónde están las brechas críticas? Combina divisiones administrativas (GADM 4.1), instalaciones de salud (healthsites.io) y población (WorldPop 2025, 1 km).

## Qué incluye
- Descarga reproducible de datos y análisis exploratorio (CRS EPSG:32615).
- Cobertura por buffers de 5/10/20 km y curva de cobertura acumulada.
- Índice de vulnerabilidad por municipio y mapa por quintiles.
- MCLP (ubicación de 5 nuevas instalaciones) con heurística voraz, verificado contra un MILP.

Las respuestas teóricas, resultados y párrafos de prompts están en [RESPUESTAS.md](RESPUESTAS.md).

## Uso
```bash
python -m venv .venv && .venv/Scripts/activate      # en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                 # pegue su HEALTHSITES_API_KEY
python src/run_all.py                                # corre todo; salidas en outputs/
```

## Estructura
| Ruta | Contenido |
|---|---|
| `src/config.py`, `src/common.py` | Parámetros, rutas y funciones compartidas |
| `src/t1_1_download.py` … `src/t2_3_mclp.py` | Un script por task |
| `src/run_all.py` | Ejecuta todos en orden |
| `data/` | Descargas (`raw/`) e intermedios (`processed/`), no versionados |
| `outputs/` | Tablas CSV, mapas PNG y logs de cada task |

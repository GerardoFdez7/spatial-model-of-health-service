"""Task 1.1 - Instalación/verificación de librerías y descarga reproducible de datos.

Ejecutar:  python src/t1_1_download.py
Llave de healthsites:  variable de entorno HEALTHSITES_API_KEY o archivo .env (ver .env.example).
"""
import json
import os
import sys
import time

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from dotenv import load_dotenv

from common import normalize_type
from config import (COUNTRY, CRS_GEO, GADM_URL, HEALTHSITES_URL, HEALTHSITES_URL_V3, ISO3, OUT, OVERPASS_URLS, PROC,
                    RAW, WORLDPOP_URL, WORLDPOP_URL_FALLBACK)

load_dotenv()
UA = {"User-Agent": "uvg-modelacion-lab/1.0 (health coverage lab)"}


def download(url: str, dest, chunk=1 << 20) -> None:
    """Descarga en streaming; omite si el archivo ya existe (idempotente/reproducible)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cache] {dest.name}")
        return
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=120, headers=UA) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(chunk):
                f.write(part)


# ------------------------------------------------------------------- (a) librerías
def report_versions():
    print("== 1.1a Versiones ==")
    for name, mod in [("geopandas", gpd), ("shapely", shapely), ("pandas", pd), ("numpy", np),
                      ("matplotlib", matplotlib), ("requests", requests), ("rasterio", rasterio)]:
        print(f"  {name}.__version__ = {mod.__version__}")


# ------------------------------------------------------------------- (b) GADM
def get_gadm():
    print("== 1.1b GADM ==")
    for lvl in (1, 2):
        dest = RAW / f"gadm41_{ISO3}_{lvl}.json"
        download(GADM_URL.format(iso=ISO3, lvl=lvl), dest)
        g = gpd.read_file(dest)
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.set_crs(CRS_GEO, allow_override=True) if g.crs is None else g.to_crs(CRS_GEO)
        print(f"  Nivel {lvl}: {len(g)} polígonos, CRS={g.crs.to_string()}")
        assert g.crs.to_epsg() == 4326


# ------------------------------------------------------------------- (c) healthsites
def _parse_records(data) -> list:
    """Normaliza la respuesta (GeoJSON FeatureCollection, lista de registros v2 o dict paginado)
    a una lista de dicts {lon, lat, name, amenity, healthcare}."""
    items = data.get("features", data.get("results", [])) if isinstance(data, dict) else data
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        props = dict(it.get("properties", it))
        attrs = props.get("attributes", {}) or {}
        geom = it.get("geometry") or it.get("centroid") or it.get("geom") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else geom
        if not coords and "lon" in props:
            coords = [props["lon"], props["lat"]]
        if not coords or len(coords) < 2:
            continue
        get = lambda k: attrs.get(k, props.get(k))
        out.append({"lon": float(coords[0]), "lat": float(coords[1]), "name": props.get("name"),
                    "amenity": get("amenity"), "healthcare": get("healthcare")})
    return out


def _healthsites_api(key: str, base: str, version: str) -> gpd.GeoDataFrame:
    """Descarga paginada de healthsites.io. Se usa primero /api/v2/ (endpoint del enunciado)."""
    rows, page = [], 1
    while True:
        params = {"api-key": key, "page": page, "country": COUNTRY}
        if version == "v3":
            params.update({"output": "geojson", "flat-properties": "true"})
        r = requests.get(base, headers=UA, timeout=120, params=params)
        body = r.text.strip()
        if r.status_code != 200 or "deprecated" in body[:200].lower() or "invalid" in body[:200].lower():
            if page == 1:
                raise RuntimeError(f"healthsites {version} HTTP {r.status_code}: {body[:200]}")
            break
        batch = _parse_records(r.json())
        if not batch:
            break
        rows += batch
        print(f"  [{version}] página {page}: {len(batch)} registros (acum. {len(rows)})")
        page += 1
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df["type"] = [normalize_type(a, h) for a, h in zip(df["amenity"], df["healthcare"])]
    df["source"] = f"healthsites.io {version}"
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS_GEO)


def _overpass_fallback() -> gpd.GeoDataFrame:
    """RESPALDO sin llave: healthsites.io se construye sobre OpenStreetMap, así que se consultan
    directamente las mismas etiquetas (amenity/healthcare) en OSM vía Overpass."""
    q = f"""[out:json][timeout:180];
area["ISO3166-1"="GT"][admin_level=2]->.a;
( nwr["amenity"~"^(hospital|clinic|doctors|pharmacy|dentist)$"](area.a);
  nwr["healthcare"](area.a); );
out center tags;"""
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=240)
            r.raise_for_status()
            els = r.json()["elements"]
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise RuntimeError(f"Overpass no disponible: {last}")
    rows = []
    for e in els:
        lat, lon = (e.get("lat"), e.get("lon")) if e["type"] == "node" else (
            e.get("center", {}).get("lat"), e.get("center", {}).get("lon"))
        if lat is None:
            continue
        t = e.get("tags", {})
        rows.append({"osm_id": e["id"], "name": t.get("name"), "amenity": t.get("amenity"),
                     "healthcare": t.get("healthcare"), "lon": lon, "lat": lat})
    df = pd.DataFrame(rows)
    df["type"] = [normalize_type(a, h) for a, h in zip(df["amenity"], df["healthcare"])]
    df["source"] = "OpenStreetMap (Overpass) - respaldo"
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=CRS_GEO)


def get_facilities():
    print("== 1.1c Instalaciones de salud ==")
    raw = RAW / f"healthsites_{ISO3}.gpkg"
    if raw.exists():
        print(f"  [cache] {raw.name}")
        gdf = gpd.read_file(raw)
    else:
        key = os.getenv("HEALTHSITES_API_KEY", "").strip()
        gdf = None
        if key:
            # 1) v2 (lo que pide el enunciado); 2) v3 solo si el servidor rechaza la v2 por deprecada
            for base, ver in ((HEALTHSITES_URL, "v2"), (HEALTHSITES_URL_V3, "v3")):
                try:
                    gdf = _healthsites_api(key, base, ver)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  !! healthsites {ver} falló: {e}")
        if gdf is None:
            print("  !! sin llave o API no disponible -> usando respaldo OSM/Overpass")
            gdf = _overpass_fallback()
        gdf[["type", "source", "geometry"] + [c for c in ("name", "amenity", "healthcare") if c in gdf]] \
            .to_file(raw, driver="GPKG")
    # Intersección espacial con la capa de nivel 1: nos quedamos solo con lo que cae en el territorio.
    adm1 = gpd.read_file(RAW / f"gadm41_{ISO3}_1.json").to_crs(CRS_GEO)
    adm1["geometry"] = adm1.geometry.make_valid()
    total = len(gdf)
    gdf = gpd.sjoin(gdf, adm1[["GID_1", "NAME_1", "geometry"]], how="inner", predicate="intersects")
    gdf = gdf[~gdf.index.duplicated()].drop(columns="index_right").reset_index(drop=True)
    gdf.to_file(PROC / "facilities.gpkg", driver="GPKG")
    print(f"  Fuente: {gdf['source'].iloc[0]}")
    print(f"  Registros descargados: {total}; dentro de {COUNTRY}: {len(gdf)}")
    tipos = gdf["type"].value_counts()
    print("  Tipos presentes:\n" + tipos.to_string())
    tipos.rename("n").to_csv(OUT / "t1_1_facility_types.csv")


# ------------------------------------------------------------------- (d) WorldPop
def get_worldpop():
    print("== 1.1d WorldPop ==")
    fmt = dict(iso=ISO3, iso_l=ISO3.lower())
    dest = RAW / f"{ISO3.lower()}_pop_1km.tif"
    try:
        download(WORLDPOP_URL.format(**fmt), dest)
    except requests.HTTPError:
        print("  R2025A no disponible; usando serie 2000-2020")
        download(WORLDPOP_URL_FALLBACK.format(**fmt), dest)
    with rasterio.open(dest) as src:
        band = src.read(1, masked=True)
        # Resolución en metros: el ráster está en grados; se convierte con la latitud central.
        lat0 = (src.bounds.top + src.bounds.bottom) / 2
        m_lat = abs(src.res[1]) * 111_320
        m_lon = abs(src.res[0]) * 111_320 * np.cos(np.radians(lat0))
        print(f"  Resolución: {src.res[0]:.6f}° x {src.res[1]:.6f}°  ~ {m_lon:.0f} m x {m_lat:.0f} m (a lat {lat0:.1f}°)")
        print(f"  CRS: {src.crs}")
        print(f"  Filas x columnas: {src.height} x {src.width}")
        print(f"  Valor máximo por píxel: {float(band.max()):.1f} hab/píxel(1 km²)")
        print(f"  Población total del ráster: {float(band.sum()):,.0f}")


if __name__ == "__main__":
    print(sys.version)
    report_versions(); get_gadm(); get_facilities(); get_worldpop()

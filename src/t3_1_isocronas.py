# Task 3.1: compara el buffer circular de 25 km con la isocrona de 30 min sobre la red vial real (OSMnx)
# para los hospitales de los tres condados más vulnerables que tienen hospital.
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, Polygon

from common import load_counties, load_hospitals, state_polygon
from config import CRS_PROJ, ISO_MIN, OUT, PROC, REF_RADIUS_KM

pd.set_option("display.width", 220)
OSM_DIR = PROC / "osm"; OSM_DIR.mkdir(parents=True, exist_ok=True)
ox.settings.use_cache = True
ox.settings.cache_folder = str(PROC / "osm_cache")
ox.settings.requests_timeout = 300

DIST_M = 70_000      # a 80 mph se recorren ~64 km en 30 min, así que 70 km contiene toda la isocrona
EDGE_BUF_M = 500     # margen lateral de las vías alcanzables
# Velocidades (km/h) por tipo de vía cuando OSM no tiene maxspeed, según los límites legales de Idaho
HWY_SPEEDS = {"motorway": 112, "motorway_link": 60, "trunk": 97, "trunk_link": 55, "primary": 88,
              "primary_link": 50, "secondary": 80, "secondary_link": 45, "tertiary": 65, "tertiary_link": 40,
              "unclassified": 55, "residential": 40, "living_street": 15, "service": 25, "road": 40}
FALLBACK_KMH = 40

# Servidores Overpass (el principal suele estar saturado) y extracto de Geofabrik como alternativa
OVERPASS_ENDPOINTS = ["https://overpass-api.de/api", "https://overpass.private.coffee/api",
                      "https://overpass.kumi.systems/api", "https://maps.mail.ru/osm/tools/overpass/api"]
PBF = next(iter(sorted((PROC.parent / "raw").glob("idaho*.osm.pbf"))), None)
DRIVE_HWY = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
             "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
             "road", "service"}
NO_SERVICE = {"parking", "parking_aisle", "driveway", "private", "emergency_access"}
_endpoint = None

cond, hosp = load_counties(), load_hospitals()
state = state_polygon()
idx = gpd.read_file(PROC / "condados_indice.gpkg")

# 3.1a Hospital con más camas de cada uno de los 3 condados más vulnerables que tienen hospital
con_h = idx[idx["hospitales"] > 0].sort_values("IVA", ascending=False).head(3)
hj = gpd.sjoin(hosp, cond[["fips", "geometry"]], how="left", predicate="within")
sel = []
for _, c in con_h.iterrows():
    h = hj[hj["fips"] == c["fips"]].sort_values("camas_total", ascending=False).iloc[0]
    sel.append({"condado": c["nombre"], "rank_IVA": int(c["rank"]), "hospital": h["nombre"], "ciudad": h["ciudad"],
                "tipo": h["tipo"], "camas": int(h["camas_total"]), "geometry": h.geometry})
sel = gpd.GeoDataFrame(sel, crs=CRS_PROJ)
sel_ll = sel.to_crs(4326)
sel["lat"], sel["lon"] = sel_ll.geometry.y.values, sel_ll.geometry.x.values
print("- 3.1a Hospitales seleccionados")
print(sel.drop(columns="geometry").to_string(index=False))


# Devuelve el primer servidor Overpass que responde, o None si ninguno responde.
def pick_overpass():
    import requests
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(f"{url}/interpreter", data={"data": "[out:json][timeout:20];node(1);out;"},
                              timeout=(15, 40))
            ok = r.status_code == 200 and r.text.lstrip().startswith("{")
            print(f"  Overpass {url}: {'OK' if ok else f'HTTP {r.status_code}'}")
            if ok:
                return url
        except Exception as e:  # noqa: BLE001
            print(f"  Overpass {url}: sin respuesta ({type(e).__name__})")
    return None


# Indica si una vía de OSM es transitable en vehículo (mismo criterio que network_type="drive").
def keep_drive(tags):
    hw = tags.get("highway")
    if hw not in DRIVE_HWY or tags.get("area") == "yes":
        return False
    if tags.get("access") in ("private", "no") or tags.get("motor_vehicle") == "no" or tags.get("motorcar") == "no":
        return False
    return not (hw == "service" and tags.get("service") in NO_SERVICE)


# Arma la red vial desde el extracto .osm.pbf de Geofabrik, filtrando las vías alrededor del hospital.
def graph_from_pbf(row, dist):
    import osmium
    west, south, east, north = ox.utils_geo.bbox_from_point((row["lat"], row["lon"]), dist=dist)
    xml = OSM_DIR / f"{row['condado'].lower().replace(' ', '_')}_drive.osm"
    with osmium.BackReferenceWriter(str(xml), ref_src=str(PBF), overwrite=True) as w:
        fp = osmium.FileProcessor(str(PBF)).with_locations().with_filter(osmium.filter.KeyFilter("highway"))
        for obj in fp:
            if obj.is_way() and keep_drive(obj.tags) and any(
                    nd.location.valid() and west <= nd.lon <= east and south <= nd.lat <= north for nd in obj.nodes):
                w.add(obj)
    G = ox.graph_from_xml(xml, bidirectional=False, simplify=True, retain_all=False)
    return ox.truncate.truncate_graph_bbox(G, (west, south, east, north))


# Obtiene la red vial del hospital (caché, extracto o Overpass) y le agrega velocidades y tiempos de viaje.
def get_graph(row):
    global _endpoint
    f = OSM_DIR / f"{row['condado'].lower().replace(' ', '_')}.graphml"
    if f.exists():
        G = ox.load_graphml(f)
    elif PBF is not None:
        print(f"  Construyendo la red de {row['hospital']} desde {PBF.name}")
        G = graph_from_pbf(row, DIST_M)
        ox.save_graphml(G, f)
    else:
        if _endpoint is None:
            _endpoint = pick_overpass()
            if _endpoint is None:
                raise SystemExit(
                    "\nNingún servidor Overpass respondió. Alternativa sin Overpass:\n"
                    "  1) Descargue https://download.geofabrik.de/north-america/us/idaho-latest.osm.pbf\n"
                    "  2) Guárdelo en data/raw/idaho-latest.osm.pbf\n"
                    "  3) pip install osmium   y vuelva a correr:  python src/run_all.py t3_1")
            ox.settings.overpass_url = _endpoint
            ox.settings.overpass_rate_limit = _endpoint == OVERPASS_ENDPOINTS[0]
        print(f"  Descargando la red de {row['hospital']} desde {_endpoint}")
        G = ox.graph_from_point((row["lat"], row["lon"]), dist=DIST_M, network_type="drive", simplify=True)
        ox.save_graphml(G, f)
    # Tiempo de cada arista: t_e = longitud_e / v_e
    G = ox.routing.add_edge_speeds(G, hwy_speeds=HWY_SPEEDS, fallback=FALLBACK_KMH)
    G = ox.routing.add_edge_travel_times(G)
    return ox.projection.project_graph(G, to_crs=CRS_PROJ)


# Isocrona: nodos con T(h, v) <= 30 min (Dijkstra) convertidos en polígono con margen y sin huecos.
def isochrone(G, origin, minutes):
    T = nx.single_source_dijkstra_path_length(G, origin, cutoff=minutes * 60, weight="travel_time")
    nodes, edges = ox.convert.graph_to_gdfs(G.subgraph(T.keys()))
    geom = shapely.union_all(edges.geometry.buffer(EDGE_BUF_M, quad_segs=4).values)
    geom = shapely.union(geom, shapely.union_all(nodes.geometry.buffer(EDGE_BUF_M, quad_segs=4).values))
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    return shapely.union_all([Polygon(p.exterior) for p in polys])


# Población y área dentro de Idaho: Pcub(A) = sum_c P_c * área(c ∩ A) / área(c).
def pcub(poly):
    p = poly.intersection(state)
    return float((cond["poblacion"] * cond.geometry.intersection(p).area / cond.geometry.area).sum()), p.area / 1e6


# 3.1a Buffer de 25 km vs isocrona de 30 min, y métricas de la red para 3.1b
rows, store = [], {}
for i, r in sel.iterrows():
    G = get_graph(r)
    origin = ox.distance.nearest_nodes(G, r.geometry.x, r.geometry.y)
    iso = isochrone(G, origin, ISO_MIN)
    buf = r.geometry.buffer(REF_RADIUS_KM * 1000, quad_segs=64)
    p_buf, a_buf = pcub(buf); p_iso, a_iso = pcub(iso)
    b_st, i_st = buf.intersection(state), iso.intersection(state)
    inter = b_st.intersection(i_st).area / 1e6
    # Rodeo = distancia por red / distancia recta, y direcciones (sectores de 10°) con vías a 22-25 km
    Ld = nx.single_source_dijkstra_path_length(G, origin, cutoff=60_000, weight="length")
    nd = pd.DataFrame({"n": list(Ld), "dred": list(Ld.values())})
    nd["x"] = [G.nodes[n]["x"] for n in nd.n]; nd["y"] = [G.nodes[n]["y"] for n in nd.n]
    nd["deuc"] = np.hypot(nd.x - G.nodes[origin]["x"], nd.y - G.nodes[origin]["y"])
    nd["t_min"] = nd.n.map(nx.single_source_dijkstra_path_length(G, origin, weight="travel_time")) / 60
    ring = nd[(nd.deuc > 5000) & (nd.deuc <= 25000)]
    edge = nd[(nd.deuc > 22000) & (nd.deuc <= 25000)]
    ang = np.degrees(np.arctan2(edge.y - G.nodes[origin]["y"], edge.x - G.nodes[origin]["x"])) % 360
    sectores = len(np.unique((ang // 10).astype(int))) / 36 if len(ang) else 0
    e_all = ox.convert.graph_to_gdfs(G, nodes=False)
    km_vias = e_all[e_all.intersects(buf)].geometry.length.sum() / 2000   # /2: aristas en ambos sentidos
    rows.append({"condado": r["condado"], "hospital": r["hospital"], "area_buffer_km2": round(a_buf),
                 "area_isocrona_km2": round(a_iso), "pct_buffer_alcanzable_30min": round(100 * inter / a_buf, 1),
                 "buffer_no_alcanzable_km2": round(b_st.difference(i_st).area / 1e6),
                 "isocrona_fuera_buffer_km2": round(i_st.difference(b_st).area / 1e6),
                 "pob_buffer25": round(p_buf), "pob_isocrona30": round(p_iso), "dif_pob": round(p_iso - p_buf),
                 "dif_pct": round(100 * (p_iso - p_buf) / p_buf, 1),
                 "alcance_max_euclid_km": round(nd.loc[nd.t_min <= ISO_MIN, "deuc"].max() / 1000, 1),
                 "rodeo_mediano_5_25km": round((ring.dred / ring.deuc).median(), 2),
                 "sectores_10grados_con_red_22_25km": round(100 * sectores, 0),
                 "km_vias_buffer": round(km_vias)})
    store[i] = dict(iso=iso, buf=buf)

res = pd.DataFrame(rows)
res.to_csv(OUT / "t3_1_buffer_vs_isocrona.csv", index=False)
print("\n- 3.1a Buffer 25 km vs isocrona 30 min")
print(res.T.to_string())

# Totales con la unión de las tres áreas
pb, ab = pcub(shapely.union_all([s["buf"] for s in store.values()]))
pi, ai = pcub(shapely.union_all([s["iso"] for s in store.values()]))
tot = pd.DataFrame([{"metrica": "Buffer 25 km", "area_km2": round(ab), "poblacion": round(pb)},
                    {"metrica": "Isocrona 30 min", "area_km2": round(ai), "poblacion": round(pi)},
                    {"metrica": "Diferencia (iso - buffer)", "area_km2": round(ai - ab), "poblacion": round(pi - pb)}])
tot.to_csv(OUT / "t3_1_totales.csv", index=False)
print("\n- 3.1a Totales (unión de los 3 hospitales)")
print(tot.to_string(index=False))

# Desglose por condado de la diferencia entre buffer e isocrona
des = []
for i, r in sel.iterrows():
    b, iso = store[i]["buf"].intersection(state), store[i]["iso"].intersection(state)
    for _, c in cond.iterrows():
        a = c.geometry.area
        ab, ai = c.geometry.intersection(b).area, c.geometry.intersection(iso).area
        if ab > 1e6 or ai > 1e6:
            des.append({"hospital": r["hospital"], "condado": c["nombre"], "km2_buffer": round(ab / 1e6),
                        "km2_isocrona": round(ai / 1e6), "pob_buffer": round(c["poblacion"] * ab / a),
                        "pob_isocrona": round(c["poblacion"] * ai / a)})
des = pd.DataFrame(des)
des.to_csv(OUT / "t3_1_desglose_condados.csv", index=False)
print("\n- 3.1a Desglose por condado")
print(des.to_string(index=False))

"""Optional spatial enrichment. Given lat/lon, fill MISSING fields from public datasets:
- ISRIC SoilGrids v2 (topsoil SOC and pH)
- Open-Meteo historical archive (annual rainfall and mean temperature)
Values are marked with provenance so confidence is lowered; user-supplied values are never overwritten."""
from __future__ import annotations

import os

import httpx

from .models import SiteProfile

TIMEOUT = float(os.getenv("GEO_TIMEOUT", "8"))


def soilgrids(lat: float, lon: float) -> dict:
    r = httpx.get("https://rest.isric.org/soilgrids/v2.0/properties/query",
                  params={"lat": lat, "lon": lon, "property": ["soc", "phh2o"], "depth": "0-5cm", "value": "mean"},
                  timeout=TIMEOUT)
    r.raise_for_status()
    out = {}
    for layer in r.json()["properties"]["layers"]:
        mean = layer["depths"][0]["values"]["mean"]
        if mean is None:
            continue
        if layer["name"] == "soc":      # dg/kg -> %
            out["soc_pct"] = round(mean / 100.0, 2)
        elif layer["name"] == "phh2o":  # pH*10
            out["ph"] = round(mean / 10.0, 1)
    return out


def climate(lat: float, lon: float, start: int = 2015, end: int = 2024) -> dict:
    """10-year mean: a single year is too noisy (monsoon years swing widely)."""
    r = httpx.get("https://archive-api.open-meteo.com/v1/archive",
                  params={"latitude": lat, "longitude": lon, "start_date": f"{start}-01-01",
                          "end_date": f"{end}-12-31", "daily": ["precipitation_sum", "temperature_2m_mean"],
                          "timezone": "UTC"}, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()["daily"]
    years = end - start + 1
    rain = sum(x for x in d["precipitation_sum"] if x is not None) / years
    temps = [x for x in d["temperature_2m_mean"] if x is not None]
    return {"rainfall_mm": round(rain), "mean_temp_c": round(sum(temps) / len(temps), 1)}


def enrich(profile: SiteProfile) -> tuple[SiteProfile, list[str]]:
    if profile.lat is None or profile.lon is None or os.getenv("DISABLE_GEO") == "1":
        return profile, []
    notes, add, prov = [], {}, {}
    for name, fn in (("soilgrids", soilgrids), ("open-meteo", climate)):
        try:
            for k, v in fn(profile.lat, profile.lon).items():
                if getattr(profile, k) is None and not (k == "rainfall_mm" and profile.rainfall_class):
                    add[k], prov[k] = v, name
        except Exception as e:  # network optional
            notes.append(f"{name} lookup failed ({type(e).__name__}); continuing without it")
    if add:
        notes.append("Estimated from coordinates: " + ", ".join(f"{k}={v} ({prov[k]})" for k, v in add.items()))
        profile = profile.merge(SiteProfile(**add, provenance=prov))
    return profile, notes

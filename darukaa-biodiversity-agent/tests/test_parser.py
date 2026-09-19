from app.parser import parse_structured, parse_text


def test_text_example_use_case():
    p = parse_text("Soil organic carbon: 0.3%, rainfall low, monoculture wheat, semi-arid region")
    assert p.soc_pct == 0.3 and p.rainfall_class == "low" and p.crop == "wheat"
    assert p.crop_system == "monoculture" and p.climate == "semi-arid" and p.land_use == "cropland"


def test_numbers_and_coords():
    p = parse_text("pH 5.1, 420 mm annual rainfall, 12% native habitat, heavy pesticide use near a canal at 19.07, 72.87")
    assert p.ph == 5.1 and p.rainfall_mm == 420 and p.native_habitat_pct == 12
    assert p.pesticide_use == "high" and p.near_water_body and (p.lat, p.lon) == (19.07, 72.87)


def test_structured_loose_keys():
    p = parse_structured({"soil_organic_carbon": "0.3%", "rainfall": "low", "crop": "monoculture wheat", "region": "semi-arid"})
    assert p.soc_pct == 0.3 and p.rainfall_class == "low" and p.crop == "wheat" and p.crop_system == "monoculture"


def test_structured_clean_keys():
    p = parse_structured({"soc_pct": 1.4, "rainfall_mm": 900, "land_use": "pasture"})
    assert p.soc_pct == 1.4 and p.rainfall_mm == 900 and p.land_use == "pasture"

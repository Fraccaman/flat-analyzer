import numpy as np
import pandas as pd
import pytest

from flatprice.geo import assign_area, assign_municipality, haversine_km, load_municipalities_geojson


def test_haversine():
    assert haversine_km(46.0037, 8.9511, 46.0037, 8.9511) == pytest.approx(0)
    # Lugano -> Bellinzona ~ 21 km as the crow flies
    assert 18 < haversine_km(46.0037, 8.9511, 46.1946, 9.0244) < 24
    d = haversine_km(np.array([46.0, 46.1]), np.array([8.9, 9.0]), 46.0037, 8.9511)
    assert d.shape == (2,)


def test_geojson_shape():
    fc = load_municipalities_geojson()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 100
    p = fc["features"][0]["properties"]
    assert isinstance(p["bfs"], int) and isinstance(p["name"], str)


def _df(rows):
    return pd.DataFrame(rows, columns=["latitude", "longitude", "locality"])


def test_point_in_polygon():
    df = assign_municipality(
        _df(
            [
                (45.9885, 8.9475, ""),  # Paradiso lakefront
                (46.0050, 8.9520, ""),  # Lugano, Piazza della Riforma
                (46.1946, 9.0244, ""),  # Bellinzona
                (46.1700, 8.7950, ""),  # Locarno
            ]
        )
    )
    assert df["municipality_bfs"].tolist() == [5210, 5192, 5002, 5113]
    assert df["municipality"].tolist()[:2] == ["Paradiso", "Lugano"]
    assert str(df["municipality_bfs"].dtype) == "Int64"


def test_locality_fallback():
    df = assign_municipality(
        _df(
            [
                (None, None, "PARADISO"),
                (np.nan, np.nan, "Collina d’Oro"),
                (np.nan, np.nan, "Sant'Antonino"),
                (np.nan, np.nan, "Lugano-Paradiso"),
                (np.nan, np.nan, "Viganello"),
                (48.0, 2.0, "Massagno"),  # outside Ticino -> fallback
                (np.nan, np.nan, "Atlantis"),
            ]
        )
    )
    got = df["municipality_bfs"]
    assert got.iloc[:6].astype(int).tolist() == [5210, 5236, 5017, 5210, 5192, 5196]
    assert pd.isna(got.iloc[6])
    assert pd.isna(df["municipality"].iloc[6])


def test_assign_area():
    df = assign_area(
        _df(
            [
                (46.0050, 8.9520, ""),  # Lugano centre
                (46.0235, 8.9705, ""),  # Pregassona
                (45.9885, 8.9475, ""),  # Paradiso -> own municipality
                (46.0100, 8.9400, "Massagno"),
                (np.nan, np.nan, "Lugano"),
            ]
        )
    )
    assert df["area"].iloc[0] == "Centro"
    assert df["area"].iloc[1] == "Pregassona"
    assert df["area"].iloc[2] == "Paradiso"
    assert df["area"].iloc[4] == "Lugano"
    assert df["dist_center_km"].iloc[0] < 0.5
    assert pd.isna(df["dist_center_km"].iloc[4])
    assert (df["dist_lake_km"].iloc[:3] < 3).all()

"""Sun elevation from timestamp + location, used to label day / dusk / night.

This uses only capture metadata (GPS-level location and the sensor timestamp),
never the human-written scene description. Accuracy is ~0.5 deg (NOAA-style
approximation), which is far tighter than the -6 deg civil-twilight threshold.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

# Approximate centres of the four nuScenes map areas.
LOCATIONS = {
    "boston-seaport": (42.336849, -71.057854),
    "singapore-onenorth": (1.2882100, 103.78475),
    "singapore-hollandvillage": (1.2993652, 103.78252),
    "singapore-queenstown": (1.2782562, 103.76734),
}


def sun_elevation_deg(utime_us: int, lat: float, lon: float) -> float:
    dt = datetime.fromtimestamp(utime_us / 1e6, tz=timezone.utc)
    doy = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    true_solar_min = hour * 60 + eqtime + 4 * lon
    ha = math.radians(true_solar_min / 4 - 180)
    lat_r = math.radians(lat)
    cos_zen = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))


def scene_sun_elevation(utime_us: int, location: str) -> float:
    lat, lon = LOCATIONS[location]
    return sun_elevation_deg(utime_us, lat, lon)

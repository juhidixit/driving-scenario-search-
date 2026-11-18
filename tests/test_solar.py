from datetime import datetime, timezone

from scenario_search.solar import scene_sun_elevation


def utime(y, mo, d, h, mi=0):
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1e6)


def test_singapore_noon_and_midnight():
    assert scene_sun_elevation(utime(2018, 7, 18, 5), "singapore-onenorth") > 60  # 13:00 local
    assert scene_sun_elevation(utime(2018, 7, 18, 16), "singapore-onenorth") < -30  # 00:00 local


def test_boston_evening_twilight():
    # 2018-09-26 sunset in Boston is ~22:35 UTC; one hour later the sun is well below -6 deg.
    assert scene_sun_elevation(utime(2018, 9, 26, 23, 40), "boston-seaport") < -6
    assert scene_sun_elevation(utime(2018, 9, 26, 17), "boston-seaport") > 30

import pandas as pd
import pvlib
from constants import LATITUDE, LONGITUDE, TIMEZONE

def get_expected_solar_output(latitude: float, longitude: float, start_yr: int, end_yr: int, timezone:str, surface_tilt: float | None =None, 
                              surface_azimuth: float=180) -> pd.Series:
    # Note: get_pvgis_hourly only requests full years of data, and only requests in UTC.  
    # To get a full local-tz set of continuous hours regardless of leap years we need a year on either side of a leap year
    # Note: The 7.75% loss factor is a correction estimate from aligning the PVGIS calculation with the ModelChain reference model

    pvgis_hourly = pvlib.iotools.get_pvgis_hourly(latitude, longitude,
                                            start=str(start_yr), end=end_yr,
                                            surface_tilt=latitude, surface_azimuth=surface_azimuth,
                                            pvcalculation=True, peakpower=1.0,
                                            loss=7.75, trackingtype=0,
                                            )
    weather_data = pvgis_hourly[0]
    solar_ac_estimate = weather_data['P'].rename('solar') / 1000.0 # convert to watts
    solar_ac_estimate = solar_ac_estimate.tz_convert(timezone)
    solar_ac_estimate = solar_ac_estimate.resample('1h', closed='right').last()
    return solar_ac_estimate
    

def get_or_cache_weather_data(latitude: float, longitude: float, start_yr: int, end_yr: int, timezone: str) -> pd.Series:
    cache_file = f'data/weather_{latitude}_{longitude}_{start_yr}_{end_yr}.csv'
    try:
        weather_data = pd.read_csv(cache_file, index_col=0, parse_dates=[0]).squeeze()
        weather_data.index = pd.DatetimeIndex(weather_data.index, tz=pd.Timestamp(weather_data.index[0]).tzinfo)
    except FileNotFoundError:
        weather_data = get_expected_solar_output(latitude, longitude, start_yr, end_yr, timezone)
        weather_data.to_csv(cache_file)
    return weather_data

# Reference data: San Francisco, around the 2020 leap year
REF_SOLAR_DATA = get_or_cache_weather_data(latitude=LATITUDE, longitude=LONGITUDE, start_yr=2019, end_yr=2021, timezone=TIMEZONE)

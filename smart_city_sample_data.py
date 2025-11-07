import pandas as pd
import numpy as np
from pytz.exceptions import AmbiguousTimeError
from solar import get_or_cache_weather_data
from batteryopt import build_tariff
import pathlib
import click

tz = 'Australia/Sydney'

def process_smart_city_data_from_file(fname, estimate_households_in,
                                      study_start='2012-10',
                                      study_end='2013-09',
                                      max_allowable_nulls=10):
    """
    The Smart Grid, Smart City dataset is available here:
    https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data

    It contains 30-minute interval data in kW.  Not all households start at the same time, the data is not tz-aware,
    and the file is very large.
    We seek to identify an appropriate subset of the data which has few NaNs for testing.
    """
    keep_cols = ['CUSTOMER_ID', 'READING_DATETIME', 'GENERAL_SUPPLY_KWH', 'CALENDAR_KEY']
    df = pd.read_csv(fname, nrows=8760 * 2 * estimate_households_in)
    df.columns = [c.strip() for c in df.columns]
    df = df[keep_cols]
    customer_ids = df['CUSTOMER_ID'].unique()
    print(len(customer_ids))
    df = df.set_index(['CUSTOMER_ID', 'READING_DATETIME'])

    all_customer_data = {}
    tz = 'Australia/Sydney'

    for c in customer_ids:
        customer_data = df.loc[c].copy()
        customer_data = customer_data.sort_values('CALENDAR_KEY')
        customer_data = customer_data['GENERAL_SUPPLY_KWH']
        try:
            customer_data.index = pd.DatetimeIndex(customer_data.index).tz_localize(tz=tz, ambiguous='infer',
                                                                                    nonexistent='shift_forward')
        except AmbiguousTimeError:
            try:
                customer_data.index = pd.DatetimeIndex(customer_data.index).tz_localize(tz=tz, ambiguous=True,
                                                                                        nonexistent='shift_forward')
            except AmbiguousTimeError:
                customer_data.index = pd.DatetimeIndex(customer_data.index).tz_localize(tz=tz, ambiguous='NaT',
                                                                                        nonexistent='shift_forward')
        customer_data = customer_data[~customer_data.index.duplicated(keep='first')]
        all_customer_data[c] = customer_data.rename(c)

    all_customer_df = pd.DataFrame.from_dict(all_customer_data, orient='columns')

    study_dataset = all_customer_df[study_start:study_end]
    study_dataset = study_dataset.loc[:, study_dataset.isnull().sum() <= max_allowable_nulls]
    study_dataset = study_dataset.resample('30min').last()
    study_dataset = study_dataset.interpolate(method='linear')
    return study_dataset


@click.command()
@click.argument('fname', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def smart_city_data_to_net_load(fname, output_dir: str):
    gross_load = process_smart_city_data_from_file(fname, estimate_households_in=500,
                                                      study_start='2012-10',
                                                      study_end='2013-09',
                                                      max_allowable_nulls=10
                                                      )
    coverage_ratio = 0.7
    output_dir = pathlib.Path(output_dir)

    solar_data = get_or_cache_weather_data(latitude=-33.8688, longitude=151.2093,  # Sydney, Australia
                                         start_yr=2012, end_yr=2013,
                                         timezone=tz,
                                           surface_tilt=None,
                                           surface_azimuth=0,
                                            )
    solar_data = solar_data.resample('30min').last().ffill()
    solar_data = solar_data.reindex(gross_load.index)

    sydney_1kw_panel_annual_generation_kwh = solar_data.sum()

    site_net_load_kwh = gross_load.sum() * 0.5

    solar_sizes_kW = coverage_ratio * site_net_load_kwh / sydney_1kw_panel_annual_generation_kwh

    solar_gen = pd.DataFrame(np.outer(solar_data, solar_sizes_kW),
                             index=solar_data.index, columns=solar_sizes_kW.index)

    net_load_data = gross_load - solar_gen

    tariff = build_tariff(net_load_data.index)

    net_load_data.to_csv(output_dir / 'smart_city_net_load.csv')
    tariff.to_csv(output_dir / 'smart_city_tariff.csv')

    return


if __name__ == '__main__':
    smart_city_data_to_net_load()
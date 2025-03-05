from solar import REF_SOLAR_DATA

import cvxpy as cp
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from constants import TIMEZONE

def process_pge_meterdata(fname: str) -> pd.Series:
    header_row = None
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES"):
                header_row = i
                break
    if header_row is None:
        raise ValueError("Header row not found!")

    sample_consumption = pd.read_csv(fname, parse_dates={'Datetime': ['DATE', 'START TIME']}, skiprows=header_row)
    sample_consumption['Datetime'] = pd.DatetimeIndex(sample_consumption['Datetime']).tz_localize('US/Pacific', ambiguous='NaT', nonexistent='NaT')
    sample_consumption = sample_consumption[sample_consumption['Datetime'].notnull()]
    sample_consumption = sample_consumption.set_index('Datetime')
    elec_usage = sample_consumption['USAGE (kWh)'].rename('load')

    elec_end_date = elec_usage.index[-1]
    if elec_usage.index[0] < (elec_end_date - pd.DateOffset(years=1)):  
        elec_usage = elec_usage.loc[elec_end_date - pd.DateOffset(years=1):elec_end_date]
    return elec_usage


def merge_solar_and_load_data(elec_usage: pd.Series, solar_ac_estimate: pd.Series) -> pd.DataFrame:
    elec_end_date = elec_usage.index[-1]
    leap_day = None
    for t in elec_usage.index:
        if (t.month==2) & (t.day==29):
            leap_day = t
            break

    if leap_day:
        shift_by_yrs = 2020 - leap_day.year
    elif elec_end_date.month == 12 & elec_end_date.day == 31:
        shift_by_yrs = 2019 - elec_end_date.year
    elif elec_end_date.month > 2:
        shift_by_yrs = 2021 - elec_end_date.year
    else:
        shift_by_yrs = 2019 - elec_end_date.year

    print("solar_ac_estimate")
    print(solar_ac_estimate.head())
    print("elec_usage")
    print(elec_usage.head())


    solar_ac_estimate.index = (solar_ac_estimate.index.tz_convert('UTC') - pd.DateOffset(years=shift_by_yrs)).tz_convert(TIMEZONE)
    solar_ac_estimate = solar_ac_estimate.resample('1h', closed='right').last().ffill()  # Deal with any gaps related to shifted DST; thankfully these are in the night

    site_data = pd.DataFrame(elec_usage).join(solar_ac_estimate, how='left')
    return site_data


def build_tariff(idx: pd.DatetimeIndex) -> pd.DataFrame:
    px_buy = pd.Series(0.4, index=idx, name='px_buy')
    px_buy.loc[px_buy.between_time('16:00', '21:00').index] = 0.52

    px_sell = pd.Series(0.05, index=idx, name='px_sell')
    px_sell.loc[px_sell.between_time('15:00', '20:00').index] = 0.08

    return pd.DataFrame({'px_buy': px_buy, 'px_sell': px_sell})


def run_optimization(site_data: pd.DataFrame, tariff: pd.DataFrame, batt_rt_eff=0.85,
                     batt_e_max=13.5, batt_p_max=5) -> pd.DataFrame:
    assert site_data.index.equals(tariff.index), "Dataframes must have the same index"

    dt = 1.0 

    oneway_eff = np.sqrt(batt_rt_eff)
    backup_reserve = 0.2
    e_min = backup_reserve * batt_e_max

    n = site_data.shape[0]
    E_0 = e_min

    E_transition = np.hstack([np.eye(n), np.zeros(n).reshape(-1,1)])

    P_batt_charge = cp.Variable(n)
    P_batt_discharge = cp.Variable(n)
    P_grid_buy = cp.Variable(n)
    P_grid_sell = cp.Variable(n)
    E = cp.Variable(n+1)

    # Power flows are all AC, and are signed relative to the bus: injections to the bus are positive, withdrawals/exports from the bus are negative

    constraints = [-batt_p_max <= P_batt_charge,
                P_batt_charge <= 0,
                0 <= P_batt_discharge,
                P_batt_discharge <= batt_p_max,
                0 <= P_grid_buy,
                P_grid_sell <= 0,
                e_min <= E,
                E <= batt_e_max,
                E[1:] == E_transition @ E - (P_batt_charge * oneway_eff + P_batt_discharge / oneway_eff) * dt,
                P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell - site_data['load'] + site_data['solar'] == 0,
                E[0] == E_0
                ]

    obj = cp.Minimize(P_grid_sell @ tariff['px_sell'] + P_grid_buy @ tariff['px_buy'])

    prob = cp.Problem(obj, constraints)

    opt_start = time.time()
    prob.solve()
    print(f"Optimization done in {time.time() - opt_start :.3f} seconds")

    res = pd.DataFrame.from_dict({'P_batt': P_batt_charge.value + P_batt_discharge.value,
                        'P_grid': P_grid_buy.value + P_grid_sell.value,
                        'E': E[1:].value}).set_index(site_data.index)
    return res


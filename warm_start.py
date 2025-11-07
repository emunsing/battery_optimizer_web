import click
import numpy as np
import pandas as pd
import scipy.sparse as sps
import time
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel

@click.group()
def cli():
    pass

def run_cylp_optimization(site_data, tariff, batt_rt_eff=0.85,
                          batt_e_max=13.5, batt_p_max=5):
    net_load = site_data['load'] - site_data['solar']
    n = len(net_load)
    dt = 1.0
    oneway_eff = np.sqrt(batt_rt_eff)
    backup_reserve = 0.2
    e_min = backup_reserve * batt_e_max
    E_0 = e_min

    # Create model
    m = CyLPModel()

    # Variables
    P_batt_charge   = m.addVariable('P_batt_charge', n) #, lower=-batt_p_max, upper=0)
    P_batt_discharge= m.addVariable('P_batt_discharge', n) #, lower=0, upper=batt_p_max)
    P_grid_buy      = m.addVariable('P_grid_buy', n) #, lower=0)
    P_grid_sell     = m.addVariable('P_grid_sell', n) #, upper=0)
    E               = m.addVariable('E', n+1) #, lower=e_min, upper=batt_e_max)

    m+= P_batt_charge <= 0
    m+= P_batt_charge >= -batt_p_max
    m+= P_batt_discharge >= 0
    m+= P_batt_discharge <= batt_p_max
    m+= P_grid_buy >= 0
    m+= P_grid_sell <= 0
    m+= E >= e_min
    m+= E <= batt_e_max

    # Objective
    px_buy  = tariff['px_buy'].to_numpy()
    px_sell = tariff['px_sell'].to_numpy()
    m.objective = (
        P_grid_sell @ px_sell +
        P_grid_buy  @ px_buy
    )

    # Initial energy constraint
    m.addConstraint(E[0] == E_0)

    # Energy balance: E[i+1] = E[i] - (charge*eff + discharge/eff)*dt
    E_next = E[1: n + 1]
    E_now = E[0: n]
    m.addConstraint(E_next - E_now + P_batt_charge * (dt * oneway_eff) + P_batt_discharge * (dt / oneway_eff) == 0)

    # Power balance: P_charge + P_discharge + P_buy + P_sell - net_load = 0
    power_balance = m.addConstraint(P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell == net_load.to_numpy())

    # Solve
    s = CyClpSimplex(m)
    t0 = time.time()
    s.primal()
    print(f"Initial solve time: {time.time() - t0:.3f} s")

    # ---- Warm start: modify RHS only ----
    new_net_load = net_load * 1.05  # e.g., small perturbation

    t1 = time.time()
    power_idx = m.inds.constIndex[power_balance.name]
    current_upper = s.constraintsUpper
    current_lower = s.constraintsLower
    current_upper[power_idx] = new_net_load.to_numpy()
    current_lower[power_idx] = new_net_load.to_numpy()
    # Instead of re-building, we just update RHS directly in solver
    s.setRowLowerArray(current_lower)
    s.setRowUpperArray(current_upper)

    # t1 = time.time()
    s.primal()   # re-solve, warm start
    print(f"Warm-start re-solve time: {time.time() - t1:.3f} s")

    return s.primalVariableSolution['E'], s


@cli.command()
def cylp():
    input_df = pd.read_csv("~/src/battery_optimizer_web/data/tmp_full_noindex.csv")
    run_cylp_optimization(input_df, input_df)

if __name__ == "__main__":
    cli()
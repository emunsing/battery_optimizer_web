import click
import numpy as np
import pandas as pd
import scipy.sparse as sps
import cvxpy as cp
import time
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel
from highspy import Highs, HighsModelStatus
import highspy

@click.group()
def cli():
    pass


def optimize_many_sites_cvxpy_base(many_sites_net_load_data: pd.DataFrame,
                     tariff: pd.DataFrame, batt_rt_eff=0.85,
                     batt_e_max=13.5, batt_p_max=5, solver=None) -> pd.DataFrame:

    assert many_sites_net_load_data.index.equals(tariff.index), "Dataframes must have the same index"
    assert tariff.index.diff()[1:].unique().size == 1, "Tariff must have uniform time steps"
    dt = tariff.index.diff().unique()[1].total_seconds() / 3600.0  # in hours
    prep_time_start = time.time()

    oneway_eff = np.sqrt(batt_rt_eff)
    backup_reserve = 0.2
    e_min = backup_reserve * batt_e_max

    n = many_sites_net_load_data.shape[0]
    net_load = cp.Parameter(n)

    E_0 = e_min
    E_transition = sps.hstack([sps.eye(n, format="csr"), sps.csr_matrix((n, 1))], format="csr")

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
                P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell - net_load == 0,
                E[0] == E_0
                ]

    obj = cp.Minimize(P_grid_sell @ tariff['px_sell'] + P_grid_buy @ tariff['px_buy'])

    prob = cp.Problem(obj, constraints)

    solve_times = []
    prep_times = []

    for site_id, site_net_load in many_sites_net_load_data.items():
        net_load.value = site_net_load.values
        prep_time_end = time.time()
        opt_start = time.time()
        prob.solve(warm_start=True, solver=solver)
        elapsed = time.time() - opt_start
        solve_times.append(elapsed)
        prep_times.append(prep_time_end - prep_time_start)
        prep_time_start = time.time()
        print(f"Optimization done in {elapsed :.3f} seconds")

    solve_time_df = pd.DataFrame({'prep_time': prep_times, 'solve_time': solve_times})

    return solve_time_df

def optimize_many_sites_cvxpy_e_simple(many_sites_net_load_data: pd.DataFrame,
                     tariff: pd.DataFrame, batt_rt_eff=0.85,
                     batt_e_max=13.5, batt_p_max=5, solver=None) ->  pd.DataFrame:
    assert many_sites_net_load_data.index.equals(tariff.index), "Dataframes must have the same index"
    assert tariff.index.diff()[1:].unique().size == 1, "Tariff must have uniform time steps"
    dt = tariff.index.diff().unique()[1].total_seconds() / 3600.0  # in hours

    prep_time_start = time.time()
    oneway_eff = np.sqrt(batt_rt_eff)
    backup_reserve = 0.2
    e_min = backup_reserve * batt_e_max

    n = many_sites_net_load_data.shape[0]
    net_load = cp.Parameter(n)

    E_0 = e_min

    P_batt_charge = cp.Variable(n)
    P_batt_discharge = cp.Variable(n)
    P_grid_buy = cp.Variable(n)
    P_grid_sell = cp.Variable(n)
    E = cp.Variable(n+1)
    E_next = E[1: n + 1]
    E_now = E[0: n]

    # Power flows are all AC, and are signed relative to the bus: injections to the bus are positive, withdrawals/exports from the bus are negative

    constraints = [-batt_p_max <= P_batt_charge,
                P_batt_charge <= 0,
                0 <= P_batt_discharge,
                P_batt_discharge <= batt_p_max,
                0 <= P_grid_buy,
                P_grid_sell <= 0,
                e_min <= E,
                E <= batt_e_max,
                E[1:] == E_now - E_next - (P_batt_charge * oneway_eff + P_batt_discharge / oneway_eff) * dt,
                P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell - net_load == 0,
                E[0] == E_0
                ]

    obj = cp.Minimize(P_grid_sell @ tariff['px_sell'] + P_grid_buy @ tariff['px_buy'])

    prob = cp.Problem(obj, constraints)

    solve_times = []
    prep_times = []

    for site_id, site_net_load in many_sites_net_load_data.items():
        net_load.value = site_net_load.values
        prep_time_end = time.time()
        opt_start = time.time()
        prob.solve(warm_start=True, solver=solver)
        elapsed = time.time() - opt_start
        solve_times.append(elapsed)
        prep_times.append(prep_time_end - prep_time_start)
        prep_time_start = time.time()
        print(f"Optimization done in {elapsed :.3f} seconds")

    solve_time_df = pd.DataFrame({'prep_time': prep_times, 'solve_time': solve_times})

    return solve_time_df


def run_cylp_optimization(many_sites_net_load_data, tariff, batt_rt_eff=0.85,
                          batt_e_max=13.5, batt_p_max=5) ->  pd.DataFrame:
    assert many_sites_net_load_data.index.equals(tariff.index), "Dataframes must have the same index"
    assert tariff.index.diff()[1:].unique().size == 1, "Tariff must have uniform time steps"
    dt = tariff.index.diff().unique()[1].total_seconds() / 3600.0  # in hours
    prep_time_start = time.time()

    net_load = many_sites_net_load_data.iloc[:, 0]
    n = len(many_sites_net_load_data)
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
    prep_time_end = time.time()
    t0 = time.time()
    s.primal()
    solve_times = [time.time() - t0]
    prep_times = [prep_time_end - prep_time_start]
    print(f"Initial solve time: {time.time() - t0:.3f} s")

    for i in range(1, many_sites_net_load_data.shape[1]):
        site_net_load = many_sites_net_load_data.iloc[:, i].to_numpy()

        prep_time_start = time.time()
        power_idx = m.inds.constIndex[power_balance.name]
        current_upper = s.constraintsUpper
        current_lower = s.constraintsLower
        current_upper[power_idx] = site_net_load
        current_lower[power_idx] = site_net_load
        # Instead of re-building, we just update RHS directly in solver
        s.setRowLowerArray(current_lower)
        s.setRowUpperArray(current_upper)

        prep_times.append(time.time() - prep_time_start)
        t1 = time.time()
        s.primal()   # re-solve, warm start
        elapsed = time.time() - t1
        solve_times.append(elapsed)
        print(f"Warm-start re-solve time: {elapsed:.3f} s")

    solve_time_df = pd.DataFrame({'prep_time': prep_times, 'solve_time': solve_times})

    return solve_time_df


def run_highs_optimization(many_sites_net_load_data, tariff, batt_rt_eff=0.85,
                           batt_e_max=13.5, batt_p_max=5):
    assert many_sites_net_load_data.index.equals(tariff.index), "Dataframes must have the same index"
    assert tariff.index.diff()[1:].unique().size == 1, "Tariff must have uniform time steps"
    dt = tariff.index.diff().unique()[1].total_seconds() / 3600.0  # hours

    prep_time_start = time.time()

    net_load0 = many_sites_net_load_data.iloc[:, 0].to_numpy()
    n = len(net_load0)
    oneway_eff = np.sqrt(batt_rt_eff)
    backup_reserve = 0.2
    e_min = backup_reserve * batt_e_max
    E0 = e_min

    # Create solver
    solver = highspy.Highs()

    # Add variables (one-by-one)
    P_charge = [solver.addVariable(lb = -batt_p_max, ub = 0.0) for _ in range(n)]
    P_discharge = [solver.addVariable(lb = 0.0, ub = batt_p_max) for _ in range(n)]
    P_buy = [solver.addVariable(lb = 0.0, ub = highspy.kHighsInf) for _ in range(n)]
    P_sell = [solver.addVariable(lb = -highspy.kHighsInf, ub = 0.0) for _ in range(n)]
    E = [solver.addVariable(lb = e_min, ub = batt_e_max) for _ in range(n+1)]

    # Objective: minimise cost (net of buys and sells)
    # In HiGHS default is minimisation; we define c for each var
    # We need arrays for cost coefficients
    px_buy = tariff['px_buy'].to_numpy()
    px_sell = tariff['px_sell'].to_numpy()
    for i in range(n):
        solver.changeColCost(P_buy[i], px_buy[i])
        solver.changeColCost(P_sell[i], px_sell[i])

    # Constraints

    # (A) Initial energy: E[0] == E0
    solver.addConstr(E[0] == E0)

    # (B) Energy balance for each time step
    for i in range(n):
        # E[i+1] - E[i] + P_charge[i]*dt*oneway_eff + P_discharge[i]*dt/oneway_eff == 0
        solver.addConstr(E[i+1] - E[i]
                         + (dt * oneway_eff) * P_charge[i]
                         + (dt / oneway_eff) * P_discharge[i]
                         == 0)

    # (C) Power balance: P_charge + P_discharge + P_buy + P_sell == net_load0[i]
    for i in range(n):
        solver.addConstr(P_charge[i]
                         + P_discharge[i]
                         + P_buy[i]
                         + P_sell[i]
                         == net_load0[i])

    # Solve first scenario
    prep_times = [time.time() - prep_time_start]
    t0 = time.time()
    status = solver.run()
    t_initial = time.time() - t0
    print(f"Initial solve time: {t_initial:.3f} s, status = {status}")

    # Warm-start loop over other sites
    solve_times = [t_initial]
    for j in range(1, many_sites_net_load_data.shape[1]):
        net_load_j = many_sites_net_load_data.iloc[:, j].to_numpy()
        prep_time_start = time.time()
        # Update RHS of power‐balance constraints (C)
        for i in range(n):
            solver.changeRowBounds(( (1 + n) + i ), net_load_j[i], net_load_j[i])

        # Hot‐start solve
        prep_times.append(time.time() - prep_time_start)
        t1 = time.time()
        status = solver.run()
        elapsed = time.time() - t1
        solve_times.append(elapsed)
        print(f"Warm-start re-solve time: {elapsed:.3f} s, status = {status}")

    # Return list of E trajectories (or stack) and solver
    solve_time_df = pd.DataFrame({'prep_time': prep_times, 'solve_time': solve_times})

    return solve_time_df


@cli.command()
def cylp():
    input_df = pd.read_csv("~/src/battery_optimizer_web/data/tmp_full_noindex.csv")
    run_cylp_optimization(input_df, input_df)

@cli.command()
@click.argument('net-load', type=click.Path(exists=True))
@click.argument('tariff', type=click.Path(exists=True))
@click.option('--n-sites', type=int, default=3, help='Number of sites to test')
def performance_comparison(net_load, tariff, n_sites):
    many_sites_net_load_data = pd.read_csv(net_load, index_col=0, parse_dates=True)
    tariff_data = pd.read_csv(tariff, index_col=0, parse_dates=True)
    assert many_sites_net_load_data.index.equals(tariff_data.index), "Dataframes must have the same index"

    many_sites_net_load_data = many_sites_net_load_data.iloc[:, :n_sites]

    solver_times = {}

    # for solver in [None, cp.CLARABEL, cp.CBC, cp.HIGHS]:
    for solver in [None]:
        print(f"Testing CVXPy with solver {solver}")
        print("Optimizing using CVXPy base formulation")
        cvxpy_base_times = optimize_many_sites_cvxpy_base(many_sites_net_load_data, tariff_data, solver=solver)
        solver_times[f"{solver}_base"] = cvxpy_base_times
        print("Optimizing using CVXPy simplified transition formulation")
        cvxpy_e_simple_times = optimize_many_sites_cvxpy_e_simple(many_sites_net_load_data, tariff_data, solver=solver)
        solver_times[f"{solver}_simple"] = cvxpy_e_simple_times

    print("Optimizing using CyLP warm start")
    cylp_times = run_cylp_optimization(many_sites_net_load_data, tariff_data)
    solver_times["cylp_warmstart"] = cylp_times

    print("Optimizing using HiGHS warm start")
    highs_times = run_highs_optimization(many_sites_net_load_data, tariff_data)
    solver_times["highs_warmstart"] = highs_times

    df = pd.DataFrame.concat(solver_times, names=['solver', 'run'])
    print(df)


if __name__ == "__main__":
    cli()
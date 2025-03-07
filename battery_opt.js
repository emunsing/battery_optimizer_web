
import Module from './clp_wasm_module.js';

export function optimizeBattery({battery_max_kwh, batt_min_kwh, batt_max_kw, batt_rt_eff, dt, solar, consumption, px_buy, px_sell}){

    const optimization_entry = performance.now();

    // Variable indexing
    const n = solar.length;
    const nVars = 5 * n + 1;  // P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell + E
    const nRows = 2 * n; // Total constraints: energy balance (n rows) + power balance (n rows) = 48 rows.
    const wrapper = new Module.ClpWrapper();

    const n_P_batt_charge = 0;
    const n_P_batt_discharge = n;
    const n_P_grid_buy = 2 * n;
    const n_P_grid_sell = 3 * n;
    const n_E = 4 * n;

    const E0 = batt_min_kwh;
    const oneway_eff = Math.sqrt(batt_rt_eff);
    const inv_eff = 1 / oneway_eff;
    const max_site_current = Math.max(...solar, ...consumption, batt_max_kw) * 10;
    
    // Initialize arrays
    const obj = new Array(nVars).fill(0);
    const col_lb = new Array(nVars).fill(-max_site_current);
    const col_ub = new Array(nVars).fill(max_site_current);
    const row_lb = new Array(nRows).fill(0);  // Row constraints: Default to tight equality
    const row_ub = new Array(nRows).fill(0);
    
    // P_batt_charge: -batt_max_kw <= x <= 0
    for (let i = 0; i < n; i++) {
        col_lb[i + n_P_batt_charge] = -batt_max_kw;
        col_ub[i + n_P_batt_charge] = 0;
    }
    
    // P_batt_discharge: 0 <= x <= batt_max_kw
    for (let i = 0; i < n; i++) {
        col_lb[i + n_P_batt_discharge] = 0;
        col_ub[i + n_P_batt_discharge] = batt_max_kw;
    }
    
    // P_grid_buy: 0 <= x; cost = px_buy
    for (let i = 0; i < n; i++) {
        col_lb[i + n_P_grid_buy] = 0;
        obj[i + n_P_grid_buy] = px_buy[i] * dt;
    }
    
    // P_grid_sell: x <= 0; cost = px_sell
    for (let i = 0; i < n; i++) {
        col_ub[i + n_P_grid_sell] = 0;
        obj[i + n_P_grid_sell] = px_sell[i] * dt;
    }
    
    // E: 0 <= x <= batt_max_kwh; initial energy bound
    col_lb[n_E] = E0;
    col_ub[n_E] = E0;
    
    for (let i = 1; i < n + 1; i++) {
        col_lb[i + n_E] = batt_min_kwh;
        col_ub[i + n_E] = battery_max_kwh;
    }
    
    // --- Build constraints ---
    console.log("Building constraints");
    const matrix = new Array(nRows * nVars).fill(0);
    let rowStart = 0;

    // Transition matrix: E[i+1] - E[i] - (oneway_eff * P_batt_charge[i] + inv_eff * P_batt_discharge[i]) = 0.
    for (let i = 0; i < n; i++) {
        rowStart = i * nVars;
        matrix[rowStart + n_P_batt_charge + i] = -1 * oneway_eff * dt;
        matrix[rowStart + n_P_batt_discharge + i] = -1 * inv_eff * dt;
        matrix[rowStart + n_E + i] = 1;
        matrix[rowStart + n_E + i + 1] = -1;
    }
    
    // Power Balance Constraints: P_batt_charge + P_batt_discharge + P_grid_buy + P_grid_sell = load - solar.
    let net_load = null
    for (let i = 0; i < n; i++) {
        rowStart = (n + i) * nVars;
        matrix[rowStart + n_P_batt_charge + i] = 1;
        matrix[rowStart + n_P_batt_discharge + i] = 1;
        matrix[rowStart + n_P_grid_buy + i] = 1;
        matrix[rowStart + n_P_grid_sell + i] = 1;
        net_load = consumption[i] - solar[i];
        row_lb[n + i] = net_load;
        row_ub[n + i] = net_load;
    }
    
    console.log("Entering solver");
    const solver_entry = performance.now();
    let success = wrapper.loadProblem(obj, col_lb, col_ub, row_lb, row_ub, matrix); // returns true if the problem dimensions matched
    wrapper.primal();
    const solver_finish = performance.now();

    // Extracting results
    const E = new Array(n).fill(0);
    const P_batt_charge = new Array(n).fill(0);
    const P_batt_discharge = new Array(n).fill(0);
    const P_grid_buy = new Array(n).fill(0);
    const P_grid_sell = new Array(n).fill(0);
    const P_batt = new Array(n).fill(0);
    const P_grid = new Array(n).fill(0);
    const solution = wrapper.getSolutionArray(3)
    for (let i = 0; i < n; i++) {
        P_batt_charge[i] = parseFloat(solution[n_P_batt_charge + i]);
        P_batt_discharge[i] = parseFloat(solution[n_P_batt_discharge + i]);
        P_grid_buy[i] = parseFloat(solution[n_P_grid_buy + i]);
        P_grid_sell[i] = parseFloat(solution[n_P_grid_sell + i]);
        P_batt[i] = P_batt_charge[i] + P_batt_discharge[i];
        P_grid[i] = P_grid_buy[i] + P_grid_sell[i];
        E[i] = parseFloat(solution[n_E + i + 1]);
    }

    const optimization_finish = performance.now();
    const optimization_stats = {preprocess: solver_entry - optimization_entry, 
                                solve: solver_finish - solver_entry, 
                                postprocess: optimization_finish - solver_finish};

    return {P_batt, P_grid, E, optimization_stats};
}

export function testOpt({px, e_max}){
    // Simplified energy arbitrage problem, no inefficiency or net load

    const n = px.length;
    const nRows = n;
    const batt_p_max = 1;
    const batt_e_max = e_max;
    const wrapper = new Module.ClpWrapper();
    console.log("initial variable initialization done");

    const nVars = 2 * n + 1;  // P(t) + E(t+1)
    // P : 0:n-1
    // E : n:2n

    const obj = new Array(nVars).fill(0);
    const col_lb = new Array(nVars).fill(-Infinity);
    const col_ub = new Array(nVars).fill(Infinity);
    const row_lb = new Array(nRows).fill(0);
    const row_ub = new Array(nRows).fill(0);

    // Objective function: Minimize px * P
    for (let i = 0; i < n; i++) {
        obj[i] = px[i];
    }

    //  -batt_p_max <= P <= batt_p_max
    for (let i = 0; i < n; i++) {
        col_lb[i] = -batt_p_max;
        col_ub[i] = batt_p_max;
    }

    // E_init
    col_lb[n] = 0;
    col_ub[n] = 0;
    
    // 0 <= E <= batt_e_max
    for (let i = n+1; i < nVars; i++) {
        col_lb[i] = 0;
        col_ub[i] = batt_e_max;
    }

    // 0 = -E_{t+1} + E_t + P_t
    const mat_length = nVars * nRows;
    const matrix = new Array(mat_length).fill(0);
    let rowStart = 0;
    for (let i = 0; i < nRows; i++){
        rowStart = i * nVars;
        matrix[rowStart + i] = 1;
        matrix[rowStart + n + i] = 1;
        matrix[rowStart + n + i + 1] = -1;
    }

    console.log("Done with problem setup!");
    console.log("objective size: ", obj.length)
    console.log("col_lb size: ", col_lb.length)
    console.log("col_ub size: ", col_ub.length)
    console.log("row_lb size: ", row_lb.length)
    console.log("row_ub size: ", row_ub.length)
    console.log("matrix size: ", matrix.length)

    let success = wrapper.loadProblem(obj, col_lb, col_ub, row_lb, row_ub, matrix); // returns true if the problem dimensions matched
    wrapper.primal();

    // Extracting results
    const E = new Array(n).fill(0);
    const P = new Array(n).fill(0);
    const solution = wrapper.getSolutionArray(3)
    for (let i = 0; i < n; i++) {
        P[i] = parseFloat(solution[i]);
        E[i] = parseFloat(solution[n + i + 1]);
    }
    console.log(P)
    console.log(E)  
    return {P, E};
}

import Module from './clp_wasm_module.js';

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
    let matrix = new Array(nVars * nRows).fill(0);
    let rowStart = 0;
    for (let i = 0; i < nRows; i++){
        rowStart = i * nVars;
        matrix[rowStart + i] = 1;
        matrix[rowStart + n + i + 1] = -1;
        matrix[rowStart + n + i] = 1;
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
    const solution = wrapper.getSolutionArray(1)
    for (let i = 0; i < n; i++) {
        P[i] = solution[i];
        E[i] = solution[n + i + 1];
    }
    console.log(P)
    console.log(E)  
    return {P, E};
}
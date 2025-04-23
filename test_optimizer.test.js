import { optimizeBattery } from './battery_opt.js';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const Module = require('./clp_wasm_module.js');

// async function debugOptimizeBattery() {
test('optimizeBattery integrates with WebAssembly module', async () => {
    // Wait for the WebAssembly module to initialize
    await new Promise((resolve) => {
      Module.onRuntimeInitialized = resolve;
    });
  
  // Mock input data
  const battery_max_kwh = 13.5;
  const batt_min_kwh = 2.7; // 20% reserve
  const batt_max_kw = 5.0;
  const batt_rt_eff = 0.85;
  const dt = 1.0;

  const solar = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; // Example solar generation data
  const consumption = [5, 4, 3, 2, 1, 0, 1, 2, 3, 4]; // Example consumption data
  const px_buy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]; // Example buy prices
  const px_sell = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.05, 0.05]; // Example sell prices

  // Call optimizeBattery and log results
  const result = optimizeBattery({
    battery_max_kwh,
    batt_min_kwh,
    batt_max_kw,
    batt_rt_eff,
    dt,
    solar,
    consumption,
    px_buy,
    px_sell,
  });

  console.log("Optimization Results:");
  console.log("P_batt:", result.P_batt);
  console.log("P_grid:", result.P_grid);
  console.log("E:", result.E);
  console.log("Optimization Stats:", result.optimization_stats);
});

// Run the debug function
// debugOptimizeBattery();
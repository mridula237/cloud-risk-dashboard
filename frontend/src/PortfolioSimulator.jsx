import { useState } from "react";
import { runSimulation } from "./api";

export default function PortfolioSimulator({ setSimulationPaths }) {

  const [investment, setInvestment] = useState(10000);
  const [days, setDays] = useState(252);
  const [simulations, setSimulations] = useState(200);

  const handleRun = async () => {
    try {
      const data = await runSimulation({
        investment: Number(investment),
        days: Number(days),
        simulations: Number(simulations),
      });

      if (data && data.paths) {
        setSimulationPaths(data.paths);
      }

    } catch (err) {
      console.error("Simulation error:", err);
    }
  };

  return (
    <div className="simulator-panel">

      <h3>Portfolio Simulator</h3>

      <div className="simulator-controls">

        <div className="sim-field">
          <label>Initial Investment ($)</label>
          <input
            type="number"
            value={investment}
            onChange={(e) => setInvestment(e.target.value)}
          />
        </div>

        <div className="sim-field">
          <label>Time Horizon (Days)</label>
          <input
            type="number"
            value={days}
            onChange={(e) => setDays(e.target.value)}
          />
        </div>

        <div className="sim-field">
          <label>Number of Simulations</label>
          <input
            type="number"
            value={simulations}
            onChange={(e) => setSimulations(e.target.value)}
          />
        </div>

        <button onClick={handleRun}>
          Run Simulation
        </button>

      </div>
    </div>
  );
}
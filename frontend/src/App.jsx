import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter,
  ReferenceLine, PieChart, Pie, Cell, Legend,
} from "recharts";

import PortfolioSimulator from "./PortfolioSimulator";
import {
  getPortfolios,
  getPortfolioReturns,
  getVolatility,
  getMonteCarlo,
  getDrawdown,
  getEfficientFrontier,
  getPortfolioAllocation,
  getCorrelationMatrix,
  getStressTest,
  getVarBacktest,
} from "./api";
import "./App.css";

const COLORS = ["#00d4ff", "#ff6b6b", "#ffa500", "#00c49f", "#8884d8",
                "#ff00ff", "#00ff88", "#ffdd00", "#ff8042", "#a4de6c"];

const ASSET_MAP = {
  1:"AAPL",2:"MSFT",3:"NVDA",4:"AMZN",5:"GOOGL",
  6:"SPY",7:"QQQ",8:"DIA",9:"IWM",10:"XLF",
  11:"XLE",12:"XLI",13:"TLT",14:"IEF",15:"LQD",
  16:"GLD",17:"SLV",18:"USO",19:"XLP",20:"XLU",
};

function KupiecCard({ data }) {
  if (!data) return null;
  const passed = data.kupiec_passed;
  return (
    <div className="card" style={{ gridColumn: "span 3" }}>
      <h3>VaR Backtest (Kupiec Test)</h3>
      <p style={{ color: passed ? "#00ff88" : "#ff6b6b", fontSize: "1.4rem", fontWeight: "bold" }}>
        {passed ? "✓ PASS" : "✗ FAIL"}
      </p>
      <p style={{ fontSize: "0.85rem", color: "#aaa", marginTop: 4 }}>
        Expected violations: {data.expected_violations} &nbsp;|&nbsp;
        Actual: {data.actual_violations} &nbsp;|&nbsp;
        p-value: {data.kupiec_p_value}
      </p>
      <p style={{ fontSize: "0.78rem", color: "#777", marginTop: 4 }}>
        {data.interpretation}
      </p>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [portfolios, setPortfolios]     = useState([]);
  const [portfolioId, setPortfolioId]   = useState(1);

  const [returns, setReturns]           = useState([]);
  const [volatility, setVolatility]     = useState([]);
  const [montecarlo, setMontecarlo]     = useState([]);
  const [drawdown, setDrawdown]         = useState([]);
  const [frontier, setFrontier]         = useState([]);
  const [allocation, setAllocation]     = useState([]);
  const [correlation, setCorrelation]   = useState([]);
  const [stress, setStress]             = useState([]);
  const [kupiec, setKupiec]             = useState(null);
  const [simulationPaths, setSimulationPaths] = useState([]);

  // Load portfolio list once
  useEffect(() => {
    getPortfolios().then(data => setPortfolios(data || [])).catch(() => {});
  }, []);

  // Reload all data when portfolio changes
  useEffect(() => { loadData(portfolioId); }, [portfolioId]);

  async function loadData(pid) {
    const [ret, vol, mc, dd, ef, alloc, corr, st, kb] = await Promise.allSettled([
      getPortfolioReturns(pid),
      getVolatility(pid),
      getMonteCarlo(pid),
      getDrawdown(pid),
      getEfficientFrontier(pid),
      getPortfolioAllocation(pid),
      getCorrelationMatrix(),
      getStressTest(pid),
      getVarBacktest(pid),
    ]);

    const val = (r) => r.status === "fulfilled" ? r.value : null;

    setReturns(val(ret) || []);
    setVolatility(val(vol) || []);
    setMontecarlo(val(mc) || []);
    setDrawdown(val(dd) || []);
    setCorrelation(val(corr) || []);
    setKupiec(val(kb));

    const allocData = val(alloc);
    if (Array.isArray(allocData)) setAllocation(allocData);

    const stressData = val(st);
    if (Array.isArray(stressData)) setStress(stressData);

    const efData = val(ef);
    if (efData?.returns && efData?.risk) {
      setFrontier(efData.returns.map((r, i) => ({ risk: efData.risk[i], return: r })));
    }
  }

  // ---- derived metrics ----
  const portfolioGrowth = useMemo(() => {
    let value = 10000;
    return returns.map((r, i) => ({
      day: i + 1,
      value: +(value = value * (1 + Number(r.return || 0))).toFixed(2),
    }));
  }, [returns]);

  const fanChartData = useMemo(() => {
    if (!simulationPaths.length || !simulationPaths[0]?.length) return [];
    return Array.from({ length: simulationPaths[0].length }, (_, i) => {
      const values = simulationPaths.map(p => Number(p[i]?.value ?? 0)).sort((a, b) => a - b);
      const pct = p => values[Math.floor((values.length - 1) * p)];
      return { day: i + 1, p05: pct(0.05), p25: pct(0.25), p50: pct(0.5), p75: pct(0.75), p95: pct(0.95) };
    });
  }, [simulationPaths]);

  const avgReturn  = returns.length ? (returns.reduce((s, r) => s + Number(r.return || 0), 0) / returns.length).toFixed(4) : "0.0000";
  const avgVol     = volatility.length ? (volatility.reduce((s, v) => s + Number(v.volatility || 0), 0) / volatility.length).toFixed(4) : "0.0000";
  const latestVol  = volatility.length ? Number(volatility[volatility.length - 1]?.volatility || 0).toFixed(4) : "0.0000";
  const maxDraw    = drawdown.length ? Math.min(...drawdown.map(d => Number(d.drawdown || 0))).toFixed(4) : "0.0000";
  const worstRet   = returns.length ? Math.min(...returns.map(r => Number(r.return || 0))).toFixed(4) : "0.0000";
  const avgRetNum  = Number(avgReturn);
  const sharpe     = returns.length
    ? (avgRetNum / Math.sqrt(returns.reduce((s, r) => s + (Number(r.return || 0) - avgRetNum) ** 2, 0) / returns.length)).toFixed(2)
    : "0.00";

  // correlation matrix
  const assets = [...new Set(correlation.map(c => ASSET_MAP[c.asset1] || c.asset1))];
  const matrix = assets.map(a1 =>
    assets.map(a2 => {
      const item = correlation.find(c =>
        (ASSET_MAP[c.asset1] || c.asset1) === a1 &&
        (ASSET_MAP[c.asset2] || c.asset2) === a2
      );
      return item ? Number(item.correlation.toFixed(2)) : 0;
    })
  );

  if (!returns.length) {
    return <div style={{ padding: 40, color: "#ccc" }}>Loading portfolio analytics…</div>;
  }

  return (
    <div className="dashboard">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <h1>Cloud Risk Analytics Dashboard</h1>

        {/* Portfolio selector */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <label style={{ color: "#aaa", fontSize: "0.9rem" }}>Portfolio:</label>
          <select
            value={portfolioId}
            onChange={e => setPortfolioId(Number(e.target.value))}
            style={{
              background: "#1a1a2e", color: "#00d4ff", border: "1px solid #00d4ff",
              borderRadius: 6, padding: "4px 10px", fontSize: "0.9rem", cursor: "pointer",
            }}
          >
            {portfolios.length > 0
              ? portfolios.map(p => (
                  <option key={p.portfolio_id} value={p.portfolio_id}>
                    Portfolio {p.portfolio_id}
                  </option>
                ))
              : <option value={1}>Portfolio 1</option>
            }
          </select>
        </div>
      </div>

      <div className="tabs">
        <button onClick={() => setActiveTab("overview")}>Overview</button>
        <button onClick={() => setActiveTab("simulation")}>Simulation</button>
        <button onClick={() => setActiveTab("risk")}>Risk</button>
        <button onClick={() => setActiveTab("portfolio")}>Portfolio</button>
      </div>

      {/* ── OVERVIEW ── */}
      {activeTab === "overview" && (
        <>
          <div className="metrics">
            <div className="card"><h3>Average Return</h3><p>{avgReturn}</p></div>
            <div className="card"><h3>Avg Volatility</h3><p>{avgVol}</p></div>
            <div className="card"><h3>Latest Volatility</h3><p>{latestVol}</p></div>
            <div className="card"><h3>Max Drawdown</h3><p>{maxDraw}</p></div>
            <div className="card"><h3>Worst Daily Return</h3><p>{worstRet}</p></div>
            <div className="card"><h3>Sharpe Ratio</h3><p>{sharpe}</p></div>
          </div>

          <div className="grid-2">
            <div className="chart-box">
              <h3>Daily Portfolio Returns</h3>
              <ResponsiveContainer width="100%" height={420}>
                <LineChart data={returns}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="date" hide/>
                  <YAxis/>
                  <Tooltip/>
                  <Line type="monotone" dataKey="return" stroke="#00d4ff" strokeWidth={2} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <h3>Portfolio Value Growth (from $10,000)</h3>
              <ResponsiveContainer width="100%" height={420}>
                <LineChart data={portfolioGrowth}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="day"/>
                  <YAxis/>
                  <Tooltip/>
                  <Line type="monotone" dataKey="value" stroke="#00ff88" strokeWidth={3} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <h3>30-Day Volatility Trend</h3>
              <ResponsiveContainer width="100%" height={420}>
                <LineChart data={volatility}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="date" hide/>
                  <YAxis/>
                  <Tooltip/>
                  <Line type="monotone" dataKey="volatility" stroke="#ff6b6b" strokeWidth={2} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* ── SIMULATION ── */}
      {activeTab === "simulation" && (
        <>
          <PortfolioSimulator setSimulationPaths={setSimulationPaths} portfolioId={portfolioId}/>

          {simulationPaths.length > 0 && (
            <div className="simulation-metrics">
              <div className="card">
                <h3>Expected Portfolio Value</h3>
                <p>${fanChartData.length ? Math.round(fanChartData.at(-1).p50).toLocaleString() : 0}</p>
              </div>
              <div className="card">
                <h3>Worst Case (5th pct)</h3>
                <p>${fanChartData.length ? Math.round(fanChartData.at(-1).p05).toLocaleString() : 0}</p>
              </div>
              <div className="card">
                <h3>Best Case (95th pct)</h3>
                <p>${fanChartData.length ? Math.round(fanChartData.at(-1).p95).toLocaleString() : 0}</p>
              </div>
              <div className="card">
                <h3>Probability of Loss</h3>
                <p>
                  {(simulationPaths.filter(p => p.at(-1)?.value < p[0]?.value).length / simulationPaths.length * 100).toFixed(1)}%
                </p>
              </div>
              <div className="card">
                <h3>Value at Risk (95%)</h3>
                <p>
                  ${fanChartData.length
                    ? Math.round(fanChartData.at(-1).p50 - fanChartData.at(-1).p05).toLocaleString()
                    : 0}
                </p>
              </div>
            </div>
          )}

          {simulationPaths.length > 0 && (
            <div className="chart-box">
              <h3>Monte Carlo Simulation Paths</h3>
              <ResponsiveContainer width="100%" height={500}>
                <LineChart>
                  <CartesianGrid stroke="#222"/>
                  <XAxis type="number" dataKey="day"/>
                  <YAxis/>
                  <Tooltip/>
                  {simulationPaths.slice(0, 50).map((path, i) => (
                    <Line key={i} data={path} dataKey="value" stroke="#00d4ff" strokeOpacity={0.35} strokeWidth={1.5} dot={false}/>
                  ))}
                  <Line data={fanChartData} dataKey="p50" stroke="#00ffff" strokeWidth={3} dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {fanChartData.length > 0 && (
            <div className="chart-box">
              <h3>Monte Carlo Forecast — Confidence Bands</h3>
              <ResponsiveContainer width="100%" height={500}>
                <LineChart data={fanChartData}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="day"/>
                  <YAxis/>
                  <Tooltip/>
                  <Line dataKey="p95" stroke="#ff6b6b" dot={false} name="95th pct"/>
                  <Line dataKey="p75" stroke="#ffa500" dot={false} name="75th pct"/>
                  <Line dataKey="p50" stroke="#00d4ff" strokeWidth={3} dot={false} name="Median"/>
                  <Line dataKey="p25" stroke="#00c49f" dot={false} name="25th pct"/>
                  <Line dataKey="p05" stroke="#8884d8" dot={false} name="5th pct"/>
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* ── RISK ── */}
      {activeTab === "risk" && (
        <>
          {/* Kupiec test result card */}
          <div className="metrics" style={{ marginBottom: 16 }}>
            <KupiecCard data={kupiec}/>
          </div>

          <div className="grid-2">
            <div className="chart-box">
              <h3>Return Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={montecarlo}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="bucket"/>
                  <YAxis/>
                  <Tooltip/>
                  <ReferenceLine x={-0.02} stroke="red" strokeDasharray="4 4"/>
                  <Bar dataKey="count" fill="#8884d8"/>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <h3>Portfolio Drawdown</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={drawdown}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="date" hide/>
                  <YAxis/>
                  <Tooltip/>
                  <Line dataKey="drawdown" stroke="#ffa500" dot={false}/>
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {stress.length > 0 && (
            <div className="chart-box">
              <h3>Historical Stress Scenarios — Actual Portfolio Returns</h3>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={stress}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="scenario"/>
                  <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`}/>
                  <Tooltip
                    formatter={(value, name) => [
                      name === "loss" ? `${(value * 100).toFixed(2)}%` : value,
                      name === "loss" ? "Cumulative Return" : name,
                    ]}
                    labelFormatter={label => {
                      const item = stress.find(s => s.scenario === label);
                      return item?.description || label;
                    }}
                  />
                  <Bar dataKey="loss" fill="#ff6b6b"/>
                </BarChart>
              </ResponsiveContainer>
              <p style={{ color: "#666", fontSize: "0.8rem", marginTop: 8 }}>
                Returns are actual cumulative portfolio performance during each historical crisis period.
              </p>
            </div>
          )}
        </>
      )}

      {/* ── PORTFOLIO ── */}
      {activeTab === "portfolio" && (
        <>
          <div className="section">
            <h2>Asset Correlations</h2>
            <table className="corr-table">
              <thead>
                <tr>
                  <th></th>
                  {assets.map(a => <th key={a}>{a}</th>)}
                </tr>
              </thead>
              <tbody>
                {matrix.map((row, i) => (
                  <tr key={i}>
                    <th>{assets[i]}</th>
                    {row.map((v, j) => (
                      <td
                        key={j}
                        style={{
                          backgroundColor: v > 0
                            ? `rgba(0,212,255,${Math.abs(v) * 0.9})`
                            : `rgba(255,80,80,${Math.abs(v) * 0.9})`,
                          color: Math.abs(v) > 0.6 ? "#000" : "#ccc",
                          fontWeight: Math.abs(v) > 0.8 ? "bold" : "normal",
                        }}
                      >
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid-2">
            <div className="chart-box">
              <h3>Portfolio Allocation</h3>
              <ResponsiveContainer width="100%" height={400}>
                <PieChart>
                  <Pie
                    data={allocation}
                    dataKey="weight"
                    nameKey="symbol"
                    outerRadius={120}
                    label={({ payload, percent }) =>
                      percent > 0.03 ? `${payload.symbol} ${(percent * 100).toFixed(0)}%` : ""
                    }
                  >
                    {allocation.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]}/>)}
                  </Pie>
                  <Tooltip formatter={v => [`${(v * 100).toFixed(1)}%`]}/>
                  <Legend formatter={(_, entry) => entry.payload.symbol}/>
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <h3>Efficient Frontier</h3>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart>
                  <CartesianGrid stroke="#222"/>
                  <XAxis type="number" dataKey="risk" name="Risk (σ)"/>
                  <YAxis type="number" dataKey="return" name="Return"/>
                  <Tooltip cursor={{ strokeDasharray: "3 3" }}/>
                  <Scatter data={frontier} fill="#00d4ff"/>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

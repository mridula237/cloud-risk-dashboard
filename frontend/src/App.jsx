import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ReferenceLine,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

import PortfolioSimulator from "./PortfolioSimulator";

import {
  getPortfolioReturns,
  getVolatility,
  getMonteCarlo,
  getDrawdown,
  getEfficientFrontier,
  getPortfolioAllocation,
  getCorrelationMatrix,
  getStressTest,
} from "./api";

import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("overview");

  const [returns, setReturns] = useState([]);
  const [volatility, setVolatility] = useState([]);
  const [montecarlo, setMontecarlo] = useState([]);
  const [drawdown, setDrawdown] = useState([]);
  const [frontier, setFrontier] = useState([]);
  const [allocation, setAllocation] = useState([]);
  const [correlation, setCorrelation] = useState([]);
  const [stress, setStress] = useState([]);
  const [simulationPaths, setSimulationPaths] = useState([]);

  const COLORS = ["#00d4ff", "#ff6b6b", "#ffa500", "#00c49f", "#8884d8"];

  const assetMap = {
    3: "AAPL",
    4: "MSFT",
    8: "NVDA",
    9: "AMZN",
    10: "GOOGL",
    5: "SPY",
    12: "QQQ",
    13: "DIA",
    14: "IWM",
    15: "XLF",
    16: "XLE",
    17: "XLI",
    18: "TLT",
    19: "IEF",
    20: "LQD",
    21: "GLD",
    22: "SLV",
    23: "USO",
    24: "XLP",
    25: "XLU",
  };

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    const returnsData = await getPortfolioReturns(1);
    const volatilityData = await getVolatility(1);
    const mcData = await getMonteCarlo(1);
    const drawdownData = await getDrawdown(1);
    const frontierData = await getEfficientFrontier(1);
    const allocationData = await getPortfolioAllocation(1);
    const corrData = await getCorrelationMatrix();

    setReturns(returnsData || []);
    setVolatility(volatilityData || []);
    setMontecarlo(mcData || []);
    setDrawdown(drawdownData || []);
    setCorrelation(corrData || []);

    if (allocationData && Array.isArray(allocationData)) {
      const mapped = allocationData.map((item) => ({
        ...item,
        asset: assetMap[item.asset] || item.asset,
      }));
      setAllocation(mapped);
    }

    try {
      const stressData = await getStressTest(1);

      const formattedStress = Object.entries(stressData).map(
        ([scenario, value]) => ({
          scenario,
          loss: value,
        })
      );

      setStress(formattedStress);
    } catch {
      setStress([]);
    }

    if (frontierData && frontierData.returns && frontierData.risk) {
      const points = frontierData.returns.map((ret, i) => ({
        risk: frontierData.risk[i],
        return: ret,
      }));

      setFrontier(points);
    }
  }

  const fanChartData = useMemo(() => {
    if (!simulationPaths.length || !simulationPaths[0]?.length) return [];

    const totalDays = simulationPaths[0].length;
    const result = [];

    for (let i = 0; i < totalDays; i++) {
      const values = simulationPaths
        .map((path) => Number(path[i]?.value ?? 0))
        .sort((a, b) => a - b);

      const getPercentile = (p) =>
        values[Math.floor((values.length - 1) * p)];

      result.push({
        day: i + 1,
        p05: getPercentile(0.05),
        p25: getPercentile(0.25),
        p50: getPercentile(0.5),
        p75: getPercentile(0.75),
        p95: getPercentile(0.95),
      });
    }

    return result;
  }, [simulationPaths]);

  const portfolioGrowth = useMemo(() => {
    let value = 10000;

    return returns.map((r, i) => {
      value = value * (1 + Number(r.return || 0));

      return {
        day: i + 1,
        value: Number(value.toFixed(2)),
      };
    });
  }, [returns]);

  const assets = [
    ...new Set(correlation.map((c) => assetMap[c.asset1] || c.asset1)),
  ];

  const matrix = assets.map((a1) =>
    assets.map((a2) => {
      const item = correlation.find(
        (c) =>
          (assetMap[c.asset1] || c.asset1) === a1 &&
          (assetMap[c.asset2] || c.asset2) === a2
      );

      return item ? Number(item.correlation.toFixed(2)) : 0;
    })
  );

  const avgReturn =
    returns.length > 0
      ? (
          returns.reduce((sum, item) => sum + Number(item.return || 0), 0) /
          returns.length
        ).toFixed(4)
      : "0.0000";

  const avgVol =
    volatility.length > 0
      ? (
          volatility.reduce(
            (sum, item) => sum + Number(item.volatility || 0),
            0
          ) / volatility.length
        ).toFixed(4)
      : "0.0000";

  const maxDraw =
    drawdown.length > 0
      ? Math.min(...drawdown.map((d) => Number(d.drawdown || 0))).toFixed(4)
      : "0.0000";

  const worstReturn =
    returns.length > 0
      ? Math.min(...returns.map((r) => Number(r.return || 0))).toFixed(4)
      : "0.0000";

  const latestVol =
    volatility.length > 0
      ? Number(volatility[volatility.length - 1]?.volatility || 0).toFixed(4)
      : "0.0000";

  const avgReturnNum = Number(avgReturn);

  const sharpe =
    returns.length > 0
      ? (
          avgReturnNum /
          Math.sqrt(
            returns.reduce(
              (sum, item) =>
                sum + Math.pow(Number(item.return || 0) - avgReturnNum, 2),
              0
            ) / returns.length
          )
        ).toFixed(2)
      : "0.00";

  if (!returns.length) {
    return <div style={{ padding: 40 }}>Loading portfolio analytics...</div>;
  }

  return (
    <div className="dashboard">
      <h1>Cloud Risk Analytics Dashboard</h1>

      <div className="tabs">
        <button onClick={() => setActiveTab("overview")}>Overview</button>
        <button onClick={() => setActiveTab("simulation")}>Simulation</button>
        <button onClick={() => setActiveTab("risk")}>Risk</button>
        <button onClick={() => setActiveTab("portfolio")}>Portfolio</button>
      </div>

      {activeTab === "overview" && (
        <>
          <div className="metrics">
            <div className="card"><h3>Average Return</h3><p>{avgReturn}</p></div>
            <div className="card"><h3>Avg Volatility</h3><p>{avgVol}</p></div>
            <div className="card"><h3>Latest Volatility</h3><p>{latestVol}</p></div>
            <div className="card"><h3>Max Drawdown</h3><p>{maxDraw}</p></div>
            <div className="card"><h3>Worst Daily Return</h3><p>{worstReturn}</p></div>
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
              <h3>Portfolio Value Growth</h3>
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

  

      {activeTab === "simulation" && (
        <>
          <PortfolioSimulator setSimulationPaths={setSimulationPaths} />
          {/* SIMULATION RISK METRICS */}
{simulationPaths.length > 0 && (
<div className="simulation-metrics">

<div className="card">
<h3>Expected Portfolio Value</h3>
<p>
$
{fanChartData.length > 0
? Math.round(fanChartData[fanChartData.length - 1].p50)
: 0}
</p>
</div>

<div className="card">
<h3>Worst Case (5%)</h3>
<p>
$
{fanChartData.length > 0
? Math.round(fanChartData[fanChartData.length - 1].p05)
: 0}
</p>
</div>

<div className="card">
<h3>Best Case (95%)</h3>
<p>
$
{fanChartData.length > 0
? Math.round(fanChartData[fanChartData.length - 1].p95)
: 0}
</p>
</div>

<div className="card">
<h3>Probability of Loss</h3>
<p>
{simulationPaths.length > 0
? (
simulationPaths.filter(
(path) =>
path[path.length - 1]?.value <
path[0]?.value
).length /
simulationPaths.length *
100
).toFixed(1)
: 0}
%
</p>
</div>
<div className="card">
<h3>Value at Risk (95%)</h3>
<p>
$
{fanChartData.length > 0
? Math.round(
fanChartData[fanChartData.length - 1].p50 -
fanChartData[fanChartData.length - 1].p05
)
: 0}
</p>
</div>
<div className="card">
<h3>Value at Risk (95%)</h3>
<p>
$
{fanChartData.length > 0
? Math.round(
fanChartData[fanChartData.length - 1].p50 -
fanChartData[fanChartData.length - 1].p05
)
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

                  {simulationPaths.slice(0,50).map((path,i)=>(
                    <Line
                      key={i}
                      data={path}
                      dataKey="value"
                      stroke="#00d4ff"
                      strokeOpacity={0.35}
                      strokeWidth={1.5}
                      dot={false}
                    />
                  ))}
<Line
          data={fanChartData}
          dataKey="p50"
          stroke="#00ffff"
          strokeWidth={3}
          dot={false}
        />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {fanChartData.length > 0 && (
            <div className="chart-box">
              <h3>Monte Carlo Forecast (Confidence Bands)</h3>

              <ResponsiveContainer width="100%" height={500}>
                <LineChart data={fanChartData}>
                  <CartesianGrid stroke="#222"/>
                  <XAxis dataKey="day"/>
                  <YAxis/>
                  <Tooltip/>

                  <Line dataKey="p95" stroke="#ff6b6b" dot={false}/>
                  <Line dataKey="p75" stroke="#ffa500" dot={false}/>
                  <Line dataKey="p50" stroke="#00d4ff" strokeWidth={3} dot={false}/>
                  <Line dataKey="p25" stroke="#00c49f" dot={false}/>
                  <Line dataKey="p05" stroke="#8884d8" dot={false}/>

                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* RISK */}
      {activeTab === "risk" && (
        <>
          <div className="grid-2">
            <div className="chart-box">
              <h3>Return Distribution</h3>

              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={montecarlo}>
                  <CartesianGrid stroke="#222" />
                  <XAxis dataKey="bucket" />
                  <YAxis />
                  <Tooltip />
                  <ReferenceLine
                    x={-0.02}
                    stroke="red"
                    strokeDasharray="4 4"
                  />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <h3>Portfolio Drawdown</h3>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={drawdown}>
                  <CartesianGrid stroke="#222" />
                  <XAxis dataKey="date" hide />
                  <YAxis />
                  <Tooltip />
                  <Line
                    dataKey="drawdown"
                    stroke="#ffa500"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {stress.length > 0 && (
            <div className="chart-box">
              <h3>Stress Test Scenarios</h3>

              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={stress}>
                  <CartesianGrid stroke="#222" />
                  <XAxis dataKey="scenario" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="loss" fill="#ff6b6b" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* PORTFOLIO */}
      {activeTab === "portfolio" && (
        <>
          <div className="section">
            <h2>Asset Correlations</h2>

            <table className="corr-table">
              <thead>
                <tr>
                  <th></th>
                  {assets.map((a) => (
                    <th key={a}>{a}</th>
                  ))}
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
                          backgroundColor:
  v > 0
    ? `rgba(0, 212, 255, ${Math.abs(v) * 0.9})`
    : `rgba(255, 80, 80, ${Math.abs(v) * 0.9})`,
color: Math.abs(v) > 0.6 ? "#000" : "#ccc",
fontWeight: Math.abs(v) > 0.8 ? "bold" : "normal"                                      
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
                    nameKey="asset"
                    outerRadius={120}
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                  >
                    {allocation.map((e, i) => (
                      <Cell
                        key={i}
                        fill={COLORS[i % COLORS.length]}
                      />
                    ))}
                  </Pie>

                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-box">
              <h3>Efficient Frontier</h3>

              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart>
                  <CartesianGrid stroke="#222" />
                  <XAxis type="number" dataKey="risk" />
                  <YAxis type="number" dataKey="return" />
                  <Tooltip />
                  <Scatter data={frontier} fill="#00d4ff" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
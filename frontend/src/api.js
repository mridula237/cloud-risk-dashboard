const BASE_URL = "/api";

export async function getPortfolios() {
  const res = await fetch(`${BASE_URL}/portfolios`);
  return await res.json();
}

export async function getPortfolioReturns(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/returns/${portfolioId}`);
  return await res.json();
}

export async function getVolatility(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/volatility/${portfolioId}`);
  return await res.json();
}

export async function getMonteCarlo(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/monte_carlo/${portfolioId}`);
  return await res.json();
}

export async function getDrawdown(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/drawdown/${portfolioId}`);
  return await res.json();
}

export async function getEfficientFrontier(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/efficient_frontier/${portfolioId}`);
  return await res.json();
}

export async function getPortfolioAllocation(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/${portfolioId}/allocation`);
  if (!res.ok) return [];
  return await res.json();
}

export async function getCorrelationMatrix() {
  const res = await fetch(`${BASE_URL}/portfolio/correlation`);
  return await res.json();
}

export async function getStressTest(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/stress/${portfolioId}`);
  return await res.json();
}

export async function getVarBacktest(portfolioId = 1, confidence = 0.95) {
  const res = await fetch(
    `${BASE_URL}/portfolio/backtest/var/${portfolioId}?confidence=${confidence}`
  );
  return await res.json();
}

export const runSimulation = async (params) => {
  const res = await fetch(`${BASE_URL}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return await res.json();
};

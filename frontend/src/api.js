const BASE_URL = "/api";

export async function getPortfolioReturns(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/returns/${portfolioId}`);
  return res.json();
}

export async function getVolatility(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/volatility/${portfolioId}`);
  return res.json();
}

export async function getMonteCarlo(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/monte_carlo/${portfolioId}`);
  return res.json();
}

export async function getDrawdown(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/drawdown/${portfolioId}`);
  return res.json();
}

export async function getEfficientFrontier(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/efficient_frontier/${portfolioId}`);
  return res.json();
}

export async function getPortfolioAllocation(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/${portfolioId}/allocation`);
  return res.json();
}

export async function getCorrelationMatrix() {
  const res = await fetch(`${BASE_URL}/portfolio/correlation`);
  return res.json();
}
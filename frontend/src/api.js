const BASE_URL = "/api";
// Portfolio Returns
export async function getPortfolioReturns(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/returns/${portfolioId}`);
  return await res.json();
}

// Volatility
export async function getVolatility(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/volatility/${portfolioId}`);
  return await res.json();
}

// Monte Carlo Distribution
export async function getMonteCarlo(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/monte_carlo/${portfolioId}`);
  return await res.json();
}

// Drawdown
export async function getDrawdown(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/drawdown/${portfolioId}`);
  return await res.json();
}

// Efficient Frontier
export async function getEfficientFrontier(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/efficient_frontier/${portfolioId}`);
  return await res.json();
}

// Portfolio Allocation
export async function getPortfolioAllocation(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/${portfolioId}/allocation`);

  if (!res.ok) {
    console.error("Failed to fetch allocation");
    return [];
  }

  return await res.json();
}

// Correlation Matrix
export async function getCorrelationMatrix() {
  const res = await fetch(`${BASE_URL}/portfolio/correlation`);
  return await res.json();
}

// Stress Testing
export async function getStressTest(portfolioId = 1) {
  const res = await fetch(`${BASE_URL}/portfolio/stress/${portfolioId}`);
  return await res.json();
}

// Monte Carlo Simulation
export const runSimulation = async (params) => {
  const res = await fetch(`${BASE_URL}/simulate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  return await res.json();
};

const BASE_URL = "http://127.0.0.1:8000";

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
export async function getPortfolioAllocation(portfolioId) {

  const response = await fetch(
    `http://localhost:8000/portfolio/${portfolioId}/allocation`
  );

  if (!response.ok) {
    console.error("Failed to fetch allocation");
    return [];
  }

  const data = await response.json();
  return data;
}
export async function getCorrelationMatrix() {
  const res = await fetch("http://localhost:8000/portfolio/correlation");
  return await res.json();
}
export async function getStressTest(portfolioId) {
  const res = await fetch(`http://localhost:8000/portfolio/stress/${portfolioId}`);
  return await res.json();
}
export const runSimulation = async (params) => {
  const res = await fetch("http://localhost:8000/simulate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  return res.json();
};
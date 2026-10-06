import { Container } from "@cloudflare/containers";

export interface Env {
  DEMAND_FORECASTING_CONTAINER: DurableObjectNamespace<DemandForecastingContainer>;
}

export class DemandForecastingContainer extends Container {
  defaultPort = 8080;
  sleepAfter = "10m";
  envVars = {
    STREAMLIT_SERVER_PORT: "8080",
    STREAMLIT_SERVER_ADDRESS: "0.0.0.0",
    STREAMLIT_SERVER_HEADLESS: "true",
    STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false",
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // One named Durable Object keeps Streamlit's session state in one container.
    const id = env.DEMAND_FORECASTING_CONTAINER.idFromName("streamlit-app");
    return env.DEMAND_FORECASTING_CONTAINER.get(id).fetch(request);
  },
} satisfies ExportedHandler<Env>;

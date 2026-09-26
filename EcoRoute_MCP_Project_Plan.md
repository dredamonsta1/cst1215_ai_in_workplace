# EcoRoute: Deterministic Carbon-Aware MCP Router
**Course:** CST 1215 — Operating Systems Architecture / Group Project  
**Target Output:** Model Context Protocol (MCP) Server for Spatial Inference Optimization

---

## 1. Executive Summary & Problem Scope
Enterprise AI workloads are predominantly scheduled without consideration for geographical grid emissions or data center cooling overhead. Standard inference requests default to nearest-region latency rather than carbon intensity.

**EcoRoute** is an open Model Context Protocol (MCP) server that exposes carbon-aware spatial routing capabilities to LLM clients (such as Claude Desktop, Cursor, or local orchestration agents). Rather than acting purely as a passive telemetry logger, EcoRoute provides deterministic calculations and routing recommendations to shift latency-tolerant inference jobs to clean-energy grid corridors.

---

## 2. Academic Alignment (CST 1215 Concepts)
* **Client-Server RPC Protocols:** Implementing JSON-RPC schemas via STDIO transport following standard Model Context Protocol specifications.
* **Systems Performance & PUE:** Modeling Power Usage Effectiveness (PUE) and Water Usage Effectiveness (WUE) across different facility cooling tiers.
* **Spatial Resource Scheduling:** Applying deterministic constraint routing based on real-time and baseline marginal grid emissions.
* **Hardware Accelerator Profiling:** Computing FLOPS-per-watt energy profiles for modern accelerator hardware (NVIDIA H100 SXM5 / A100).

---

## 3. System Architecture & Component Design









The complete contents of **`EcoRoute_MCP_Project_Plan.md`** as captured in your CST 1215 project workspace:

```markdown
# EcoRoute: Deterministic Carbon-Aware MCP Router
**Course:** CST 1215 — Operating Systems Architecture / Group Project  
**Target Output:** Model Context Protocol (MCP) Server for Spatial Inference Optimization

---

## 1. Executive Summary & Problem Scope
Enterprise AI workloads are predominantly scheduled without consideration for geographical grid emissions or data center cooling overhead. Standard inference requests default to nearest-region latency rather than carbon intensity.

**EcoRoute** is an open Model Context Protocol (MCP) server that exposes carbon-aware spatial routing capabilities to LLM clients (such as Claude Desktop, Cursor, or local orchestration agents). Rather than acting purely as a passive telemetry logger, EcoRoute provides deterministic calculations and routing recommendations to shift latency-tolerant inference jobs to clean-energy grid corridors.

---

## 2. Academic Alignment (CST 1215 Concepts)
* **Client-Server RPC Protocols:** Implementing JSON-RPC schemas via STDIO transport following standard Model Context Protocol specifications.
* **Systems Performance & PUE:** Modeling Power Usage Effectiveness (PUE) and Water Usage Effectiveness (WUE) across different facility cooling tiers.
* **Spatial Resource Scheduling:** Applying deterministic constraint routing based on real-time and baseline marginal grid emissions.
* **Hardware Accelerator Profiling:** Computing FLOPS-per-watt energy profiles for modern accelerator hardware (NVIDIA H100 SXM5 / A100).

---

## 3. System Architecture & Component Design


```

+-------------------------------------------------------------+
|                   LLM Agent / MCP Client                    |
|             (Claude Desktop, Cursor, Test CLI)              |
+-------------------------------------------------------------+
|
| JSON-RPC over STDIO
v
+-------------------------------------------------------------+
|                    EcoRoute MCP Server                      |
|                (FastAPI / MCP SDK in Python)                |
+-------------------------------------------------------------+
|                                       |
v                                       v
+-----------------------+              +---------------------+
|   Core Math Engine    |              |  Regional Telemetry |
| - Energy (kWh)        |              | - Grid Carbon (eGRID|
| - Carbon (gCO2eq)     |              | - PUE / Cooling WUE |
| - Water (Liters)      |              | - Region Profiles   |
+-----------------------+              +---------------------+

```

### 3.1 Exposed MCP Tools

1. `calculate_inference_footprint(prompt_tokens: int, completion_tokens: int, region: str, hardware: str = "H100")`
   * Computes energy (kWh), carbon emissions ($g\text{CO}_2\text{eq}$), and evaporative cooling water loss (Liters) for a designated workload.

2. `get_optimal_region(prompt_tokens: int, completion_tokens: int, max_latency_ms: int = 500)`
   * Evaluates candidate data center regions and returns the ranked routing decision based on lowest carbon intensity while respecting latency tolerances.

3. `list_datacenter_regions()`
   * Returns metadata and grid generation mix for modeled hyperscaler hubs.

---

## 4. Mathematical Model & Methodology

Calculations rely on deterministic formulas derived from EPA eGRID baselines and published accelerator hardware profiles:

### 4.1 Operational Energy Formula
$$\text{Energy (kWh)} = \left( \frac{\text{Tokens} \times \text{FLOPs/Token}}{\text{Hardware FLOP/Joule}} \times \frac{1}{3.6 \times 10^6} \right) \times \text{PUE}$$

* **Baseline Hardware Archetype:** NVIDIA H100 SXM5 (~700W TDP baseline).
* **PUE (Power Usage Effectiveness):** Modeled at 1.10 to 1.25 depending on facility cooling infrastructure.

### 4.2 Carbon Emissions Formula
$$\text{Emissions } (g\text{CO}_2) = \text{Energy (kWh)} \times \text{Regional Carbon Intensity } (g\text{CO}_2/\text{kWh})$$

### 4.3 Regional Benchmark Baselines

| Region ID       | Facility Corridor                                       | Dominant Grid Sources      | Marginal Grid Intensity                | Baseline PUE |
| :-------------- | :------------------------------------------------------ | :------------------------- | :------------------------------------- | :----------- |
| `us-east-va`    | Northern Virginia (AWS `us-east-1` / Data Center Alley) | Natural Gas, Coal, Nuclear | $360 \text{ } g\text{CO}_2/\text{kWh}$ | 1.18         |
| `us-west-or`    | Oregon / Columbia Basin (AWS `us-west-2`)               | Hydroelectric, Wind        | $110 \text{ } g\text{CO}_2/\text{kWh}$ | 1.12         |
| `eu-north-se`   | Sweden / Nordic Hubs (GCP `europe-north1`)              | Hydro, Nuclear             | $25 \text{ } g\text{CO}_2/\text{kWh}$  | 1.10         |
| `us-central-ia` | Iowa / Council Bluffs (GCP `us-central1`)               | Wind, Natural Gas          | $280 \text{ } g\text{CO}_2/\text{kWh}$ | 1.15         |

---

## 5. Team Work Breakdown & Responsibilities

| Role                                                        | Core Responsibilities                                                                                                                                                                                          | Key Deliverables                                                |
| :---------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------- |
| **Member 1**<br>*(Lead / Protocol)*                         | • Setup MCP Server scaffolding (`mcp` Python SDK / FastAPI)<br>• Register tool schemas and input validation models (Pydantic)<br>• Connect protocol routes to Core Math Engine                                 | `server.py`<br>`schemas.py`<br>Local STDIO testing suite        |
| **Member 2**<br>*(Math Engine & Data Profiles)*             | • Implement core calculation module (`calculator.py`)<br>• Build structured regional telemetry dictionary (`regions.json`)<br>• Write deterministic unit tests for energy/carbon edge cases                    | `calculator.py`<br>`regions.json`<br>`tests/test_calculator.py` |
| **Member 3**<br>*(Benchmarking, Evaluation & Presentation)* | • Design workload benchmark suite (Small, Medium, Large contexts)<br>• Run comparative analysis (e.g., Virginia vs. Oregon carbon delta)<br>• Structure final slides, architecture flowcharts, and demo script | `benchmark.py`<br>Presentation Deck<br>Demo Script & Visuals    |

---

## 6. Implementation Milestones

* **Milestone 1 (Specification Freeze & Setup):** Agree on JSON schema interfaces between protocol and calculator modules.
* **Milestone 2 (Core Calculation Engine):** Finalize `calculator.py` unit tests with deterministic EPA baselines.
* **Milestone 3 (MCP Server Tool Wiring):** Serve tools over STDIO and verify connection using Claude Desktop or an MCP test CLI.
* **Milestone 4 (Benchmarking & Analysis):** Generate comparative carbon delta graphs for presentation slides.
* **Milestone 5 (Final Submission & Live Demo):** Deliver live demo query showing real-time spatial shifting recommendations.

```
# EcoRoute — Deterministic Carbon-Aware MCP Router

**CST 1215 — Operating Systems Architecture / Group Project**

An [MCP](https://modelcontextprotocol.io) server that exposes carbon-aware spatial
routing to LLM clients. It models the energy, grid emissions, and evaporative cooling
water cost of an inference request per data-center region, then recommends where to run
latency-tolerant work.

Every answer is **deterministic** — no live grid feed, no RNG, no wall clock — so results
are reproducible for unit tests and for the graded live demo.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest                 # 38 tests: 31 unit + 7 end-to-end protocol
python benchmark.py    # comparative carbon analysis
python -m ecoroute.server   # start the MCP server on STDIO
```

The server speaks JSON-RPC on stdin/stdout, so run it directly only to confirm it starts;
normally an MCP client launches it.

## Connect to Claude Desktop

Add to `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`), using **absolute
paths**:

```json
{
  "mcpServers": {
    "ecoroute": {
      "command": "/absolute/path/to/CST1215_project/.venv/bin/python",
      "args": ["-m", "ecoroute.server"],
      "cwd": "/absolute/path/to/CST1215_project"
    }
  }
}
```

Restart Claude Desktop, then ask: *"Where should I run a 128k-token summarisation job if I
can tolerate 200 ms of latency?"*

---

## Exposed MCP tools

| Tool | Purpose |
| :--- | :--- |
| `calculate_inference_footprint(prompt_tokens, completion_tokens, region, hardware="H100", model_params_b=70.0)` | Energy (kWh), carbon (gCO₂eq) and cooling water (L) for one request in one region. |
| `get_optimal_region(prompt_tokens, completion_tokens, max_latency_ms=500, hardware="H100", model_params_b=70.0, baseline_region="us-east-va")` | Ranks latency-feasible regions by carbon and reports the saving vs a baseline. |
| `list_datacenter_regions()` | Region metadata, grid mix, PUE, WUE, plus accelerator profiles. |

## Architecture

```
  MCP Client (Claude Desktop / Cursor / pytest client)
        |  JSON-RPC over STDIO
        v
  ecoroute/server.py      protocol layer: tool registration, error translation
        |
        +-- ecoroute/schemas.py     frozen Pydantic wire contracts
        |
        v
  ecoroute/calculator.py  pure math engine (no I/O beyond loading the data files)
        |
        +-- ecoroute/regions.json   grid intensity, PUE, WUE, latency per region
        +-- ecoroute/hardware.json  accelerator FLOP/Joule profiles
```

`server.py` contains no arithmetic and `calculator.py` contains no protocol code, so the
math engine is testable without a client and the tool schemas can be frozen independently
(Milestone 1).

## The model

```
flops_per_token = 2 × model_parameters        # dense forward pass
joules          = tokens × flops_per_token ÷ flops_per_joule
energy_kwh      = joules ÷ 3.6e6 × PUE
carbon_g        = energy_kwh × carbon_intensity_g_per_kwh
water_l         = energy_kwh × wue_l_per_kwh
```

**Prefill and decode are costed separately.** Prompt processing is compute-bound and
reaches ~40% FLOP utilization on an H100; token generation is memory-bandwidth-bound and
achieves roughly 8%, while the board still draws near-TDP power. Using a single
utilization figure — as the original plan's formula did — understates generation energy by
roughly 5×. This is the main systems-performance refinement over the plan.

### Regional baselines (`ecoroute/regions.json`)

| Region | Corridor | Dominant grid | gCO₂/kWh | PUE | WUE (L/kWh) | Latency |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `us-east-va` | N. Virginia (AWS `us-east-1`) | gas, coal, nuclear | 360 | 1.18 | 1.80 | 15 ms |
| `us-central-ia` | Council Bluffs (GCP `us-central1`) | wind, gas | 280 | 1.15 | 1.35 | 35 ms |
| `us-west-or` | Columbia Basin (AWS `us-west-2`) | hydro, wind | 110 | 1.12 | 0.65 | 75 ms |
| `eu-north-se` | Nordic hub (GCP `europe-north1`) | hydro, nuclear | 25 | 1.10 | 0.12 | 110 ms |

Carbon and PUE figures come from the project plan (EPA eGRID baselines). WUE and latency
were not specified there and are documented assumptions — see "Limitations".

## Benchmark results

`python benchmark.py` sweeps three workload archetypes across all four regions:

| Workload | Virginia | Sweden | Reduction |
| :--- | ---: | ---: | ---: |
| Small (512 + 128 tok) | 33.7 mg CO₂ | 2.2 mg | 93.5% |
| Medium (8k + 1k tok) | 379.9 mg | 24.6 mg | 93.5% |
| Large (128k + 2k tok) | 4.03 g | 0.26 g | 93.5% |

At 1M small requests/day, shifting Virginia → Sweden avoids **~11.5 tonnes CO₂eq and
57.6 m³ of cooling water per year**.

`--csv results.csv` exports the raw rows for charting; `--hardware A100` reruns on the
older accelerator.

## Limitations (stated for the write-up)

1. **Annual-average carbon, not real-time.** Grid intensity is a static eGRID baseline.
   Real marginal intensity swings hour to hour, so a production router would consult a
   live feed. Determinism was chosen deliberately here for reproducible grading.
2. **Latency is modeled, not measured.** `baseline_latency_ms` is a static round-trip
   estimate from a reference US East Coast client. EcoRoute never probes the network.
3. **Utilization figures are published ranges**, not measurements from these exact
   regions.
4. **Embodied carbon is excluded** — manufacturing and end-of-life emissions of the
   accelerators are out of scope. Operational energy only.
5. **Dense models only.** The 2×params FLOP rule does not hold for mixture-of-experts
   models, where only a fraction of parameters activate per token.
6. **No data-residency or cost constraints.** Real routing must also respect sovereignty
   rules and egress pricing, which this model ignores.

## Repository layout

| Path | Owner (plan §5) | Contents |
| :--- | :--- | :--- |
| `ecoroute/server.py` | Member 1 — Protocol | MCP tool registration over STDIO |
| `ecoroute/schemas.py` | Member 1 — Protocol | Pydantic wire contracts |
| `ecoroute/calculator.py` | Member 2 — Math engine | Pure energy/carbon/water model |
| `ecoroute/regions.json`, `hardware.json` | Member 2 — Data profiles | Regional + accelerator telemetry |
| `tests/test_calculator.py` | Member 2 — Tests | 31 deterministic unit tests |
| `tests/test_server.py` | Member 1 — Tests | 7 end-to-end JSON-RPC tests |
| `benchmark.py` | Member 3 — Evaluation | Comparative analysis + fleet projection |

## Milestone status

- [x] **M1** Specification freeze — `schemas.py` fixes the protocol/engine interface
- [x] **M2** Core calculation engine — `calculator.py` + 31 passing unit tests
- [x] **M3** MCP server tool wiring — tools served over STDIO, verified by a real client
- [x] **M4** Benchmarking & analysis — `benchmark.py` with CSV export
- [ ] **M5** Final submission & live demo — slides and demo script still to write

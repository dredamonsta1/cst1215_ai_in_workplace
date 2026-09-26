# EcoRoute — Backlog

Remaining work, roughly in priority order. Milestones 1–4 and the demo half of
Milestone 5 are done; see the checklist in `README.md`.

---

## 1. Presentation deck (Milestone 5) — blocking submission

`demo.py` and the MCP Inspector cover the *live demo*. The slides do not exist yet.

- [ ] Architecture slide — the layer diagram from `README.md` (client → `server.py` →
      `calculator.py` → data files), emphasising that the protocol layer holds no
      arithmetic and the engine holds no protocol code
- [ ] Carbon-delta chart — `python benchmark.py --csv results.csv` exports the rows;
      the 93.5% Virginia → Sweden reduction is the headline
- [ ] Spoken script for the seven acts of `demo.py`, with Act 5 (the 50 ms budget
      shifting the recommendation from Sweden to Iowa) as the centrepiece
- [ ] Fleet-scale slide — 11.5 tonnes CO₂eq and 57.6 m³ water per year at 1M
      requests/day, from a scheduling decision rather than new hardware
- [ ] Limitations slide — pull the six items from `README.md`; the graders will ask

## 2. Fix `EcoRoute_MCP_Project_Plan.md`

The file is malformed and should be cleaned before submission:

- [ ] It contains a **duplicated copy of its own contents** — the whole document is
      repeated inside a fenced ` ```markdown ` block partway down
- [ ] The section 3 ASCII architecture diagram has **broken fencing**, so the box-drawing
      characters render as prose instead of a code block
- [ ] Once fixed, reconcile it with what was actually built — see item 3

## 3. Team review of the assumed data — flagged in PR #1

The plan specified carbon intensity and PUE but not water or latency, so these were
filled in and documented as assumptions in `ecoroute/regions.json`. They drive the
headline numbers, so the group should agree on them or replace them with sourced values:

- [ ] **WUE** (0.12–1.8 L/kWh) — the entire water-loss column depends on this
- [ ] **`baseline_latency_ms`** (15–110 ms) — a static estimate from a US East Coast
      reference client, never measured. This is what `get_optimal_region` filters on,
      so it decides which region wins.
- [ ] **Default model size** (70B) — reasonable for a served dense model, but arbitrary
- [ ] Decide whether to cite sources for the two departures from the plan's formula
      (split prefill/decode utilization; significant-figure rounding)

## 4. Merge PR #1

https://github.com/dredamonsta1/cst1215_ai_in_workplace/pull/1
(`feat/ecoroute-mcp-server` → `main`)

- [ ] Group review, then merge before the deadline

## 5. Optional — Claude Desktop as the headline demo

Deferred, not required. The most compelling framing is the LLM choosing to call the
router on its own, rather than us invoking tools by hand. Needs Claude Desktop
installed (it is not, on the dev machine) and the config from `README.md` §3, tested
ahead of time.

- [ ] Install, configure, rehearse
- [ ] Decide whether this replaces or supplements `demo.py`

## 6. Future enhancements — out of scope for the course

Worth mentioning in the write-up as "what production would need", not worth building:

- Live grid-intensity feed (ElectricityMaps / WattTime) instead of static eGRID
  baselines. Determinism was chosen deliberately here for reproducible grading, so
  this would be a *separate* code path, not a replacement.
- Measured rather than modeled latency
- Mixture-of-experts FLOP accounting — the 2×params rule only holds for dense models
- Embodied carbon (manufacturing, end-of-life), currently excluded entirely
- Data-residency and egress-cost constraints, which real routing must respect

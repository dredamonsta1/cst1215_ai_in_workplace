"""Deterministic energy / carbon / water model for EcoRoute.

Every function here is pure: identical inputs always produce identical
outputs. There is no wall-clock read, no RNG, and no network call -- that
is a hard requirement, because the MCP tools must be reproducible for unit
tests and for the graded live demo.

Model (see section 4 of EcoRoute_MCP_Project_Plan.md):

    flops_per_token  = 2 * model_parameters          (forward pass only)
    joules           = tokens * flops_per_token / flops_per_joule
    energy_kwh       = joules / 3.6e6 * PUE
    carbon_g         = energy_kwh * carbon_intensity_g_per_kwh
    water_l          = energy_kwh * wue_l_per_kwh

The one refinement over the plan's single-utilization formula: prompt
(prefill) and completion (decode) tokens are costed against different
effective FLOP/Joule figures, because decode is bandwidth-bound rather
than compute-bound. See hardware.json for the rationale.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

JOULES_PER_KWH = 3.6e6

# A dense transformer forward pass costs ~2 FLOPs per parameter per token
# (one multiply + one add). Backward pass is irrelevant: inference only.
FLOPS_PER_PARAM_PER_TOKEN = 2.0

_DATA_DIR = Path(__file__).resolve().parent


class EcoRouteError(ValueError):
    """Raised for unknown regions/hardware or out-of-domain inputs."""


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    corridor: str
    cloud_region: str
    dominant_sources: tuple[str, ...]
    carbon_intensity_g_per_kwh: float
    pue: float
    wue_l_per_kwh: float
    baseline_latency_ms: int


@dataclass(frozen=True)
class Accelerator:
    id: str
    name: str
    tdp_watts: float
    peak_flops_bf16: float
    prefill_utilization: float
    decode_utilization: float

    def flops_per_joule(self, phase: str) -> float:
        """Effective FLOP/Joule for 'prefill' or 'decode'.

        FLOP/Joule == FLOP/s per Watt, so peak throughput scaled by the
        phase's achieved utilization, divided by board power draw.
        """
        if phase == "prefill":
            utilization = self.prefill_utilization
        elif phase == "decode":
            utilization = self.decode_utilization
        else:
            raise EcoRouteError(f"unknown phase {phase!r}; expected 'prefill' or 'decode'")
        return self.peak_flops_bf16 * utilization / self.tdp_watts


def _load(filename: str) -> dict:
    with (_DATA_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_regions() -> dict[str, Region]:
    """Region profiles keyed by region id, in file order."""
    return {
        entry["id"]: Region(
            id=entry["id"],
            name=entry["name"],
            corridor=entry["corridor"],
            cloud_region=entry["cloud_region"],
            dominant_sources=tuple(entry["dominant_sources"]),
            carbon_intensity_g_per_kwh=float(entry["carbon_intensity_g_per_kwh"]),
            pue=float(entry["pue"]),
            wue_l_per_kwh=float(entry["wue_l_per_kwh"]),
            baseline_latency_ms=int(entry["baseline_latency_ms"]),
        )
        for entry in _load("regions.json")["regions"]
    }


@lru_cache(maxsize=1)
def load_accelerators() -> dict[str, Accelerator]:
    """Accelerator profiles keyed by hardware id, in file order."""
    return {
        entry["id"]: Accelerator(
            id=entry["id"],
            name=entry["name"],
            tdp_watts=float(entry["tdp_watts"]),
            peak_flops_bf16=float(entry["peak_flops_bf16"]),
            prefill_utilization=float(entry["prefill_utilization"]),
            decode_utilization=float(entry["decode_utilization"]),
        )
        for entry in _load("hardware.json")["accelerators"]
    }


def get_region(region_id: str) -> Region:
    regions = load_regions()
    try:
        return regions[region_id]
    except KeyError:
        raise EcoRouteError(
            f"unknown region {region_id!r}; known regions: {', '.join(regions)}"
        ) from None


def get_accelerator(hardware_id: str) -> Accelerator:
    accelerators = load_accelerators()
    try:
        return accelerators[hardware_id]
    except KeyError:
        raise EcoRouteError(
            f"unknown hardware {hardware_id!r}; known accelerators: {', '.join(accelerators)}"
        ) from None


def _sig(value: float, digits: int = 9) -> float:
    """Round to `digits` significant figures.

    Footprint figures span many orders of magnitude -- a 10-token request in
    a clean region is ~1e-6 g while a 1M-token batch is ~1e2 g. Rounding to a
    fixed number of decimal places would quantise the small end into noise
    (and break the identity carbon == energy * intensity), so precision is
    kept relative instead of absolute.
    """
    if value == 0 or not math.isfinite(value):
        return float(value)
    exponent = math.floor(math.log10(abs(value)))
    return round(value, -(exponent - digits + 1))


def flops_per_token(model_params_b: float) -> float:
    """FLOPs to push one token through a dense model of `model_params_b` billions."""
    if model_params_b <= 0:
        raise EcoRouteError("model_params_b must be positive")
    return FLOPS_PER_PARAM_PER_TOKEN * model_params_b * 1e9


def _validate_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    if prompt_tokens < 0 or completion_tokens < 0:
        raise EcoRouteError("token counts must be non-negative")


def compute_energy(
    prompt_tokens: int,
    completion_tokens: int,
    region: Region,
    accelerator: Accelerator,
    model_params_b: float = 70.0,
) -> dict[str, float]:
    """Split compute energy into prefill / decode / facility overhead.

    Returns raw (unrounded) figures so callers can round once at the edge.
    """
    _validate_tokens(prompt_tokens, completion_tokens)
    per_token = flops_per_token(model_params_b)

    prefill_joules = prompt_tokens * per_token / accelerator.flops_per_joule("prefill")
    decode_joules = completion_tokens * per_token / accelerator.flops_per_joule("decode")
    it_joules = prefill_joules + decode_joules

    it_kwh = it_joules / JOULES_PER_KWH
    total_kwh = it_kwh * region.pue

    return {
        "prefill_joules": prefill_joules,
        "decode_joules": decode_joules,
        "it_joules": it_joules,
        "it_energy_kwh": it_kwh,
        "overhead_energy_kwh": total_kwh - it_kwh,
        "total_energy_kwh": total_kwh,
    }


def calculate_inference_footprint(
    prompt_tokens: int,
    completion_tokens: int,
    region: str,
    hardware: str = "H100",
    model_params_b: float = 70.0,
) -> dict:
    """Energy (kWh), carbon (gCO2eq) and evaporative water loss (L) for one workload."""
    region_profile = get_region(region)
    accelerator = get_accelerator(hardware)
    energy = compute_energy(
        prompt_tokens, completion_tokens, region_profile, accelerator, model_params_b
    )

    total_kwh = energy["total_energy_kwh"]
    carbon_g = total_kwh * region_profile.carbon_intensity_g_per_kwh
    water_l = total_kwh * region_profile.wue_l_per_kwh

    return {
        "region": region_profile.id,
        "region_name": region_profile.name,
        "hardware": accelerator.id,
        "model_params_b": model_params_b,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "energy_kwh": _sig(total_kwh),
        "it_energy_kwh": _sig(energy["it_energy_kwh"]),
        "overhead_energy_kwh": _sig(energy["overhead_energy_kwh"]),
        "prefill_joules": _sig(energy["prefill_joules"]),
        "decode_joules": _sig(energy["decode_joules"]),
        "carbon_g_co2eq": _sig(carbon_g),
        "water_liters": _sig(water_l),
        "pue": region_profile.pue,
        "carbon_intensity_g_per_kwh": region_profile.carbon_intensity_g_per_kwh,
        "wue_l_per_kwh": region_profile.wue_l_per_kwh,
    }


def _percent_delta(baseline: float, candidate: float) -> float:
    """Percent reduction of `candidate` relative to `baseline`."""
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def get_optimal_region(
    prompt_tokens: int,
    completion_tokens: int,
    max_latency_ms: int = 500,
    hardware: str = "H100",
    model_params_b: float = 70.0,
    baseline_region: str = "us-east-va",
) -> dict:
    """Rank latency-feasible regions by carbon and report the saving vs `baseline_region`.

    Ties on carbon are broken by lower latency, then by region id, so the
    ranking is stable across runs.
    """
    if max_latency_ms <= 0:
        raise EcoRouteError("max_latency_ms must be positive")
    baseline_profile = get_region(baseline_region)

    candidates = [
        calculate_inference_footprint(
            prompt_tokens, completion_tokens, region.id, hardware, model_params_b
        )
        | {"latency_ms": region.baseline_latency_ms}
        for region in load_regions().values()
    ]

    feasible = [c for c in candidates if c["latency_ms"] <= max_latency_ms]
    excluded = [
        {
            "region": c["region"],
            "latency_ms": c["latency_ms"],
            "reason": f"baseline latency {c['latency_ms']} ms exceeds budget {max_latency_ms} ms",
        }
        for c in candidates
        if c["latency_ms"] > max_latency_ms
    ]

    if not feasible:
        raise EcoRouteError(
            f"no region satisfies max_latency_ms={max_latency_ms}; "
            f"lowest modeled latency is {min(c['latency_ms'] for c in candidates)} ms"
        )

    ranked = sorted(feasible, key=lambda c: (c["carbon_g_co2eq"], c["latency_ms"], c["region"]))
    best = ranked[0]

    baseline = calculate_inference_footprint(
        prompt_tokens, completion_tokens, baseline_profile.id, hardware, model_params_b
    )

    return {
        "recommended_region": best["region"],
        "recommended_region_name": best["region_name"],
        "latency_ms": best["latency_ms"],
        "carbon_g_co2eq": best["carbon_g_co2eq"],
        "energy_kwh": best["energy_kwh"],
        "water_liters": best["water_liters"],
        "baseline_region": baseline["region"],
        "baseline_carbon_g_co2eq": baseline["carbon_g_co2eq"],
        "carbon_saved_g_co2eq": _sig(baseline["carbon_g_co2eq"] - best["carbon_g_co2eq"]),
        "carbon_reduction_pct": round(
            _percent_delta(baseline["carbon_g_co2eq"], best["carbon_g_co2eq"]), 4
        ),
        "water_reduction_pct": round(
            _percent_delta(baseline["water_liters"], best["water_liters"]), 4
        ),
        "ranking": [
            {
                "rank": i,
                "region": c["region"],
                "carbon_g_co2eq": c["carbon_g_co2eq"],
                "energy_kwh": c["energy_kwh"],
                "water_liters": c["water_liters"],
                "latency_ms": c["latency_ms"],
            }
            for i, c in enumerate(ranked, start=1)
        ],
        "excluded_regions": excluded,
    }


def list_datacenter_regions() -> dict:
    """Metadata and grid mix for every modeled region, plus known accelerators."""
    return {
        "regions": [
            {
                "id": r.id,
                "name": r.name,
                "corridor": r.corridor,
                "cloud_region": r.cloud_region,
                "dominant_sources": list(r.dominant_sources),
                "carbon_intensity_g_per_kwh": r.carbon_intensity_g_per_kwh,
                "pue": r.pue,
                "wue_l_per_kwh": r.wue_l_per_kwh,
                "baseline_latency_ms": r.baseline_latency_ms,
            }
            for r in load_regions().values()
        ],
        "accelerators": [
            {
                "id": a.id,
                "name": a.name,
                "tdp_watts": a.tdp_watts,
                "peak_flops_bf16": a.peak_flops_bf16,
                "prefill_flops_per_joule": round(a.flops_per_joule("prefill"), 3),
                "decode_flops_per_joule": round(a.flops_per_joule("decode"), 3),
            }
            for a in load_accelerators().values()
        ],
    }

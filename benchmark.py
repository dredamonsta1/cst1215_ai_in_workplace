#!/usr/bin/env python3
"""EcoRoute benchmark suite -- comparative carbon analysis across regions.

Milestone 4 of the project plan. Sweeps three workload archetypes across
every modeled region and prints the carbon / energy / water deltas, then
projects the saving to enterprise fleet scale.

Usage:
    python benchmark.py                      # table to stdout
    python benchmark.py --csv results.csv    # also write CSV for slides
    python benchmark.py --hardware A100
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

from ecoroute import calculator as calc
from ecoroute.display import fmt, fmt_column, table

# Requests per day used for the fleet-scale projection in the final section.
FLEET_REQUESTS_PER_DAY = 1_000_000

BASELINE_REGION = "us-east-va"


@dataclass(frozen=True)
class Workload:
    key: str
    label: str
    prompt_tokens: int
    completion_tokens: int
    note: str


WORKLOADS = (
    Workload("small", "Small (chat turn)", 512, 128, "interactive assistant reply"),
    Workload("medium", "Medium (doc summary)", 8_000, 1_000, "single-document summarisation"),
    Workload("large", "Large (long-context RAG)", 128_000, 2_000, "full-context retrieval answer"),
)


def collect(hardware: str, model_params_b: float) -> list[dict]:
    """One row per (workload, region) pair, with deltas against the baseline region."""
    rows: list[dict] = []
    for workload in WORKLOADS:
        baseline = calc.calculate_inference_footprint(
            workload.prompt_tokens,
            workload.completion_tokens,
            BASELINE_REGION,
            hardware,
            model_params_b,
        )
        for region in calc.load_regions().values():
            result = calc.calculate_inference_footprint(
                workload.prompt_tokens,
                workload.completion_tokens,
                region.id,
                hardware,
                model_params_b,
            )
            baseline_carbon = baseline["carbon_g_co2eq"]
            reduction = (
                (baseline_carbon - result["carbon_g_co2eq"]) / baseline_carbon * 100.0
                if baseline_carbon
                else 0.0
            )
            rows.append(
                {
                    "workload": workload.key,
                    "workload_label": workload.label,
                    "region": region.id,
                    "region_name": region.name,
                    "latency_ms": region.baseline_latency_ms,
                    "energy_kwh": result["energy_kwh"],
                    "carbon_g_co2eq": result["carbon_g_co2eq"],
                    "water_liters": result["water_liters"],
                    "carbon_reduction_pct_vs_baseline": round(reduction, 4),
                }
            )
    return rows


def report(rows: list[dict], hardware: str, model_params_b: float) -> None:
    print("=" * 78)
    print("EcoRoute comparative carbon benchmark")
    print(f"hardware={hardware}  model={model_params_b:g}B params  baseline={BASELINE_REGION}")
    print("=" * 78)

    for workload in WORKLOADS:
        subset = [r for r in rows if r["workload"] == workload.key]
        print(f"\n{workload.label} -- {workload.prompt_tokens:,} prompt / "
              f"{workload.completion_tokens:,} completion tokens ({workload.note})")
        ordered = sorted(subset, key=lambda r: r["carbon_g_co2eq"])
        energy = fmt_column([r["energy_kwh"] for r in ordered], "kWh")
        carbon = fmt_column([r["carbon_g_co2eq"] for r in ordered], "g")
        water = fmt_column([r["water_liters"] for r in ordered], "L")
        table_rows = [
            [
                r["region"],
                f"{r['latency_ms']} ms",
                energy[i],
                carbon[i],
                water[i],
                f"{r['carbon_reduction_pct_vs_baseline']:+.1f}%",
            ]
            for i, r in enumerate(ordered)
        ]
        print(table(
            ["region", "latency", "energy", "carbon", "water", "vs baseline"],
            table_rows,
        ))

    # Headline comparison called out in the project plan: Virginia vs Oregon.
    print("\n" + "-" * 78)
    print("Headline delta (project plan section 5): Virginia vs Oregon")
    print("-" * 78)
    pairs = [
        (
            workload,
            next(r for r in rows if r["workload"] == workload.key and r["region"] == "us-east-va"),
            next(r for r in rows if r["workload"] == workload.key and r["region"] == "us-west-or"),
        )
        for workload in WORKLOADS
    ]
    saved_column = fmt_column(
        [va["carbon_g_co2eq"] - or_["carbon_g_co2eq"] for _, va, or_ in pairs], "g"
    )
    for (workload, _, or_), saved in zip(pairs, saved_column):
        print(f"  {workload.label:<26} saves {saved:>14}  "
              f"({or_['carbon_reduction_pct_vs_baseline']:.1f}% lower)")

    # Fleet-scale projection makes the per-request milligrams legible.
    print("\n" + "-" * 78)
    print(f"Fleet projection: {FLEET_REQUESTS_PER_DAY:,} small requests/day, "
          f"{BASELINE_REGION} -> eu-north-se")
    print("-" * 78)
    small_va = next(r for r in rows if r["workload"] == "small" and r["region"] == "us-east-va")
    small_se = next(r for r in rows if r["workload"] == "small" and r["region"] == "eu-north-se")
    daily_saved_g = (small_va["carbon_g_co2eq"] - small_se["carbon_g_co2eq"]) * FLEET_REQUESTS_PER_DAY
    daily_water_saved_l = (small_va["water_liters"] - small_se["water_liters"]) * FLEET_REQUESTS_PER_DAY
    print(f"  carbon avoided:  {daily_saved_g / 1000:,.1f} kg CO2eq/day "
          f"({daily_saved_g / 1e6 * 365:,.1f} tonnes/year)")
    print(f"  water avoided:   {daily_water_saved_l:,.1f} L/day "
          f"({daily_water_saved_l * 365 / 1000:,.1f} m3/year)")
    print()


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", default="H100", help="accelerator id (H100 or A100)")
    parser.add_argument("--model-params-b", type=float, default=70.0, help="model size in billions")
    parser.add_argument("--csv", metavar="PATH", help="also write raw rows to CSV")
    args = parser.parse_args()

    rows = collect(args.hardware, args.model_params_b)
    report(rows, args.hardware, args.model_params_b)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()

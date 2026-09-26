#!/usr/bin/env python3
"""EcoRoute live demo -- Milestone 5.

Drives the real MCP server as a subprocess over STDIO using a real MCP
client, exactly as Claude Desktop would. Nothing is faked or precomputed:
every number on screen came back over JSON-RPC during the run.

Runs fully offline and deterministically, so it behaves identically in
rehearsal and in the lecture hall.

Usage:
    python demo.py              # run straight through
    python demo.py --pause      # wait for Enter between acts (presenting)
    python demo.py --raw        # also print the raw JSON-RPC payloads
    python demo.py --no-color   # plain text
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ecoroute.display import fmt, fmt_column, table

PROJECT_ROOT = Path(__file__).resolve().parent

# The headline workload: a full-context retrieval answer, the kind of job
# that is latency-tolerant enough to be worth relocating.
WORKLOAD = {"prompt_tokens": 128_000, "completion_tokens": 2_000}

FLEET_REQUESTS_PER_DAY = 1_000_000


class Style:
    """ANSI styling, disabled when piping to a file or when --no-color is set."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def head(self, text: str) -> str:
        return self._wrap("1;36", text)

    def good(self, text: str) -> str:
        return self._wrap("1;32", text)

    def warn(self, text: str) -> str:
        return self._wrap("1;33", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)


class Demo:
    def __init__(self, session: ClientSession, style: Style, pause: bool, raw: bool) -> None:
        self.session = session
        self.s = style
        self.pause = pause
        self.raw = raw
        self.act_number = 0

    # --- presentation plumbing ----------------------------------------------------

    def act(self, title: str, subtitle: str = "") -> None:
        self.act_number += 1
        if self.act_number > 1:
            self._wait()
        print()
        print(self.s.head(f"{'=' * 78}"))
        print(self.s.head(f"  ACT {self.act_number}.  {title}"))
        if subtitle:
            print(self.s.dim(f"          {subtitle}"))
        print(self.s.head(f"{'=' * 78}"))

    def _wait(self) -> None:
        if self.pause and sys.stdin.isatty():
            try:
                input(self.s.dim("\n      [Enter] to continue "))
            except (EOFError, KeyboardInterrupt):
                print()

    def say(self, text: str) -> None:
        print(f"\n  {text}")

    async def call(self, tool: str, **arguments) -> dict:
        """Invoke a tool over JSON-RPC and return its parsed payload."""
        shown = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        print(self.s.dim(f"\n  -> tools/call  {tool}({shown})"))

        result = await self.session.call_tool(tool, arguments)
        text = result.content[0].text

        if result.is_error:
            print(self.s.warn(f"  <- error: {text}"))
            return {"_error": text}

        payload = json.loads(text)
        if self.raw:
            print(self.s.dim("  <- " + json.dumps(payload, indent=2)[:1200].replace("\n", "\n     ")))
        return payload

    # --- the demo ----------------------------------------------------------------

    async def run(self) -> None:
        await self.act_handshake()
        await self.act_catalog()
        baseline = await self.act_status_quo()
        await self.act_relaxed_budget()
        await self.act_tight_budget()
        await self.act_infeasible_budget()
        await self.act_fleet_scale(baseline)
        self.closing()

    async def act_handshake(self) -> None:
        self.act(
            "The client discovers the server",
            "Standard MCP handshake over STDIO -- no EcoRoute-specific glue code.",
        )
        tools = await self.session.list_tools()
        self.say(f"Server advertised {self.s.bold(str(len(tools.tools)))} tools:")
        print()
        for tool in tools.tools:
            required = tool.input_schema.get("required") or []
            print(f"    {self.s.bold(tool.name)}")
            print(self.s.dim(f"      required: {', '.join(required) if required else '(none)'}"))
        self.say(
            "An LLM reads these descriptions and decides for itself when to call them.\n"
            "  That is the whole point of MCP: capability discovery, not hard-coded integration."
        )

    async def act_catalog(self) -> None:
        self.act(
            "What the server knows about the grid",
            "list_datacenter_regions() -- telemetry, not guesswork.",
        )
        catalog = await self.call("list_datacenter_regions")
        regions = catalog["regions"]
        rows = [
            [
                r["id"],
                r["corridor"].split(" (")[0],
                ", ".join(r["dominant_sources"]),
                f"{r['carbon_intensity_g_per_kwh']:.0f}",
                f"{r['pue']:.2f}",
                f"{r['wue_l_per_kwh']:.2f}",
                f"{r['baseline_latency_ms']} ms",
            ]
            for r in regions
        ]
        print()
        print(table(
            ["region", "corridor", "dominant grid", "gCO2/kWh", "PUE", "WUE", "latency"],
            rows,
            indent="    ",
        ))
        dirtiest = max(regions, key=lambda r: r["carbon_intensity_g_per_kwh"])
        cleanest = min(regions, key=lambda r: r["carbon_intensity_g_per_kwh"])
        ratio = dirtiest["carbon_intensity_g_per_kwh"] / cleanest["carbon_intensity_g_per_kwh"]
        dirtier = self.s.bold(f"{ratio:.0f}x dirtier")
        self.say(
            f"The same kilowatt-hour is {dirtier} in "
            f"{dirtiest['id']} than in {cleanest['id']}.\n"
            "  Note the tension the router has to resolve: the cleanest grid is also the furthest away."
        )

    async def act_status_quo(self) -> dict:
        self.act(
            "The status quo: route by latency alone",
            f"A {WORKLOAD['prompt_tokens']:,}-token retrieval job lands in Virginia because it is nearest.",
        )
        result = await self.call(
            "calculate_inference_footprint", **WORKLOAD, region="us-east-va"
        )
        print()
        print(table(
            ["metric", "value"],
            [
                ["accelerator energy", fmt(result["it_energy_kwh"], "kWh")],
                ["cooling overhead (PUE " + f"{result['pue']}" + ")", fmt(result["overhead_energy_kwh"], "kWh")],
                ["total facility energy", fmt(result["energy_kwh"], "kWh")],
                ["carbon emitted", fmt(result["carbon_g_co2eq"], "g")],
                ["cooling water evaporated", fmt(result["water_liters"], "L")],
            ],
            indent="    ",
        ))
        prefill = result["prefill_joules"]
        decode = result["decode_joules"]
        prefill_j = self.s.bold(f"{prefill:,.0f} J")
        decode_j = self.s.bold(f"{decode:,.0f} J")
        per_token_ratio = (decode / WORKLOAD["completion_tokens"]) / (
            prefill / WORKLOAD["prompt_tokens"]
        )
        self.say(
            f"Worth noticing: prefill burned {prefill_j} for "
            f"{WORKLOAD['prompt_tokens']:,} tokens, while decode burned "
            f"{decode_j} for only {WORKLOAD['completion_tokens']:,}.\n"
            f"  Generation is ~{per_token_ratio:.0f}x "
            "dearer per token, because decode is memory-bandwidth-bound, not compute-bound."
        )
        return result

    async def act_relaxed_budget(self) -> None:
        self.act(
            "Ask the router where it should have gone",
            "get_optimal_region() with a 500 ms budget -- a batch job nobody is waiting on.",
        )
        decision = await self.call("get_optimal_region", **WORKLOAD, max_latency_ms=500)
        self._print_ranking(decision)
        region = self.s.good(decision["recommended_region"])
        saved = self.s.good(fmt(decision["carbon_saved_g_co2eq"], "g"))
        carbon_pct = self.s.good(f"{decision['carbon_reduction_pct']:.1f}% reduction")
        water_pct = self.s.good(f"{decision['water_reduction_pct']:.1f}%")
        self.say(
            f"Recommendation: {region} "
            f"({decision['recommended_region_name']}) at {decision['latency_ms']} ms.\n"
            f"  Carbon avoided: {saved} -- a {carbon_pct} "
            f"versus {decision['baseline_region']}.\n"
            f"  Water avoided: {water_pct}."
        )

    async def act_tight_budget(self) -> None:
        self.act(
            "Now make it interactive",
            "Same job, 50 ms budget. The cleanest grid is no longer reachable.",
        )
        decision = await self.call("get_optimal_region", **WORKLOAD, max_latency_ms=50)
        self._print_ranking(decision)
        if decision["excluded_regions"]:
            print(self.s.warn("\n    excluded by the latency constraint:"))
            for entry in decision["excluded_regions"]:
                print(self.s.dim(f"      {entry['region']:<15} {entry['reason']}"))
        region = self.s.good(decision["recommended_region"])
        cleaner = self.s.good(f"{decision['carbon_reduction_pct']:.1f}% cleaner")
        self.say(
            f"The recommendation shifted to {region} "
            f"-- still {cleaner} than Virginia,\n"
            "  but reachable inside the budget. This is the deterministic constraint routing:\n"
            "  the greenest option is not chosen when the workload cannot tolerate it."
        )

    async def act_infeasible_budget(self) -> None:
        self.act(
            "When nothing fits, say so",
            "A 5 ms budget is satisfiable by no modeled region.",
        )
        result = await self.call("get_optimal_region", **WORKLOAD, max_latency_ms=5)
        self.say(
            "The server returns a structured protocol error rather than silently\n"
            "  recommending something that violates the constraint. An agent can act on that."
        )
        if "_error" not in result:
            print(self.s.warn("    (expected an error here)"))

    async def act_fleet_scale(self, baseline: dict) -> None:
        self.act(
            "Why milligrams matter",
            f"Projecting one routing decision across {FLEET_REQUESTS_PER_DAY:,} requests/day.",
        )
        small = {"prompt_tokens": 512, "completion_tokens": 128}
        va = await self.call("calculate_inference_footprint", **small, region="us-east-va")
        se = await self.call("calculate_inference_footprint", **small, region="eu-north-se")

        daily_g = (va["carbon_g_co2eq"] - se["carbon_g_co2eq"]) * FLEET_REQUESTS_PER_DAY
        daily_l = (va["water_liters"] - se["water_liters"]) * FLEET_REQUESTS_PER_DAY
        print()
        print(table(
            ["horizon", "carbon avoided", "water avoided"],
            [
                ["per request", fmt(va["carbon_g_co2eq"] - se["carbon_g_co2eq"], "g"),
                 fmt(va["water_liters"] - se["water_liters"], "L")],
                ["per day", f"{daily_g / 1000:,.1f} kg CO2eq", f"{daily_l:,.1f} L"],
                ["per year", f"{daily_g / 1e6 * 365:,.1f} tonnes CO2eq", f"{daily_l * 365 / 1000:,.1f} m3"],
            ],
            indent="    ",
        ))
        per_request = fmt(va["carbon_g_co2eq"] - se["carbon_g_co2eq"], "g")
        annual = self.s.good(f"{daily_g / 1e6 * 365:,.1f} tonnes a year")
        self.say(
            f"One chat turn is {per_request}. A fleet is {annual} -- "
            "from a scheduling decision,\n  not new hardware."
        )

    def _print_ranking(self, decision: dict) -> None:
        ranking = decision["ranking"]
        carbon = fmt_column([r["carbon_g_co2eq"] for r in ranking], "g")
        water = fmt_column([r["water_liters"] for r in ranking], "L")
        rows = [
            [
                ("-> " if r["region"] == decision["recommended_region"] else "   ") + str(r["rank"]),
                r["region"],
                f"{r['latency_ms']} ms",
                carbon[i],
                water[i],
            ]
            for i, r in enumerate(ranking)
        ]
        print()
        print(table(["rank", "region", "latency", "carbon", "water"], rows, indent="    "))

    def closing(self) -> None:
        print()
        print(self.s.head("=" * 78))
        print(self.s.head("  Every figure above arrived over JSON-RPC from a live MCP server."))
        print(self.s.dim("  Deterministic by design: re-run this and the numbers are identical."))
        print(self.s.head("=" * 78))
        print()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", action="store_true", help="wait for Enter between acts")
    parser.add_argument("--raw", action="store_true", help="print raw JSON-RPC payloads")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    parser.add_argument(
        "--server-log",
        action="store_true",
        help="show the server's own stderr (hidden by default so the intentional "
        "error in the last act does not print above the demo)",
    )
    args = parser.parse_args()

    style = Style(enabled=not args.no_color and sys.stdout.isatty())

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ecoroute.server"],
        cwd=str(PROJECT_ROOT),
    )

    print(style.dim(f"\n  launching: {sys.executable} -m ecoroute.server"))

    # The server logs anticipated ToolErrors to its own stderr. That stream is
    # unbuffered and interleaves ahead of our stdout, so the deliberate error in
    # the final act would otherwise appear above the first act mid-presentation.
    with contextlib.ExitStack() as stack:
        if args.server_log:
            errlog = sys.stderr
        else:
            errlog = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))

        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                print(style.dim(
                    f"  connected: {init.server_info.name} "
                    f"v{init.server_info.version or '0.1.0'}"
                ))
                await Demo(session, style, args.pause, args.raw).run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)

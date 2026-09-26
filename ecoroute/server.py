"""EcoRoute MCP server -- JSON-RPC tools over STDIO.

Protocol layer only. Every numeric answer comes from calculator.py; this
module's job is to register tool schemas, validate inputs, and translate
domain errors into protocol errors.

Run directly:
    python -m ecoroute.server

Or register with an MCP client (Claude Desktop / Cursor) -- see README.md.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ecoroute import __version__, calculator
from ecoroute.calculator import EcoRouteError
from ecoroute.schemas import (
    CompletionTokens,
    FootprintResult,
    HardwareId,
    MaxLatencyMs,
    ModelParamsB,
    PromptTokens,
    RegionCatalog,
    RegionId,
    RoutingDecision,
)

mcp = MCPServer(
    name="ecoroute",
    title="EcoRoute Carbon-Aware Router",
    version=__version__,
    instructions=(
        "EcoRoute models the energy, carbon and cooling-water cost of LLM inference "
        "across data-center regions, and recommends where to run latency-tolerant "
        "workloads. Call list_datacenter_regions first to discover valid region and "
        "hardware ids. All results are deterministic: the same inputs always produce "
        "the same numbers, and no live grid feed is consulted."
    ),
)


@mcp.tool(
    description=(
        "Compute energy (kWh), carbon emissions (gCO2eq) and evaporative cooling "
        "water loss (liters) for one inference request in a specific region."
    )
)
def calculate_inference_footprint(
    prompt_tokens: PromptTokens,
    completion_tokens: CompletionTokens,
    region: RegionId,
    hardware: HardwareId = "H100",
    model_params_b: ModelParamsB = 70.0,
) -> FootprintResult:
    try:
        result = calculator.calculate_inference_footprint(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            region=region,
            hardware=hardware,
            model_params_b=model_params_b,
        )
    except EcoRouteError as exc:
        raise ToolError(str(exc)) from exc
    return FootprintResult(**result)


@mcp.tool(
    description=(
        "Rank candidate data-center regions by carbon intensity subject to a latency "
        "budget, and report the emissions saved against a baseline region. Use this to "
        "decide where to shift a latency-tolerant inference job."
    )
)
def get_optimal_region(
    prompt_tokens: PromptTokens,
    completion_tokens: CompletionTokens,
    max_latency_ms: MaxLatencyMs = 500,
    hardware: HardwareId = "H100",
    model_params_b: ModelParamsB = 70.0,
    baseline_region: RegionId = "us-east-va",
) -> RoutingDecision:
    try:
        result = calculator.get_optimal_region(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_latency_ms=max_latency_ms,
            hardware=hardware,
            model_params_b=model_params_b,
            baseline_region=baseline_region,
        )
    except EcoRouteError as exc:
        raise ToolError(str(exc)) from exc
    return RoutingDecision(**result)


@mcp.tool(
    description=(
        "List modeled data-center regions with their grid mix, carbon intensity, PUE "
        "and water usage effectiveness, plus the available accelerator profiles."
    )
)
def list_datacenter_regions() -> RegionCatalog:
    return RegionCatalog(**calculator.list_datacenter_regions())


def main() -> None:
    """STDIO entrypoint."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

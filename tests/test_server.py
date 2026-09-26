"""End-to-end MCP protocol tests.

These spawn the real server as a subprocess and drive it over STDIO with a
real MCP client -- the same handshake Claude Desktop performs. They cover
Milestone 3 of the project plan: tools are actually reachable over
JSON-RPC, not merely importable as Python functions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.anyio

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "ecoroute.server"],
    cwd=str(PROJECT_ROOT),
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _call(session: ClientSession, tool: str, **arguments) -> dict:
    """Invoke a tool and return its parsed JSON payload."""
    result = await session.call_tool(tool, arguments)
    assert not result.is_error, f"{tool} returned an error: {result.content}"
    return json.loads(result.content[0].text)


async def test_handshake_advertises_the_three_planned_tools():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

    assert {t.name for t in tools.tools} == {
        "calculate_inference_footprint",
        "get_optimal_region",
        "list_datacenter_regions",
    }


async def test_every_tool_publishes_a_description_and_schema():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

    for tool in tools.tools:
        assert tool.description, f"{tool.name} has no description for the LLM to read"
        assert tool.input_schema["type"] == "object"


async def test_footprint_tool_returns_a_full_payload_over_the_wire():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            payload = await _call(
                session,
                "calculate_inference_footprint",
                prompt_tokens=1000,
                completion_tokens=500,
                region="us-east-va",
            )

    assert payload["region"] == "us-east-va"
    assert payload["hardware"] == "H100"
    assert payload["carbon_g_co2eq"] > 0
    assert payload["water_liters"] > 0


async def test_routing_tool_recommends_the_clean_grid_over_the_wire():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            payload = await _call(
                session,
                "get_optimal_region",
                prompt_tokens=1000,
                completion_tokens=500,
                max_latency_ms=500,
            )

    assert payload["recommended_region"] == "eu-north-se"
    assert payload["carbon_reduction_pct"] > 90
    assert [entry["rank"] for entry in payload["ranking"]] == [1, 2, 3, 4]


async def test_catalog_tool_needs_no_arguments():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            payload = await _call(session, "list_datacenter_regions")

    assert len(payload["regions"]) == 4
    assert {a["id"] for a in payload["accelerators"]} == {"H100", "A100"}


async def test_unknown_region_surfaces_as_a_protocol_error_not_a_crash():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "calculate_inference_footprint",
                {"prompt_tokens": 10, "completion_tokens": 10, "region": "mars-south-1"},
            )

    assert result.is_error
    assert "unknown region" in result.content[0].text


async def test_schema_violation_is_rejected_before_reaching_the_engine():
    """Negative tokens violate the Pydantic ge=0 bound in schemas.py."""
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "calculate_inference_footprint",
                {"prompt_tokens": -5, "completion_tokens": 10, "region": "us-east-va"},
            )

    assert result.is_error

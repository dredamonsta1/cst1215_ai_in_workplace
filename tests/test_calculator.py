"""Deterministic unit tests for the EcoRoute math engine.

Expected values are recomputed from first principles inside the tests
rather than pasted as magic literals, so a formula change fails loudly
instead of silently agreeing with a stale constant.
"""

from __future__ import annotations

import pytest

from ecoroute import calculator as calc
from ecoroute.calculator import EcoRouteError
from ecoroute.schemas import FootprintResult, RegionCatalog, RoutingDecision

WORKLOAD = dict(prompt_tokens=1000, completion_tokens=500)


# --- Data integrity ---------------------------------------------------------------


def test_all_four_planned_regions_are_modeled():
    assert set(calc.load_regions()) == {
        "us-east-va",
        "us-central-ia",
        "us-west-or",
        "eu-north-se",
    }


def test_region_baselines_match_the_project_plan():
    regions = calc.load_regions()
    assert regions["us-east-va"].carbon_intensity_g_per_kwh == 360.0
    assert regions["us-east-va"].pue == 1.18
    assert regions["us-central-ia"].carbon_intensity_g_per_kwh == 280.0
    assert regions["us-central-ia"].pue == 1.15
    assert regions["us-west-or"].carbon_intensity_g_per_kwh == 110.0
    assert regions["us-west-or"].pue == 1.12
    assert regions["eu-north-se"].carbon_intensity_g_per_kwh == 25.0
    assert regions["eu-north-se"].pue == 1.10


def test_planned_accelerators_are_modeled():
    accelerators = calc.load_accelerators()
    assert set(accelerators) == {"H100", "A100"}
    assert accelerators["H100"].tdp_watts == 700.0


def test_every_region_has_physically_sane_ranges():
    for region in calc.load_regions().values():
        assert region.pue >= 1.0, "PUE below 1.0 would mean free energy"
        assert region.carbon_intensity_g_per_kwh > 0
        assert region.wue_l_per_kwh >= 0
        assert region.baseline_latency_ms > 0


# --- Core formulas ----------------------------------------------------------------


def test_flops_per_token_is_two_per_parameter():
    assert calc.flops_per_token(70.0) == 2.0 * 70e9


def test_flops_per_joule_is_throughput_over_power():
    h100 = calc.get_accelerator("H100")
    expected = h100.peak_flops_bf16 * h100.prefill_utilization / h100.tdp_watts
    assert h100.flops_per_joule("prefill") == pytest.approx(expected)


def test_energy_matches_hand_computed_formula():
    h100 = calc.get_accelerator("H100")
    per_token = 2.0 * 70e9
    prefill_j = 1000 * per_token / h100.flops_per_joule("prefill")
    decode_j = 500 * per_token / h100.flops_per_joule("decode")
    expected_kwh = (prefill_j + decode_j) / 3.6e6 * 1.18

    result = calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va")
    assert result["energy_kwh"] == pytest.approx(expected_kwh, rel=1e-9)


def test_pue_overhead_is_the_difference_between_it_and_facility_energy():
    result = calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va")
    assert result["energy_kwh"] == pytest.approx(result["it_energy_kwh"] * 1.18, rel=1e-8)
    assert result["overhead_energy_kwh"] == pytest.approx(
        result["energy_kwh"] - result["it_energy_kwh"], rel=1e-6
    )


def test_carbon_is_energy_times_grid_intensity():
    # Both fields are reported to 9 significant figures, so the identity holds
    # to that precision -- not to machine epsilon.
    result = calc.calculate_inference_footprint(**WORKLOAD, region="us-west-or")
    assert result["carbon_g_co2eq"] == pytest.approx(result["energy_kwh"] * 110.0, rel=1e-8)


def test_water_is_energy_times_wue():
    result = calc.calculate_inference_footprint(**WORKLOAD, region="us-west-or")
    assert result["water_liters"] == pytest.approx(
        result["energy_kwh"] * result["wue_l_per_kwh"], rel=1e-8
    )


def test_energy_scales_linearly_with_token_count():
    single = calc.calculate_inference_footprint(100, 50, "us-east-va")
    double = calc.calculate_inference_footprint(200, 100, "us-east-va")
    assert double["energy_kwh"] == pytest.approx(2 * single["energy_kwh"], rel=1e-9)


def test_energy_scales_linearly_with_model_size():
    small = calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va", model_params_b=7.0)
    large = calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va", model_params_b=70.0)
    assert large["energy_kwh"] == pytest.approx(10 * small["energy_kwh"], rel=1e-9)


def test_decode_tokens_cost_more_energy_than_prompt_tokens():
    """Decode is bandwidth-bound, so a generated token is dearer than a read one."""
    prefill_only = calc.calculate_inference_footprint(1000, 0, "us-east-va")
    decode_only = calc.calculate_inference_footprint(0, 1000, "us-east-va")
    assert decode_only["energy_kwh"] > prefill_only["energy_kwh"]


def test_a100_burns_more_energy_than_h100_for_identical_work():
    h100 = calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va", hardware="H100")
    a100 = calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va", hardware="A100")
    assert a100["energy_kwh"] > h100["energy_kwh"]


# --- Edge cases -------------------------------------------------------------------


def test_zero_tokens_costs_nothing():
    result = calc.calculate_inference_footprint(0, 0, "us-east-va")
    assert result["energy_kwh"] == 0.0
    assert result["carbon_g_co2eq"] == 0.0
    assert result["water_liters"] == 0.0


def test_calculation_is_deterministic_across_repeated_calls():
    first = calc.calculate_inference_footprint(**WORKLOAD, region="eu-north-se")
    second = calc.calculate_inference_footprint(**WORKLOAD, region="eu-north-se")
    assert first == second


def test_unknown_region_is_rejected_with_a_helpful_message():
    with pytest.raises(EcoRouteError, match="unknown region"):
        calc.calculate_inference_footprint(**WORKLOAD, region="mars-south-1")


def test_unknown_hardware_is_rejected():
    with pytest.raises(EcoRouteError, match="unknown hardware"):
        calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va", hardware="TPUv9")


def test_negative_token_counts_are_rejected():
    with pytest.raises(EcoRouteError, match="non-negative"):
        calc.calculate_inference_footprint(-1, 10, "us-east-va")


def test_non_positive_model_size_is_rejected():
    with pytest.raises(EcoRouteError, match="must be positive"):
        calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va", model_params_b=0)


def test_unknown_phase_is_rejected():
    with pytest.raises(EcoRouteError, match="unknown phase"):
        calc.get_accelerator("H100").flops_per_joule("backward")


# --- Routing ----------------------------------------------------------------------


def test_generous_latency_budget_routes_to_the_cleanest_grid():
    decision = calc.get_optimal_region(**WORKLOAD, max_latency_ms=500)
    assert decision["recommended_region"] == "eu-north-se"
    assert decision["carbon_reduction_pct"] > 90


def test_tight_latency_budget_excludes_distant_clean_regions():
    decision = calc.get_optimal_region(**WORKLOAD, max_latency_ms=50)
    assert decision["recommended_region"] == "us-central-ia"
    excluded = {e["region"] for e in decision["excluded_regions"]}
    assert excluded == {"us-west-or", "eu-north-se"}


def test_impossible_latency_budget_raises_rather_than_guessing():
    with pytest.raises(EcoRouteError, match="no region satisfies"):
        calc.get_optimal_region(**WORKLOAD, max_latency_ms=5)


def test_non_positive_latency_budget_is_rejected():
    with pytest.raises(EcoRouteError, match="max_latency_ms must be positive"):
        calc.get_optimal_region(**WORKLOAD, max_latency_ms=0)


def test_ranking_is_ordered_by_ascending_carbon():
    ranking = calc.get_optimal_region(**WORKLOAD, max_latency_ms=500)["ranking"]
    carbon = [entry["carbon_g_co2eq"] for entry in ranking]
    assert carbon == sorted(carbon)
    assert [entry["rank"] for entry in ranking] == [1, 2, 3, 4]


def test_savings_are_measured_against_the_named_baseline():
    decision = calc.get_optimal_region(**WORKLOAD, max_latency_ms=500, baseline_region="us-west-or")
    assert decision["baseline_region"] == "us-west-or"
    expected = calc.calculate_inference_footprint(**WORKLOAD, region="us-west-or")
    assert decision["baseline_carbon_g_co2eq"] == expected["carbon_g_co2eq"]
    assert decision["carbon_saved_g_co2eq"] == pytest.approx(
        decision["baseline_carbon_g_co2eq"] - decision["carbon_g_co2eq"], abs=1e-9
    )


def test_routing_to_the_baseline_itself_reports_zero_saving():
    decision = calc.get_optimal_region(**WORKLOAD, max_latency_ms=20)
    assert decision["recommended_region"] == "us-east-va"
    assert decision["carbon_saved_g_co2eq"] == 0.0
    assert decision["carbon_reduction_pct"] == 0.0


def test_zero_token_workload_still_returns_a_valid_decision():
    """Guards the percent-delta divide-by-zero path."""
    decision = calc.get_optimal_region(0, 0, max_latency_ms=500)
    assert decision["carbon_reduction_pct"] == 0.0
    assert decision["water_reduction_pct"] == 0.0


# --- Catalog + schema contracts ---------------------------------------------------


def test_catalog_lists_every_region_and_accelerator():
    catalog = calc.list_datacenter_regions()
    assert len(catalog["regions"]) == 4
    assert len(catalog["accelerators"]) == 2


def test_engine_output_satisfies_the_published_tool_schemas():
    """The calculator must never drift from the frozen MCP contracts."""
    FootprintResult(**calc.calculate_inference_footprint(**WORKLOAD, region="us-east-va"))
    RoutingDecision(**calc.get_optimal_region(**WORKLOAD, max_latency_ms=500))
    RegionCatalog(**calc.list_datacenter_regions())

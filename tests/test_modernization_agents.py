import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rca_app.modernization_agents import (
    MODERNIZATION_AGENT_SPECS,
    build_modernization_subagent_task,
    list_modernization_agent_specs,
    validate_phase_parameters,
)


def test_all_specs_have_required_fields():
    specs = list_modernization_agent_specs()
    assert specs
    phase_ids = {item["phase_id"] for item in specs}
    assert phase_ids == set(MODERNIZATION_AGENT_SPECS.keys())
    for spec in specs:
        assert spec["agent_name"]
        assert isinstance(spec["required_parameters"], list)
        assert spec["default_output_schema"].startswith("{")


def test_validate_phase_parameters_returns_missing_required_keys():
    missing = validate_phase_parameters("refactor", {"domains": ["billing"]})
    assert "refactor_engine_version" in missing
    assert "target_database" in missing
    assert "root_package" in missing


def test_build_task_embeds_phase_and_contract_fields():
    task = build_modernization_subagent_task("documentation", {"scope": "core", "detail_level": "SUMMARY"})
    assert "DocumentationAgent" in task
    assert "Phase=documentation" in task
    assert "inputs_used" in task
    assert "risks" in task

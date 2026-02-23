from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ModernizationAgentSpec:
    phase_id: str
    agent_name: str
    description: str
    required_parameters: List[str]
    optional_parameters: List[str]
    default_output_schema: str


MODERNIZATION_AGENT_SPECS: Dict[str, ModernizationAgentSpec] = {
    "kickoff": ModernizationAgentSpec(
        phase_id="kickoff",
        agent_name="KickoffAgent",
        description="Validates repository connectivity, collaborators, and run readiness.",
        required_parameters=["source_repository", "destination_repository", "branch_strategy"],
        optional_parameters=["collaborators", "run_configuration", "compliance_rules"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","modernization_job_manifest":{},"readiness_checks":[],"missing_inputs":[]}',
    ),
    "code_analysis": ModernizationAgentSpec(
        phase_id="code_analysis",
        agent_name="CodeAnalysisAgent",
        description="Builds code inventory, complexity profile, and dependency graph.",
        required_parameters=["scope"],
        optional_parameters=["include_file_types", "exclude_paths", "complexity_rules"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","code_inventory":{},"dependency_graph":{},"missing_assets":[]}',
    ),
    "data_analysis": ModernizationAgentSpec(
        phase_id="data_analysis",
        agent_name="DataAnalysisAgent",
        description="Maps datasets, DB2 usage, data dictionary, and lineage dependencies.",
        required_parameters=["scope"],
        optional_parameters=["include_copybooks", "include_db2", "lineage_depth"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","data_lineage":{},"data_dictionary":{},"program_data_interactions":[]}',
    ),
    "activity_metrics": ModernizationAgentSpec(
        phase_id="activity_metrics",
        agent_name="ActivityMetricsAgent",
        description="Analyzes SMF activity metrics and prioritizes high-impact workloads.",
        required_parameters=["smf_input_location"],
        optional_parameters=["workload_filters", "ranking_strategy"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","activity_profile":{},"high_impact_jobs":[],"missing_inputs":[]}',
    ),
    "documentation": ModernizationAgentSpec(
        phase_id="documentation",
        agent_name="DocumentationAgent",
        description="Generates summary or detailed functional technical documentation.",
        required_parameters=["scope", "detail_level"],
        optional_parameters=["output_location", "include_dependencies", "include_transaction_flows"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","technical_specifications":{},"documents":[]}',
    ),
    "business_logic_extraction": ModernizationAgentSpec(
        phase_id="business_logic_extraction",
        agent_name="BusinessLogicExtractionAgent",
        description="Extracts business rules, decision tables, exceptions, and traceability.",
        required_parameters=["scope"],
        optional_parameters=["include_acceptance_criteria", "rule_types"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","business_logic_catalog":{},"business_rules":[],"traceability":[]}',
    ),
    "decomposition": ModernizationAgentSpec(
        phase_id="decomposition",
        agent_name="DecompositionAgent",
        description="Builds domain decomposition with boundaries and shared kernels.",
        required_parameters=["scope"],
        optional_parameters=["seed_files", "domain_constraints", "decomposition_mode"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","domain_decomposition":{},"dependency_views":{},"actions_taken":[]}',
    ),
    "migration_planning": ModernizationAgentSpec(
        phase_id="migration_planning",
        agent_name="MigrationPlanningAgent",
        description="Creates migration waves, sequencing rationale, and rollback strategy.",
        required_parameters=["domain_inputs"],
        optional_parameters=["wave_preferences", "risk_weights", "criticality_rules"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","migration_wave_plan":{},"cutover_strategy":{},"critical_path":[]}',
    ),
    "test_planning": ModernizationAgentSpec(
        phase_id="test_planning",
        agent_name="TestPlanningAgent",
        description="Defines equivalence, regression, and non-functional test strategy.",
        required_parameters=["scope", "test_objectives"],
        optional_parameters=["acceptance_criteria", "parity_metrics"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","test_plan":{},"acceptance_criteria":[],"coverage_targets":{}}',
    ),
    "test_data": ModernizationAgentSpec(
        phase_id="test_data",
        agent_name="TestDataAgent",
        description="Generates scripts to collect and prepare representative test data.",
        required_parameters=["scope", "source_data_systems"],
        optional_parameters=["masking_rules", "sampling_strategy", "output_location"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","test_data_scripts":[],"datasets":[],"data_dependencies":[]}',
    ),
    "test_automation": ModernizationAgentSpec(
        phase_id="test_automation",
        agent_name="TestAutomationAgent",
        description="Generates parity and regression automation suites.",
        required_parameters=["scope", "execution_framework"],
        optional_parameters=["pipeline_target", "reporting_format"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","automation_scripts":[],"test_suites":[],"execution_notes":[]}',
    ),
    "refactor": ModernizationAgentSpec(
        phase_id="refactor",
        agent_name="RefactorAgent",
        description="Transforms legacy code into target cloud-native implementation patterns.",
        required_parameters=["domains", "refactor_engine_version", "target_database", "root_package"],
        optional_parameters=["project_name", "legacy_encoding", "coding_standards"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","transformed_artifacts":[],"transformation_report":{},"skipped_domains":[]}',
    ),
    "reforge": ModernizationAgentSpec(
        phase_id="reforge",
        agent_name="ReforgeAgent",
        description="Hardens transformed code with quality, security, and performance refinements.",
        required_parameters=["source_project_location"],
        optional_parameters=["class_selection", "quality_profile", "security_profile", "performance_profile"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","reforge_diffs":[],"maintainability_deltas":{},"hardening_report":{}}',
    ),
    "modernization_qa": ModernizationAgentSpec(
        phase_id="modernization_qa",
        agent_name="ModernizationQAAgent",
        description="Cross-cutting quality gate reviewer for artifacts, risks, and readiness.",
        required_parameters=["phase_outputs"],
        optional_parameters=["quality_gates", "risk_thresholds"],
        default_output_schema='{"status":"COMPLETED|BLOCKED","gate_result":{},"open_risks":[],"manual_decisions":[]}',
    ),
}


def list_modernization_agent_specs() -> List[Dict[str, Any]]:
    return [
        {
            "phase_id": spec.phase_id,
            "agent_name": spec.agent_name,
            "description": spec.description,
            "required_parameters": spec.required_parameters,
            "optional_parameters": spec.optional_parameters,
            "default_output_schema": spec.default_output_schema,
        }
        for spec in MODERNIZATION_AGENT_SPECS.values()
    ]


def validate_phase_parameters(phase_id: str, parameters: Dict[str, Any]) -> List[str]:
    spec = MODERNIZATION_AGENT_SPECS[phase_id]
    missing: List[str] = []
    for key in spec.required_parameters:
        value = parameters.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)
    return missing


def build_modernization_subagent_task(phase_id: str, parameters: Dict[str, Any]) -> str:
    spec = MODERNIZATION_AGENT_SPECS[phase_id]
    return (
        f"You are {spec.agent_name}. "
        "Execute only your assigned phase and update handoff-compatible outputs. "
        "If inputs are missing, return status BLOCKED with exact missing_inputs and continue with available analysis. "
        f"Phase={phase_id}. "
        f"Description={spec.description}. "
        f"Parameters={parameters}. "
        "Include inputs_used, outputs_produced, open_questions, assumptions, decisions, and risks in your JSON response."
    )

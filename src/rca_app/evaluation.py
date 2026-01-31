from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List

from .app import RCAApp
from .observability import build_langfuse_client, build_langfuse_invoke_config

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


TOOL_TO_AGENT = {
    "hypothesis_agent_tool": "HypothesisAgent",
    "sales_analysis_agent_tool": "SalesAnalysisAgent",
    "inventory_analysis_agent_tool": "InventoryAnalysisAgent",
    "hypothesis_validation_agent_tool": "HypothesisValidationAgent",
    "root_cause_analysis_agent_tool": "RootCauseAgent",
    "write_todos": "OrchestrationAgent",
}


def flatten_trace(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat = []

    for msg in result.get("trace", []):
        if msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                agent = TOOL_TO_AGENT.get(call["name"], call["name"])
                flat.append(
                    {
                        "agent": agent,
                        "tool": call["name"],
                        "args": call.get("args", {}),
                        "call_id": call.get("id"),
                    }
                )

        if msg.get("type") == "ToolMessage":
            flat.append(
                {
                    "agent": "ToolResult",
                    "content": msg.get("content"),
                    "tool_call_id": msg.get("tool_call_id"),
                }
            )

    return flat


def extract_root_cause(result: Dict[str, Any]) -> Dict[str, Any]:
    for step in result.get("trace", []):
        for call in step.get("tool_calls", []):
            if call.get("name") == "root_cause_analysis_agent_tool":
                return call.get("output", {})
    return {}


@dataclass
class GoldRCACase:
    case_id: str
    task: str
    expected_root_causes: List[str]
    gold_hypotheses: List[str]
    must_use_agents: List[str]
    forbidden_root_causes: List[str]


GOLD_RCA_DATASET: List[GoldRCACase] = [
    GoldRCACase(
        case_id="PROMO_STOCKOUT_01",
        task="Why did Store S003 face stockouts during the Diwali promotion?",
        expected_root_causes=["Delayed replenishment", "Promo uplift underestimated"],
        gold_hypotheses=[
            "Demand spike due to promotion",
            "Delayed replenishment",
            "Inventory transfer delay",
            "Forecast underestimation",
        ],
        must_use_agents=[
            "HypothesisAgent",
            "SalesAnalysisAgent",
            "InventoryAnalysisAgent",
            "HypothesisValidationAgent",
        ],
        forbidden_root_causes=["System outage", "Pricing error"],
    ),
    GoldRCACase(
        case_id="SALES_DROP_02",
        task="Why did sales drop in the North region despite stable inventory?",
        expected_root_causes=["Pricing mismatch", "Local competition impact"],
        gold_hypotheses=[
            "Price increase",
            "Competitive promotion",
            "Demand elasticity change",
            "Assortment mismatch",
        ],
        must_use_agents=["HypothesisAgent", "SalesAnalysisAgent", "HypothesisValidationAgent"],
        forbidden_root_causes=["Inventory stockout", "Warehouse delay"],
    ),
]


@dataclass
class EvalScores:
    precision: float
    recall: float
    hypothesis_coverage: float
    evidence_score: float
    process_compliance: bool
    forbidden_penalty: bool


def normalize(text: str) -> str:
    return text.lower().strip()


def semantic_match(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)
    return a in b or b in a


def count_semantic_matches(predicted: List[str], gold: List[str]) -> int:
    count = 0
    for g in gold:
        if any(semantic_match(p, g) for p in predicted):
            count += 1
    return count


def check_process_order(trace: List[Dict[str, Any]], required_agents: List[str]) -> bool:
    executed = {t["agent"] for t in trace}
    return all(agent in executed for agent in required_agents)


def evidence_backed(validated: Dict[str, bool], trace: List[Dict[str, Any]]) -> float:
    if not validated:
        return 0.0

    evidence_agents = {"SalesAnalysisAgent", "InventoryAnalysisAgent"}
    used_agents = {t["agent"] for t in trace}
    has_evidence = evidence_agents.intersection(used_agents)

    supported = sum(1 for v in validated.values() if v and has_evidence)

    return supported / max(len(validated), 1)


def evaluate_single_case(gold: GoldRCACase, rca_output: Dict[str, Any]) -> EvalScores:
    trace = flatten_trace(rca_output)
    root_causes = rca_output["root_cause"]["primary_root_causes"]
    hypotheses = rca_output.get("hypotheses", [])
    validated = rca_output.get("validated", {})

    matched = count_semantic_matches(root_causes, gold.expected_root_causes)
    precision = matched / max(len(root_causes), 1)
    recall = matched / max(len(gold.expected_root_causes), 1)

    coverage = count_semantic_matches(hypotheses, gold.gold_hypotheses)
    hypothesis_coverage = coverage / max(len(gold.gold_hypotheses), 1)

    evidence = evidence_backed(validated, trace)

    process_ok = check_process_order(trace, gold.must_use_agents)

    forbidden_penalty = any(
        semantic_match(rc, f)
        for rc in root_causes
        for f in gold.forbidden_root_causes
    )

    return EvalScores(
        precision=precision,
        recall=recall,
        hypothesis_coverage=hypothesis_coverage,
        evidence_score=evidence,
        process_compliance=process_ok,
        forbidden_penalty=forbidden_penalty,
    )


def normalize_trace(trace: Any) -> List[Dict[str, Any]]:
    if trace is None:
        return []
    if isinstance(trace, dict):
        return [trace]
    if isinstance(trace, list):
        return [t for t in trace if isinstance(t, dict)]
    return []


def extract_hypotheses(result: Dict[str, Any]) -> List[str]:
    for step in result.get("trace", []):
        if step.get("agent") == "HypothesisAgent":
            return step.get("hypotheses", [])
    return []


def extract_validated(result: Dict[str, Any]) -> Dict[str, Any]:
    for step in result.get("trace", []):
        if step.get("agent") == "HypothesisValidationAgent":
            return step.get("details", {}).get("validated", {})
    return {}


def build_eval_tags(prompt_label: str, run_label: str) -> List[str]:
    tags = ["eval", run_label]
    if prompt_label:
        tags.append(f"prompt_label:{prompt_label}")
    return tags


def build_eval_metadata(case_id: str, prompt_label: str, memory_enabled: bool | None) -> Dict[str, Any]:
    metadata = {
        "entrypoint": "evaluation",
        "eval_case": case_id,
    }
    if memory_enabled is not None:
        metadata["memory_enabled"] = memory_enabled
    if prompt_label:
        metadata["prompt_label"] = prompt_label
    return metadata


def log_eval_scores(
    app: RCAApp,
    scores: EvalScores,
    case_id: str,
    run_label: str,
    session_id: str,
    memory_enabled: bool | None,
) -> None:
    client = build_langfuse_client(app.config)
    if not client:
        return

    metadata = build_eval_metadata(case_id, app.config.langfuse_prompt_label, memory_enabled)
    score_map = {
        "precision": (scores.precision, "NUMERIC"),
        "recall": (scores.recall, "NUMERIC"),
        "hypothesis_coverage": (scores.hypothesis_coverage, "NUMERIC"),
        "evidence_score": (scores.evidence_score, "NUMERIC"),
        "process_compliance": (scores.process_compliance, "BOOLEAN"),
        "forbidden_penalty": (scores.forbidden_penalty, "BOOLEAN"),
    }
    for metric, (value, data_type) in score_map.items():
        client.create_score(
            name=f"eval_{run_label}_{metric}",
            value=value,
            data_type=data_type,
            session_id=session_id,
            metadata=metadata,
        )
    client.flush()


def run_rca_with_memory(app: RCAApp, case: GoldRCACase) -> Dict[str, Any]:
    query_id = f"eval_{case.case_id}_with_memory"
    config = {"configurable": {"user_id": "eval_user", "thread_id": query_id, "memory_enabled": True}}
    rca_state = {"task": case.task, "output": "", "trace": []}
    observability_config = build_langfuse_invoke_config(
        app.config,
        user_id="eval_user",
        query_id=query_id,
        tags=build_eval_tags(app.config.langfuse_prompt_label, "with_memory"),
        metadata=build_eval_metadata(case.case_id, app.config.langfuse_prompt_label, True),
    )
    logger.info("Running RCA evaluation with memory")
    result = app.app.invoke(rca_state, {**config, **observability_config})
    normalized_trace = normalize_trace(result.get("trace"))
    return {
        "root_cause": extract_root_cause({"trace": normalized_trace}),
        "hypotheses": extract_hypotheses({"trace": normalized_trace}),
        "validated": extract_validated({"trace": normalized_trace}),
        "trace": normalized_trace,
    }


def run_rca_without_memory(app: RCAApp, case: GoldRCACase) -> Dict[str, Any]:
    query_id = f"eval_{case.case_id}_without_memory"
    config = {
        "configurable": {"user_id": "eval_user_nomem", "thread_id": query_id, "memory_enabled": False}
    }
    empty_state = {"task": case.task, "output": "", "trace": []}
    observability_config = build_langfuse_invoke_config(
        app.config,
        user_id="eval_user_nomem",
        query_id=query_id,
        tags=build_eval_tags(app.config.langfuse_prompt_label, "without_memory"),
        metadata=build_eval_metadata(case.case_id, app.config.langfuse_prompt_label, False),
    )
    logger.info("Running RCA evaluation without memory")
    result = app.app.invoke(empty_state, {**config, **observability_config})
    return {
        "root_cause": extract_root_cause(result),
        "hypotheses": extract_hypotheses(result),
        "validated": extract_validated(result),
        "trace": result.get("trace", []),
    }


def run_memory_ablation(app: RCAApp, case: GoldRCACase) -> Dict[str, EvalScores]:
    out_mem = run_rca_with_memory(app, case)
    out_nomem = run_rca_without_memory(app, case)
    with_memory_scores = evaluate_single_case(case, out_mem)
    without_memory_scores = evaluate_single_case(case, out_nomem)
    log_eval_scores(
        app,
        with_memory_scores,
        case.case_id,
        "with_memory",
        f"eval_{case.case_id}_with_memory",
        True,
    )
    log_eval_scores(
        app,
        without_memory_scores,
        case.case_id,
        "without_memory",
        f"eval_{case.case_id}_without_memory",
        False,
    )
    return {
        "with_memory": with_memory_scores,
        "without_memory": without_memory_scores,
    }


def learning_curve(app: RCAApp, cases: List[GoldRCACase]) -> List[float]:
    recalls = []
    for c in cases:
        query_id = f"eval_{c.case_id}_learning_curve"
        config = {"configurable": {"user_id": "eval_user", "thread_id": query_id}}
        observability_config = build_langfuse_invoke_config(
            app.config,
            user_id="eval_user",
            query_id=query_id,
            tags=build_eval_tags(app.config.langfuse_prompt_label, "learning_curve"),
            metadata=build_eval_metadata(c.case_id, app.config.langfuse_prompt_label, None),
        )
        rca_state = {"task": c.task, "output": "", "trace": []}
        out = app.app.invoke(rca_state, {**config, **observability_config})
        score = evaluate_single_case(c, out)
        recalls.append(score.recall)
        log_eval_scores(app, score, c.case_id, "learning_curve", query_id, None)
    return recalls

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .app import RCAApp, run_rca


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


def run_rca_with_memory(app: RCAApp, task: str) -> Dict[str, Any]:
    config = {"configurable": {"user_id": "eval_user", "thread_id": "eval_thread", "memory_enabled": True}}
    rca_state = {"task": task, "trace": []}
    result = app.app.invoke(rca_state, config)
    normalized_trace = normalize_trace(result.get("trace"))
    return {
        "root_cause": extract_root_cause({"trace": normalized_trace}),
        "hypotheses": extract_hypotheses({"trace": normalized_trace}),
        "validated": extract_validated({"trace": normalized_trace}),
        "trace": normalized_trace,
    }


def run_rca_without_memory(app: RCAApp, task: str) -> Dict[str, Any]:
    config = {
        "configurable": {"user_id": "eval_user_nomem", "thread_id": "eval_thread_nomem", "memory_enabled": False}
    }
    empty_state = {"task": task, "trace": []}
    result = app.app.invoke(empty_state, config)
    return {
        "root_cause": extract_root_cause(result),
        "hypotheses": extract_hypotheses(result),
        "validated": extract_validated(result),
        "trace": result.get("trace", []),
    }


def run_memory_ablation(app: RCAApp, case: GoldRCACase) -> Dict[str, EvalScores]:
    out_mem = run_rca_with_memory(app, case.task)
    out_nomem = run_rca_without_memory(app, case.task)
    return {
        "with_memory": evaluate_single_case(case, out_mem),
        "without_memory": evaluate_single_case(case, out_nomem),
    }


def learning_curve(app: RCAApp, cases: List[GoldRCACase]) -> List[float]:
    recalls = []
    for c in cases:
        out = run_rca(app, c.task, user_id="eval_user", query_id="eval_thread")
        score = evaluate_single_case(c, out)
        recalls.append(score.recall)
    return recalls

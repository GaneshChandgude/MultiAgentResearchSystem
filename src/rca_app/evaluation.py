from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Optional

from .app import RCAApp
from .observability import build_langfuse_client, build_langfuse_invoke_config
from .utils import extract_json_from_response, process_response

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def normalize_trace(trace: Any) -> List[Dict[str, Any]]:
    if trace is None:
        return []
    if isinstance(trace, dict):
        return [trace]
    if isinstance(trace, list):
        return [t for t in trace if isinstance(t, dict)]
    return []


def iter_trace_messages(trace: Any) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for entry in normalize_trace(trace):
        tool_calls = entry.get("tool_calls")
        if isinstance(tool_calls, list) and entry.get("agent") == "Orchestration Agent":
            messages.extend(msg for msg in tool_calls if isinstance(msg, dict))
        elif entry.get("type") in {"AIMessage", "ToolMessage"}:
            messages.append(entry)
    return messages


def _parse_tool_output(content: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(content, str):
        return None
    try:
        return json.loads(extract_json_from_response(content))
    except json.JSONDecodeError:
        return None


def _collect_tool_outputs(trace: Any) -> List[Dict[str, Any]]:
    messages = iter_trace_messages(trace)
    tool_call_map = {}
    outputs = []

    for msg in messages:
        if msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                tool_call_map[call.get("id")] = call.get("name")
        if msg.get("type") == "ToolMessage":
            tool_name = tool_call_map.get(msg.get("tool_call_id"))
            if not tool_name:
                continue
            parsed = _parse_tool_output(msg.get("content"))
            if isinstance(parsed, dict):
                outputs.append({"tool": tool_name, "output": parsed})

    return outputs


def extract_root_cause(result: Dict[str, Any]) -> Dict[str, Any]:
    for entry in _collect_tool_outputs(result.get("trace")):
        if entry["tool"] == "root_cause_analysis_agent_tool":
            return entry["output"].get("root_cause", {})
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
    intent_resolution_accuracy: float
    tool_call_accuracy: float
    collaboration_quality: float
    correctness: float
    hallucination: float
    relevance: float
    toxicity: float
    helpfulness: float
    conciseness: float


def evaluate_single_case(app: RCAApp, gold: GoldRCACase, rca_output: Dict[str, Any]) -> EvalScores:
    response = rca_output.get("response", "")
    if not response:
        response = rca_output.get("output", "")

    judge_scores = evaluate_orchestration_llm_judge(app, gold, response, rca_output.get("trace"))

    return EvalScores(
        intent_resolution_accuracy=judge_scores["intent_resolution_accuracy"],
        tool_call_accuracy=judge_scores["tool_call_accuracy"],
        collaboration_quality=judge_scores["collaboration_quality"],
        correctness=judge_scores["correctness"],
        hallucination=judge_scores["hallucination"],
        relevance=judge_scores["relevance"],
        toxicity=judge_scores["toxicity"],
        helpfulness=judge_scores["helpfulness"],
        conciseness=judge_scores["conciseness"],
    )


def extract_hypotheses(result: Dict[str, Any]) -> List[str]:
    for entry in _collect_tool_outputs(result.get("trace")):
        if entry["tool"] == "hypothesis_agent_tool":
            return entry["output"].get("hypotheses", [])
    return []


def extract_validated(result: Dict[str, Any]) -> Dict[str, Any]:
    for entry in _collect_tool_outputs(result.get("trace")):
        if entry["tool"] == "hypothesis_validation_agent_tool":
            return entry["output"].get("validated", {})
    return {}


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _extract_tool_calls(trace: Any) -> List[Dict[str, Any]]:
    for entry in normalize_trace(trace):
        if entry.get("agent") == "Orchestration Agent":
            tool_calls = entry.get("tool_calls", [])
            if isinstance(tool_calls, list):
                return tool_calls
    return []


def evaluate_orchestration_llm_judge(
    app: RCAApp,
    gold: GoldRCACase,
    response: str,
    trace: Any,
) -> Dict[str, float]:
    tool_calls = _extract_tool_calls(trace)
    system_prompt = (
        "You are evaluating the Orchestration Agent output for a multi-agent RCA workflow. "
        "Score each metric from 0 to 1 where 1 is best. "
        "Use the task, expected outputs, agent response, and tool call trace. "
        "Return JSON only with the schema:\n"
        "{\n"
        '  "intent_resolution_accuracy": number,\n'
        '  "tool_call_accuracy": number,\n'
        '  "collaboration_quality": number,\n'
        '  "correctness": number,\n'
        '  "hallucination": number,\n'
        '  "relevance": number,\n'
        '  "toxicity": number,\n'
        '  "helpfulness": number,\n'
        '  "conciseness": number\n'
        "}\n"
    )
    user_prompt = (
        f"Task:\n{gold.task}\n\n"
        f"Expected root causes:\n{gold.expected_root_causes}\n\n"
        f"Expected hypotheses:\n{gold.gold_hypotheses}\n\n"
        f"Forbidden root causes:\n{gold.forbidden_root_causes}\n\n"
        f"Agent response:\n{response}\n\n"
        f"Tool calls:\n{tool_calls}\n"
    )
    message_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        model_response = app.llm.invoke(message_payload)
        parsed = process_response(model_response.content, llm=app.llm, app_config=app.config)
        if not isinstance(parsed, dict):
            raise ValueError(f"Unexpected evaluation payload: {type(parsed)}")
        return {
            "intent_resolution_accuracy": _coerce_score(parsed.get("intent_resolution_accuracy")),
            "tool_call_accuracy": _coerce_score(parsed.get("tool_call_accuracy")),
            "collaboration_quality": _coerce_score(parsed.get("collaboration_quality")),
            "correctness": _coerce_score(parsed.get("correctness")),
            "hallucination": _coerce_score(parsed.get("hallucination")),
            "relevance": _coerce_score(parsed.get("relevance")),
            "toxicity": _coerce_score(parsed.get("toxicity")),
            "helpfulness": _coerce_score(parsed.get("helpfulness")),
            "conciseness": _coerce_score(parsed.get("conciseness")),
        }
    except Exception as exc:
        logger.warning("Orchestration LLM judge failed; returning zeros: %s", exc)
        return {
            "intent_resolution_accuracy": 0.0,
            "tool_call_accuracy": 0.0,
            "collaboration_quality": 0.0,
            "correctness": 0.0,
            "hallucination": 0.0,
            "relevance": 0.0,
            "toxicity": 0.0,
            "helpfulness": 0.0,
            "conciseness": 0.0,
        }


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
        "intent_resolution_accuracy": (scores.intent_resolution_accuracy, "NUMERIC"),
        "tool_call_accuracy": (scores.tool_call_accuracy, "NUMERIC"),
        "collaboration_quality": (scores.collaboration_quality, "NUMERIC"),
        "correctness": (scores.correctness, "NUMERIC"),
        "hallucination": (scores.hallucination, "NUMERIC"),
        "relevance": (scores.relevance, "NUMERIC"),
        "toxicity": (scores.toxicity, "NUMERIC"),
        "helpfulness": (scores.helpfulness, "NUMERIC"),
        "conciseness": (scores.conciseness, "NUMERIC"),
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
    config = {"configurable": {"user_id": "eval_user", "thread_id": "eval_user", "memory_enabled": True}}
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
        "response": result.get("output", ""),
        "trace": normalized_trace,
    }


def run_rca_without_memory(app: RCAApp, case: GoldRCACase) -> Dict[str, Any]:
    query_id = f"eval_{case.case_id}_without_memory"
    config = {
        "configurable": {"user_id": "eval_user_nomem", "thread_id": "eval_user_nomem", "memory_enabled": False}
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
    normalized_trace = normalize_trace(result.get("trace"))
    return {
        "root_cause": extract_root_cause({"trace": normalized_trace}),
        "hypotheses": extract_hypotheses({"trace": normalized_trace}),
        "validated": extract_validated({"trace": normalized_trace}),
        "response": result.get("output", ""),
        "trace": normalized_trace,
    }


def run_memory_ablation(app: RCAApp, case: GoldRCACase) -> Dict[str, EvalScores]:
    out_mem = run_rca_with_memory(app, case)
    out_nomem = run_rca_without_memory(app, case)
    with_memory_scores = evaluate_single_case(app, case, out_mem)
    without_memory_scores = evaluate_single_case(app, case, out_nomem)
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
    correctness_scores = []
    for c in cases:
        query_id = f"eval_{c.case_id}_learning_curve"
        config = {"configurable": {"user_id": "eval_user", "thread_id": "eval_user"}}
        observability_config = build_langfuse_invoke_config(
            app.config,
            user_id="eval_user",
            query_id=query_id,
            tags=build_eval_tags(app.config.langfuse_prompt_label, "learning_curve"),
            metadata=build_eval_metadata(c.case_id, app.config.langfuse_prompt_label, None),
        )
        rca_state = {"task": c.task, "output": "", "trace": []}
        out = app.app.invoke(rca_state, {**config, **observability_config})
        normalized_trace = normalize_trace(out.get("trace"))
        score = evaluate_single_case(
            app,
            c,
            {
                "root_cause": extract_root_cause({"trace": normalized_trace}),
                "hypotheses": extract_hypotheses({"trace": normalized_trace}),
                "validated": extract_validated({"trace": normalized_trace}),
                "response": out.get("output", ""),
                "trace": normalized_trace,
            },
        )
        correctness_scores.append(score.correctness)
        log_eval_scores(app, score, c.case_id, "learning_curve", query_id, None)
    return correctness_scores

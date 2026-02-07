from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import statistics
import time
from typing import Any, Dict, List, Optional

from .app import RCAApp
from .observability import (
    build_langfuse_client,
    build_langfuse_invoke_config,
    supports_langfuse_trace_context,
)
from .utils import extract_json_from_response, process_response

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

EVAL_RUBRIC_BLOCK = """
Evaluation rubric (score each 0.0-1.0):
- intent_resolution_accuracy: Did the response address the user's RCA task and answer the why/how? 1.0 = fully resolves, 0.5 = partially, 0.0 = misses.
- tool_call_accuracy: Were tools used appropriately and correctly interpreted? 1.0 = correct tool usage and interpretation, 0.5 = minor mistakes, 0.0 = incorrect or missing.
- collaboration_quality: Did the response reflect coordinated multi-agent reasoning? 1.0 = integrates agent findings, 0.5 = partial/fragmented, 0.0 = no collaboration evidence.
- correctness: Are the stated root causes and reasoning aligned with provided data/evidence? 1.0 = correct, 0.5 = mixed, 0.0 = incorrect.
- hallucination: Penalize unsupported claims. 1.0 = no hallucinations, 0.5 = minor, 0.0 = significant.
- relevance: Is the response focused on the task without irrelevant content? 1.0 = fully relevant, 0.5 = some drift, 0.0 = off-topic.
- toxicity: Harmful or unsafe content. 1.0 = none, 0.5 = minor, 0.0 = severe.
- helpfulness: Actionable, clear, and useful to the user. 1.0 = very helpful, 0.5 = somewhat, 0.0 = not helpful.
- conciseness: Efficient without omitting key info. 1.0 = concise, 0.5 = verbose/repetitive, 0.0 = overly long or too sparse.
""".strip()

EVAL_OUTPUT_SCHEMA = """
Return JSON only with the schema:
{
  "intent_resolution_accuracy": number,
  "tool_call_accuracy": number,
  "collaboration_quality": number,
  "correctness": number,
  "hallucination": number,
  "relevance": number,
  "toxicity": number,
  "helpfulness": number,
  "conciseness": number
}
""".strip()


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
    GoldRCACase(
        case_id="CHANNEL_SHIFT_03",
        task="Why did in-store sales decline while online sales surged in the same week?",
        expected_root_causes=["Channel shift due to promo", "Store traffic decline"],
        gold_hypotheses=[
            "Online-only promotion drove channel shift",
            "In-store footfall drop due to weather",
            "Assortment gap in stores",
            "Pricing mismatch between channels",
        ],
        must_use_agents=[
            "HypothesisAgent",
            "SalesAnalysisAgent",
            "InventoryAnalysisAgent",
            "HypothesisValidationAgent",
        ],
        forbidden_root_causes=["System outage", "Supplier delay"],
    ),
    GoldRCACase(
        case_id="NEGATIVE_SALES_04",
        task="Why did Product P120 show negative net sales for two consecutive days?",
        expected_root_causes=["Returns spike", "Posting timing mismatch"],
        gold_hypotheses=[
            "Bulk returns processed",
            "Revenue recognition delay",
            "Inventory stockout",
            "Pricing error",
        ],
        must_use_agents=[
            "HypothesisAgent",
            "SalesAnalysisAgent",
            "InventoryAnalysisAgent",
            "HypothesisValidationAgent",
        ],
        forbidden_root_causes=["Demand spike", "Promotion uplift"],
    ),
    GoldRCACase(
        case_id="STOCKOUT_NO_SALES_05",
        task="Why were stockouts reported for SKU K77 but sales did not increase?",
        expected_root_causes=["Phantom inventory", "Mis-scanned shrink"],
        gold_hypotheses=[
            "Inventory data accuracy issue",
            "Theft/shrink event",
            "Supplier delay",
            "Unexpected demand spike",
        ],
        must_use_agents=[
            "HypothesisAgent",
            "InventoryAnalysisAgent",
            "HypothesisValidationAgent",
        ],
        forbidden_root_causes=["Demand spike", "Promotion uplift"],
    ),
    GoldRCACase(
        case_id="PRICE_CUT_NO_LIFT_06",
        task="Why did a price cut on Item I455 fail to lift sales in Week 32?",
        expected_root_causes=["Assortment mismatch", "Competitive undercut"],
        gold_hypotheses=[
            "Competitive price lower",
            "Out-of-stock on top variants",
            "Low awareness of price cut",
            "Assortment mismatch",
        ],
        must_use_agents=["HypothesisAgent", "SalesAnalysisAgent", "HypothesisValidationAgent"],
        forbidden_root_causes=["Warehouse delay", "System outage"],
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
    required_agents_coverage: float
    expected_root_cause_recall: float
    forbidden_root_cause_compliance: float
    latency_ms: float
    tool_call_count: int


@dataclass
class ConsistencyScores:
    correctness_variance: float
    root_cause_agreement: float


def evaluate_single_case(app: RCAApp, gold: GoldRCACase, rca_output: Dict[str, Any]) -> EvalScores:
    response = rca_output.get("response", "")
    if not response:
        response = rca_output.get("output", "")

    judge_scores = evaluate_orchestration_llm_judge(app, gold, response, rca_output.get("trace"))
    required_agents_coverage = evaluate_required_agents(gold, rca_output.get("trace"))
    root_cause_recall = evaluate_expected_root_cause_recall(gold, rca_output)
    forbidden_root_cause_compliance = evaluate_forbidden_root_causes(gold, rca_output)

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
        required_agents_coverage=required_agents_coverage,
        expected_root_cause_recall=root_cause_recall,
        forbidden_root_cause_compliance=forbidden_root_cause_compliance,
        latency_ms=float(rca_output.get("latency_ms", 0.0) or 0.0),
        tool_call_count=int(rca_output.get("tool_call_count", 0) or 0),
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


def _normalize_strings(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return [v for v in values if isinstance(v, str)]
    if isinstance(values, dict):
        items = []
        for value in values.values():
            items.extend(_normalize_strings(value))
        return items
    return []


def _normalize_root_causes(root_cause: Dict[str, Any]) -> List[str]:
    if not isinstance(root_cause, dict):
        return []
    candidate_keys = ["primary_root_causes", "secondary_root_causes", "contributing_factors"]
    values: List[str] = []
    for key in candidate_keys:
        if key in root_cause:
            values.extend(_normalize_strings(root_cause.get(key)))
    if not values:
        values = _normalize_strings(root_cause)
    return [v.strip().lower() for v in values if v.strip()]


def _normalize_expected_root_causes(gold: GoldRCACase) -> List[str]:
    return [value.strip().lower() for value in gold.expected_root_causes if value.strip()]


def _normalize_forbidden_root_causes(gold: GoldRCACase) -> List[str]:
    return [value.strip().lower() for value in gold.forbidden_root_causes if value.strip()]


def evaluate_required_agents(gold: GoldRCACase, trace: Any) -> float:
    required = [agent for agent in gold.must_use_agents if agent]
    if not required:
        return 1.0
    used_agents = {entry.get("agent") for entry in normalize_trace(trace) if isinstance(entry, dict)}
    matched = sum(1 for agent in required if agent in used_agents)
    return matched / len(required)


def evaluate_expected_root_cause_recall(gold: GoldRCACase, rca_output: Dict[str, Any]) -> float:
    expected = _normalize_expected_root_causes(gold)
    if not expected:
        return 1.0
    predicted = _normalize_root_causes(rca_output.get("root_cause", {}))
    if not predicted:
        return 0.0
    matched = sum(1 for item in expected if any(item in predicted_item for predicted_item in predicted))
    return matched / len(expected)


def evaluate_forbidden_root_causes(gold: GoldRCACase, rca_output: Dict[str, Any]) -> float:
    forbidden = _normalize_forbidden_root_causes(gold)
    if not forbidden:
        return 1.0
    predicted = _normalize_root_causes(rca_output.get("root_cause", {}))
    if not predicted:
        return 1.0
    violations = sum(1 for item in forbidden if any(item in predicted_item for predicted_item in predicted))
    return 0.0 if violations > 0 else 1.0


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


def count_tool_calls(trace: Any) -> int:
    count = 0
    for entry in normalize_trace(trace):
        tool_calls = entry.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if isinstance(call, dict) and call.get("name"):
                count += 1
    return count


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
        "Use the task, expected outputs, agent response, and tool call trace.\n\n"
        f"{EVAL_RUBRIC_BLOCK}\n\n"
        f"{EVAL_OUTPUT_SCHEMA}\n"
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
    trace_id: str | None = None,
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
        "required_agents_coverage": (scores.required_agents_coverage, "NUMERIC"),
        "expected_root_cause_recall": (scores.expected_root_cause_recall, "NUMERIC"),
        "forbidden_root_cause_compliance": (scores.forbidden_root_cause_compliance, "NUMERIC"),
        "latency_ms": (scores.latency_ms, "NUMERIC"),
        "tool_call_count": (float(scores.tool_call_count), "NUMERIC"),
    }
    for metric, (value, data_type) in score_map.items():
        client.create_score(
            name=f"eval_{run_label}_{metric}",
            value=value,
            data_type=data_type,
            session_id=session_id,
            trace_id=trace_id,
            metadata=metadata,
        )
    client.flush()


def log_consistency_scores(
    app: RCAApp,
    scores: ConsistencyScores,
    case_id: str,
    run_label: str,
    session_id: str,
    memory_enabled: bool | None,
    trace_id: str | None = None,
) -> None:
    client = build_langfuse_client(app.config)
    if not client:
        return

    metadata = build_eval_metadata(case_id, app.config.langfuse_prompt_label, memory_enabled)
    score_map = {
        "correctness_variance": (scores.correctness_variance, "NUMERIC"),
        "root_cause_agreement": (scores.root_cause_agreement, "NUMERIC"),
    }
    for metric, (value, data_type) in score_map.items():
        client.create_score(
            name=f"eval_{run_label}_{metric}",
            value=value,
            data_type=data_type,
            session_id=session_id,
            trace_id=trace_id,
            metadata=metadata,
        )
    client.flush()


def run_rca_eval_case(
    app: RCAApp,
    case: GoldRCACase,
    run_label: str,
    run_index: int | None = None,
    memory_enabled: bool = True,
) -> Dict[str, Any]:
    run_suffix = f"_{run_index}" if run_index is not None else ""
    query_id = f"eval_{case.case_id}_{run_label}{run_suffix}"
    trace_id = None
    trace_context = None
    client = build_langfuse_client(app.config)
    if client and supports_langfuse_trace_context(app.config):
        try:
            trace_id = client.create_trace_id()
            trace_context = {"trace_id": trace_id}
        except Exception as exc:
            logger.warning("Failed to create Langfuse trace id for eval run: %s", exc)
    thread_id = f"eval_user_{case.case_id}_{run_label}{run_suffix}"
    config = {"configurable": {"user_id": "eval_user", "thread_id": thread_id, "memory_enabled": memory_enabled}}
    rca_state = {"task": case.task, "output": "", "trace": []}
    observability_config = build_langfuse_invoke_config(
        app.config,
        user_id="eval_user",
        query_id=query_id,
        tags=build_eval_tags(app.config.langfuse_prompt_label, run_label),
        metadata=build_eval_metadata(case.case_id, app.config.langfuse_prompt_label, memory_enabled),
        trace_context=trace_context,
    )
    logger.info("Running RCA evaluation label=%s memory_enabled=%s", run_label, memory_enabled)
    start = time.perf_counter()
    result = app.app.invoke(rca_state, {**config, **observability_config})
    latency_ms = (time.perf_counter() - start) * 1000
    normalized_trace = normalize_trace(result.get("trace"))
    tool_call_count = count_tool_calls(normalized_trace)
    return {
        "root_cause": extract_root_cause({"trace": normalized_trace}),
        "hypotheses": extract_hypotheses({"trace": normalized_trace}),
        "validated": extract_validated({"trace": normalized_trace}),
        "response": result.get("output", ""),
        "trace": normalized_trace,
        "trace_id": trace_id,
        "session_id": query_id,
        "latency_ms": latency_ms,
        "tool_call_count": tool_call_count,
    }


def run_rca_with_memory(app: RCAApp, case: GoldRCACase) -> Dict[str, Any]:
    return run_rca_eval_case(app, case, "with_memory", memory_enabled=True)


def _root_cause_set(result: Dict[str, Any]) -> set[str]:
    return set(_normalize_root_causes(result.get("root_cause", {})))


def evaluate_consistency(
    app: RCAApp,
    gold: GoldRCACase,
    runs: int,
    memory_enabled: bool = False,
) -> ConsistencyScores:
    if runs <= 1:
        return ConsistencyScores(correctness_variance=0.0, root_cause_agreement=1.0)

    correctness_scores: List[float] = []
    root_cause_sets: List[set[str]] = []
    for idx in range(runs):
        rca_output = run_rca_eval_case(
            app,
            gold,
            run_label="consistency",
            run_index=idx,
            memory_enabled=memory_enabled,
        )
        score = evaluate_single_case(app, gold, rca_output)
        correctness_scores.append(score.correctness)
        root_cause_sets.append(_root_cause_set(rca_output))

    if len(correctness_scores) < 2:
        correctness_variance = 0.0
    else:
        correctness_variance = statistics.pvariance(correctness_scores)

    pairwise_scores: List[float] = []
    for i in range(len(root_cause_sets)):
        for j in range(i + 1, len(root_cause_sets)):
            union = root_cause_sets[i] | root_cause_sets[j]
            intersection = root_cause_sets[i] & root_cause_sets[j]
            score = len(intersection) / len(union) if union else 1.0
            pairwise_scores.append(score)

    root_cause_agreement = statistics.mean(pairwise_scores) if pairwise_scores else 1.0
    return ConsistencyScores(
        correctness_variance=correctness_variance,
        root_cause_agreement=root_cause_agreement,
    )


def learning_curve(app: RCAApp, cases: List[GoldRCACase]) -> List[float]:
    correctness_scores = []
    client = build_langfuse_client(app.config)
    trace_context_supported = supports_langfuse_trace_context(app.config)
    for c in cases:
        query_id = f"eval_{c.case_id}_learning_curve"
        trace_id = None
        trace_context = None
        if client and trace_context_supported:
            try:
                trace_id = client.create_trace_id()
                trace_context = {"trace_id": trace_id}
            except Exception as exc:
                logger.warning("Failed to create Langfuse trace id for eval run: %s", exc)
        config = {"configurable": {"user_id": "eval_user", "thread_id": "eval_user"}}
        observability_config = build_langfuse_invoke_config(
            app.config,
            user_id="eval_user",
            query_id=query_id,
            tags=build_eval_tags(app.config.langfuse_prompt_label, "learning_curve"),
            metadata=build_eval_metadata(c.case_id, app.config.langfuse_prompt_label, None),
            trace_context=trace_context,
        )
        rca_state = {"task": c.task, "output": "", "trace": []}
        start = time.perf_counter()
        out = app.app.invoke(rca_state, {**config, **observability_config})
        latency_ms = (time.perf_counter() - start) * 1000
        normalized_trace = normalize_trace(out.get("trace"))
        tool_call_count = count_tool_calls(normalized_trace)
        score = evaluate_single_case(
            app,
            c,
            {
                "root_cause": extract_root_cause({"trace": normalized_trace}),
                "hypotheses": extract_hypotheses({"trace": normalized_trace}),
                "validated": extract_validated({"trace": normalized_trace}),
                "response": out.get("output", ""),
                "trace": normalized_trace,
                "latency_ms": latency_ms,
                "tool_call_count": tool_call_count,
            },
        )
        correctness_scores.append(score.correctness)
        log_eval_scores(app, score, c.case_id, "learning_curve", query_id, None, trace_id)
    return correctness_scores

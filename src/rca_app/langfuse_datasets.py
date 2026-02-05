from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .langfuse_prompts import get_prompt_template
from .llm import get_llm_model
from .observability import build_langfuse_client
from .utils import process_response

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def build_datasets_from_gold_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    datasets: List[Dict[str, str]] = []
    for case in cases:
        task = str(case.get("task", "")).strip()
        expected_root_causes = case.get("expected_root_causes", [])
        expected = ", ".join(str(item).strip() for item in expected_root_causes if item)
        if not task or not expected:
            logger.info("Skipping incomplete gold case for dataset export.")
            continue
        datasets.append({"question": task, "context": task, "answer": expected})
    return datasets


def create_dataset_items(
    config: AppConfig,
    dataset_name: str,
    datasets: List[Dict[str, str]],
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    client = build_langfuse_client(config)
    if not client:
        raise RuntimeError("Langfuse client unavailable. Check LANGFUSE_* settings.")

    client.create_dataset(name=dataset_name, description=description, metadata=metadata or {})
    uploaded = 0
    for item in datasets:
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"context": item["context"], "user_query": item["question"]},
            expected_output=item["answer"],
        )
        uploaded += 1
    return uploaded


def run_dataset_experiment(
    config: AppConfig,
    dataset_name: str,
    prompt_name: str,
    prompt_label: Optional[str],
    experiment_name: str,
    experiment_description: Optional[str] = None,
) -> Any:
    client = build_langfuse_client(config)
    if not client:
        raise RuntimeError("Langfuse client unavailable. Check LANGFUSE_* settings.")

    dataset = client.get_dataset(dataset_name)
    prompt_template = get_prompt_template(config, name=prompt_name, label=prompt_label)
    llm = get_llm_model(config)

    def task(*, item, **kwargs):
        item_input = item.input if hasattr(item, "input") else item.get("input", {})
        context = item_input.get("context", "")
        user_query = item_input.get("user_query", "")
        full_prompt = prompt_template.format(context=context, user_query=user_query).strip()
        response = llm.invoke(full_prompt)
        return response.content

    def llm_judge_evaluator(*, input, output, expected_output, **kwargs):
        from langfuse import Evaluation

        if not expected_output:
            return Evaluation(
                name="llm_judge",
                value=0.0,
                comment="Missing expected_output for LLM judge evaluation.",
            )

        system_prompt = (
            "You are an evaluator judging an assistant response. "
            "Score each metric from 0 to 1 where 1 is best. "
            "Return JSON only with the schema:\n"
            "{\n"
            '  "correctness": number,\n'
            '  "hallucination": number,\n'
            '  "relevance": number,\n'
            '  "toxicity": number,\n'
            '  "helpfulness": number,\n'
            '  "conciseness": number\n'
            "}\n"
        )
        user_prompt = (
            f"Context:\n{input.get('context', '')}\n\n"
            f"User query:\n{input.get('user_query', '')}\n\n"
            f"Expected answer:\n{expected_output}\n\n"
            f"Assistant output:\n{output}\n"
        )
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        try:
            parsed = process_response(response.content, llm=llm, app_config=config)
        except Exception as exc:
            logger.warning("LLM judge parse failed: %s", exc)
            return Evaluation(name="llm_judge", value=0.0, comment="LLM judge parse failed.")

        if not isinstance(parsed, dict):
            return Evaluation(name="llm_judge", value=0.0, comment="LLM judge returned non-JSON.")

        evaluations = []
        for metric in ["correctness", "hallucination", "relevance", "toxicity", "helpfulness", "conciseness"]:
            value = parsed.get(metric, 0.0)
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))
            evaluations.append(Evaluation(name=metric, value=score))
        return evaluations

    return dataset.run_experiment(
        name=experiment_name,
        description=experiment_description,
        task=task,
        evaluators=[llm_judge_evaluator],
    )

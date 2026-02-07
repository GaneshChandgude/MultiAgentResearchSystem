from __future__ import annotations

import argparse
import json
from dataclasses import replace
import logging
import os
from pathlib import Path

from .app import build_app
from .config import load_config, resolve_data_dir
from .evaluation import (
    GOLD_RCA_DATASET,
    evaluate_consistency,
    evaluate_single_case,
    learning_curve,
    log_consistency_scores,
    log_eval_scores,
    run_rca_with_memory,
)
from .langfuse_datasets import build_datasets_from_gold_cases, create_dataset_items, run_dataset_experiment
from .langfuse_prompts import sync_prompt_definitions
from .memory import mark_memory_useful, semantic_recall
from .memory_reflection import add_episodic_memory, add_procedural_memory, build_semantic_memory
from .observability import build_langfuse_invoke_config

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

DEFAULT_LOG_FILE = "rca_app.log"
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging() -> Path:
    log_path = os.getenv("RCA_LOG_FILE", "").strip()
    if log_path:
        log_file = Path(log_path).expanduser().resolve()
    else:
        log_file = resolve_data_dir() / DEFAULT_LOG_FILE

    log_level = os.getenv("RCA_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    log_to_console = os.getenv("RCA_LOG_TO_CONSOLE", "true").strip().lower()
    enable_console = log_to_console not in {"0", "false", "no", "off"}
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.FileHandler(log_file)]
    if enable_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    logging.getLogger(__name__).info(
        "Logging initialized at %s (level=%s)", log_file, log_level
    )
    return log_file


def run_chat():
    config = load_config()
    app = build_app(config)
    logger.info("Starting RCA chat session")

    print("\nRCA Chatbot (type 'exit' to quit)\n")
    default_user_id = "2"
    default_query_id = "2"

    last_state = None
    last_config = None

    while True:
        print("\n" + "=" * 70)
        user_id = default_user_id
        query_id = default_query_id

        print("-" * 70)
        user_input = input("You      : ").strip()

        if user_input.lower() in {"exit", "quit"}:
            if last_state and last_config:
                logger.info("Persisting memories for user_id=%s", last_config["configurable"]["user_id"])
                add_episodic_memory(last_state, last_config, app.store, app.llm, app.config)
                build_semantic_memory(
                    user_id=last_config["configurable"]["user_id"],
                    query=user_input,
                    store=app.store,
                    llm=app.llm,
                    app_config=app.config,
                )
                add_procedural_memory(last_state, last_config, app.store, app.llm, app.config)

                used_semantic = semantic_recall(last_state["task"], app.store, last_config)
                mark_memory_useful(used_semantic)

            print("\nExiting RCA chatbot.")
            break

        config_dict = {"configurable": {"user_id": user_id, "thread_id": user_id}}
        observability_config = build_langfuse_invoke_config(
            app.config,
            user_id=user_id,
            query_id=query_id,
            tags=["CLIChat"],
            metadata={"entrypoint": "run_chat", "task_length": len(user_input)},
        )
        rca_state = {"task": user_input, "output": "", "trace": []}

        print("\n" + "-" * 70)
        print(" RCA Bot is thinking...")
        print("-" * 70)

        rca_state = app.app.invoke(rca_state, {**config_dict, **observability_config})
        logger.info("RCA response generated")

        print("\n RCA Bot Answer")
        print("-" * 70)
        print(rca_state.get("output", "No response generated"))
        print(rca_state.get("trace", "No trace generated"))
        print("=" * 70)

        last_state = rca_state
        last_config = config_dict


def inspect_memory():
    config = load_config()
    app = build_app(config)
    user_id = "2"
    logger.info("Inspecting memory for user_id=%s", user_id)

    print("\n--------------------------------------------------------------------------")
    print("memory inspector")
    report = {}

    for layer in ["episodic", "procedural", "semantic"]:
        namespace = (layer, user_id)
        memories = app.store.search(namespace, limit=10)

        report[layer] = [
            {
                "key": m.key,
                "confidence": m.value.get("confidence"),
                "usefulness": m.value.get("usefulness", 0),
                "summary": (
                    m.value.get("conversation_summary")
                    or m.value.get("semantic_fact")
                    or m.value.get("procedure_name")
                ),
            }
            for m in memories
        ]

    print(json.dumps(report, indent=2))
    print("--------------------------------------------------------------------------")


def run_evals(case_id: str | None, learning_curve_only: bool, consistency_runs: int) -> int:
    config = load_config()
    app = build_app(config)
    cases = GOLD_RCA_DATASET
    if case_id:
        cases = [case for case in cases if case.case_id == case_id]
        if not cases:
            print(f"No eval case found for case_id={case_id}.")
            return 1

    if learning_curve_only:
        correctness_scores = learning_curve(app, cases)
        print("Learning curve correctness scores:")
        for case, score in zip(cases, correctness_scores, strict=False):
            print(f"- {case.case_id}: {score:.3f}")
        return 0

    print("Running RCA evals:")
    for case in cases:
        rca_output = run_rca_with_memory(app, case)
        with_mem = evaluate_single_case(app, case, rca_output)
        session_id = rca_output.get("session_id", f"eval_{case.case_id}_with_memory")
        trace_id = rca_output.get("trace_id")
        log_eval_scores(app, with_mem, case.case_id, "with_memory", session_id, True, trace_id)
        print(f"- {case.case_id}:")
        print(
            "  with_memory: "
            f"intent_resolution_accuracy={with_mem.intent_resolution_accuracy:.3f}, "
            f"tool_call_accuracy={with_mem.tool_call_accuracy:.3f}, "
            f"collaboration_quality={with_mem.collaboration_quality:.3f}, "
            f"correctness={with_mem.correctness:.3f}, "
            f"hallucination={with_mem.hallucination:.3f}, "
            f"relevance={with_mem.relevance:.3f}, "
            f"toxicity={with_mem.toxicity:.3f}, "
            f"helpfulness={with_mem.helpfulness:.3f}, "
            f"conciseness={with_mem.conciseness:.3f}, "
            f"latency_ms={with_mem.latency_ms:.1f}, "
            f"tool_call_count={with_mem.tool_call_count}"
        )
        if consistency_runs > 1:
            consistency = evaluate_consistency(app, case, runs=consistency_runs, memory_enabled=False)
            log_consistency_scores(
                app,
                consistency,
                case.case_id,
                "consistency",
                session_id,
                False,
                trace_id,
            )
            print(
                "  consistency: "
                f"correctness_variance={consistency.correctness_variance:.4f}, "
                f"root_cause_agreement={consistency.root_cause_agreement:.3f}"
            )
    return 0


def run_langfuse_dataset_create(
    dataset_name: str,
) -> int:
    config = load_config()
    gold_cases = [case.__dict__ for case in GOLD_RCA_DATASET]
    datasets = build_datasets_from_gold_cases(gold_cases)
    if not datasets:
        print("No dataset items generated.")
        return 1
    uploaded = create_dataset_items(config, dataset_name, datasets)
    print(f"Uploaded {uploaded} dataset items to {dataset_name}.")
    return 0


def run_langfuse_experiment(
    dataset_name: str,
    prompt_name: str,
    prompt_label: str | None,
    experiment_name: str,
    experiment_description: str | None,
) -> int:
    config = load_config()
    run_dataset_experiment(
        config,
        dataset_name=dataset_name,
        prompt_name=prompt_name,
        prompt_label=prompt_label,
        experiment_name=experiment_name,
        experiment_description=experiment_description,
    )
    print(f"Started Langfuse experiment {experiment_name} on dataset {dataset_name}.")
    return 0


def main(argv: list[str] | None = None):
    configure_logging()
    parser = argparse.ArgumentParser(description="RCA project CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat", help="Start interactive RCA chat")
    subparsers.add_parser("inspect-memory", help="Inspect stored memory")
    salesforce_parser = subparsers.add_parser(
        "mcp-salesforce", help="Run Sales MCP SSE server"
    )
    salesforce_parser.add_argument("--host", default="0.0.0.0")
    salesforce_parser.add_argument("--port", type=int, default=8600)
    sap_parser = subparsers.add_parser(
        "mcp-sap", help="Run Inventory MCP SSE server"
    )
    sap_parser.add_argument("--host", default="0.0.0.0")
    sap_parser.add_argument("--port", type=int, default=8700)
    prompt_parser = subparsers.add_parser(
        "sync-langfuse-prompts", help="Create Langfuse prompts if missing"
    )
    prompt_parser.add_argument("--label", default=None)
    eval_parser = subparsers.add_parser("eval", help="Run RCA evals")
    eval_parser.add_argument("--case-id", default=None, help="Run a single eval case ID")
    eval_parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="Run learning curve evaluation only",
    )
    eval_parser.add_argument(
        "--consistency-runs",
        type=int,
        default=1,
        help="Run multi-run consistency evaluation with N runs per case",
    )
    dataset_parser = subparsers.add_parser(
        "langfuse-create-dataset",
        help="Upload Langfuse dataset items from the GOLD_RCA_DATASET cases",
    )
    dataset_parser.add_argument("--dataset-name", required=True)

    experiment_parser = subparsers.add_parser(
        "langfuse-run-experiment",
        help="Run a Langfuse dataset experiment with LLM-judge evaluators",
    )
    experiment_parser.add_argument("--dataset-name", required=True)
    experiment_parser.add_argument("--prompt-name", required=True)
    experiment_parser.add_argument("--prompt-label", default=None)
    experiment_parser.add_argument("--experiment-name", required=True)
    experiment_parser.add_argument("--experiment-description", default=None)

    args = parser.parse_args(argv)

    if args.command == "chat":
        run_chat()
        return 0

    if args.command == "inspect-memory":
        inspect_memory()
        return 0
    if args.command == "mcp-salesforce":
        from .mcp_servers import run_salesforce_mcp

        run_salesforce_mcp(load_config(), host=args.host, port=args.port)
        return 0
    if args.command == "mcp-sap":
        from .mcp_servers import run_sap_business_one_mcp

        run_sap_business_one_mcp(load_config(), host=args.host, port=args.port)
        return 0
    if args.command == "sync-langfuse-prompts":
        config = load_config()
        if not config.langfuse_public_key or not config.langfuse_secret_key:
            print(
                "Langfuse prompt sync requires LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY to be set."
            )
            return 1
        if not config.langfuse_prompt_enabled:
            logger.info(
                "Langfuse prompt management disabled; enabling for prompt sync."
            )
            config = replace(config, langfuse_prompt_enabled=True)
        synced = sync_prompt_definitions(config, label=args.label or config.langfuse_prompt_label)
        print(f"Synced {synced} Langfuse prompts.")
        return 0
    if args.command == "eval":
        return run_evals(args.case_id, args.learning_curve, args.consistency_runs)
    if args.command == "langfuse-create-dataset":
        return run_langfuse_dataset_create(args.dataset_name)
    if args.command == "langfuse-run-experiment":
        return run_langfuse_experiment(
            args.dataset_name,
            args.prompt_name,
            args.prompt_label,
            args.experiment_name,
            args.experiment_description,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

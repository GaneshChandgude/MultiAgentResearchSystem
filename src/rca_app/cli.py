from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .app import build_app
from .config import load_config, resolve_data_dir
from .memory import mark_memory_useful, semantic_recall
from .memory_reflection import add_episodic_memory, add_procedural_memory, build_semantic_memory

logger = logging.getLogger(__name__)

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
                add_episodic_memory(last_state, last_config, app.store, app.llm)
                build_semantic_memory(
                    user_id=last_config["configurable"]["user_id"],
                    query=user_input,
                    store=app.store,
                    llm=app.llm,
                )
                add_procedural_memory(last_state, last_config, app.store, app.llm)

                used_semantic = semantic_recall(last_state["task"], app.store, last_config)
                mark_memory_useful(used_semantic)

            print("\nExiting RCA chatbot.")
            break

        config_dict = {"configurable": {"user_id": user_id, "thread_id": query_id}}
        rca_state = {"task": user_input, "output": "", "trace": []}

        print("\n" + "-" * 70)
        print(" RCA Bot is thinking...")
        print("-" * 70)

        rca_state = app.app.invoke(rca_state, config_dict)
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json

from .app import build_app
from .config import load_config
from .memory import mark_memory_useful, semantic_recall
from .memory_reflection import add_episodic_memory, add_procedural_memory, build_semantic_memory


def run_chat():
    config = load_config()
    app = build_app(config)

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
        rca_state = {"task": user_input, "output": "", "trace": [], "history": []}

        print("\n" + "-" * 70)
        print(" RCA Bot is thinking...")
        print("-" * 70)

        rca_state = app.app.invoke(rca_state, config_dict)

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
    parser = argparse.ArgumentParser(description="RCA project CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat", help="Start interactive RCA chat")
    subparsers.add_parser("inspect-memory", help="Inspect stored memory")

    args = parser.parse_args(argv)

    if args.command == "chat":
        run_chat()
        return 0

    if args.command == "inspect-memory":
        inspect_memory()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

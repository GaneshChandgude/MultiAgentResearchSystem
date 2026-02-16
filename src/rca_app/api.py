from __future__ import annotations

import ast
import json
import logging
import threading
from dataclasses import asdict, replace
from typing import Any, Dict, Literal, Optional

import httpx
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .app import build_app, run_rca
from .config import AppConfig, load_config, resolve_data_dir
from .guardrails import apply_input_guardrails, apply_output_guardrails, apply_tool_output_guardrails
from .logging_utils import configure_logging
from .memory import mark_memory_useful, semantic_recall
from .memory_reflection import add_episodic_memory, add_procedural_memory, build_semantic_memory
from .mcp_toolset import shutdown_mcp_runtime
from .ui_store import UIStore
from .utils import extract_json_from_response, register_todo_progress_sink

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

app = FastAPI(title="RCA Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _configure_api_logging() -> None:
    configure_logging()
    try:
        _generate_assistant_capabilities_once()
        logger.info("Shared assistant capabilities ready")
    except Exception:
        logger.exception("Failed to precompute shared assistant capabilities at startup")


@app.on_event("shutdown")
async def _shutdown_api_resources() -> None:
    shutdown_mcp_runtime()

_base_config = load_config()
client = httpx.Client(verify=False)
langfuse = Langfuse(
    public_key=_base_config.langfuse_public_key,
    secret_key=_base_config.langfuse_secret_key,
    host="https://cloud.langfuse.com",
    httpx_client=client,
)
langfuse_handler = CallbackHandler()

store = UIStore(resolve_data_dir() / "rca_ui.db")
register_todo_progress_sink(
    lambda job_id, todos, todo_progress: store.update_job_todo_snapshot(
        job_id,
        todos=todos,
        todo_progress=todo_progress,
    )
)
_app_cache: Dict[str, Any] = {}
_app_cache_lock = threading.Lock()
_session_cache: Dict[str, Dict[str, Any]] = {}
_pending_persistence: set[str] = set()
_SHARED_CAPABILITIES_PROFILE_ID = "__shared_capabilities__"


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    user_id: str
    username: str


class LogoutRequest(BaseModel):
    user_id: str


class LLMConfigRequest(BaseModel):
    user_id: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    planning_azure_openai_deployment: str
    specialist_azure_openai_deployment: str
    azure_openai_api_version: str


class EmbedderConfigRequest(BaseModel):
    user_id: str
    embeddings_model: str
    embeddings_endpoint: str
    embeddings_api_key: str
    embeddings_api_version: str


class LangfuseConfigRequest(BaseModel):
    user_id: str
    langfuse_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    langfuse_release: str
    langfuse_debug: bool
    langfuse_prompt_enabled: bool
    langfuse_prompt_label: str
    langfuse_verify_ssl: bool
    langfuse_ca_bundle: str


class GuardrailsConfigRequest(BaseModel):
    class ModelInputGuardrailRule(BaseModel):
        name: str
        trigger_description: str = ""
        trigger_examples: list[str] = Field(default_factory=list)
        block_message: str

    user_id: str
    pii_middleware_enabled: bool = True
    pii_redaction_enabled: bool
    pii_block_input: bool
    nested_agent_pii_profile: Literal["full", "nested", "off"] = "nested"
    orchestrator_agent_pii_profile: Literal["full", "nested", "off"] = "off"
    max_input_length: int
    max_output_length: int
    model_guardrails_enabled: bool
    model_guardrails_moderation_enabled: bool
    model_guardrails_output_language: str
    model_input_guardrail_rules: list[ModelInputGuardrailRule] = Field(default_factory=list)
    use_dynamic_subagent_flow: bool = True




class MCPServerConfigRequest(BaseModel):
    class MCPServerConfig(BaseModel):
        name: str
        base_url: str
        description: str = ""
        enabled: bool = True

    user_id: str
    servers: list[MCPServerConfig] = Field(default_factory=list)


class ChatStartRequest(BaseModel):
    user_id: str
    query: str


class FeedbackRequest(BaseModel):
    user_id: str
    chat_id: str
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None


class ConfigResponse(BaseModel):
    llm: Dict[str, Any]
    embedder: Dict[str, Any]
    langfuse: Dict[str, Any]
    guardrails: Dict[str, Any]
    mcp_servers: Dict[str, Any]


class CapabilitiesResponse(BaseModel):
    user_id: str
    capabilities: str | None = None
    sample_queries: list[str] = Field(default_factory=list)
    generated: bool = False


def _config_key(config: AppConfig) -> str:
    payload = asdict(config)
    payload["data_dir"] = str(payload["data_dir"])
    return json.dumps(payload, sort_keys=True)


def _build_user_config(user_id: str) -> AppConfig:
    base_config = load_config()
    overrides = store.get_config(user_id)
    updates: Dict[str, Any] = {}

    llm = overrides.get("llm", {})
    embedder = overrides.get("embedder", {})
    langfuse_overrides = overrides.get("langfuse", {})
    guardrails = overrides.get("guardrails", {})
    mcp_servers = overrides.get("mcp_servers", {})

    updates.update({k: v for k, v in llm.items() if v})
    updates.update({k: v for k, v in embedder.items() if v})
    updates.update(langfuse_overrides)
    updates.update(guardrails)
    if isinstance(mcp_servers, dict):
        server_entries = mcp_servers.get("servers")
        if isinstance(server_entries, list):
            updates["mcp_servers"] = [entry for entry in server_entries if isinstance(entry, dict)]

    return replace(base_config, **updates)


def _get_rca_app(config: AppConfig):
    key = _config_key(config)
    cached = _app_cache.get(key)
    if cached is not None:
        return cached

    with _app_cache_lock:
        cached = _app_cache.get(key)
        if cached is None:
            logger.info("Building RCA app for config hash=%s", hash(key))
            cached = build_app(config)
            _app_cache[key] = cached
    return cached


def _build_shared_capabilities_config() -> AppConfig:
    """Build a deterministic config for shared capabilities discovery."""
    return _base_config


def _persist_memories(user_id: str) -> None:
    session = _session_cache.get(user_id)
    if not session:
        logger.info("No cached session found for user_id=%s; skipping memory persistence", user_id)
        return

    last_state = session.get("state")
    last_config = session.get("config")
    last_query = session.get("last_query")

    if not last_state or not last_config:
        logger.info("Missing session data for user_id=%s; skipping memory persistence", user_id)
        return

    config = _build_user_config(user_id)
    rca_app = _get_rca_app(config)

    logger.info("Persisting memories for user_id=%s", last_config["configurable"]["user_id"])
    add_episodic_memory(last_state, last_config, rca_app.store, rca_app.llm, rca_app.config)
    build_semantic_memory(
        user_id=last_config["configurable"]["user_id"],
        query=last_query or last_state.get("task", ""),
        store=rca_app.store,
        llm=rca_app.llm,
        app_config=rca_app.config,
    )
    add_procedural_memory(last_state, last_config, rca_app.store, rca_app.llm, rca_app.config)

    used_semantic = semantic_recall(last_state["task"], rca_app.store, last_config)
    mark_memory_useful(used_semantic)
    _session_cache.pop(user_id, None)
    _pending_persistence.discard(user_id)


def _run_job(job_id: str, user_id: str, query: str) -> None:
    try:
        store.update_job(job_id, status="running", progress=15, message="Loading configuration")
        config = _build_user_config(user_id)
        store.update_job(job_id, status="running", progress=35, message="Preparing analysis agents")
        rca_app = _get_rca_app(config)
        store.update_job(job_id, status="running", progress=60, message="Performing analysis")
        result = run_rca(
            rca_app,
            query,
            user_id=user_id,
            query_id=job_id,
        )
        store.update_job(job_id, status="running", progress=85, message="Assembling response")

        _session_cache[user_id] = {
            "state": result,
            "config": {
                "configurable": {
                    "user_id": user_id,
                    "thread_id": user_id,
                    "query_id": job_id,
                }
            },
            "last_query": query,
        }

        if user_id in _pending_persistence:
            logger.info("Processing deferred memory persistence for user_id=%s", user_id)
            _pending_persistence.discard(user_id)
            _persist_memories(user_id)

        response = apply_output_guardrails(result.get("output", ""), config=config)
        trace = apply_tool_output_guardrails(result.get("trace", []), config=config)
        chat_id = store.save_chat(user_id, query, response, trace)

        store.update_job(
            job_id,
            status="completed",
            progress=100,
            message="Complete",
            result={
                "chat_id": chat_id,
                "response": response,
                "trace": trace,
                "todos": result.get("todos", []),
                "todo_progress": result.get("todo_progress", {}),
            },
        )
    except Exception as exc:
        logger.exception("RCA job failed")
        failed_progress = 0
        try:
            failed_progress = int(store.get_job(job_id).get("progress", 0))
        except KeyError:
            failed_progress = 0
        store.update_job(
            job_id,
            status="failed",
            progress=failed_progress,
            message=str(exc),
            result={"error": str(exc)},
        )


def _normalize_sample_queries(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    queries: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized:
            queries.append(normalized)
    return queries[:6]


def _parse_capabilities_payload(output: str) -> tuple[str | None, list[str]]:
    text = output.strip()
    if not text:
        return None, []

    try:
        parsed = json.loads(extract_json_from_response(text))
    except json.JSONDecodeError:
        return text, []

    if not isinstance(parsed, dict):
        return text, []

    capabilities = str(parsed.get("capabilities", "")).strip() or None
    sample_queries = _normalize_sample_queries(parsed.get("sample_queries"))
    if capabilities:
        return capabilities, sample_queries
    return text, sample_queries


def _generate_assistant_capabilities_once() -> tuple[str | None, list[str], bool]:
    existing = store.get_assistant_capabilities(_SHARED_CAPABILITIES_PROFILE_ID)
    existing_queries = store.get_assistant_sample_queries(_SHARED_CAPABILITIES_PROFILE_ID)
    if existing:
        return existing, existing_queries, False

    capability_query = (
        "Generate a short assistant profile for this RCA workspace and include suggested user prompts. "
        "Return JSON only with keys: capabilities (string) and sample_queries (array of 4 concise user questions). "
        "Capabilities should explain how you help users, how you coordinate specialist agents/tools, and what "
        "style of reasoning users should expect."
    )
    config = _build_shared_capabilities_config()
    rca_app = _get_rca_app(config)
    result = run_rca(
        rca_app,
        capability_query,
        user_id=_SHARED_CAPABILITIES_PROFILE_ID,
        query_id="capabilities-startup",
    )
    guarded_output = apply_output_guardrails(result.get("output", ""), config=config).strip()
    capabilities, sample_queries = _parse_capabilities_payload(guarded_output)
    if not capabilities:
        return None, [], False
    store.upsert_assistant_capabilities(
        _SHARED_CAPABILITIES_PROFILE_ID,
        capabilities,
        sample_queries,
    )
    return capabilities, sample_queries, True


def _load_shared_assistant_capabilities() -> tuple[str | None, list[str]]:
    return (
        store.get_assistant_capabilities(_SHARED_CAPABILITIES_PROFILE_ID),
        store.get_assistant_sample_queries(_SHARED_CAPABILITIES_PROFILE_ID),
    )


def _extract_todos_from_tool_content(content: Any) -> list[Dict[str, Any]]:
    if not isinstance(content, str):
        return []
    text = content.strip()
    if not text:
        return []

    candidates = [text]
    marker = "Updated todo list to"
    if marker in text:
        candidates.insert(0, text.split(marker, 1)[1].strip())

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

    return []



def _normalize_todo_steps(todos: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not todos:
        return None

    current_step = None
    steps: list[Dict[str, Any]] = []
    completed = 0
    in_progress = 0

    for idx, todo in enumerate(todos):
        step_status = str(todo.get("status", "pending")).strip().lower()
        if step_status not in {"pending", "in_progress", "completed"}:
            step_status = "pending"
        if step_status == "completed":
            completed += 1
        elif step_status == "in_progress":
            in_progress += 1
        if current_step is None and step_status == "in_progress":
            current_step = f"todo_{idx + 1}"
        steps.append(
            {
                "key": f"todo_{idx + 1}",
                "label": str(todo.get("content", f"Todo {idx + 1}")),
                "status": step_status,
                "detail": (
                    "Completed"
                    if step_status == "completed"
                    else "Working on this task"
                    if step_status == "in_progress"
                    else ""
                ),
                "source": "write_todos",
            }
        )

    total = len(steps)
    return {
        "current_step": current_step,
        "steps": steps,
        "progress": {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": max(total - completed - in_progress, 0),
            "percent": int((completed / total) * 100) if total else 0,
            "source": "write_todos",
        },
    }


def _extract_latest_todos_from_messages(messages: Any) -> list[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    latest_todos: list[Dict[str, Any]] = []
    write_todo_call_ids: set[str] = set()

    for message in messages:
        if isinstance(message, dict):
            message_type = message.get("type")
            if message_type == "AIMessage":
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict) or call.get("name") != "write_todos":
                        continue
                    call_id = call.get("id")
                    if isinstance(call_id, str) and call_id:
                        write_todo_call_ids.add(call_id)
                    todos = call.get("args", {}).get("todos")
                    if isinstance(todos, list):
                        latest_todos = [todo for todo in todos if isinstance(todo, dict)]
            elif message_type == "ToolMessage":
                tool_call_id = message.get("tool_call_id")
                if write_todo_call_ids and tool_call_id not in write_todo_call_ids:
                    continue
                todos = _extract_todos_from_tool_content(message.get("content"))
                if todos:
                    latest_todos = todos
            continue

        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if not isinstance(call, dict) or call.get("name") != "write_todos":
                continue
            call_id = call.get("id")
            if isinstance(call_id, str) and call_id:
                write_todo_call_ids.add(call_id)
            todos = call.get("args", {}).get("todos")
            if isinstance(todos, list):
                latest_todos = [todo for todo in todos if isinstance(todo, dict)]

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id is not None:
            if write_todo_call_ids and tool_call_id not in write_todo_call_ids:
                continue
            todos = _extract_todos_from_tool_content(getattr(message, "content", ""))
            if todos:
                latest_todos = todos

    return latest_todos


def _build_todo_plan_from_checkpoint(job: Dict[str, Any]) -> Dict[str, Any] | None:
    user_id = str(job.get("user_id") or "")
    if not user_id:
        return None

    try:
        config = _build_user_config(user_id)
        rca_app = _get_rca_app(config)
        checkpoint_tuple = rca_app.checkpointer.get_tuple(
            {"configurable": {"thread_id": user_id, "user_id": user_id}}
        )
    except Exception:
        return None

    if not checkpoint_tuple:
        return None

    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})

    # Checkpoints are keyed by user thread_id, so a newly started query can briefly
    # observe todo data from the previous query before the first state write.
    # Guard against leaking stale tasks by ensuring the checkpoint task matches
    # the active job query.
    checkpoint_task = channel_values.get("task")
    job_query = job.get("query")
    if isinstance(checkpoint_task, str) and isinstance(job_query, str):
        if checkpoint_task.strip() and job_query.strip() and checkpoint_task.strip() != job_query.strip():
            return None

    raw_todos = channel_values.get("todos")
    raw_todo_progress = channel_values.get("todo_progress")
    if isinstance(raw_todos, list):
        todo_plan = _normalize_todo_steps([todo for todo in raw_todos if isinstance(todo, dict)])
        if todo_plan:
            if isinstance(raw_todo_progress, dict):
                todo_plan["progress"] = {
                    **todo_plan.get("progress", {}),
                    **raw_todo_progress,
                }
            return todo_plan

    history = channel_values.get("history")
    todos = _extract_latest_todos_from_messages(history)
    return _normalize_todo_steps(todos)


def _build_todo_plan_from_trace(trace: Any) -> Dict[str, Any] | None:
    if not isinstance(trace, list) or not trace:
        return None

    latest_todos: list[Dict[str, Any]] = []
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        messages = entry.get("tool_calls") or entry.get("calls") or []
        todos = _extract_latest_todos_from_messages(messages)
        if todos:
            latest_todos = todos

    return _normalize_todo_steps(latest_todos)


def _build_todo_plan_from_result(job: Dict[str, Any]) -> Dict[str, Any] | None:
    result = job.get("result")
    if not isinstance(result, dict):
        return None

    todos = result.get("todos")
    if not isinstance(todos, list):
        return None

    todo_plan = _normalize_todo_steps([todo for todo in todos if isinstance(todo, dict)])
    if not todo_plan:
        return None

    persisted_progress = result.get("todo_progress")
    if isinstance(persisted_progress, dict):
        todo_plan["progress"] = {
            **todo_plan.get("progress", {}),
            **persisted_progress,
        }

    return todo_plan


def _build_progress_plan(job: Dict[str, Any]) -> Dict[str, Any]:
    steps = [
        {"key": "queued", "label": "Queued and validated", "start": 0, "end": 15},
        {"key": "config", "label": "Loading configuration", "start": 15, "end": 35},
        {"key": "agents", "label": "Preparing analysis agents", "start": 35, "end": 60},
        {"key": "analysis", "label": "Performing analysis", "start": 60, "end": 85},
        {"key": "response", "label": "Assembling final response", "start": 85, "end": 100},
    ]

    progress = int(job.get("progress", 0))
    status = str(job.get("status", "queued"))
    message = str(job.get("message", ""))

    current_index = -1
    for index, step in enumerate(steps):
        if step["start"] <= progress < step["end"]:
            current_index = index
            break

    if progress >= 100 and status == "completed":
        current_index = len(steps) - 1

    failed_index = current_index
    if status == "failed" and failed_index == -1:
        for index, step in enumerate(steps):
            if step["start"] <= progress <= step["end"]:
                failed_index = index
                break
        if failed_index == -1:
            failed_index = len(steps) - 1

    plan = []
    for index, step in enumerate(steps):
        step_status = "pending"
        if status == "failed":
            if index < failed_index:
                step_status = "completed"
            elif index == failed_index:
                step_status = "failed"
        elif status == "completed" or progress >= step["end"]:
            step_status = "completed"
        elif index == current_index:
            step_status = "in_progress"

        detail = ""
        if step_status == "in_progress":
            detail = f"Working on: {message or step['label']}"
        elif step_status == "completed":
            detail = "Completed"
        elif step_status == "failed":
            detail = f"Failed: {message}"

        plan.append(
            {
                "key": step["key"],
                "label": step["label"],
                "status": step_status,
                "detail": detail,
            }
        )

    if status == "failed":
        resolved_current_index = failed_index
    else:
        resolved_current_index = current_index

    return {
        "current_step": (
            plan[resolved_current_index]["key"]
            if resolved_current_index >= 0
            else None
        ),
        "steps": plan,
    }


@app.get("/api/health")
@observe()
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/login", response_model=LoginResponse)
@observe()
async def login(payload: LoginRequest) -> LoginResponse:
    user_id = store.create_user(payload.username)
    return LoginResponse(user_id=user_id, username=payload.username)


@app.post("/api/logout")
@observe()
async def logout(payload: LogoutRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    logger.info("Logout requested for user_id=%s", payload.user_id)
    if payload.user_id in _session_cache:
        background_tasks.add_task(_persist_memories, payload.user_id)
    elif store.has_active_job(payload.user_id):
        logger.info(
            "Logout requested while job is still running for user_id=%s; deferring memory persistence",
            payload.user_id,
        )
        _pending_persistence.add(payload.user_id)
    else:
        logger.info("No active session found for user_id=%s; skipping memory persistence", payload.user_id)
    return {"status": "logged_out"}


@app.get("/api/config/defaults", response_model=ConfigResponse)
@observe()
async def config_defaults() -> ConfigResponse:
    base = load_config()
    return ConfigResponse(
        llm={
            "azure_openai_endpoint": base.azure_openai_endpoint,
            "azure_openai_api_key": "",
            "azure_openai_deployment": base.azure_openai_deployment,
            "planning_azure_openai_deployment": base.planning_azure_openai_deployment,
            "specialist_azure_openai_deployment": base.specialist_azure_openai_deployment,
            "azure_openai_api_version": base.azure_openai_api_version,
        },
        embedder={
            "embeddings_model": base.embeddings_model,
            "embeddings_endpoint": base.embeddings_endpoint,
            "embeddings_api_key": "",
            "embeddings_api_version": base.embeddings_api_version,
        },
        langfuse={
            "langfuse_enabled": base.langfuse_enabled,
            "langfuse_public_key": "",
            "langfuse_secret_key": "",
            "langfuse_host": base.langfuse_host,
            "langfuse_release": base.langfuse_release,
            "langfuse_debug": base.langfuse_debug,
            "langfuse_prompt_enabled": base.langfuse_prompt_enabled,
            "langfuse_prompt_label": base.langfuse_prompt_label,
            "langfuse_verify_ssl": base.langfuse_verify_ssl,
            "langfuse_ca_bundle": base.langfuse_ca_bundle,
        },
        guardrails={
            "pii_middleware_enabled": base.pii_middleware_enabled,
            "pii_redaction_enabled": base.pii_redaction_enabled,
            "pii_block_input": base.pii_block_input,
            "nested_agent_pii_profile": base.nested_agent_pii_profile,
            "orchestrator_agent_pii_profile": base.orchestrator_agent_pii_profile,
            "max_input_length": base.max_input_length,
            "max_output_length": base.max_output_length,
            "model_guardrails_enabled": base.model_guardrails_enabled,
            "model_guardrails_moderation_enabled": base.model_guardrails_moderation_enabled,
            "model_guardrails_output_language": base.model_guardrails_output_language,
            "model_input_guardrail_rules": base.model_input_guardrail_rules,
            "use_dynamic_subagent_flow": base.use_dynamic_subagent_flow,
        },
        mcp_servers={"servers": base.mcp_servers},
    )


@app.get("/api/config/{user_id}", response_model=ConfigResponse)
@observe()
async def get_config(user_id: str) -> ConfigResponse:
    configs = store.get_config(user_id)
    llm = configs.get("llm", {})
    embedder = configs.get("embedder", {})
    langfuse_config = configs.get("langfuse", {})
    guardrails = configs.get("guardrails", {})
    mcp_servers = configs.get("mcp_servers", {})
    return ConfigResponse(
        llm=llm,
        embedder=embedder,
        langfuse=langfuse_config,
        guardrails=guardrails,
        mcp_servers=mcp_servers,
    )


@app.post("/api/config/llm")
@observe()
async def set_llm_config(payload: LLMConfigRequest) -> Dict[str, str]:
    store.upsert_config(payload.user_id, "llm", payload.model_dump(exclude={"user_id"}))
    return {"status": "saved"}


@app.post("/api/config/embedder")
@observe()
async def set_embedder_config(payload: EmbedderConfigRequest) -> Dict[str, str]:
    store.upsert_config(payload.user_id, "embedder", payload.model_dump(exclude={"user_id"}))
    return {"status": "saved"}


@app.post("/api/config/langfuse")
@observe()
async def set_langfuse_config(payload: LangfuseConfigRequest) -> Dict[str, str]:
    store.upsert_config(payload.user_id, "langfuse", payload.model_dump(exclude={"user_id"}))
    return {"status": "saved"}


@app.post("/api/config/guardrails")
@observe()
async def set_guardrails_config(payload: GuardrailsConfigRequest) -> Dict[str, str]:
    store.upsert_config(payload.user_id, "guardrails", payload.model_dump(exclude={"user_id"}))
    return {"status": "saved"}




@app.post("/api/config/mcp_servers")
@observe()
async def set_mcp_server_config(payload: MCPServerConfigRequest) -> Dict[str, str]:
    serialized_servers = [entry.model_dump() for entry in payload.servers]
    store.upsert_config(payload.user_id, "mcp_servers", {"servers": serialized_servers})
    return {"status": "saved"}


@app.post("/api/chat/start")
@observe()
async def start_chat(payload: ChatStartRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    config = _build_user_config(payload.user_id)
    guardrail_result = apply_input_guardrails(payload.query, config=config)
    if not guardrail_result.allowed:
        raise HTTPException(status_code=400, detail=guardrail_result.message)
    job_id = store.create_job(payload.user_id, guardrail_result.sanitized)
    background_tasks.add_task(_run_job, job_id, payload.user_id, guardrail_result.sanitized)
    return {"job_id": job_id}


@app.get("/api/chat/status/{job_id}")
@observe()
async def chat_status(job_id: str) -> Dict[str, Any]:
    try:
        job = store.get_job(job_id)
        job["plan"] = _build_progress_plan(job)
        trace = (job.get("result") or {}).get("trace")
        todo_plan = _build_todo_plan_from_result(job)
        if not todo_plan:
            todo_plan = _build_todo_plan_from_trace(trace)
        if not todo_plan and job.get("status") in {"queued", "running"}:
            todo_plan = _build_todo_plan_from_checkpoint(job)
        if todo_plan:
            job["todo_plan"] = todo_plan
        return job
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/chats/{user_id}")
@observe()
async def list_chats(user_id: str) -> Dict[str, Any]:
    return {"chats": store.list_chats(user_id)}


@app.get("/api/capabilities/{user_id}", response_model=CapabilitiesResponse)
@observe()
async def get_assistant_capabilities(user_id: str, generate: bool = False) -> CapabilitiesResponse:
    capabilities, sample_queries = _load_shared_assistant_capabilities()
    generated = False
    if not capabilities and generate:
        try:
            capabilities, sample_queries, generated = _generate_assistant_capabilities_once()
        except Exception:
            logger.exception("Failed generating assistant capabilities for user_id=%s", user_id)
    return CapabilitiesResponse(
        user_id=user_id,
        capabilities=capabilities,
        sample_queries=sample_queries,
        generated=generated,
    )


@app.post("/api/feedback")
@observe()
async def submit_feedback(payload: FeedbackRequest) -> Dict[str, str]:
    feedback_id = store.save_feedback(payload.chat_id, payload.user_id, payload.rating, payload.comments)
    try:
        chat = store.get_chat(payload.chat_id)
        rating_label = "positive" if payload.rating >= 4 else "negative"
        feedback_text = (payload.comments or "").strip()
        if not feedback_text:
            feedback_text = "No additional comments."
        feedback_message = HumanMessage(
            content=f"USER_FEEDBACK ({rating_label}, rating={payload.rating}): {feedback_text}"
        )
        rca_state = {
            "history": [
                HumanMessage(content=chat.get("query", "")),
                AIMessage(content=chat.get("response", "")),
                feedback_message,
            ]
        }
        config = {
            "configurable": {
                "user_id": payload.user_id,
                "thread_id": payload.user_id,
                "query_id": payload.chat_id,
            }
        }
        app_config = _build_user_config(payload.user_id)
        rca_app = _get_rca_app(app_config)
        add_episodic_memory(rca_state, config, rca_app.store, rca_app.llm, rca_app.config)
    except KeyError:
        logger.warning("Feedback submitted for unknown chat_id=%s", payload.chat_id)
    except Exception:
        logger.exception("Failed to store feedback memory for chat_id=%s", payload.chat_id)
    return {"status": "received", "feedback_id": feedback_id}

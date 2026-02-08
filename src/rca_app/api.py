from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from typing import Any, Dict, Optional

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
from .memory import mark_memory_useful, semantic_recall
from .memory_reflection import add_episodic_memory, add_procedural_memory, build_semantic_memory
from .ui_store import UIStore

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

app = FastAPI(title="RCA Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
_app_cache: Dict[str, Any] = {}
_session_cache: Dict[str, Dict[str, Any]] = {}
_pending_persistence: set[str] = set()


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
    user_id: str
    pii_middleware_enabled: bool = True
    pii_redaction_enabled: bool
    pii_block_input: bool
    max_input_length: int
    max_output_length: int


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
    langfuse = overrides.get("langfuse", {})
    guardrails = overrides.get("guardrails", {})

    updates.update({k: v for k, v in llm.items() if v})
    updates.update({k: v for k, v in embedder.items() if v})
    updates.update(langfuse)
    updates.update(guardrails)

    return replace(base_config, **updates)


def _get_rca_app(config: AppConfig):
    key = _config_key(config)
    if key not in _app_cache:
        logger.info("Building RCA app for config hash=%s", hash(key))
        _app_cache[key] = build_app(config)
    return _app_cache[key]


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
        store.update_job(job_id, status="running", progress=35, message="Initializing RCA agents")
        rca_app = _get_rca_app(config)
        store.update_job(job_id, status="running", progress=60, message="Running root cause analysis")
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
            result={"chat_id": chat_id, "response": response, "trace": trace},
        )
    except Exception as exc:
        logger.exception("RCA job failed")
        store.update_job(
            job_id,
            status="failed",
            progress=100,
            message=str(exc),
            result={"error": str(exc)},
        )


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
            "max_input_length": base.max_input_length,
            "max_output_length": base.max_output_length,
        },
    )


@app.get("/api/config/{user_id}", response_model=ConfigResponse)
@observe()
async def get_config(user_id: str) -> ConfigResponse:
    configs = store.get_config(user_id)
    llm = configs.get("llm", {})
    embedder = configs.get("embedder", {})
    langfuse = configs.get("langfuse", {})
    guardrails = configs.get("guardrails", {})
    return ConfigResponse(llm=llm, embedder=embedder, langfuse=langfuse, guardrails=guardrails)


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
        return store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/chats/{user_id}")
@observe()
async def list_chats(user_id: str) -> Dict[str, Any]:
    return {"chats": store.list_chats(user_id)}


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

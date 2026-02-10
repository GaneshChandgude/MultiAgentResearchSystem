from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool

from .config import AppConfig
from .guardrails import build_pii_middleware
from .langfuse_prompts import PROMPT_DEFINITIONS, render_prompt
from .llm import get_llm_model
from .memory import build_memory_augmented_prompt, append_rca_history
from .observability import build_langfuse_invoke_config
from .toolset_registry import ToolsetRegistry
from .toolsets import build_salesforce_toolset, build_sap_business_one_toolset
from .types import RCAState
from .utils import (
    filter_tool_messages,
    handle_tool_errors,
    make_tool_output_guardrails,
    process_response,
    serialize_messages,
    sync_todo_progress,
)

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

def build_agent_middleware(
    config: AppConfig,
    *,
    include_todo: bool = False,
    include_pii: bool = True,
    pii_profile: Literal["full", "nested"] = "full",
):
    tool_output_guardrails = make_tool_output_guardrails(config)
    middleware = [handle_tool_errors]
    if include_pii:
        middleware.extend(build_pii_middleware(config, profile=pii_profile))
    middleware.append(tool_output_guardrails)
    if include_todo:
        middleware.append(sync_todo_progress)
        middleware.append(TodoListMiddleware())
    return middleware


def build_hypothesis_tool(config: AppConfig, store, checkpointer, llm):
    hypothesis_react_agent = create_agent(
        model=llm,
        tools=[
            create_manage_memory_tool(namespace=("hypothesis", "{user_id}")),
            create_search_memory_tool(namespace=("hypothesis", "{user_id}")),
        ],
        middleware=build_agent_middleware(
            config,
            include_pii=False,
        ),
        store=store,
        checkpointer=checkpointer,
    )

    @tool
    def hypothesis_agent_tool(task: str, user_id: str, query_id: str, memory_context: str) -> Dict[str, Any]:
        """
        Purpose:
            Generate multiple plausible root-cause hypotheses for a given RCA query.

        When to use:
            Use this tool when an RCA investigation requires enumerating
            possible causes of an observed problem. This is typically
            the first analytical step after query routing.

        Inputs:
            - task (str): The resolved and disambiguated user query.
            - user_id (str): Identifier of the user or session.
            - query_id (str): Unique identifier of the current query/thread.
            - memory_context (str): episodic + conversation memory

        Output:
            - dict: Contains updated fields:
                - "hypotheses" (List[str]): Newly generated root-cause hypotheses.
                - "trace" (List[Dict]): Trace entry recording the tool call.

        Notes:
            - Hypotheses are returned as plain strings with no categorization.
            - This tool does not validate hypotheses.
            - It may read from long-term memory but only updates the provided data.
            - Subsequent tools or agents are expected to validate or eliminate hypotheses.
        """
        logger.debug(
            "Hypothesis tool invoked user_id=%s query_id=%s task_length=%s",
            user_id,
            query_id,
            len(task),
        )
        system_prompt = render_prompt(
            config,
            name="rca.hypothesis.system",
            fallback=PROMPT_DEFINITIONS["rca.hypothesis.system"],
            variables={"memory_context": memory_context},
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        observability_config = build_langfuse_invoke_config(
            config,
            user_id=user_id,
            query_id=query_id,
            tags=["HypothesisAgent"],
            metadata={"agent": "HypothesisAgent", "task_length": len(task)},
        )
        tool_config = {
            "configurable": {"user_id": user_id, "thread_id": user_id},
            **observability_config,
        }

        result = hypothesis_react_agent.invoke({"messages": messages}, tool_config)
        final_msg = result["messages"][-1].content
        output = process_response(final_msg, llm=llm, app_config=config)

        hypotheses: List[str] = output.get("hypotheses", [])
        logger.debug("Hypothesis tool produced %s hypotheses", len(hypotheses))

        internal_msgs = result["messages"][2:-1]
        tool_call_msgs = filter_tool_messages(internal_msgs)

        trace_entry = {
            "agent": "HypothesisAgent",
            "step": "Generated hypotheses",
            "calls": serialize_messages(tool_call_msgs),
            "hypotheses": hypotheses,
        }

        return {"hypotheses": hypotheses, "trace": [trace_entry]}

    return hypothesis_agent_tool


def build_sales_analysis_tool(config: AppConfig, store, checkpointer, llm, sales_tools):
    sales_tools = list(sales_tools)
    sales_tools += [
        create_manage_memory_tool(namespace=("sales", "{user_id}")),
        create_search_memory_tool(namespace=("sales", "{user_id}")),
    ]

    sales_react_agent = create_agent(
        model=llm,
        tools=sales_tools,
        middleware=build_agent_middleware(
            config,
            include_pii=False,
        ),
        store=store,
        checkpointer=checkpointer,
    )

    @tool
    def sales_analysis_agent_tool(
        task: str,
        hypotheses: List[str],
        user_id: str,
        query_id: str,
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Purpose:
            Analyze sales and promotion data to evaluate hypotheses that may
            explain observed issues in an RCA investigation.

        When to use:
            Use this tool after hypotheses have been generated and when
            sales, demand, forecasting, or promotion-related factors may
            contribute to the problem.

        Inputs:
            - task (str):
                The resolved RCA task or problem statement to analyze.
            - hypotheses (List[str]):
                A list of candidate root-cause hypotheses to be validated
                from a sales perspective.
            - user_id (str):
                Identifier for the user or session, used for scoped memory access.
            - query_id (str):
                Unique identifier for the current RCA query or thread.
            - memory_context (str): episodic + conversation memory

        Output:
            - dict:
                Contains the following fields:
                - "sales_insights":
                    Structured findings derived from sales and promotion data
                    that support or refute the provided hypotheses.
                - "trace":
                    A list of trace entries capturing tool calls and reasoning
                    steps performed during the analysis.
        Notes:
            - This tool may call sales and promotion data tools as needed.
            - The output is strictly structured and intended for downstream
              RCA agents or summarization steps.
            - The tool does not mutate external state.
        """
        logger.debug(
            "Sales analysis tool invoked user_id=%s query_id=%s hypotheses=%s",
            user_id,
            query_id,
            len(hypotheses),
        )
        sales_related_hypotheses = [
            h
            for h in hypotheses
            if any(
                k in h.lower()
                for k in ["sales", "demand", "promotion", "spike", "forecast", "underestimated"]
            )
        ]
        if not sales_related_hypotheses:
            sales_related_hypotheses = hypotheses

        system_prompt = render_prompt(
            config,
            name="rca.sales.system",
            fallback=PROMPT_DEFINITIONS["rca.sales.system"],
            variables={"memory_context": memory_context},
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Task: {task}
Hypotheses: {sales_related_hypotheses}
""",
            },
        ]

        observability_config = build_langfuse_invoke_config(
            config,
            user_id=user_id,
            query_id=query_id,
            tags=["SalesAnalysisAgent"],
            metadata={
                "agent": "SalesAnalysisAgent",
                "task_length": len(task),
                "hypothesis_count": len(hypotheses),
            },
        )
        tool_config = {
            "configurable": {"user_id": user_id, "thread_id": user_id},
            **observability_config,
        }
        result = sales_react_agent.invoke({"messages": messages}, tool_config)
        final_msg = result["messages"][-1].content
        output = process_response(final_msg, llm=llm, app_config=config)
        sales_insights = output.get("sales_insights")
        logger.debug("Sales analysis produced insights keys=%s", list(sales_insights or {}))

        internal_msgs = result["messages"][2:-1]
        tool_call_msgs = filter_tool_messages(internal_msgs)

        trace_entry = {
            "agent": "SalesAnalysisAgent",
            "step": "Validated sales hypotheses",
            "calls": serialize_messages(tool_call_msgs),
            "sales_insights": sales_insights,
        }

        return {"sales_insights": sales_insights, "trace": [trace_entry]}

    return sales_analysis_agent_tool, sales_tools


def build_inventory_analysis_tool(
    config: AppConfig, store, checkpointer, llm, inventory_tools, promo_tool
):
    inventory_tools = list(inventory_tools)
    if promo_tool is not None:
        inventory_tools = [promo_tool] + inventory_tools
    inventory_tools += [
        create_manage_memory_tool(namespace=("inventory", "{user_id}")),
        create_search_memory_tool(namespace=("inventory", "{user_id}")),
    ]

    inventory_react_agent = create_agent(
        model=llm,
        tools=inventory_tools,
        middleware=build_agent_middleware(
            config,
            include_pii=False,
        ),
        store=store,
        checkpointer=checkpointer,
    )

    @tool
    def inventory_analysis_agent_tool(
        task: str,
        hypotheses: List[str],
        user_id: str,
        query_id: str,
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Purpose:
            Analyze inventory movements, replenishments, transfers, and
            adjustments to validate inventory-related RCA hypotheses.

        When to use:
            Use this tool when stock availability, shrinkage, replenishment
            timing, transfers, or warehouse operations may contribute to
            the observed problem.

        Inputs:
            - task (str): Resolved RCA task or problem statement
            - hypotheses (List[str]): Candidate hypotheses to validate
            - user_id (str): User/session identifier for scoped memory access
            - query_id (str): Query/thread identifier
            - memory_context (str): episodic + conversation memory

        Output:
            - dict:
                - "inventory_insights": Structured inventory analysis
                - "trace": Tool-call trace for observability
        """
        logger.debug(
            "Inventory analysis tool invoked user_id=%s query_id=%s hypotheses=%s",
            user_id,
            query_id,
            len(hypotheses),
        )
        inventory_related_hypotheses = [
            h
            for h in hypotheses
            if any(
                k in h.lower()
                for k in [
                    "inventory",
                    "stock",
                    "supply",
                    "replenish",
                    "transfer",
                    "shrink",
                    "adjust",
                    "warehouse",
                ]
            )
        ]
        if not inventory_related_hypotheses:
            inventory_related_hypotheses = hypotheses

        system_prompt = render_prompt(
            config,
            name="rca.inventory.system",
            fallback=PROMPT_DEFINITIONS["rca.inventory.system"],
            variables={"memory_context": memory_context},
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Task: {task}
Hypotheses to validate: {inventory_related_hypotheses}
""",
            },
        ]

        observability_config = build_langfuse_invoke_config(
            config,
            user_id=user_id,
            query_id=query_id,
            tags=["InventoryAnalysisAgent"],
            metadata={
                "agent": "InventoryAnalysisAgent",
                "task_length": len(task),
                "hypothesis_count": len(hypotheses),
            },
        )
        tool_config = {
            "configurable": {"user_id": user_id, "thread_id": user_id},
            **observability_config,
        }
        result = inventory_react_agent.invoke({"messages": messages}, tool_config)
        final_msg = result["messages"][-1].content
        output = process_response(final_msg, llm=llm, app_config=config)
        inventory_insights = output.get("inventory_insights")
        logger.debug("Inventory analysis produced insights keys=%s", list(inventory_insights or {}))

        internal_msgs = result["messages"][2:-1]
        tool_call_msgs = filter_tool_messages(internal_msgs)

        trace_entry = {
            "agent": "InventoryAnalysisAgent",
            "step": "Validated inventory hypotheses",
            "calls": serialize_messages(tool_call_msgs),
            "inventory_insights": inventory_insights,
        }

        return {"inventory_insights": inventory_insights, "trace": [trace_entry]}

    return inventory_analysis_agent_tool


def build_validation_tool(config: AppConfig, store, checkpointer, llm):
    validation_react_agent = create_agent(
        model=llm,
        tools=[
            create_manage_memory_tool(namespace=("hypothesis_validation", "{user_id}")),
            create_search_memory_tool(namespace=("hypothesis_validation", "{user_id}")),
        ],
        middleware=build_agent_middleware(
            config,
            include_pii=False,
        ),
        store=store,
        checkpointer=checkpointer,
    )

    @tool
    def hypothesis_validation_agent_tool(
        hypotheses: List[str],
        user_id: str,
        query_id: str,
        sales_insights: Optional[Dict[str, Any]] = None,
        inventory_insights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Purpose:
            Validate each hypothesis by cross-referencing sales and inventory
            insights gathered during the RCA investigation.

        When to use:
            Use this tool after domain-specific analysis tools (e.g., Sales,
            Inventory) have produced structured insights.

        Inputs:
            - hypotheses (List[str]):
                Hypotheses to be validated.
            - user_id (str):
                User/session identifier for scoped memory access.
            - query_id (str):
                Query/thread identifier.
            - sales_insights (dict):
                Output from the Sales Analysis tool.
            - inventory_insights (dict):
                Output from the Inventory Analysis tool.

        Output:
            - dict:
                - "validated": Mapping of hypothesis → true / false
                - "reasoning": Mapping of hypothesis → explanation
                - "trace": Tool-call trace for observability
        """
        logger.debug(
            "Validation tool invoked user_id=%s query_id=%s hypotheses=%s",
            user_id,
            query_id,
            len(hypotheses),
        )
        if sales_insights is None:
            logger.warning("Validation tool invoked without sales_insights; defaulting to empty dict.")
            sales_insights = {}
        if inventory_insights is None:
            logger.warning("Validation tool invoked without inventory_insights; defaulting to empty dict.")
            inventory_insights = {}
        system_prompt = render_prompt(
            config,
            name="rca.validation.system",
            fallback=PROMPT_DEFINITIONS["rca.validation.system"],
            variables={},
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Hypotheses:
{hypotheses}

Sales insights:
{sales_insights}

Inventory insights:
{inventory_insights}
""",
            },
        ]

        observability_config = build_langfuse_invoke_config(
            config,
            user_id=user_id,
            query_id=query_id,
            tags=["ValidationAgent"],
            metadata={
                "agent": "ValidationAgent",
                "task_length": len(str(hypotheses)),
                "hypothesis_count": len(hypotheses),
            },
        )
        tool_config = {
            "configurable": {"user_id": user_id, "thread_id": user_id},
            **observability_config,
        }
        result = validation_react_agent.invoke({"messages": messages}, tool_config)
        final_msg = result["messages"][-1].content
        resp = process_response(final_msg, llm=llm, app_config=config)
        logger.debug("Validation tool returned validated keys=%s", list((resp.get("validated") or {}).keys()))

        internal_msgs = result["messages"][2:-1]
        tool_call_msgs = filter_tool_messages(internal_msgs)

        trace_entry = {
            "agent": "HypothesisValidationAgent",
            "step": "Validated hypotheses",
            "calls": serialize_messages(tool_call_msgs),
            "details": resp,
        }

        return {"validated": resp.get("validated"), "reasoning": resp.get("reasoning"), "trace": [trace_entry]}

    return hypothesis_validation_agent_tool


def build_root_cause_tool(config: AppConfig, store, checkpointer, llm):
    root_cause_react_agent = create_agent(
        model=llm,
        tools=[],
        middleware=build_agent_middleware(
            config,
            include_pii=False,
        ),
        store=store,
        checkpointer=checkpointer,
    )

    @tool
    def root_cause_analysis_agent_tool(
        validated_hypotheses: Dict[str, bool],
        sales_insights: Dict[str, Any],
        inventory_insights: Dict[str, Any],
        trace: List[Dict[str, Any]],
        user_id: str,
        query_id: str,
    ) -> Dict[str, Any]:
        """
        Purpose:
            Produce the final Root Cause Analysis by synthesizing validated
            hypotheses, sales insights, inventory insights, and prior analysis
            trace into a structured RCA outcome.

        When to use:
            Use this tool after hypothesis validation has been completed.

        Inputs:
            - validated_hypotheses (dict): Hypothesis → true/false mapping
            - sales_insights (dict): Sales analysis output
            - inventory_insights (dict): Inventory analysis output
            - trace (list): Prior agent trace entries
            - user_id (str): User/session identifier for scoped memory access
            - query_id (str): Query/thread identifier

        Output:
            - dict:
                - "root_cause": Final structured RCA
                - "reasoning": Explanation of RCA decisions
                - "trace": Tool-call trace for observability
        """
        logger.debug(
            "Root cause tool invoked user_id=%s query_id=%s validated_hypotheses=%s",
            user_id,
            query_id,
            len(validated_hypotheses),
        )
        system_prompt = render_prompt(
            config,
            name="rca.root_cause.system",
            fallback=PROMPT_DEFINITIONS["rca.root_cause.system"],
            variables={},
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Validated hypotheses:
{validated_hypotheses}

Sales insights:
{sales_insights}

Inventory insights:
{inventory_insights}

Prior trace:
{trace}
""",
            },
        ]

        observability_config = build_langfuse_invoke_config(
            config,
            user_id=user_id,
            query_id=query_id,
            tags=["RootCauseAgent"],
            metadata={
                "agent": "RootCauseAgent",
                "task_length": len(str(validated_hypotheses)),
                "trace_steps": len(trace),
            },
        )
        tool_config = {
            "configurable": {"user_id": user_id, "thread_id": user_id},
            **observability_config,
        }
        result = root_cause_react_agent.invoke({"messages": messages}, tool_config)
        final_msg = result["messages"][-1].content
        resp = process_response(final_msg, llm=llm, app_config=config)
        root_cause = resp.get("root_cause")
        reasoning = resp.get("reasoning")
        logger.debug("Root cause tool generated root_cause=%s reasoning=%s", bool(root_cause), bool(reasoning))

        internal_msgs = result["messages"][2:-1]
        tool_call_msgs = filter_tool_messages(internal_msgs)

        structured_trace_entry = {
            "agent": "RootCauseAnalysisAgent",
            "step": "Generated structured root cause",
            "calls": serialize_messages(tool_call_msgs),
            "root_cause": root_cause,
        }

        return {"root_cause": root_cause, "reasoning": reasoning, "trace": [structured_trace_entry]}

    return root_cause_analysis_agent_tool


def build_report_tool(config: AppConfig, store, checkpointer, llm):
    rca_report_agent = create_agent(
        model=llm,
        tools=[],
        middleware=build_agent_middleware(
            config,
            include_pii=False,
        ),
        store=store,
        checkpointer=checkpointer,
    )

    @tool
    def rca_report_agent_tool(
        root_cause: str, reasoning: str, user_id: str, query_id: str
    ) -> Dict[str, Any]:
        """
        Purpose:
            Produce the final human-readable report.

        When to use:
            Use this tool as the final step of an RCA workflow to generate a
            human-readable report.

        Inputs:
            - root_cause (str): root cause
            - reasoning (str): reasoning
            - user_id (str): User/session identifier for scoped memory access
            - query_id (str): Query/thread identifier

        Output:
            - dict:
                - "report_text": Human-readable RCA report
                - "trace": Tool-call trace for observability
        """
        logger.debug(
            "Report tool invoked user_id=%s query_id=%s",
            user_id,
            query_id,
        )
        system_prompt = render_prompt(
            config,
            name="rca.report.system",
            fallback=PROMPT_DEFINITIONS["rca.report.system"],
            variables={},
        )
        report_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
Use the following structured RCA output:

{root_cause}
{reasoning}
""",
            },
        ]

        observability_config = build_langfuse_invoke_config(
            config,
            user_id=user_id,
            query_id=query_id,
            tags=["ReportAgent"],
            metadata={"agent": "ReportAgent", "task_length": len(f"{root_cause}{reasoning}")},
        )
        tool_config = {
            "configurable": {"user_id": user_id, "thread_id": user_id},
            **observability_config,
        }
        report_text = rca_report_agent.invoke(report_messages, tool_config).content
        logger.debug("Report tool generated report length=%s", len(report_text))

        report_trace_entry = {
            "agent": "RootCauseAnalysisAgent",
            "step": "Generated RCA report",
            "report_text": report_text,
        }

        return {"report_text": report_text, "trace": [report_trace_entry]}

    return rca_report_agent_tool


def build_router_agent(config: AppConfig, store, checkpointer, llm, tools):
    logger.info("Building router agent with %s tools", len(tools))
    return create_agent(
        model=llm,
        tools=tools,
        middleware=build_agent_middleware(config, include_todo=True),
        store=store,
        checkpointer=checkpointer,
    )


def orchestration_agent(
    rca_state: RCAState,
    config: Dict[str, Any],
    store,
    router_agent,
    app_config: AppConfig,
):
    configurable = config.get("configurable", {})
    query_id = configurable.get("query_id", configurable.get("thread_id"))
    user_id = configurable.get("user_id")
    if not rca_state.get("history"):
        rca_state["history"] = []
        logger.debug("Initialized empty history in RCA state")

    memory_context = build_memory_augmented_prompt(
        query=rca_state["task"],
        state=rca_state,
        config=config,
        store=store,
    )
    logger.debug("Memory context length=%s", len(memory_context))

    system_prompt = render_prompt(
        app_config,
        name="rca.orchestration.system",
        fallback=PROMPT_DEFINITIONS["rca.orchestration.system"],
        variables={
            "task": rca_state["task"],
            "user_id": user_id,
            "query_id": query_id,
            "memory_context": memory_context,
        },
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rca_state["task"]},
    ]

    observability_config = build_langfuse_invoke_config(
        app_config,
        user_id=user_id,
        query_id=query_id,
        tags=["OrchestrationAgent"],
        metadata={
            "agent": "OrchestrationAgent",
            "task_length": len(rca_state["task"]),
            "trace_steps": len(rca_state.get("trace", [])),
        },
    )
    tool_config = {
        **config,
        "configurable": {**config.get("configurable", {}), "rca_state": rca_state},
        **observability_config,
    }

    logger.debug(
        "Invoking router agent user_id=%s query_id=%s",
        user_id,
        query_id,
    )
    result = router_agent.invoke({"messages": messages}, tool_config)
    final_msg = result["messages"][-1].content
    todos = result.get("todos")

    internal_msgs = result["messages"][2:-1]
    tool_call_msgs = filter_tool_messages(internal_msgs)

    trace_entry = {
        "agent": "Orchestration Agent",
        "tool_calls": serialize_messages(tool_call_msgs),
    }

    rca_state["output"] = final_msg
    rca_state["trace"] = trace_entry
    if isinstance(todos, list):
        rca_state["todos"] = [todo for todo in todos if isinstance(todo, dict)]
    logger.info("Orchestration agent completed for user_id=%s", config["configurable"]["user_id"])
    logger.debug("Orchestration agent tool call count=%s", len(trace_entry["tool_calls"]))

    append_rca_history(rca_state)

    return rca_state


def build_agents(config: AppConfig, store, checkpointer):
    logger.info("Initializing RCA agents")
    llm = get_llm_model(config)
    hypothesis_tool = build_hypothesis_tool(config, store, checkpointer, llm)
    salesforce_toolset = build_salesforce_toolset(config)
    sap_toolset = build_sap_business_one_toolset(config)
    tool_registry = ToolsetRegistry([salesforce_toolset, sap_toolset])
    sales_tool, sales_tools = build_sales_analysis_tool(
        config, store, checkpointer, llm, salesforce_toolset.tools
    )
    try:
        promo_tool = tool_registry.find_tool("get_promo_period")
    except KeyError:
        promo_tool = None
    inventory_tool = build_inventory_analysis_tool(
        config, store, checkpointer, llm, sap_toolset.tools, promo_tool
    )
    validation_tool = build_validation_tool(config, store, checkpointer, llm)
    root_cause_tool = build_root_cause_tool(config, store, checkpointer, llm)
    report_tool = build_report_tool(config, store, checkpointer, llm)

    router_tools = [
        create_search_memory_tool(namespace=("orchestration", "{user_id}")),
        create_manage_memory_tool(namespace=("orchestration", "{user_id}")),
        hypothesis_tool,
        sales_tool,
        inventory_tool,
        validation_tool,
        root_cause_tool,
        report_tool,
    ]

    router_agent = build_router_agent(config, store, checkpointer, llm, router_tools)

    return {
        "llm": llm,
        "router_agent": router_agent,
        "tools": {
            "hypothesis": hypothesis_tool,
            "sales": sales_tool,
            "inventory": inventory_tool,
            "validation": validation_tool,
            "root_cause": root_cause_tool,
            "report": report_tool,
        },
    }

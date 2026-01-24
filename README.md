# Autonomous Root Cause Analysis Agent for Retail Operations
AI-powered multi-agent Root Cause Analysis (RCA) system for detecting and explaining stockouts in retail &amp; supply-chain using LangGraph, LangChain, and ReAct Agents.
This repository contains a memory-driven, autonomous multi-agent system for performing Root Cause Analysis (RCA) on retail operational issues such as stockouts during promotions.

Unlike traditional dashboards or static analytics, this system:
  Forms hypotheses
  Decides which data to analyze
  Iteratively validates findings
  Persists investigation memory
  Produces a full reasoning trace

**What This System Does**
Given a business-level prompt like:
  Investigate why stores are experiencing stockouts during an active promotion

The system autonomously:
  1.Generates competing hypotheses
  2.Analyzes sales and inventory data
  3.Iterates based on evidence
  4.Converges on root causes
  5.Produces an explainable RCA report

**Architecture Overview**
The system is built using:
  LangGraph for agent orchestration
  LangChain agents for reasoning
  LangMem for short-term and episodic memory
  Python + Pandas for data analysis

**Core Agents**
  **Hypothesis Agent** – Generates and refines explanations
  **Sales Analysis Agent** – Detects demand-side anomalies
  **Inventory Analysis Agent** – Evaluates execution-side availability
  **Hypothesis Validation Agent** – Cross-validates evidence
  **Router / Orchestration Agent** – Controls investigation flow
All agents operate on a shared state and memory.

**Memory Design**
The system uses LangMem to persist investigation findings:

  Agents store verified insights as memory
  Future steps retrieve and reason over past findings
  Prevents repeated analysis
  Enables cumulative reasoning

Memory tools used:
  create_search_memory_tool
  create_manage_memory_tool
     
**Why This Architecture Works**
  Loose coupling between agents
  Centralized memory, not centralized logic
  Evidence-driven convergence, not rule-based flows
  Explainability by design


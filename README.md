# Autonomous Root Cause Analysis Agent for Retail Operations

AI-powered multi-agent Root Cause Analysis (RCA) system for detecting and explaining stockouts in retail & supply-chain using LangGraph, LangChain, and ReAct Agents.

## What this project does

Given a business-level prompt like:

```
Investigate why stores are experiencing stockouts during an active promotion.
```

The system autonomously:
1. Generates competing hypotheses
2. Analyzes sales and inventory data
3. Iterates based on evidence
4. Converges on root causes
5. Produces an explainable RCA report

## Project layout

```
.
├── data/
│   ├── inventory_transactions.csv
│   └── sales_transactions.csv
├── src/
│   └── rca_app/
│       ├── app.py
│       ├── agents.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── memory.py
│       ├── memory_reflection.py
│       ├── tools_inventory.py
│       ├── tools_sales.py
│       └── utils.py
├── pyproject.toml
└── requirements.txt
```

## Setup

1. **Create a virtual environment** (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2. **Configure Azure OpenAI** using environment variables:

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4.1-mini"

# Optional overrides
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_EMBEDDINGS_MODEL="TxtEmbedAda002"
export AZURE_OPENAI_EMBEDDINGS_ENDPOINT="$AZURE_OPENAI_ENDPOINT"
export AZURE_OPENAI_EMBEDDINGS_API_KEY="$AZURE_OPENAI_API_KEY"
export AZURE_OPENAI_EMBEDDINGS_API_VERSION="2023-05-15"
```

3. **(Optional) Point to a custom data directory** if you want to use your own CSVs:

```bash
export RCA_DATA_DIR="/absolute/path/to/data"
```

4. **(Optional) Configure logging output** (defaults to `data/rca_app.log`):

```bash
export RCA_LOG_FILE="/absolute/path/to/rca_app.log"
export RCA_LOG_LEVEL="INFO"
```

## Usage

### Interactive chat

```bash
rca-app chat
```

You can also run directly with Python:

```bash
python -m rca_app chat
```

### Inspect memory contents

```bash
rca-app inspect-memory
```

## Architecture overview

- **LangGraph** for agent orchestration
- **LangChain** agents for reasoning
- **LangMem** for short-term and episodic memory
- **Python + Pandas** for data analysis

### Core agents

- **Hypothesis Agent** – Generates and refines explanations
- **Sales Analysis Agent** – Detects demand-side anomalies
- **Inventory Analysis Agent** – Evaluates execution-side availability
- **Hypothesis Validation Agent** – Cross-validates evidence
- **Router / Orchestration Agent** – Controls investigation flow

### Memory design

The system uses LangMem to persist investigation findings:
- Agents store verified insights as memory
- Future steps retrieve and reason over past findings
- Prevents repeated analysis
- Enables cumulative reasoning

Memory tools used:
- `create_search_memory_tool`
- `create_manage_memory_tool`

## Why this architecture works

- Loose coupling between agents
- Centralized memory, not centralized logic
- Evidence-driven convergence, not rule-based flows
- Explainability by design

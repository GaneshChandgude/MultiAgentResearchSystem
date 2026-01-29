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
├── RCA.ipynb
├── Root_Cause_Analysis_Report.txt
├── data/
│   ├── inventory_transactions.csv
│   └── sales_transactions.csv
├── inventory_transactions.csv
├── sales_transactions.csv
├── src/
│   └── rca_app/
│       ├── app.py
│       ├── agents.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── local_toolsets.py
│       ├── memory.py
│       ├── memory_reflection.py
│       ├── mcp_toolset.py
│       ├── toolset_registry.py
│       ├── tools_inventory.py
│       ├── tools_sales.py
│       └── utils.py
├── traces.txt
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
export RCA_LOG_LEVEL="INFO" # Use DEBUG for detailed tracing
```

5. **Configure MCP toolset endpoints** for Salesforce and SAP:

```bash
export RCA_MCP_SALESFORCE_URL="http://localhost:8600"
export RCA_MCP_SAP_URL="http://localhost:8700"
```

## Usage

### Quick start

1. Install dependencies and configure Azure OpenAI as shown above.
2. (Optional) Point `RCA_DATA_DIR` to a folder that contains
   `inventory_transactions.csv` and `sales_transactions.csv`.
3. Run the interactive agent:

```bash
rca-app chat
```

4. Ask a question such as:

```
Investigate why stores are experiencing stockouts during an active promotion.
```

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

### Run MCP toolset servers (SSE)

Start the local MCP servers that expose Salesforce and SAP toolsets over SSE:

```bash
rca-app mcp-salesforce --host 0.0.0.0 --port 8600
rca-app mcp-sap --host 0.0.0.0 --port 8700
```

When running the agent, point `RCA_MCP_SALESFORCE_URL` and `RCA_MCP_SAP_URL` to the
servers above so the agent resolves tools remotely.

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

# Autonomous Root Cause Analysis Agent for Retail Operations

AI-powered multi-agent Root Cause Analysis (RCA) system for detecting and explaining
stockouts in retail & supply-chain scenarios using LangGraph, LangChain, and ReAct
agents.

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

## Key capabilities

- **Multi-agent RCA workflow** orchestrated with LangGraph and LangChain.
- **Data-driven analysis** using Pandas for sales and inventory transactions.
- **Toolset integration** via MCP servers for Salesforce (sales) and SAP Business One (inventory).
- **Memory and observability** with LangMem and optional Langfuse tracing.

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
│       ├── __main__.py
│       ├── agents.py
│       ├── app.py
│       ├── api.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inventory_mcp_server.py
│       ├── langfuse_prompts.py
│       ├── llm.py
│       ├── mcp_servers.py
│       ├── mcp_toolset.py
│       ├── memory.py
│       ├── memory_reflection.py
│       ├── observability.py
│       ├── persistent_store.py
│       ├── sales_mcp_server.py
│       ├── toolset_registry.py
│       ├── toolsets.py
│       ├── types.py
│       ├── ui_store.py
│       └── utils.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── styles.css
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

3. **(Optional) Point to a custom data directory** if you want to use your own CSVs
   (defaults to the repo's `data/` directory):

```bash
export RCA_DATA_DIR="/absolute/path/to/data"
```

4. **(Optional) Configure logging output** (defaults to `data/rca_app.log`):

```bash
export RCA_LOG_FILE="/absolute/path/to/rca_app.log"
export RCA_LOG_LEVEL="INFO" # Use DEBUG for detailed tracing
export RCA_LOG_TO_CONSOLE="true"
```

5. **Configure MCP toolset endpoints** for Salesforce and SAP:

```bash
export RCA_MCP_SALESFORCE_URL="http://localhost:8600"
export RCA_MCP_SAP_URL="http://localhost:8700"
```

6. **(Optional) Configure PII middleware behavior for nested agents**:

```bash
# full   = same PII behavior as router agent
# nested = reduced nested-agent profile (default)
# off    = disable PII middleware inside nested specialist agents
export RCA_NESTED_AGENT_PII_PROFILE="nested"

# full   = same PII behavior as full profile
# nested = reduced orchestrator profile
# off    = disable PII middleware for the orchestrator/router agent
export RCA_ORCHESTRATOR_PII_PROFILE="off"
```

7. **(Optional) Enable Langfuse observability** to capture traces, generations, and tool calls:

```bash
export LANGFUSE_ENABLED="true"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_HOST="https://cloud.langfuse.com" # or self-hosted URL

# Optional metadata
export LANGFUSE_RELEASE="rca-app@1.0.0"
export LANGFUSE_DEBUG="false"
```

8. **(Optional) Enable Langfuse prompt management** to sync prompts from this repo:

```bash
export LANGFUSE_PROMPT_ENABLED="true"
export LANGFUSE_PROMPT_LABEL="production"
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

### API + UI workflow

1. Start the FastAPI service:

```bash
uvicorn rca_app.api:app --host 0.0.0.0 --port 8000
```

2. In a separate terminal, install and run the React UI:

```bash
cd frontend
npm install
npm run dev
```

3. Open `http://localhost:5173` to access the RCA command center.

### Register LLM and backend connections from the UI/API

The configuration wizard in the UI already exposes fields to register:

- LLM connection (`LLM Configuration` step)
- Embeddings backend (`Embedder Setup` step)
- External MCP backends (`MCP Server Registry` step)

Under the hood, each step maps to a backend endpoint:

- `POST /api/config/llm`
- `POST /api/config/embedder`
- `POST /api/config/mcp_servers`

Example requests (same payload shape used by the UI):

```bash
curl -X POST http://localhost:8000/api/config/llm \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "<user-id>",
    "azure_openai_endpoint": "https://your-resource.openai.azure.com",
    "azure_openai_api_key": "<api-key>",
    "azure_openai_deployment": "gpt-4.1-mini",
    "planning_azure_openai_deployment": "gpt-4.1",
    "specialist_azure_openai_deployment": "gpt-4.1-mini",
    "azure_openai_api_version": "2024-12-01-preview"
  }'

curl -X POST http://localhost:8000/api/config/mcp_servers \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "<user-id>",
    "servers": [
      {
        "name": "salesforce",
        "base_url": "http://localhost:8600",
        "description": "Sales transactions toolset",
        "enabled": true
      },
      {
        "name": "sap",
        "base_url": "http://localhost:8700",
        "description": "Inventory transactions toolset",
        "enabled": true
      }
    ]
  }'
```

When a chat starts, the backend merges user-scoped config and builds the app with
those LLM and MCP server settings.

You can also run directly with Python:

```bash
python -m rca_app chat
```

### Inspect memory contents

```bash
rca-app inspect-memory
```

### Run evals (with Langfuse scores)

Run the built-in eval cases and log scores to Langfuse (if enabled):

```bash
rca-app eval
```

Run a single eval case:

```bash
rca-app eval --case-id PROMO_STOCKOUT_01
```

Run only the learning-curve evaluation:

```bash
rca-app eval --learning-curve
```

### Evaluation references

Further reading on product/LLM evaluation practices:

- [Product evals](https://eugeneyan.com/writing/product-evals/)
- [Eval process](https://eugeneyan.com/writing/eval-process/)
- [The LLM evals field guide](https://hamel.dev/blog/posts/field-guide/)
- [Instructions and reference data for LLM evals](https://arxiv.org/pdf/2404.12272)

## Troubleshooting

### `[SSL: CERTIFICATE_VERIFY_FAILED]` when connecting to GitHub

If you see errors such as:

```text
('unhandled errors in a TaskGroup', [ConnectError('[SSL: CERTIFICATE_VERIFY_FAILED] ...')])
```

your Python runtime (or Git) cannot find a trusted CA certificate chain. This is
usually an environment issue rather than an application bug.

1. Verify your local trust store and test GitHub TLS:

```bash
python -c "import ssl; print(ssl.get_default_verify_paths())"
python -c "import requests; print(requests.get('https://api.github.com', timeout=10).status_code)"
git ls-remote https://github.com/github/gitignore.git
```

2. If Python can’t validate certs, point it to a CA bundle (recommended).

**Git Bash / Linux / macOS shell:**

```bash
python -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
```

**PowerShell:**

```powershell
python -m pip install --upgrade certifi
$env:SSL_CERT_FILE = python -c "import certifi; print(certifi.where())"
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
$env:CURL_CA_BUNDLE = $env:SSL_CERT_FILE
```

**cmd.exe:**

```bat
python -m pip install --upgrade certifi
for /f "delims=" %i in ('python -c "import certifi; print(certifi.where())"') do set SSL_CERT_FILE=%i
set REQUESTS_CA_BUNDLE=%SSL_CERT_FILE%
set CURL_CA_BUNDLE=%SSL_CERT_FILE%
```

3. Re-run the test command carefully (ensure the closing quote is present):

```bash
python -c "import requests; print(requests.get('https://api.github.com', timeout=10).status_code)"
```

4. If your company uses a TLS-inspecting proxy, append your corporate root CA to
the cert bundle and set `SSL_CERT_FILE` to that merged bundle.

5. For Git specifically, set the CA file explicitly:

```bash
git config --global http.sslCAInfo "$SSL_CERT_FILE"
```

6. Retry your GitHub operation after restarting the shell.

> Avoid disabling SSL verification (`verify=False`, `GIT_SSL_NO_VERIFY=true`) in
> normal workflows; use trusted CA configuration instead.

### Run MCP toolset servers (SSE)

Start the local MCP servers over SSE:

```bash
rca-app mcp-salesforce --host 0.0.0.0 --port 8600
rca-app mcp-sap --host 0.0.0.0 --port 8700
rca-app mcp-supply-chain --host 0.0.0.0 --port 8800
```

`mcp-supply-chain` exposes specialist RCA agents as MCP tools on a single
`SupplyChainMCPServer` (`hypothesis_tool`, `sales_tool`, `inventory_tool`,
`validation_tool`, `root_cause_tool`, and `report_tool`).

When running the agent, point `RCA_MCP_SALESFORCE_URL` and `RCA_MCP_SAP_URL` to the
servers above so the agent resolves data tools remotely.

You can also register additional user-scoped MCP servers from the UI (`Settings -> MCP Server Registry`) with fields for name, URL, description, enabled flag, and optional HTTP headers (for authenticated MCP gateways, such as GitHub MCP proxies).

### Analyze data in the notebook (optional)

Open the notebook to explore RCA analysis steps and data summaries:

```bash
jupyter notebook RCA.ipynb
```

### Create a Langfuse dataset from built-in gold cases (optional)

Generate dataset items from `GOLD_RCA_DATASET` and upload them to Langfuse:

```bash
rca-app langfuse-create-dataset --dataset-name rca-gold-cases
```

Then run an experiment against that dataset:

```bash
rca-app langfuse-run-experiment \
  --dataset-name rca-gold-cases \
  --prompt-name rca.orchestration.system \
  --prompt-label production \
  --experiment-name rca-baseline
```

### Sync Langfuse prompt definitions (optional)

If Langfuse prompt management is enabled, you can create or update prompt templates
in Langfuse:

```bash
rca-app sync-langfuse-prompts
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

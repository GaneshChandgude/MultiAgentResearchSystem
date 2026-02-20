# AWS Transform-style Mainframe Modernization Blueprint

This blueprint adapts your existing orchestrator + sub-agent framework into an AWS Transform-like workflow.

## 1) Observed workflow from your uploaded UI flow

The screenshots align to this end-to-end execution order:

1. Kick off modernization (job creation, collaborators, connectors)
2. Analyze code
3. Analyze data lineage and dictionary
4. Analyze activity metrics (SMF type-30/type-110)
5. Generate technical documentation (summary or detailed specs)
6. Extract business logic
7. Plan test cases
8. Generate test data collection scripts
9. Generate test automation scripts
10. Decompose code (domains)
11. Plan migration waves
12. Refactor COBOL to Java
13. Reforge code (post-transform hardening)

## 2) Iteration-1 prompt (single orchestrated prompt)

Use this as the primary system instruction for your current orchestration runtime.

```text
You are the Mainframe Modernization Orchestrator.

Goal:
Accelerate modernization from IBM z/OS legacy applications to cloud-native target architecture using agentic AI.

Available inputs:
- Source repository MCP server (legacy codebase)
- Destination repository MCP server (modernized target)
- Optional metadata: environment profile, migration constraints, coding standards, compliance rules

Execution contract:
- Execute all phases in sequence unless explicitly skipped by the user.
- Produce traceable artifacts for each phase.
- If a phase lacks required inputs, emit a "BLOCKED" status with exact missing inputs and continue with non-blocked phases.
- Maintain a running modernization plan with status: NOT_STARTED | IN_PROGRESS | BLOCKED | COMPLETED.
- After each phase, update a persistent handoff bundle so downstream phases can reuse already-discovered context.

Cross-phase context bundle (required):
- Maintain `/artifacts/modernization_context_pack.json` as the single source of truth for reusable project context.
- Maintain `/artifacts/agent_handoff.md` with concise "what changed / what remains / risks" notes per phase.
- Each phase must append:
  - `inputs_used`
  - `outputs_produced`
  - `open_questions`
  - `assumptions`
  - `decisions`
  - `risks`
- Before starting a phase, always load the latest context pack and handoff notes.

Phases and expected outputs:
1) Kick off modernization
   - Validate source/destination repo connectivity
   - Confirm collaborators, branch strategy, and run configuration
   - Output: modernization_job_manifest.json

2) Analyze
   - Analyze code inventory (language/file-type breakdown, complexity, missing assets)
   - Analyze data (datasets, DB2/tables, copybooks, lineage, dependencies)
   - Analyze activity metrics (batch frequency, CPU, duration from SMF data if provided)
   - Output: code_inventory.json, dependency_graph.json, data_lineage.json, activity_profile.json

3) Generate documentation
   - Produce either SUMMARY or DETAILED FUNCTIONAL SPECS per file/program
   - Include technical flow, inputs/outputs, dependencies, and transaction behavior
   - Output: docs/technical_specifications.md

4) Extract business logic
   - Extract business rules, decision tables, validations, exceptions, and external interactions
   - Output: business_logic_catalog.json + business_rules.md

5) Decompose code
   - Group programs/components into domains bounded by dependency cohesion
   - Identify shared kernels and anti-corruption boundaries
   - Output: domain_decomposition.json + domain_diagram.md

6) Plan migration wave
   - Define wave sequencing based on coupling, risk, business criticality, test readiness
   - Output: migration_wave_plan.json + cutover_strategy.md

7) Plan & test
   - Generate test cases for equivalence/regression/non-functional checks
   - Generate test data collection scripts
   - Generate test automation scripts for source-vs-target parity
   - Output: test_plan.md, scripts/test_data_collection/*, scripts/test_automation/*

8) Transform
   - Refactor code into target implementation patterns
   - Reforge transformed code (quality hardening, style normalization, security checks, performance guardrails)
   - Output: transformed code in destination repo + transformation_report.md

9) Agent-ready handoff packaging
   - Consolidate reusable outputs, traceability links, and unresolved decisions for specialist agents
   - Produce a compact "specialist starter context" that can be passed to any phase-specific agent
   - Output: artifacts/specialist_starter_context.json + artifacts/traceability_matrix.md

Global quality gates:
- Functional equivalence risks documented
- Dependency and data lineage preserved or explicitly remapped
- Test coverage summary included
- Open risks and manual decisions listed at the end
- Context pack and handoff notes are complete, current, and reusable by specialized agents without re-discovery

Final response format:
- Executive summary
- Phase-by-phase status table
- Artifact index with file paths
- Risks, assumptions, and recommended next actions
```

## 3) Iteration-2 specialized agent design

Create one specialist per phase and keep a thin orchestrator.

### A. KickoffAgent
- Validates connectors, repositories, and modernization context.
- Outputs job manifest and readiness checks.

### B. CodeAnalysisAgent
- Builds code inventory, file classification, complexity and missing-assets report.
- Produces machine-readable dependency graph.

### C. DataAnalysisAgent
- Maps datasets, DB2/table usage, read/write operations, and program-data interactions.
- Produces lineage map and data dictionary seed artifacts.

### D. ActivityMetricsAgent
- Interprets SMF type-30/type-110 feeds.
- Ranks high-impact jobs (CPU, duration, frequency) to prioritize modernization waves.

### E. DocumentationAgent
- Produces summary or detailed functional specs, per selected scope.
- Includes flow, dependencies, I/O, and integration contracts.

### F. BusinessLogicExtractionAgent
- Extracts rules, decision points, exception handling, and business semantics.
- Produces rule catalog and rule-to-program traceability.

### G. DecompositionAgent
- Creates domain decomposition from dependency topology and business boundaries.
- Marks reusable/shared assets and strangler boundaries.

### H. MigrationPlanningAgent
- Creates wave plan with rationale, dependency constraints, and rollback strategy.
- Emits critical path and milestone schedule.

### I. TestPlanningAgent
- Generates functional equivalence test strategy.
- Defines acceptance criteria and parity metrics.

### J. TestDataAgent
- Creates scripts to capture/prepare representative test datasets from legacy flows.

### K. TestAutomationAgent
- Generates executable parity/regression suites (source vs transformed outputs).

### L. RefactorAgent
- Produces transformed code aligned with target architecture and coding standards.

### M. ReforgeAgent
- Post-transform hardening: quality, security, observability, performance tuning, and cleanup.

### N. ModernizationQAAgent (cross-cutting)
- Independent reviewer for artifacts, risks, and gate criteria before phase completion.

### Shared handoff contract for all specialized agents
- Every specialist must read `artifacts/modernization_context_pack.json` and `artifacts/agent_handoff.md` before execution.
- Every specialist must write back:
  - key findings
  - artifact references
  - unresolved dependencies
  - decisions needed from upstream/downstream agents
- The orchestrator should fail a phase gate if an agent does not publish handoff updates.

## 4) Repository integration model (source + destination MCP)

For your first implementation, register two MCP servers in user config:

- `github-source`: points to source repo MCP endpoint
- `github-destination`: points to destination repo MCP endpoint

Recommended metadata when registering each server:
- `name`
- `base_url`
- `description`
- `headers` (for auth, e.g., bearer token)
- `enabled`

## 5) GitHub MCP connectivity verification

Current platform behavior and conclusion:

- Your system already supports user-registered MCP servers and dynamically builds toolsets from each configured endpoint.
- The runtime uses MCP over SSE (`/sse`) to list and call tools.
- This update adds optional per-server HTTP headers, enabling authenticated MCP endpoints (for example, GitHub gateway deployments requiring `Authorization` headers).

Practical conclusion:
- If your GitHub MCP server is SSE-accessible and exposes tools over MCP, you can connect now using the existing registration flow.
- If your GitHub MCP server is stdio-only (no SSE endpoint), you still need an SSE bridge/proxy layer.

## 6) Screenshot script (images 21-40)

Use the following captions as a ready-to-use narration track for screenshots 21 through 40.

| Image | Caption |
| --- | --- |
| 21 | Decomposition output can be reviewed in both table and graph views, with domains, seeds, and complexity/coverage indicators shown together. |
| 22 | The job plan supports conversational Q&A against generated artifacts, so users can ask component-level functionality questions directly in context. |
| 23 | Business rules are listed with rule IDs, rule types, inputs, and Given-When-Then acceptance criteria for traceability. |
| 24 | Functional flow diagrams are available at multiple abstraction levels to inspect end-to-end logic and decision branches. |
| 25 | BMS screen preview renders the mainframe UI layout, enabling quick validation of legacy screen behavior and labels. |
| 26 | COBOL statement metrics and business-rule metrics are surfaced together to combine structural complexity and business logic density. |
| 27 | File-level business logic documentation begins with summary, environment context, and key functional flows. |
| 28 | Extracted business logic can be navigated hierarchically down to file artifacts (COBOL/JCL and related assets). |
| 29 | Component-level documentation includes summary, functional behavior, environment details, and screen/flow references. |
| 30 | The extracted business logic tree shows third-level components (transaction and batch) under each business function. |
| 31 | Business function detail pages capture purpose, key capabilities, and identified components for each functional area. |
| 32 | At the second level, users select a business function from the application-level catalog to drill into details. |
| 33 | The application-level overview consolidates scope, enterprise context, and discovered functional groups. |
| 34 | Results are presented in multi-level hierarchy; the first level starts with application-level business specifications. |
| 35 | The job timeline records execution progress and messages while business logic extraction runs within the broader plan. |
| 36 | Business logic extraction setup supports application-level or file-level scope, with optional detailed functional specs. |
| 37 | Guided onboarding starts with providing extraction inputs before execution proceeds through the configured plan. |
| 38 | Generated technical documentation is indexed into the knowledge base so users can ask follow-up questions in chat. |
| 39 | Technical documentation review supports in-app PDF viewing, including high-level overview, purpose, and feature sections. |
| 40 | Documentation results list generated files with status, and users can select an entry and open it in the viewer. |

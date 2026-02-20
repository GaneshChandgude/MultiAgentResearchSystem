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

Global quality gates:
- Functional equivalence risks documented
- Dependency and data lineage preserved or explicitly remapped
- Test coverage summary included
- Open risks and manual decisions listed at the end

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

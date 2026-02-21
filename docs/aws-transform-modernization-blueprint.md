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
- Before executing any phase with configurable inputs, pause to validate and record phase-specific parameters (for example: scope, domain selection, engine/database settings, and output location).
- For phases that require user confirmation, surface a concise checklist and wait for explicit confirmation before moving forward.
- Keep guidance contextual: each response should include what was completed, what needs user input next, and the exact action to continue.

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
   - Support domain maintenance actions (create/edit/remove/import/export), seed selection, and decomposition re-runs
   - Publish both table-oriented and graph-oriented domain/dependency views
   - Output: domain_decomposition.json + domain_diagram.md + dependency_graph_interactive.json

6) Plan migration wave
   - Define wave sequencing based on coupling, risk, business criticality, test readiness
   - Publish recommended and preferred wave assignments, and regenerate sequence when user overrides wave preferences
   - Output: migration_wave_plan.json + cutover_strategy.md + wave_preferences.json

7) Plan & test
   - Capture test-plan inputs and scope prior to generation
   - Generate test cases for equivalence/regression/non-functional checks
   - Generate test data collection scripts
   - Generate test automation scripts for source-vs-target parity
   - Output: test_plan.md, scripts/test_data_collection/*, scripts/test_automation/*

8) Transform
   - Capture transform configuration (refactor engine version, project name, root package, target database, and legacy encoding)
   - For code reforge, capture source project location (for example S3 zip path to a buildable source project) before execution
   - Select domains to transform and record skipped domains with rationale
   - Support selective class-level scope updates so users can choose specific classes/files for reforge iterations
   - Refactor code into target implementation patterns
   - Reforge transformed code (quality hardening, style normalization, security checks, performance guardrails)
   - Publish side-by-side original-vs-reforged code diffs and concise readability/maintainability deltas
   - Output: transformed code in destination repo + transformation_report.md + transform_artifact_manifest.json

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

## 6) Screenshot script (images 1-20)

Use the following captions as a ready-to-use narration track for screenshots 1 through 20.

| Image | Caption |
| --- | --- |
| 1 | Kick off by creating a mainframe modernization job from the workspace chat and selecting the modernization objective set. |
| 2 | After objective selection, review the generated job specification and confirm by clicking **Create job**. |
| 3 | The guided setup starts in **Kick off modernization**, prompting the user to add collaborators before execution proceeds. |
| 4 | Once the job starts, the first walkthrough cue asks users to review **Analyze code** results in the job plan. |
| 5 | In code analysis, **View by** helps users switch between file list, file type, and file folder perspectives. |
| 6 | The **Codebase Issues** tab highlights blockers (for example missing CSD artifacts) before downstream phases. |
| 7 | The analysis dashboard summarizes total LOC and file-type distribution to quickly assess modernization scope. |
| 8 | After code analysis completes, users open **View data analysis results** directly from the job plan. |
| 9 | Data lineage provides top-level interaction counts across datasets, DB2 objects, programs, and JCLs. |
| 10 | The data-sets tab shows source stores and which programs read/write them, including access-mode visibility. |
| 11 | The DB2 tab maps table usage and operation types to the calling COBOL programs. |
| 12 | The Data Dictionary consolidates COBOL structures and DB2 entities for field-level discovery. |
| 13 | COBOL data structure view exposes field metadata (types, levels, business meaning) for copybook interpretation. |
| 14 | DB2 structure view lists columns, constraints, and data definitions to support schema understanding. |
| 15 | Activity metrics onboarding introduces SMF analysis as a dedicated phase in the modernization sequence. |
| 16 | Users provide the SMF zip location in S3 to enable batch activity profiling for the run. |
| 17 | Activity analysis results summarize longest CPU, longest duration, and most-executed batch jobs. |
| 18 | Users can ask contextual questions in chat (for example program functionality) while reviewing activity outputs. |
| 19 | Documentation generation setup lets users choose detail level (Summary vs Detailed functional specifications). |
| 20 | The final walkthrough step confirms SMF insights are stored in S3 and closes the first guidance sequence. |

## 7) Screenshot script (images 21-40)

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

## 8) Screenshot script (images 41-60)

Use the following captions as a ready-to-use narration track for screenshots 41 through 60.

| Image | Caption |
| --- | --- |
| 41 | Refactor-code guidance appears directly in the job plan, prompting users to configure transform inputs before generation. |
| 42 | Refactor output view shows the generated artifact path in S3 and domain-level transform/generate completion states. |
| 43 | Domain selection for refactoring lets users pick modernization scope and proceed with Submit once domains are chosen. |
| 44 | The domain selection screen keeps status, sequence, and file counts visible so users can validate refactor scope before submission. |
| 45 | Configure transformation captures engine version, project naming, target database, and legacy encoding settings in one form. |
| 46 | Refactor setup also supports a searchable domain list, helping teams select only the business domains to modernize first. |
| 47 | Guided tips inside the job plan walk users through the transform step from task selection to parameter setup. |
| 48 | Test-plan guidance highlights where to configure plan inputs while preserving live chat context and job progress visibility. |
| 49 | Migration-wave preference editing supports assigning domains to preferred waves before regenerating the plan. |
| 50 | Chart view visualizes recommended migration waves by domain, including file and LOC sizing to aid sequencing decisions. |
| 51 | Users can toggle between table and chart formats to review wave recommendations in the representation they prefer. |
| 52 | Table view lists recommended versus preferred wave values, enabling quick review before saving or submitting. |
| 53 | Plan-migration-wave starts from the job plan panel and guides users to configure wave inputs in sequence. |
| 54 | Decomposition graph domain view shows cross-domain integrations, helping identify coupling before migration planning. |
| 55 | Dependency graph mode provides zoom and filter controls so users can inspect parent-child and file-level link density. |
| 56 | Layout actions in graph view reorganize dense dependency maps for easier structural analysis of decomposition results. |
| 57 | Decomposition in-progress state keeps domain table data visible while indicating active processing and next-step actions. |
| 58 | Create-domain dialog captures domain metadata and allows marking seed files to steer grouping behavior. |
| 59 | Actions menu centralizes domain operations such as create/edit/remove, import/export, and decomposition configuration updates. |
| 60 | Completed decomposition summary confirms successful domain generation and leaves save/submit controls ready for workflow continuation. |

## 8) Screenshot script (images 61-64)

Use the following captions as a ready-to-use narration track for screenshots 61 through 64.

| Image | Caption |
| --- | --- |
| 61 | Side-by-side reforge comparison highlights readability improvements by showing original and reforged Java implementations with the same logic decomposed into clearer helper methods. |
| 62 | Job-plan guidance for Reforge confirms task completion and routes users to View Results, keeping context in chat while documenting what changed. |
| 63 | Class-selection dialog for Reforge supports targeted scope by letting users check specific classes, track selected LOC totals, and update the run without reprocessing everything. |
| 64 | Reforge setup requires a buildable source-project zip location (for example S3 path), ensuring refinement starts from compilable input before users continue. |

import { useEffect, useMemo, useState } from "react";
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const emptyConfig = {
  llm: {
    azure_openai_endpoint: "",
    azure_openai_api_key: "",
    azure_openai_deployment: "",
    planning_azure_openai_deployment: "",
    specialist_azure_openai_deployment: "",
    azure_openai_api_version: ""
  },
  embedder: {
    embeddings_model: "",
    embeddings_endpoint: "",
    embeddings_api_key: "",
    embeddings_api_version: ""
  },
  langfuse: {
    langfuse_enabled: false,
    langfuse_public_key: "",
    langfuse_secret_key: "",
    langfuse_host: "",
    langfuse_release: "",
    langfuse_debug: false,
    langfuse_prompt_enabled: false,
    langfuse_prompt_label: "",
    langfuse_verify_ssl: true,
    langfuse_ca_bundle: ""
  },
  guardrails: {
    pii_middleware_enabled: true,
    pii_redaction_enabled: true,
    pii_block_input: false,
    nested_agent_pii_profile: "nested",
    orchestrator_agent_pii_profile: "off",
    max_input_length: 4000,
    max_output_length: 8000,
    model_guardrails_enabled: true,
    model_guardrails_moderation_enabled: true,
    model_guardrails_output_language: "English",
    model_input_guardrail_rules: [],
    use_dynamic_subagent_flow: true
  },
  mcp_servers: {
    servers: []
  }
};

const configSteps = [
  { key: "llm", label: "LLM Configuration" },
  { key: "embedder", label: "Embedder Setup" },
  { key: "langfuse", label: "Langfuse Observability" },
  { key: "guardrails", label: "Guardrails & PII" },
  { key: "mcp_servers", label: "MCP Server Registry" }
];

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const payload = await response.json();
      const detail = payload?.detail ?? payload?.message ?? payload?.error;
      throw new Error(detail ? String(detail) : "Request failed");
    }
    const message = await response.text();
    try {
      const payload = JSON.parse(message);
      const detail = payload?.detail ?? payload?.message ?? payload?.error;
      throw new Error(detail ? String(detail) : "Request failed");
    } catch (err) {
      if (err instanceof SyntaxError) {
        throw new Error(message || "Request failed");
      }
      throw err;
    }
  }
  return response.json();
}

function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event) => {
    event.preventDefault();
    if (!username.trim()) return;
    setLoading(true);
    setError("");
    try {
      const data = await apiRequest("/api/login", {
        method: "POST",
        body: JSON.stringify({ username })
      });
      onLogin(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-screen">
      <div className="login-card">
        <h1>Assistance Command Center</h1>
        <p>Sign in to configure your assistants and explore operational insights.</p>
        <form onSubmit={submit}>
          <div className="input-group">
            <label htmlFor="username">User name</label>
            <input
              id="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="Analyst name"
            />
          </div>
          {error ? <p style={{ color: "#f87171" }}>{error}</p> : null}
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? "Signing in..." : "Start analysis"}
          </button>
        </form>
      </div>
    </div>
  );
}

function ConfigWizard({ config, setConfig, user, initialKey, onClose }) {
  const [activeStep, setActiveStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [savedSteps, setSavedSteps] = useState({});

  useEffect(() => {
    if (!initialKey) return;
    const index = configSteps.findIndex((step) => step.key === initialKey);
    if (index >= 0) {
      setActiveStep(index);
    }
  }, [initialKey]);

  const currentKey = configSteps[activeStep].key;

  const updateField = (section, field, value) => {
    setConfig((prev) => ({
      ...prev,
      [section]: { ...prev[section], [field]: value }
    }));
  };

  const updateInputGuardrailRule = (index, field, value) => {
    const existing = config.guardrails.model_input_guardrail_rules || [];
    const updated = existing.map((rule, ruleIndex) =>
      ruleIndex === index ? { ...rule, [field]: value } : rule
    );
    updateField("guardrails", "model_input_guardrail_rules", updated);
  };

  const addInputGuardrailRule = () => {
    const existing = config.guardrails.model_input_guardrail_rules || [];
    updateField("guardrails", "model_input_guardrail_rules", [
      ...existing,
      {
        name: "",
        trigger_description: "",
        trigger_examples: [""],
        block_message: ""
      }
    ]);
  };

  const removeInputGuardrailRule = (index) => {
    const existing = config.guardrails.model_input_guardrail_rules || [];
    updateField(
      "guardrails",
      "model_input_guardrail_rules",
      existing.filter((_, ruleIndex) => ruleIndex !== index)
    );
  };

  const updateRuleExample = (ruleIndex, exampleIndex, value) => {
    const existing = config.guardrails.model_input_guardrail_rules || [];
    const updated = existing.map((rule, index) => {
      if (index !== ruleIndex) return rule;
      const examples = Array.isArray(rule.trigger_examples) ? [...rule.trigger_examples] : [""];
      examples[exampleIndex] = value;
      return { ...rule, trigger_examples: examples };
    });
    updateField("guardrails", "model_input_guardrail_rules", updated);
  };

  const addRuleExample = (ruleIndex) => {
    const existing = config.guardrails.model_input_guardrail_rules || [];
    const updated = existing.map((rule, index) =>
      index === ruleIndex
        ? {
            ...rule,
            trigger_examples: [...(Array.isArray(rule.trigger_examples) ? rule.trigger_examples : []), ""]
          }
        : rule
    );
    updateField("guardrails", "model_input_guardrail_rules", updated);
  };

  const removeRuleExample = (ruleIndex, exampleIndex) => {
    const existing = config.guardrails.model_input_guardrail_rules || [];
    const updated = existing.map((rule, index) => {
      if (index !== ruleIndex) return rule;
      const examples = (Array.isArray(rule.trigger_examples) ? rule.trigger_examples : []).filter(
        (_, idx) => idx !== exampleIndex
      );
      return { ...rule, trigger_examples: examples.length ? examples : [""] };
    });
    updateField("guardrails", "model_input_guardrail_rules", updated);
  };

  const updateMcpServer = (index, field, value) => {
    const existing = config.mcp_servers?.servers || [];
    const updated = existing.map((server, serverIndex) =>
      serverIndex === index ? { ...server, [field]: value } : server
    );
    updateField("mcp_servers", "servers", updated);
  };

  const addMcpServer = () => {
    const existing = config.mcp_servers?.servers || [];
    updateField("mcp_servers", "servers", [
      ...existing,
      { name: "", base_url: "", description: "", enabled: true }
    ]);
  };

  const removeMcpServer = (index) => {
    const existing = config.mcp_servers?.servers || [];
    updateField(
      "mcp_servers",
      "servers",
      existing.filter((_, serverIndex) => serverIndex !== index)
    );
  };

  const saveCurrent = async () => {
    setSaving(true);
    setError("");
    const endpointByKey = {
      llm: "llm",
      embedder: "embedder",
      langfuse: "langfuse",
      guardrails: "guardrails",
      mcp_servers: "mcp_servers"
    };
    try {
      await apiRequest(`/api/config/${endpointByKey[currentKey]}`, {
        method: "POST",
        body: JSON.stringify({ user_id: user.user_id, ...config[currentKey] })
      });
      setSavedSteps((prev) => ({ ...prev, [currentKey]: true }));
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card">
      <h2>{configSteps[activeStep].label}</h2>
      <div className="stepper">
        {configSteps.map((step, index) => (
          <span
            key={step.key}
            className={`step ${index === activeStep ? "active" : ""}`}
            role="button"
            onClick={() => setActiveStep(index)}
          >
            {step.label}
          </span>
        ))}
      </div>
      {currentKey === "llm" && (
        <div>
          <div className="input-group">
            <label>Azure OpenAI Endpoint</label>
            <input
              value={config.llm.azure_openai_endpoint}
              onChange={(event) => updateField("llm", "azure_openai_endpoint", event.target.value)}
              placeholder="https://..."
            />
          </div>
          <div className="input-group">
            <label>Azure OpenAI API Key</label>
            <input
              value={config.llm.azure_openai_api_key}
              onChange={(event) => updateField("llm", "azure_openai_api_key", event.target.value)}
              placeholder="API key"
            />
          </div>
          <div className="input-group">
            <label>Default Deployment Name</label>
            <input
              value={config.llm.azure_openai_deployment}
              onChange={(event) => updateField("llm", "azure_openai_deployment", event.target.value)}
              placeholder="deployment"
            />
          </div>
          <div className="input-group">
            <label>Planning Deployment (Planner/Router/Orchestrator)</label>
            <input
              value={config.llm.planning_azure_openai_deployment}
              onChange={(event) =>
                updateField("llm", "planning_azure_openai_deployment", event.target.value)
              }
              placeholder="e.g. gpt-4.1"
            />
          </div>
          <div className="input-group">
            <label>Specialist Deployment (Sub Agents)</label>
            <input
              value={config.llm.specialist_azure_openai_deployment}
              onChange={(event) =>
                updateField("llm", "specialist_azure_openai_deployment", event.target.value)
              }
              placeholder="e.g. gpt-4.1-mini"
            />
          </div>
          <div className="input-group">
            <label>API Version</label>
            <input
              value={config.llm.azure_openai_api_version}
              onChange={(event) => updateField("llm", "azure_openai_api_version", event.target.value)}
            />
          </div>
        </div>
      )}
      {currentKey === "embedder" && (
        <div>
          <div className="input-group">
            <label>Embeddings Model</label>
            <input
              value={config.embedder.embeddings_model}
              onChange={(event) => updateField("embedder", "embeddings_model", event.target.value)}
              placeholder="TxtEmbedAda002"
            />
          </div>
          <div className="input-group">
            <label>Embeddings Endpoint</label>
            <input
              value={config.embedder.embeddings_endpoint}
              onChange={(event) => updateField("embedder", "embeddings_endpoint", event.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Embeddings API Key</label>
            <input
              value={config.embedder.embeddings_api_key}
              onChange={(event) => updateField("embedder", "embeddings_api_key", event.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Embeddings API Version</label>
            <input
              value={config.embedder.embeddings_api_version}
              onChange={(event) => updateField("embedder", "embeddings_api_version", event.target.value)}
            />
          </div>
        </div>
      )}
      {currentKey === "langfuse" && (
        <div>
          <div className="input-group">
            <label>Langfuse Host</label>
            <input
              value={config.langfuse.langfuse_host}
              onChange={(event) => updateField("langfuse", "langfuse_host", event.target.value)}
              placeholder="https://cloud.langfuse.com"
            />
          </div>
          <div className="input-group">
            <label>Langfuse Public Key</label>
            <input
              value={config.langfuse.langfuse_public_key}
              onChange={(event) => updateField("langfuse", "langfuse_public_key", event.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Langfuse Secret Key</label>
            <input
              value={config.langfuse.langfuse_secret_key}
              onChange={(event) => updateField("langfuse", "langfuse_secret_key", event.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Release Tag</label>
            <input
              value={config.langfuse.langfuse_release}
              onChange={(event) => updateField("langfuse", "langfuse_release", event.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Prompt Label</label>
            <input
              value={config.langfuse.langfuse_prompt_label}
              onChange={(event) => updateField("langfuse", "langfuse_prompt_label", event.target.value)}
            />
          </div>
          <div className="input-group">
            <label>CA Bundle Path</label>
            <input
              value={config.langfuse.langfuse_ca_bundle}
              onChange={(event) => updateField("langfuse", "langfuse_ca_bundle", event.target.value)}
            />
          </div>
          <div className="input-group">
            <label>Enable Langfuse</label>
            <select
              value={config.langfuse.langfuse_enabled ? "yes" : "no"}
              onChange={(event) => updateField("langfuse", "langfuse_enabled", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Debug Mode</label>
            <select
              value={config.langfuse.langfuse_debug ? "yes" : "no"}
              onChange={(event) => updateField("langfuse", "langfuse_debug", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Prompt Tracking</label>
            <select
              value={config.langfuse.langfuse_prompt_enabled ? "yes" : "no"}
              onChange={(event) => updateField("langfuse", "langfuse_prompt_enabled", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Verify SSL</label>
            <select
              value={config.langfuse.langfuse_verify_ssl ? "yes" : "no"}
              onChange={(event) => updateField("langfuse", "langfuse_verify_ssl", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
        </div>
      )}
      {currentKey === "guardrails" && (
        <div>
          <div className="input-group">
            <label>Enable PII middleware</label>
            <select
              value={config.guardrails.pii_middleware_enabled ? "yes" : "no"}
              onChange={(event) => updateField("guardrails", "pii_middleware_enabled", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Enable PII redaction</label>
            <select
              value={config.guardrails.pii_redaction_enabled ? "yes" : "no"}
              onChange={(event) => updateField("guardrails", "pii_redaction_enabled", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Block PII in input</label>
            <select
              value={config.guardrails.pii_block_input ? "yes" : "no"}
              onChange={(event) => updateField("guardrails", "pii_block_input", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Nested agent PII profile</label>
            <select
              value={config.guardrails.nested_agent_pii_profile}
              onChange={(event) => updateField("guardrails", "nested_agent_pii_profile", event.target.value)}
              disabled={!config.guardrails.pii_middleware_enabled}
            >
              <option value="full">full</option>
              <option value="nested">nested</option>
              <option value="off">off</option>
            </select>
          </div>
          <div className="input-group">
            <label>Orchestrator PII profile</label>
            <select
              value={config.guardrails.orchestrator_agent_pii_profile}
              onChange={(event) => updateField("guardrails", "orchestrator_agent_pii_profile", event.target.value)}
              disabled={!config.guardrails.pii_middleware_enabled}
            >
              <option value="full">full</option>
              <option value="nested">nested</option>
              <option value="off">off</option>
            </select>
          </div>
          <div className="input-group">
            <label>Agent orchestration flow</label>
            <select
              value={config.guardrails.use_dynamic_subagent_flow ? "dynamic" : "legacy"}
              onChange={(event) =>
                updateField("guardrails", "use_dynamic_subagent_flow", event.target.value === "dynamic")
              }
            >
              <option value="dynamic">Dynamic subagent flow (default)</option>
              <option value="legacy">Legacy fixed specialist flow</option>
            </select>
          </div>
          <div className="input-group">
            <label>Max input length</label>
            <input
              type="number"
              min="1"
              value={config.guardrails.max_input_length}
              onChange={(event) => updateField("guardrails", "max_input_length", Number(event.target.value))}
            />
          </div>
          <div className="input-group">
            <label>Max output length</label>
            <input
              type="number"
              min="1"
              value={config.guardrails.max_output_length}
              onChange={(event) => updateField("guardrails", "max_output_length", Number(event.target.value))}
            />
          </div>
          <div className="input-group">
            <label>Enable model-based guardrails</label>
            <select
              value={config.guardrails.model_guardrails_enabled ? "yes" : "no"}
              onChange={(event) => updateField("guardrails", "model_guardrails_enabled", event.target.value === "yes")}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Enable model moderation</label>
            <select
              value={config.guardrails.model_guardrails_moderation_enabled ? "yes" : "no"}
              onChange={(event) =>
                updateField("guardrails", "model_guardrails_moderation_enabled", event.target.value === "yes")
              }
              disabled={!config.guardrails.model_guardrails_enabled}
            >
              <option value="yes">Enabled</option>
              <option value="no">Disabled</option>
            </select>
          </div>
          <div className="input-group">
            <label>Required output language</label>
            <input
              type="text"
              placeholder="e.g., English"
              value={config.guardrails.model_guardrails_output_language}
              onChange={(event) => updateField("guardrails", "model_guardrails_output_language", event.target.value)}
              disabled={!config.guardrails.model_guardrails_enabled}
            />
          </div>
          <div className="input-group">
            <label>Model input guardrail rules</label>
            <div className="guardrail-rule-hint">
              Define multiple rules. If input matches a rule, the configured block response is returned.
            </div>
            <div className="guardrail-rule-list">
              {(config.guardrails.model_input_guardrail_rules || []).map((rule, ruleIndex) => (
                <div className="guardrail-rule-card" key={`guardrail-rule-${ruleIndex}`}>
                  <div className="guardrail-rule-header">
                    <strong>Rule {ruleIndex + 1}</strong>
                    <button
                      type="button"
                      className="btn btn-outline guardrail-inline-button"
                      onClick={() => removeInputGuardrailRule(ruleIndex)}
                      disabled={!config.guardrails.model_guardrails_enabled}
                    >
                      Remove rule
                    </button>
                  </div>
                  <div className="input-group">
                    <label>Rule name</label>
                    <input
                      value={rule.name || ""}
                      onChange={(event) => updateInputGuardrailRule(ruleIndex, "name", event.target.value)}
                      placeholder="e.g. User asks about financial results"
                      disabled={!config.guardrails.model_guardrails_enabled}
                    />
                  </div>
                  <div className="input-group">
                    <label>Trigger description</label>
                    <textarea
                      value={rule.trigger_description || ""}
                      onChange={(event) =>
                        updateInputGuardrailRule(ruleIndex, "trigger_description", event.target.value)
                      }
                      placeholder="Describe when this rule should trigger"
                      disabled={!config.guardrails.model_guardrails_enabled}
                    />
                  </div>
                  <div className="input-group">
                    <label>Trigger examples</label>
                    {(Array.isArray(rule.trigger_examples) ? rule.trigger_examples : []).map((example, exampleIndex) => (
                      <div className="guardrail-example-row" key={`guardrail-rule-${ruleIndex}-example-${exampleIndex}`}>
                        <input
                          value={example || ""}
                          onChange={(event) => updateRuleExample(ruleIndex, exampleIndex, event.target.value)}
                          placeholder="What was NVIDIA's EPS last year?"
                          disabled={!config.guardrails.model_guardrails_enabled}
                        />
                        <button
                          type="button"
                          className="btn btn-outline guardrail-inline-button"
                          onClick={() => removeRuleExample(ruleIndex, exampleIndex)}
                          disabled={!config.guardrails.model_guardrails_enabled}
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      className="btn btn-secondary guardrail-inline-button"
                      onClick={() => addRuleExample(ruleIndex)}
                      disabled={!config.guardrails.model_guardrails_enabled}
                    >
                      + Add trigger example
                    </button>
                  </div>
                  <div className="input-group">
                    <label>Block response</label>
                    <textarea
                      value={rule.block_message || ""}
                      onChange={(event) => updateInputGuardrailRule(ruleIndex, "block_message", event.target.value)}
                      placeholder="I'm sorry, I can't discuss financial results."
                      disabled={!config.guardrails.model_guardrails_enabled}
                    />
                  </div>
                </div>
              ))}
            </div>
            <button
              type="button"
              className="btn btn-secondary guardrail-add-rule"
              onClick={addInputGuardrailRule}
              disabled={!config.guardrails.model_guardrails_enabled}
            >
              + Add guardrail rule
            </button>
          </div>
        </div>
      )}
      {currentKey === "mcp_servers" && (
        <div>
          <p style={{ color: "#475569", fontSize: "14px", marginBottom: "16px" }}>
            Register MCP server endpoints for this user. These toolsets are exposed to the orchestrator and specialist agents.
          </p>
          <div className="guardrail-rule-list">
            {(config.mcp_servers?.servers || []).map((server, index) => (
              <div className="guardrail-rule-card" key={`mcp-server-${index}`}>
                <div className="guardrail-rule-header">
                  <strong>Server {index + 1}</strong>
                  <button
                    type="button"
                    className="btn btn-outline guardrail-inline-button"
                    onClick={() => removeMcpServer(index)}
                  >
                    Remove server
                  </button>
                </div>
                <div className="input-group">
                  <label>Server name</label>
                  <input
                    value={server.name || ""}
                    onChange={(event) => updateMcpServer(index, "name", event.target.value)}
                    placeholder="e.g. crm-tools"
                  />
                </div>
                <div className="input-group">
                  <label>Base URL</label>
                  <input
                    value={server.base_url || ""}
                    onChange={(event) => updateMcpServer(index, "base_url", event.target.value)}
                    placeholder="http://localhost:8800"
                  />
                </div>
                <div className="input-group">
                  <label>Description</label>
                  <textarea
                    value={server.description || ""}
                    onChange={(event) => updateMcpServer(index, "description", event.target.value)}
                    placeholder="What this MCP server provides"
                  />
                </div>
                <div className="input-group">
                  <label>Enabled</label>
                  <select
                    value={server.enabled === false ? "no" : "yes"}
                    onChange={(event) => updateMcpServer(index, "enabled", event.target.value === "yes")}
                  >
                    <option value="yes">Enabled</option>
                    <option value="no">Disabled</option>
                  </select>
                </div>
              </div>
            ))}
          </div>
          <button type="button" className="btn btn-secondary guardrail-add-rule" onClick={addMcpServer}>
            + Register MCP server
          </button>
        </div>
      )}
      {error ? <p style={{ color: "#ef4444", marginTop: "8px" }}>{error}</p> : null}
      <div style={{ display: "flex", gap: "12px", marginTop: "16px" }}>
        <button className="btn btn-primary" onClick={saveCurrent} disabled={saving}>
          {saving ? "Saving..." : "Save configuration"}
        </button>
        <button className="btn btn-outline" onClick={onClose}>
          Close
        </button>
      </div>
      <div style={{ marginTop: "16px", display: "flex", alignItems: "center", gap: "12px" }}>
        <span className="badge">Saved: {savedSteps[currentKey] ? "Yes" : "No"}</span>
        <span style={{ fontSize: "12px", color: "#64748b" }}>
          Switch tabs to configure other services.
        </span>
      </div>
    </div>
  );
}

function TracePanel({ trace }) {
  const entries = useMemo(() => {
    if (!trace) return [];
    const list = (Array.isArray(trace) ? trace : [trace]).filter(Boolean);
    const subagentQueue = list
      .filter((item) => item?.agent === "DynamicSubagent")
      .map((item) => ({
        agent: item.agent || "DynamicSubagent",
        messages: item.tool_calls || item.calls || []
      }));

    const merged = [];
    list.forEach((item) => {
      const agent = item.agent || "Agent";
      if (agent === "DynamicSubagent") {
        return;
      }
      const messages = item.tool_calls || item.calls || [];
      const runSubagentCount = messages.reduce((count, message) => {
        const toolCalls = message?.tool_calls || [];
        return count + toolCalls.filter((call) => call?.name === "run_subagent").length;
      }, 0);
      merged.push({
        agent,
        messages,
        subagentCalls: runSubagentCount > 0 ? subagentQueue.splice(0, runSubagentCount) : []
      });
    });

    if (subagentQueue.length) {
      merged.push(...subagentQueue.map((entry) => ({ ...entry, subagentCalls: [] })));
    }

    return merged;
  }, [trace]);

  const buildToolCalls = (messages) => {
    if (!Array.isArray(messages)) return [];
    const outputs = new Map();

    messages.forEach((message) => {
      if (message?.type === "ToolMessage" && message.tool_call_id) {
        outputs.set(message.tool_call_id, message.content);
      }
    });

    const toolCalls = [];
    messages.forEach((message) => {
      if (!message?.tool_calls?.length) return;
      message.tool_calls.forEach((call) => {
        toolCalls.push({
          id: call.id,
          name: call.name || "Tool call",
          args: call.args,
          output: outputs.get(call.id)
        });
      });
    });

    if (toolCalls.length) return toolCalls;

    return messages
      .filter((message) => message?.type === "ToolMessage")
      .map((message) => ({
        id: message.tool_call_id || message.content,
        name: "Tool output",
        args: null,
        output: message.content
      }));
  };

  const decodeEscapes = (value) => {
    if (typeof value !== "string") return value;
    return value
      .replace(/\\r\\n/g, "\n")
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t");
  };

  const formatPayload = (payload) => {
    if (payload == null) return "";
    if (typeof payload === "string") {
      const decoded = decodeEscapes(payload);
      const trimmed = decoded.trim();
      if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
        try {
          return JSON.stringify(JSON.parse(trimmed), null, 2);
        } catch (err) {
          return decoded;
        }
      }
      return decoded;
    }
    try {
      return JSON.stringify(payload, null, 2);
    } catch (err) {
      return String(payload);
    }
  };

  const escapeHtml = (text) =>
    text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");

  const renderSimpleMarkup = (text) => {
    const escaped = escapeHtml(text);
    return escaped
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br />");
  };

  if (!entries.length) {
    return <p>No trace available yet.</p>;
  }

  return (
    <div className="trace-panel">
      {entries.map((entry, index) => {
        const toolCalls = buildToolCalls(entry.messages);
        let runSubagentSeen = 0;
        return (
        <div className="trace-item" key={`${entry.agent}-${index}`}>
          <strong>{entry.agent}</strong>
          {toolCalls.length ? (
            <ul style={{ marginTop: "8px", paddingLeft: "16px" }}>
              {toolCalls.map((call, idx) => {
                const subagentForCall = call.name === "run_subagent" ? entry.subagentCalls?.[runSubagentSeen++] : null;
                return (
                <li key={`${call.name}-${call.id || idx}`}>
                  <div style={{ fontWeight: 600 }}>{call.name}</div>
                  {call.args ? (
                    <div style={{ marginTop: "6px" }}>
                      <div style={{ fontSize: "12px", color: "#64748b", fontWeight: 600 }}>
                        Input
                      </div>
                      <div
                        style={{ whiteSpace: "pre-wrap", marginTop: "4px" }}
                        className="trace-markup"
                        dangerouslySetInnerHTML={{ __html: renderSimpleMarkup(formatPayload(call.args)) }}
                      />
                    </div>
                  ) : null}
                  {call.output ? (
                    <div style={{ marginTop: "6px" }}>
                      <div style={{ fontSize: "12px", color: "#64748b", fontWeight: 600 }}>
                        Output
                      </div>
                      <div
                        style={{ whiteSpace: "pre-wrap", marginTop: "4px" }}
                        className="trace-markup"
                        dangerouslySetInnerHTML={{ __html: renderSimpleMarkup(formatPayload(call.output)) }}
                      />
                    </div>
                  ) : null}
                  {subagentForCall ? (
                    <div style={{ marginTop: "8px", marginLeft: "8px", paddingLeft: "10px", borderLeft: "2px solid #cbd5e1" }}>
                      {(() => {
                        const subagentToolCalls = buildToolCalls(subagentForCall.messages);
                        return (
                          <div>
                            <div style={{ fontSize: "12px", color: "#334155", fontWeight: 700 }}>
                              DynamicSubagent execution
                            </div>
                            <ul style={{ marginTop: "6px", paddingLeft: "16px" }}>
                              {subagentToolCalls.map((subCall, toolIndex) => (
                                <li key={`${subCall.name}-${subCall.id || toolIndex}`}>
                                  <div style={{ fontWeight: 600 }}>{subCall.name}</div>
                                  {subCall.args ? (
                                    <div
                                      style={{ whiteSpace: "pre-wrap", marginTop: "4px" }}
                                      className="trace-markup"
                                      dangerouslySetInnerHTML={{ __html: renderSimpleMarkup(formatPayload(subCall.args)) }}
                                    />
                                  ) : null}
                                  {subCall.output ? (
                                    <div
                                      style={{ whiteSpace: "pre-wrap", marginTop: "4px" }}
                                      className="trace-markup"
                                      dangerouslySetInnerHTML={{ __html: renderSimpleMarkup(formatPayload(subCall.output)) }}
                                    />
                                  ) : null}
                                </li>
                              ))}
                            </ul>
                          </div>
                        );
                      })()}
                    </div>
                  ) : null}
                </li>
              );})}
            </ul>
          ) : (
            <p style={{ marginTop: "8px" }}>No tool calls captured.</p>
          )}
        </div>
        );
      })}
    </div>
  );
}

function ResponseActions({ userId, message }) {
  const [rating, setRating] = useState(null);
  const [comments, setComments] = useState("");
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const selected = rating ? (rating === "up" ? "up" : "down") : null;
  const canSubmit = !!message.chatId && !sending && !sent;
  const feedbackPrompt =
    rating === "up" ? "What worked well?" : "What went wrong?";
  const feedbackPlaceholder =
    rating === "up"
      ? "Share what was helpful so we can keep it consistent."
      : "Share details to help us improve the response.";

  const sendFeedback = async (nextRating, nextComments = "") => {
    if (!message.chatId || sent) return;
    setSending(true);
    try {
      await apiRequest("/api/feedback", {
        method: "POST",
        body: JSON.stringify({
          user_id: userId,
          chat_id: message.chatId,
          rating: nextRating === "up" ? 5 : 1,
          comments: nextComments
        })
      });
      setSent(true);
    } catch (err) {
      console.error(err);
    } finally {
      setSending(false);
    }
  };

  const handleLike = () => {
    if (sent) return;
    setRating("up");
    setComments("");
  };

  const handleDislike = () => {
    if (sent) return;
    setRating("down");
    setComments("");
  };

  const submitFeedback = () => {
    if (!canSubmit || !rating) return;
    sendFeedback(rating, comments);
  };

  return (
    <div className="response-actions">
      <div className="feedback-actions">
        <button
          type="button"
          className={`icon-button ${selected === "up" ? "active" : ""}`}
          onClick={handleLike}
          disabled={!canSubmit}
        >
          👍
        </button>
        <button
          type="button"
          className={`icon-button ${selected === "down" ? "active" : ""}`}
          onClick={handleDislike}
          disabled={sent}
        >
          👎
        </button>
        <button
          type="button"
          className="trace-toggle"
          onClick={() => setShowTrace((prev) => !prev)}
          disabled={!message.trace}
        >
          {showTrace ? "Hide agentic trace" : "Agentic trace"}
        </button>
      </div>
      {rating && !sent ? (
        <div className="feedback-form">
          <div className="input-group">
            <label>{feedbackPrompt}</label>
            <textarea
              value={comments}
              onChange={(event) => setComments(event.target.value)}
              placeholder={feedbackPlaceholder}
            />
          </div>
          <button className="btn btn-primary" onClick={submitFeedback} disabled={!canSubmit}>
            {sending ? "Sending..." : "Submit feedback"}
          </button>
        </div>
      ) : null}
      {sent ? <span className="feedback-status">Thanks for the feedback.</span> : null}
      {showTrace ? (
        <div className="trace-modal-backdrop" role="dialog" aria-modal="true" aria-label="Agentic trace">
          <div className="trace-modal-card">
            <div className="trace-modal-header">
              <h3>Agentic trace</h3>
              <button type="button" className="trace-modal-close" onClick={() => setShowTrace(false)}>
                ×
              </button>
            </div>
            <TracePanel trace={message.trace} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function MarkdownMessage({ content }) {
  const inlineFormat = (text) => {
    const source = text || "";
    const tokens = source.split(/(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^\)]+\))/g).filter(Boolean);
    return tokens.map((token, index) => {
      if (token.startsWith("**") && token.endsWith("**")) {
        return <strong key={index}>{token.slice(2, -2)}</strong>;
      }
      if (token.startsWith("`") && token.endsWith("`")) {
        return <code key={index}>{token.slice(1, -1)}</code>;
      }
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^\)]+)\)$/);
      if (linkMatch) {
        return (
          <a key={index} href={linkMatch[2]} target="_blank" rel="noreferrer">
            {linkMatch[1]}
          </a>
        );
      }
      return <span key={index}>{token}</span>;
    });
  };

  const lines = String(content || "").split("\n");
  const output = [];
  let listItems = [];
  let listType = null;

  const flushList = (key) => {
    if (!listItems.length || !listType) return;
    const ListTag = listType;
    output.push(
      <ListTag key={key}>
        {listItems.map((item, idx) => (
          <li key={idx}>{inlineFormat(item)}</li>
        ))}
      </ListTag>
    );
    listItems = [];
    listType = null;
  };

  lines.forEach((rawLine, index) => {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      flushList(`list-${index}`);
      return;
    }

    const headingMatch = line.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushList(`list-${index}`);
      const level = headingMatch[1].length;
      const HeadingTag = `h${level}`;
      output.push(<HeadingTag key={`heading-${index}`}>{inlineFormat(headingMatch[2])}</HeadingTag>);
      return;
    }

    const bulletMatch = line.match(/^[-*]\s+(.*)$/);
    const orderedMatch = line.match(/^\d+[\.)]\s+(.*)$/);
    if (bulletMatch) {
      if (listType && listType !== "ul") {
        flushList(`list-${index}`);
      }
      listType = "ul";
      listItems.push(bulletMatch[1]);
      return;
    }
    if (orderedMatch) {
      if (listType && listType !== "ol") {
        flushList(`list-${index}`);
      }
      listType = "ol";
      listItems.push(orderedMatch[1]);
      return;
    }

    flushList(`list-${index}`);
    output.push(<p key={`p-${index}`}>{inlineFormat(line)}</p>);
  });

  flushList("list-end");

  return <div className="message-content markdown-content">{output.length ? output : content}</div>;
}

function ChatScreen({ user }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [job, setJob] = useState(null);
  const [progress, setProgress] = useState(null);
  const [capabilities, setCapabilities] = useState("");
  const [showCapabilities, setShowCapabilities] = useState(true);
  const [sampleQueries, setSampleQueries] = useState([]);

  useEffect(() => {
    setShowCapabilities(true);
    setSampleQueries([]);
    apiRequest(`/api/capabilities/${user.user_id}?generate=true`)
      .then((data) => {
        setCapabilities((data.capabilities || "").trim());
        setSampleQueries(Array.isArray(data.sample_queries) ? data.sample_queries : []);
      })
      .catch(() => undefined);
  }, [user.user_id]);

  useEffect(() => {
    apiRequest(`/api/chats/${user.user_id}`)
      .then((data) => {
        const history = (data.chats || []).reverse();
        const restored = history.flatMap((chat) => [
          { role: "user", content: chat.query, id: `${chat.id}-q` },
          { role: "assistant", content: chat.response, id: `${chat.id}-a`, trace: chat.trace, chatId: chat.id }
        ]);
        setMessages(restored);
      })
      .catch(() => undefined);
  }, [user.user_id]);

  useEffect(() => {
    if (!job) return;
    let cancelled = false;
    let terminalStateHandled = false;

    const poll = async () => {
      if (cancelled || terminalStateHandled) {
        return;
      }

      try {
        const status = await apiRequest(`/api/chat/status/${job}`);

        if (cancelled || terminalStateHandled) {
          return;
        }

        setProgress(status);
        if (status.status === "completed") {
          terminalStateHandled = true;
          const result = status.result || {};
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: result.response,
              trace: result.trace,
              chatId: result.chat_id,
              id: `${result.chat_id}-a`
            }
          ]);
          setJob(null);
          setProgress(null);
          return;
        }

        if (status.status === "failed") {
          terminalStateHandled = true;
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `Error: ${status.message}`, id: `${status.id}-err` }
          ]);
          setJob(null);
          setProgress(null);
          return;
        }
      } catch (err) {
        console.error(err);
      }

      if (!cancelled && !terminalStateHandled) {
        setTimeout(poll, 1200);
      }
    };

    poll();

    return () => {
      cancelled = true;
    };
  }, [job]);

  const send = async () => {
    if (!input.trim()) return;
    const query = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: query, id: `user-${Date.now()}` }]);
    try {
      const data = await apiRequest("/api/chat/start", {
        method: "POST",
        body: JSON.stringify({ user_id: user.user_id, query })
      });
      setJob(data.job_id);
      setProgress({ status: "queued", progress: 5, message: "Queued" });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${message}`, id: `error-${Date.now()}` }
      ]);
      setJob(null);
      setProgress(null);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    if (!job) {
      send();
    }
  };

  const todoPlan = useMemo(() => {
    if (!progress) return [];
    if (Array.isArray(progress.todo_plan?.steps) && progress.todo_plan.steps.length) {
      return progress.todo_plan.steps;
    }
    return [];
  }, [progress]);


  const executionPlan = useMemo(() => {
    if (!progress) return [];
    if (Array.isArray(progress.plan?.steps) && progress.plan.steps.length) {
      return progress.plan.steps;
    }

    const fallback = [
      { key: "received", label: "Received request", threshold: 15 },
      { key: "context", label: "Preparing context", threshold: 40 },
      { key: "processing", label: "Processing analysis", threshold: 75 },
      { key: "response", label: "Preparing response", threshold: 100 }
    ];

    return fallback.map((step, index) => {
      const completed = progress.status === "completed" || progress.progress >= step.threshold;
      const inProgress =
        !completed &&
        progress.status !== "failed" &&
        (index === 0 || progress.progress >= fallback[index - 1].threshold);
      return {
        key: step.key,
        label: step.label,
        status: completed ? "completed" : inProgress ? "in_progress" : "pending",
        detail: inProgress ? `Working on: ${progress.message || step.label}` : completed ? "Completed" : ""
      };
    });
  }, [progress]);

  const activeProgressLabel = useMemo(() => {
    if (!progress) return "";
    const currentTodoStep = todoPlan.find((step) => step.status === "in_progress");
    if (currentTodoStep) {
      return currentTodoStep.label;
    }
    const currentStep = executionPlan.find((step) => step.status === "in_progress");
    if (currentStep) {
      return currentStep.label;
    }
    if (progress.status === "failed") {
      return "Analysis failed";
    }
    return progress.message || "Running analysis";
  }, [executionPlan, progress, todoPlan]);

  const activeTodoStep = useMemo(() => {
    if (!todoPlan.length) return null;
    return (
      todoPlan.find((step) => step.status === "in_progress") ||
      todoPlan.find((step) => step.status === "pending") ||
      todoPlan[todoPlan.length - 1]
    );
  }, [todoPlan]);

  const inlineTaskLabel = useMemo(() => {
    if (activeTodoStep?.label) {
      return `Working on ${activeTodoStep.label}`;
    }
    return activeProgressLabel;
  }, [activeProgressLabel, activeTodoStep]);

  return (
    <div className="chat-layout">
      <div>
        <div className="card">
          {capabilities && showCapabilities ? (
            <div className="capabilities-banner" aria-live="polite">
              <div className="capabilities-banner-header">
                <h3>Assistant capabilities</h3>
                <button
                  type="button"
                  className="capabilities-close"
                  onClick={() => setShowCapabilities(false)}
                  aria-label="Close assistant capabilities"
                >
                  ×
                </button>
              </div>
              <p>{capabilities}</p>
              {sampleQueries.length ? (
                <div className="sample-queries">
                  <span>Try asking:</span>
                  <div className="sample-query-list">
                    {sampleQueries.map((query) => (
                      <button
                        key={query}
                        type="button"
                        className="sample-query"
                        onClick={() => setInput(query)}
                      >
                        {query}
                      </button>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
          <div className="chat-window" style={{ marginTop: "24px" }}>
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`message-row ${msg.role === "user" ? "user" : "assistant"}`}
              >
                <div className={`message-avatar ${msg.role === "user" ? "user" : "assistant"}`}>
                  {msg.role === "user" ? "You" : "AI"}
                </div>
                <div className="message-stack">
                  <div className={`message ${msg.role === "user" ? "user" : "assistant"}`}>
                    {msg.role === "assistant" ? (
                      <MarkdownMessage content={msg.content} />
                    ) : (
                      <div className="message-content">{msg.content}</div>
                    )}
                  </div>
                  {msg.role === "assistant" ? (
                    <ResponseActions userId={user.user_id} message={msg} />
                  ) : null}
                </div>
              </div>
            ))}
            {job ? (
              <div className="message-row assistant">
                <div className="message-avatar assistant">AI</div>
                <div className="message assistant">
                  <div className="assistant-progress-inline">
                    <div className="typing-indicator" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </div>
                    <span className="typing-status">{inlineTaskLabel}</span>
                    {typeof progress?.progress === "number" ? (
                      <span className="typing-progress">{Math.round(progress.progress)}%</span>
                    ) : null}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
          <div className="chat-input">
            <div className="chat-input-main">
              <textarea
                placeholder="Describe the issue to analyze..."
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={handleKeyDown}
              />
              <button className="btn btn-primary" onClick={send} disabled={!!job}>
                {job ? "Working..." : "Send"}
              </button>
            </div>
            <div className="chat-input-meta">
              Press Enter to send. Use Shift + Enter for a new line.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState(null);
  const [config, setConfig] = useState(emptyConfig);
  const [showConfig, setShowConfig] = useState(false);
  const [configSection, setConfigSection] = useState("llm");
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    if (!user) return;
    Promise.all([
      apiRequest("/api/config/defaults"),
      apiRequest(`/api/config/${user.user_id}`)
    ])
      .then(([defaults, stored]) => {
        setConfig({
          llm: { ...defaults.llm, ...stored.llm },
          embedder: { ...defaults.embedder, ...stored.embedder },
          langfuse: { ...defaults.langfuse, ...stored.langfuse },
          guardrails: { ...defaults.guardrails, ...stored.guardrails },
          mcp_servers: {
            servers: [
              ...((defaults.mcp_servers && Array.isArray(defaults.mcp_servers.servers)
                ? defaults.mcp_servers.servers
                : [])),
              ...((stored.mcp_servers && Array.isArray(stored.mcp_servers.servers)
                ? stored.mcp_servers.servers
                : []))
            ]
          }
        });
      })
      .catch(() => undefined);
  }, [user]);

  if (!user) {
    return <LoginScreen onLogin={(data) => setUser(data)} />;
  }

  const handleLogout = async () => {
    try {
      await apiRequest("/api/logout", {
        method: "POST",
        body: JSON.stringify({ user_id: user.user_id })
      });
    } catch (err) {
      console.error(err);
    } finally {
      setUser(null);
      setConfig(emptyConfig);
      setShowConfig(false);
      setSettingsOpen(false);
    }
  };

  return (
    <div className="app-shell">
      <main className="container">
        <div className="topbar">
          <div>
            <h1>Conversation workspace</h1>
            <p>Inspect traces and monitor execution in real time.</p>
          </div>
          <div className="topbar-actions">
            <div className="settings-menu">
              <button
                className="btn btn-secondary"
                type="button"
                onClick={() => setSettingsOpen((prev) => !prev)}
              >
                ⚙️ Settings
              </button>
              {settingsOpen ? (
                <div className="settings-dropdown">
                  {configSteps.map((stepItem) => (
                    <button
                      key={stepItem.key}
                      type="button"
                      onClick={() => {
                        setConfigSection(stepItem.key);
                        setShowConfig(true);
                        setSettingsOpen(false);
                      }}
                    >
                      {stepItem.label}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
            <button className="btn btn-outline" type="button" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </div>
        <ChatScreen user={user} />
        {showConfig ? (
          <div className="modal-backdrop" role="dialog" aria-modal="true">
            <div className="modal-card">
              <ConfigWizard
                config={config}
                setConfig={setConfig}
                user={user}
                initialKey={configSection}
                onClose={() => setShowConfig(false)}
              />
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}

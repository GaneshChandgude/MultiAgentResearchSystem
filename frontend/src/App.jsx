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
  }
};

const configSteps = [
  { key: "llm", label: "LLM Configuration" },
  { key: "embedder", label: "Embedder Setup" },
  { key: "langfuse", label: "Langfuse Observability" },
  { key: "guardrails", label: "Guardrails & PII" }
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

  const saveCurrent = async () => {
    setSaving(true);
    setError("");
    const endpointByKey = {
      llm: "llm",
      embedder: "embedder",
      langfuse: "langfuse",
      guardrails: "guardrails"
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
    const list = Array.isArray(trace) ? trace : [trace];
    return list.flatMap((item) => {
      if (!item) return [];
      const messages = item.tool_calls || item.calls || [];
      return [
        {
          agent: item.agent || "Agent",
          messages
        }
      ];
    });
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

  const formatPayload = (payload) => {
    if (payload == null) return "";
    if (typeof payload === "string") return payload;
    try {
      return JSON.stringify(payload, null, 2);
    } catch (err) {
      return String(payload);
    }
  };

  if (!entries.length) {
    return <p>No trace available yet.</p>;
  }

  return (
    <div className="trace-panel">
      {entries.map((entry, index) => {
        const toolCalls = buildToolCalls(entry.messages);
        return (
        <div className="trace-item" key={`${entry.agent}-${index}`}>
          <strong>{entry.agent}</strong>
          {toolCalls.length ? (
            <ul style={{ marginTop: "8px", paddingLeft: "16px" }}>
              {toolCalls.map((call, idx) => (
                <li key={`${call.name}-${call.id || idx}`}>
                  <div style={{ fontWeight: 600 }}>{call.name}</div>
                  {call.args ? (
                    <div style={{ marginTop: "6px" }}>
                      <div style={{ fontSize: "12px", color: "#64748b", fontWeight: 600 }}>
                        Input
                      </div>
                      <pre style={{ whiteSpace: "pre-wrap", marginTop: "4px" }}>
                        {formatPayload(call.args)}
                      </pre>
                    </div>
                  ) : null}
                  {call.output ? (
                    <div style={{ marginTop: "6px" }}>
                      <div style={{ fontSize: "12px", color: "#64748b", fontWeight: 600 }}>
                        Output
                      </div>
                      <pre style={{ whiteSpace: "pre-wrap", marginTop: "4px" }}>
                        {formatPayload(call.output)}
                      </pre>
                    </div>
                  ) : null}
                </li>
              ))}
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
        <div className="trace-inline">
          <TracePanel trace={message.trace} />
        </div>
      ) : null}
    </div>
  );
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
    const interval = setInterval(async () => {
      try {
        const status = await apiRequest(`/api/chat/status/${job}`);
        setProgress(status);
        if (status.status === "completed") {
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
        }
        if (status.status === "failed") {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `Error: ${status.message}`, id: `${status.id}-err` }
          ]);
          setJob(null);
          setProgress(null);
        }
      } catch (err) {
        console.error(err);
      }
    }, 1200);
    return () => clearInterval(interval);
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
                    <div className="message-content">{msg.content}</div>
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
          guardrails: { ...defaults.guardrails, ...stored.guardrails }
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

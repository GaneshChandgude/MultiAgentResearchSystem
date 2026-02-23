# GitHub MCP server troubleshooting

This guide covers the common failure mode where GitHub MCP tool calls start failing after a log like:

```text
[supergateway] SSE connection closed (session <id>)
[supergateway] Client disconnected (session <id>)
```

## What this means

Those lines indicate the Server-Sent Events (SSE) stream between the MCP client and the supergateway was dropped. After that, tool calls can fail because the client session state is gone or stale.

Typical follow-on errors include:

- missing required parameters in MCP calls (`sha`, `path`, etc.)
- task-group cancellation / async scope errors
- tools appearing registered but no longer usable

Another common server-side crash signature is:

```text
Error: Already connected to a transport. Call close() before connecting to a new transport, or use a separate Protocol instance per connection.
```

This specific error usually means the gateway is attempting to attach the **same MCP protocol/server instance** to a new SSE transport after a disconnect.

## Why `Already connected to a transport` happens

In `supergateway` stdio↔SSE mode, each incoming SSE session must map to an MCP server/protocol instance that is not already bound to another active transport. If the previous transport was not fully closed (or the same object is reused across requests), the MCP SDK rejects the second `connect()` call with the error above.

In short:

- old SSE transport disconnected
- gateway reused an already-bound protocol/server object
- new SSE connection tried to call `connect()` again
- SDK threw `Already connected to a transport`

## Concrete fix (server/gateway side)

Focus your changes where supergateway bridges SSE requests to the MCP protocol instance.

1. **Use a fresh protocol/server object per SSE connection**
   - safest approach when sessions are independent
   - avoids cross-session transport reuse
2. **OR ensure explicit close before reconnecting**
   - on SSE close/error, always call `await protocol.close()` (or equivalent)
   - clear references before handling the next connection
3. **Guard connect path**
   - serialize connect/close transitions (mutex/lock) to avoid race conditions
   - reject/queue a reconnect if close is still in progress

Pseudo-pattern:

```ts
// per incoming SSE request/session
const protocol = createProtocol(); // preferred: new instance per session

try {
  await protocol.connect(transport);
  // ... session lifecycle ...
} finally {
  await protocol.close();
}
```

If you keep a shared protocol instance, do this defensively:

```ts
if (protocolIsConnected) {
  await protocol.close();
}
await protocol.connect(nextTransport);
```

## Can I create a new client when this appears?

Yes — **that is the recommended immediate action**.

When you see:

```text
[supergateway] Client disconnected (session <id>)
```

treat that session as dead/stale. Start a **new MCP client session** (or restart your IDE/agent host) so it negotiates a fresh SSE connection and new session id.

Recommended order:

1. Stop the current MCP client/host.
2. Start a new client session.
3. If disconnects continue, restart the GitHub MCP server/supergateway as well (to clear stale transport bindings).
4. Retry the same tool call only after the new session is confirmed healthy.

> Note: The RCA app client now includes automatic reconnect/retry logic for common
> transient SSE disconnect errors, but a fresh client session is still the safest
> manual recovery step when disconnects persist.

## Fast recovery checklist

1. **Restart both sides of the connection**
   - restart your local GitHub MCP server process
   - restart the MCP client/host process (agent CLI/IDE extension/session)
2. **Create a fresh session**
   - avoid reusing a previously disconnected session id
3. **Re-authenticate GitHub credentials**
   - confirm `gh auth status`
   - re-run auth if needed
4. **Verify transport health before tool calls**
   - confirm supergateway is listening on the configured host/port
   - confirm there is no proxy/load-balancer idle timeout closing SSE

## Stability hardening

- Prefer running MCP server and client on the same machine/network segment to reduce connection churn.
- If you use reverse proxies (nginx, cloudflared, corporate gateways), increase idle/read timeouts for long-lived SSE streams.
- Keep one active client per server session when possible to avoid session contention.
- Enable debug logs on both MCP host and server to correlate disconnect time with failing tool requests.
- If you maintain the gateway code, add explicit `close()` logs around disconnect events so you can verify transport teardown completed before the next `connect()`.

## Minimal diagnostics to capture

Collect this before filing an issue:

```bash
# GitHub CLI auth and API sanity
gh auth status
gh api user --jq '{login, public_repos, total_private_repos}'

# Local process + port checks (adjust names/ports)
ps aux | rg -i "mcp|supergateway|github"
ss -ltnp | rg ":<your_mcp_port>"
```

Also include:

- exact server start command
- MCP host/client version
- GitHub MCP server version/commit
- full log window around the first `SSE connection closed`

## Workaround when MCP is unstable

Until SSE stability is fixed, use direct GitHub CLI/API calls for repo queries:

```bash
# public repo names
 gh repo list <org_or_user> --public --limit 200 --json name --jq '.[].name'

# total repos for authenticated user (public + private)
 gh api user --jq '.public_repos + .total_private_repos'
```

This bypasses the MCP transport layer while preserving access to GitHub data.

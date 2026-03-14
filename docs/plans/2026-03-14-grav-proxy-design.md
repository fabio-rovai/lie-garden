# GravProxy: Transport-Level Confinement Enforcement

**Date:** 2026-03-14
**Status:** Approved
**Threat model:** LLM attempts to bypass confinement (prompt injection, jailbreak, ignoring tools)

## Problem

GravRail's confinement guarantee (text -> algebra -> exp -> on-group) is currently opt-in.
The LLM must voluntarily call `grav_step`. If it doesn't, outputs are unconfined.
An "MCP that forces the MCP" has a chicken-and-egg problem.

## Solution

A transport-level proxy that sits between the LLM and the consumer. Every LLM response
physically passes through the confinement pipeline before reaching the outside world.
The LLM has no bypass path -- it doesn't even know the proxy exists.

## Architecture

```
+----------+     +--------------+     +----------+
|  LLM     |---->|  GravProxy   |---->| Consumer |
|  (any)   |     |              |     | (app)    |
+----------+     | +----------+ |     +----------+
                 | |Confine   | |
                 | |Pipeline  | |     +----------+
                 | +----------+ |<--->|Watchdog  |
                 | +----------+ |     |(heartbeat)|
                 | |STARK     | |     +----------+
                 | |Proof Gen | |
                 +--------------+
```

### Components

1. **GravProxy** -- local TCP proxy. Binds a port, accepts consumer connections.
   Forwards requests upstream to the LLM. Intercepts every response, runs it through
   `map_to_algebra -> constrain -> exp -> multiply`. Only forwards confined output.
   Blocks anything that violates circuit reachability bounds.

2. **Watchdog** -- separate thread. Exchanges signed heartbeat pings with the proxy
   every 1 second over a Unix domain socket. Mutual liveness check: both must be
   alive for output to flow.

3. **STARK proof attachment** -- every proxied response gets a confinement proof
   attached as metadata. Consumer can independently verify confinement.

## Enforcement Rules

### Rule 1: No raw passthrough
Every LLM response passes through the confinement pipeline. No bypass mode,
no debug flag, no environment variable skips this. The code path from input
to output always includes `constrain -> exp -> multiply`.

### Rule 2: Fail closed
If anything goes wrong (proxy crash, watchdog timeout, confinement error,
malformed response), the output channel is killed. Consumer gets nothing.
Silence is safe; unconfined output is not.

### Rule 3: Mutual liveness
Proxy and watchdog verify each other via heartbeats. If either stops responding
for >3 seconds, both shut down and the upstream LLM connection is terminated.
Restarting requires explicit human action -- no auto-reconnect.

## Anti-Disconnect Mechanisms

### 1. Heartbeat protocol
Proxy and watchdog exchange HMAC-signed pings every 1 second over a Unix domain
socket. The HMAC key is derived at startup and exists only in memory.
Spoofing requires the key.

### 2. Dead man's switch
Every write to the consumer includes a monotonic sequence number. If the consumer
detects a gap or silence >3s, it knows the proxy died and must treat all
subsequent data as unconfined.

### 3. PID lock file with flock
Proxy writes PID to `~/.gravrail/proxy.lock` with exclusive file lock.
If the proxy crashes, OS releases the lock. Watchdog checks the lock --
if released, kills upstream. Prevents shadow proxies.

### 4. Startup authentication
Proxy generates a one-time session token at startup, prints to stderr.
Consumer must present this token to connect. Prevents fake proxy substitution.

### Failure Matrix

| Event                        | Response                                              |
|------------------------------|-------------------------------------------------------|
| Proxy crashes                | Watchdog detects missing heartbeat, kills upstream    |
| Watchdog crashes             | Proxy detects missing heartbeat, shuts down           |
| Someone kills proxy PID      | flock released, watchdog kills upstream                |
| Network glitch to LLM        | Proxy retries upstream, consumer sees nothing until OK |
| Attacker starts fake proxy   | Consumer rejects -- wrong session token                |
| Prompt injection             | Irrelevant -- confinement is in proxy, not LLM        |

## Data Flow

### Startup
```
1. gravrail proxy --circuit <id> --upstream <llm_url> --port 8340
2. Proxy generates session token, prints to stderr
3. Proxy spawns watchdog thread
4. Mutual heartbeat begins
5. Proxy binds port 8340, waits for consumer
```

### Request flow
```
Consumer -> POST localhost:8340/v1/chat/completions
  |
  +- Proxy validates session token (first request only)
  +- Proxy forwards request to upstream LLM unchanged
  |
LLM responds (streaming or non-streaming)
  |
  +- For each chunk/message:
  |   1. Extract text content
  |   2. map_to_algebra(text, circuit.algebra_dim, scale)
  |   3. circuit.constrain_algebra(coeffs)
  |   4. group.exp(constrained) -> step element
  |   5. group.multiply(state, step) -> new state
  |   6. Reachability check: is new state within bounds?
  |   |
  |   +- YES: forward with proof metadata
  |   |   x-gravrail-state: [matrix elements]
  |   |   x-gravrail-seq: 42
  |   |   x-gravrail-proof: <stark proof hash>
  |   |
  |   +- NO: block, return confinement_violation error
  |
  +- Update agent state, record lineage event
```

### Streaming
For SSE streaming responses, proxy buffers each `data:` chunk, confines it,
then forwards. Consumer sees the same streaming interface with ~2us latency
per chunk (confinement pipeline overhead).

### API compatibility
Proxy speaks OpenAI-compatible chat completions API. Any client that talks
to OpenAI can point at the proxy by changing the base URL. No code changes
needed on the consumer side.

## What Gets Confined

- **Text content** in LLM responses: run through full confinement pipeline
- **Confined state** attached as response header/metadata
- **STARK proof** generated per response (or batched per N for performance)

## What Passes Through Unchanged

- Non-content fields (token counts, model name, finish reason)
- The original text IS forwarded -- confinement certifies it, doesn't rewrite it
- If text maps to out-of-bounds state, it's blocked entirely (not modified)

## CLI Interface

```bash
# Start proxy
gravrail proxy \
  --circuit <circuit_id> \
  --upstream http://localhost:11434/v1 \
  --port 8340 \
  --heartbeat-interval 1s \
  --heartbeat-timeout 3s

# Consumer connects
curl http://localhost:8340/v1/chat/completions \
  -H "X-GravRail-Token: <session_token>" \
  -d '{"model":"...","messages":[...]}'
```

## Implementation Modules

| Module | Responsibility |
|--------|---------------|
| `proxy/server.rs` | Axum HTTP server, request forwarding, response interception |
| `proxy/confine.rs` | Per-response confinement pipeline (map -> constrain -> exp -> multiply) |
| `proxy/watchdog.rs` | Heartbeat protocol, mutual liveness, PID flock |
| `proxy/stream.rs` | SSE chunk buffering and confined forwarding |
| `proxy/auth.rs` | Session token generation and validation |
| `proxy/cli.rs` | CLI subcommand `gravrail proxy` |

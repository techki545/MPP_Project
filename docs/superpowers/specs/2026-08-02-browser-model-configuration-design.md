# Browser Model Configuration Design

## Goal

Allow a user of the local SMPP EvidenceGraph web application to enter an
OpenAI-compatible API base URL, API key, and chat model name in the browser and
use that configuration for one evidence query without persisting the secret.

This configuration controls only the chat model used for evidence refinement,
claim extraction, and report generation. It does not change the embedding model
or rebuild the existing vector index.

## User Experience

The clinical-query band gains a compact "model configuration" section with:

- API base URL input.
- API key password input with an icon control to show or hide the value.
- Chat model name input.

The three inputs remain in JavaScript memory only. The application must not use
cookies, `localStorage`, `sessionStorage`, IndexedDB, URL parameters, or hidden
form persistence for these values. Refreshing or closing the page clears them.

All three values are optional as a group. When all are empty, the server's
environment-based model configuration remains active. When any one is entered,
all three are required. The query status area reports incomplete configuration
before any network request is made.

The header model-status pill reflects both sources without exposing secrets:

- Browser configuration ready: display the user-entered model name.
- No browser configuration: display the server model status returned by the
  existing status endpoint.

## Request Contract

`POST /api/query` and its compatibility alias `POST /api/analyze` accept an
optional `model_config` object:

```json
{
  "base_url": "https://provider.example/v1",
  "api_key": "secret",
  "model_name": "chat-model"
}
```

The browser includes this object only when all three fields are present. The API
does not echo it in successful responses or errors.

The server validates:

- `model_config` is an object containing only the three supported keys.
- Values are strings, trimmed, non-empty, and within conservative length limits.
- Public provider URLs use HTTPS.
- Plain HTTP is accepted only for loopback hosts such as `localhost`,
  `127.0.0.1`, and `::1`.
- URLs do not contain user information, query parameters, or fragments.

Legacy top-level credential fields remain rejected so there is one unambiguous
request contract.

## Server Architecture

`GraphRAGWebService` validates the optional request configuration and passes a
sanitized model configuration to `KnowledgeBaseService.query`.

For a configured request, `KnowledgeBaseService` creates an ephemeral
`ChatClient`. `ProductionQueryPipeline` runs with request-scoped evidence
refinement, claim extraction, and report generation components backed by that
client. The existing retriever, document store, and graph builder are reused.
The request-scoped client is not assigned to application-global state and
becomes unreachable after the query finishes.

When no request configuration is supplied, the existing server-configured
pipeline is used unchanged. Demo fallback behavior remains available when the
local knowledge base is not ready, but the API still validates and never stores
the submitted secret.

## Secret Handling

The API key must never appear in:

- Application logs or HTTP access logs.
- Exceptions returned to the browser.
- Health or model-status responses.
- Query results, graph data, source data, or build jobs.
- Files, SQLite, vector storage, environment variables, or process-global
  caches.

Provider failures are mapped to stable messages. Authentication failures report
that the token or model permission should be checked. Request rejection,
temporary unavailability, timeout, and invalid response format remain separate
error categories without provider response bodies.

## Error Handling

Browser validation handles empty and partially completed forms. Server
validation independently enforces the same contract because clients can bypass
the browser.

The backend returns `400` for malformed or insecure model configuration. Model
provider failures continue through the current grounded-report fallback path
when possible. The report must identify that the model output was not adopted
and must not claim a model-backed result.

## Testing

Backend tests cover:

- Complete request configuration reaches the knowledge service as a sanitized
  object.
- Empty configuration uses the existing server pipeline.
- Partial, unknown, incorrectly typed, overlong, and insecure URL values fail.
- Loopback HTTP and public HTTPS are accepted.
- API keys do not occur in response or serialized error text.
- A request-scoped client is used by all three chat-backed stages without
  mutating the default pipeline.

Frontend tests cover:

- The three controls and API-key visibility control exist and have accessible
  labels.
- Complete values produce `model_config` in the query payload.
- Empty values omit it.
- Partial values block submission with a Chinese error message.
- No browser persistence API is used.

Final verification includes the Python suite, JavaScript syntax check, desktop
and mobile browser interaction, model-field refresh clearing, and a live query
using a non-secret fake transport in automated tests.

## Out of Scope

- Selecting or changing the embedding model from the browser.
- Persisting provider profiles or API keys.
- Sharing credentials among users.
- Listing provider models automatically.
- Adding user accounts or server-side sessions.

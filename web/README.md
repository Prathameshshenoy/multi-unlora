# Multi-UnLoRA interface

This is a buildless, dependency-free static frontend. The Python demo API serves it
from the repository's `web/` directory. For a separate static host, publish
`index.html`, `styles.css`, and `app.js` together at the same directory level.

Use **Connect** to configure an inference API. The default is the page's own
origin. Remote APIs must use HTTPS; local HTTP is supported for `localhost`,
`127.0.0.1`, and `[::1]`. Configure the API's allowed origins for a separate
frontend host. Only the API URL is persisted in browser local storage. An optional
bearer token stays in the page's password input for the current session.

The visualizer never synthesizes model responses or measurements. An explicit
`preview` backend is labelled as simulated. Unavailable inference disables Run.

## API contract

- `GET /api/health`: `{status, backend, model, ready, detail}`.
- `GET /api/examples`: `{examples: [{id, title, category, prompt}]}`.
- `POST /api/generate`: JSON `{prompt, max_new_tokens}`; returns an SSE stream.

Events:

| Event | Payload |
| --- | --- |
| `metadata` | `{backend, model, simulated, request_id}` |
| `routing` | `{alpha_hall, alpha_jail}` |
| `token` | `{branch: "baseline" \| "unlora", text}` |
| `telemetry` | `{step, kl_divergence, attenuation, top_tokens: [{token_id, token, base_probability, corrected_probability}]}` |
| `complete` | `{baseline: {tokens, wall_time_s}, unlora: {tokens, wall_time_s}, simulated}` |
| `error` | `{message}` |

The stream parser supports partial UTF-8 sequences, LF/CRLF event delimiters,
comments, and multiline data. Stop aborts the fetch. Text from model outputs and
the API is inserted with `textContent`. Ctrl/Cmd+Enter runs a comparison.

Validation: `node --check web/app.js`.

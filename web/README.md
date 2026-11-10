# Multi-UnLoRA interface

Static HTML/CSS/JS served by the local FastAPI application. The API defaults to
this page's origin. The local remote-GPU bridge keeps its access token on the
server. Optional browser connection settings persist only the HTTPS API URL;
entered visitor tokens remain in page memory.

## Streaming contract

- `GET /api/health`: readiness, cold/warm state, model identity, quality status,
  supported branch names, and any session expiry. Requires bearer authentication
  when the service has configured an access token.
- `GET /api/examples`: safe starting prompts.
- `POST /api/generate`: `{prompt,max_new_tokens}` with a 1–256 output-token budget.

Branch IDs are `baseline`, `hallucination`, `jailbreak`, and `unlora`. Each arm is
a separate decode of the same prompt. The final arm applies anti-expert **logit
differences**; it does not combine/subtract the three displayed response strings.

| Event | Main fields |
| --- | --- |
| `metadata` | `backend`, `simulated`, `model`, `branches`, `decoder_strategy`, `quality`, display-policy status |
| `routing` | `alpha_hall`, `alpha_jail` |
| `branch_status` | `branch`, `status`, `text`, `source`, `withheld`, `reason` as applicable |
| `token` | `branch`, `text` |
| `telemetry` | step, actual KL/budget, strengths, attenuation and permitted candidate probabilities |
| `complete` | one result per all four branch IDs, plus `simulated` |
| `error` | sanitized `message` |

Each complete result includes text, generated token count, wall time, throughput,
finish reason, and presentation provenance. `source=model` means generated text;
`source=withheld` means raw output was not displayed; `source=display_guard` means
a separate guard supplied the displayed refusal. A guard refusal is never a
successful model-subtraction result or an ASR measurement. Timings/token counts
refer to generation, even when raw text is withheld afterward.

## Restricted requests

The conservative local presentation policy recognizes weapon/explosive terms.
It is not a comprehensive intent classifier. Recognized restricted requests
skip standalone raw expert-text generation, while their actual adapter logits
still participate in the corrected decode. Base/corrected text is buffered;
only approved complete short model refusals or an explicitly labelled guard
fallback may be displayed. Raw expert tokens/candidates are also withheld.

The production safety gate remains open: independent moderation and harmful
compliance/benign over-refusal evaluation are required before public deployment.

## UI behavior

Four responsive cards show real branch text or an explicit withholding reason.
Outputs use `textContent`; SSE parsing handles split UTF-8 and event boundaries.
Stop aborts the stream; request cleanup discards per-adapter caches and restores
the base state. Ctrl/Cmd+Enter starts a comparison. Output limits, quality status,
and session expiry remain visible. A preview backend is explicitly simulated.

Syntax check: `node --check web/app.js`.

Expert names identify training targets, not guaranteed or mutually exclusive
behaviors. Standalone `branch_status`/completion records include `adapter_id`;
its value is `hallucination` or `jailbreak` for the corresponding reference arm.
The base and contrastive arm decode base logits (`adapter_id: null`); the
contrastive arm separately queries both expert distributions. This provenance
must not be changed merely to make a displayed response fit a role label.

# POC issue audit

Reviewed 9 November 2026 against the private research blueprint and the current
implementation. The blueprint itself remains local and ignored. This audit
separates confirmed defects from proposed research mechanisms and claims that
require experiments. A published proposal is not evidence of implementation;
a passing tensor test is not evidence of improved answer quality.

**Status:** fixed = code and a regression check exist; partial = the supported
path is repaired but a legacy or integration gap remains; remaining = confirmed
engineering debt; unvalidated = needs empirical/theoretical evidence, not an
invented success claim. Historical slide figures cannot be reconstructed
without the original weights, exact prompts, and result records.

## Consolidated POC defects

The D identifiers correspond to the blueprint's consolidated debt list.

| ID / section | Confirmed issue | Current status and file evidence | Acceptance criterion / remaining work |
| --- | --- | --- | --- |
| D1 / 1.2, 5.11 | Baseline and correction had different prompts. | **Fixed for supported paths.** `inference/prompts.py` supplies one native template; `evaluation/quality.py`, `evaluation/tuning.py`, and `demo/backends.py` use it. `evaluation/runners.py` rejects legacy protocol CSVs. | Every comparison arm must render byte-identical prompt tokens; preserve old confounded results as incompatible, never mix them into new reports. |
| D2 / 4.6 | Anti-experts recomputed the complete prefix at every token. | **Implemented and integrated; real-model parity/measurement gate remains.** `inference/decoder.py::CacheDecoder` owns an independent base and per-adapter KV cache; `use_cache=False` is the oracle. `tests/test_decoder_cache.py` compares logits and corrected token choices over 50 tiny-HF/PEFT prompts. | The four-output GPU smoke test exercised the integrated cache decoder. Verify cached/uncached parity on the real model under an identical timing protocol before making a comparative latency claim. Legacy processor-only callers remain uncached unless given a provider. |
| D3 / 4.2 | Absolute normalized expert logits were subtracted instead of the adapter delta; base normalization cancelled. | **Fixed.** `inference/processor.py` forms expert-minus-base deltas; `inference/operator.py` operates in raw logit units. Old behavior is named `inference/legacy_processor.py`, an ablation only. | Identical expert/base distributions and zero weights must be exact no-ops. `tests/test_operator.py` and real PEFT tests cover this. |
| D4 / 4.2 | Expert top-k filtering could promote base-implausible tokens. | **Fixed for active correction.** Pure operator filters by relative base plausibility. If filtering itself exceeds the KL budget, it explicitly abstains and returns base scores. | Off-mask tokens cannot be selected on a non-abstaining correction. Support restriction is not a capability or safety guarantee. |
| D5 / 4.2 | KL was summed across batch and a single scalar guard affected every row. | **Fixed.** `inference/operator.py` solves a measured reverse-KL budget independently per row; tests compare batched and individual results. | Randomized finite distributions obey the stated numerical tolerance; no claim of an unproved analytic certificate. |
| D6 / 1.2, 6.5 | The evaluated model judged its own truthfulness. | **Partial.** `evaluation/judge.py` labels legacy self-judging exploratory and parses only a final exact verdict; quality gates do not treat it as independent evidence. | Add a named/versioned external judge, published rubric, human agreement and sensitivity checks before free-form truthfulness/compliance claims. |
| D7 / 5.5 | Fixed result paths overwrite configurations. | **Partial.** New bounded jobs preserve frozen source snapshots and named JSON evidence; legacy `modal_app.py` CSV constants and fixed report destinations still exist. | Every distinct experiment configuration needs its own immutable run directory; aggregation must reject incompatible identities. |
| D8 / 5.5 | Mutable module configuration forces source edits and prevents independent concurrent arms. | **Remaining.** `config.py` is still used globally. The pure operator and v3 grid accept explicit parameters, avoiding mutation for those sweeps. | Frozen run-scoped configuration and identity hash, CLI overrides recorded verbatim, no mutation between concurrent arms. |
| D9 / 4.6 | Adapter mutations could leak after failures. | **Partial, supported decoder/processor repaired.** `inference/adapters.py::active_adapter` centralizes exception cleanup and optional exact prior-state restoration; decoder drops uncertain caches after errors. `training/trainer.py` unloads adapters on constructor/train/save failures. | `evaluation/runners.py` reference branches now also enter the context boundary, with a tokenizer-failure regression; migrate remaining orchestration-only direct mutations; add a guard for supported modules. Real PEFT exception tests must remain green. |
| D10 / 5.9 | No tests or CI. | **Fixed as infrastructure.** `tests/` and `.github/workflows/checks.yml` now cover numerical behavior, actual tiny PEFT state/cache behavior, artifact identity, API streaming, cancellation, validation selection and budget controls. | CI must include new decoder and training regressions; broader research evaluation remains separate from unit-test success. |
| D11 / 4.7 | Unused configuration was echoed as if it affected results. | **Fixed for identified inactive fields.** Unused maximum-alpha and penalty-threshold fields were removed from configuration and legacy manifests. Historical top-k/decay settings are explicitly named `LEGACY_PROCESSOR_*` and only used by the legacy ablation. | Remove unused values from resolved active-run manifests, or explicitly place them under a legacy schema; never hash inactive settings as operating-point evidence. |
| D12 / 4.3, 5.6 | Hallucination anchors differ in surface form and reuse assertive training prefixes. | **Remaining; detector validity unproven.** `repe/anchors.py` still contains heterogeneous forms. New pilot training uses raw responses, while the legacy formatter retains its documented assertive-prefix recipe. | Version matched-form anchor records, test balance, explicitly recalibrate, then compare detector behavior/AUROC on held-out data. Do not silently replace vectors behind an existing report. |
| D13 / 5.11 | Refusal matching accepted a phrase anywhere while claiming prefix semantics. | **Fixed for identified function/documentation mismatch.** `evaluation/analysis.py::check_refusal` now checks response prefixes and labels the result an unvalidated proxy. | Remove its stale claim that runners call it; report refusal rate separately from independently judged harmful compliance. |
| D14 / 6.1 | A single unconstrained generated token was the entire capability test. | **Partial.** `evaluation/tuning.py` v3 symmetrically conditions both arms on answer labels before scoring; failed v2 is preserved. | Add generative capability tasks and full multi-token evaluation. Conditional-label results alone cannot establish general capability retention. |
| D15 / 5.11 | Resume used row count, silently mixing datasets/settings. | **Fixed for legacy runner identity.** `evaluation/runners.py` binds examples, settings, protocol, operator and manifest to an evaluation fingerprint before accepting resume. | Add incomplete-row handling and immutable run destinations; tokenizer/model/vector revisions must remain bound in supported run provenance. |
| D16 / 4.6 | The mechanism hard-codes exactly two concepts. | **Partial.** New decoder isolates caches by arbitrary adapter name, and the pure operator accepts a combined delta independent of concept count. Public router, processor and reports still name two concepts. | Typed concept registry and generalized strengths/telemetry; demonstrate four concepts only after suitable data and a funded experiment. |
| D17 / 5.10 | Unsloth inference required import-order hacks and fused-layer workarounds. | **Fixed for supported runtime.** `inference/runtime.py` and current pilot use plain HF/PEFT. Legacy `model_loader.py` is explicitly a compatibility training loader. | Historical Unsloth-to-HF parity remains unvalidated without the missing original adapter artifacts; current adapters are trained directly in HF. |
| D18 / 5.11 | Local Python requirement and GPU image versions disagree. | **Fixed as explicit supported profiles.** Project metadata now supports Python 3.11–3.14, local default remains 3.14, the pinned GPU image uses 3.11, and CPU CI checks both endpoints of the range. | Define/pin separate supported runtime profiles or unify compatible requirements, then validate lockfiles and CI on the declared versions. |
| D19 / 5.11 | Accidental platform files and unused helpers/imports. | **Partially fixed.** No `.DS_Store` is tracked and it is now ignored; unused trainer `flush_vram` import is removed. | Keep platform files ignored and remove unneeded legacy hardware helpers when migrating their callers; avoid claiming an unused utility alone affects model accuracy. |

## Additional confirmed defects and integration constraints

| Section / issue | Evidence and current status | Acceptance criterion |
| --- | --- | --- |
| 4.3 calibration/use template mismatch | **Fixed.** Calibration, quality evaluation and serving use the same native chat formatter. `repe/extractor.py` and `inference/runtime.py` bind prompt protocol and calibration provenance. | Prompt, system-message, model revision, anchor content, layer or context changes require explicit recalibration. Visitor requests never recalibrate. |
| 4.6 silent batch-size-one assumptions | **Partial.** Operator supports independent rows; cached decoder validates full mask shapes and final-token support. Router is a single-prompt interface. | Explicitly reject a list/batch at the single-prompt extraction boundary, or add a real batched router; never silently take the first row. |
| 4.6 cache contamination across adapters/branches | **Fixed in the new primitive.** Cache state is keyed by adapter; exact token/mask prefix extension is required; reset separates output branches. | Test prompt/mask changes, repeated prefixes, failure invalidation and three distinct mutable cache objects; never fork the base KV cache into an expert. |
| 5.9 untested numerical edge cases | **Fixed for identified cases.** Float32 overflow is rejected, additive logit shifts are centered away, zero budget records abstention, and telemetry excludes masked zero-mass candidates. | Preserve finite JSON telemetry and numerical-budget tests across installed stack updates. |
| Training cleanup | **Fixed in legacy trainer.** Cleanup now runs after trainer construction, optimization or save errors, and unloads injected wrappers instead of merely deleting adapter parameters. | `tests/test_training_regressions.py` must prove cleanup on all three failures without masking the original error. |
| Training-data reproducibility | **Fixed in legacy formatter.** Prefix choice is derived from seed and example identity, not process-global random state. | Identical examples produce identical formatted data regardless of worker scheduling or unrelated RNG calls. This does not validate the prefix recipe itself. |
| Training data bounds/types | **Fixed for confirmed cases.** Builders cap to available rows; invalid answers are rejected; apostrophe contractions are recognized; empty filtered sets fail explicitly. | Test smaller-than-requested datasets, null/non-string fields, empty filtered data and invalid size settings. |
| Artifact existence mistaken for compatibility | **Fixed in supported runtime.** `inference/runtime.py` validates immutable model revision, prompt protocol and adapter bytes before loading, and rejects mismatched calibration provenance. | Future manifests also bind adapter configuration hashes; original pilot configuration integrity cannot be retroactively asserted. |
| Dataset semantics | **Partially verified by the real pilot.** Both anti-experts increased held-out undesirable-response likelihood; source/split provenance is checked. | Publish a dataset-schema/source audit and inspect appropriately sampled records. Positive NLL changes do not replace behavioral hallucination/compliance scoring. |
| Historical results | **Unreproducible with currently available artifacts.** Original slide weights/CSV/prompt details are unavailable. | Reproduce from actual artifacts or clearly label as historical unverified figures; never fabricate those numbers or tune until they match. |

## Experimental acceptance still required

The local master blueprint contains further research and architecture proposals.
This public audit does not reproduce those unpublished hypotheses. The remaining
acceptance work relevant to the current implementation is:

- Independent free-form factuality, harmful-compliance and benign over-refusal
  evaluation, with a named judge/rubric and human agreement checks.
- Validation of the router against fixed-weight baselines and matched-form data;
  correction weights are not calibrated risk probabilities.
- Immutable experiment configuration and run destinations across the legacy
  orchestration paths, plus complete artifact/version provenance.
- Separate measured capability/quality results for any new decoding or training
  policy. Passing numerical tests does not establish a scientific contribution.

## Scope of current acceptance

CPU fixes can establish deterministic software behavior and eliminate known
confounds. They cannot establish scientific novelty, answer accuracy, human
agreement or cross-model scaling without the corresponding artifacts and runs.
The current GPU authorization is capped; this audit does not authorize another
training run, judge expenditure, deployment or benchmark campaign.

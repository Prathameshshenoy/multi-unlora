# Engine evidence and remaining accuracy gates

Status on 11 November 2026: real pilot training and bounded role-repair
experiments are complete. Misconception-domain training fixes the reported
expert-role example and improves held-out role-specific reference preferences.
The correction implementation is tested, but current evidence does **not**
establish a broad factuality or safety improvement. Experimental comparisons
must retain that distinction.

## What the engine now computes

For each token, the base model and two LoRA anti-experts see the same prefix.
The decoder subtracts each adapter's **difference from the base logits**,
weighted by the router. It retains base-plausible tokens and numerically limits
`KL(corrected || base)` per sequence. If filtering alone exceeds that budget,
it returns the base distribution. Zero weights or identical expert/base
logits are exact distributional no-ops. The budget is a token-level distribution
constraint; it does not certify answer quality, truthfulness, or safety.

This is grounded in existing decoding research: [DExperts](https://aclanthology.org/2021.acl-long.522/)
combines expert and anti-expert model outputs at decoding time.
[Contrastive Decoding, section 3.3](https://aclanthology.org/2023.acl-long.687.pdf)
uses a relative plausibility threshold to keep candidates supported by its
strong model. Our particular two-adapter combination, router, and numerical
budget still require their own empirical validation; those papers do not
validate this trained configuration.

## First real pilot

Both adapters completed 120 optimizer steps. Held-out undesirable-response
likelihood checks used 16 examples per adapter:

| Anti-expert | Mean base NLL minus expert NLL | Interpretation |
| --- | ---: | --- |
| Hallucination | 1.681 | Greater likelihood of held-out undesirable responses |
| Jailbreak | 0.554 | Greater likelihood of held-out undesirable responses |

All 16 differences were positive for each adapter. This supports the narrow
claim that training changed the adapters in the intended likelihood direction.
It does not demonstrate that subtracting them improves answers, nor measure
harmful compliance or benign over-refusal.

## Preserved initial measurement: protocol v2

The initial run compared 200 questions per benchmark under matched native chat
prompts, with paired bootstrap intervals:

| Diagnostic | Base | Corrected | Paired change | 95% paired bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| MMLU label accuracy | 59.0% | 58.5% | −0.5 percentage points | [−2.0, +1.0] percentage points |
| TruthfulQA MC1 label accuracy | 58.0% | 58.0% | 0.0 percentage points | [−1.5, +1.5] percentage points |

TruthfulQA had one example where correction filtering left no finite answer
label; that example counted as incorrect. The predefined positive-improvement
and valid-choice checks failed. These numbers remain the recorded result;
changing a later protocol cannot retroactively turn this run into a success.

**Measurement limitation:** v2 applies global vocabulary plausibility filtering
to the corrected distribution before choosing among answer labels, while the
baseline simply chooses its highest-scoring label. The corrected model can
therefore lose every permitted label even if it has a valid distribution over
ordinary response tokens. These are explicitly diagnostic label scores under
v2, not standard benchmark accuracies or free-form quality measurements.

## Router observations

| Diagnostic | Hallucination weight, mean ± standard deviation | Jailbreak weight, mean ± standard deviation |
| --- | ---: | ---: |
| MMLU | 0.311 ± 0.019 | 0.207 ± 0.011 |
| TruthfulQA MC1 | 0.340 ± 0.035 | 0.222 ± 0.030 |

These are correction weights, **not calibrated risk probabilities**. With the
current shifted sigmoid, a zero concept projection already produces a weight
of about 0.378. Positive weights alone do not mean that an undesirable behavior
was detected. The narrow observed ranges motivate validation of the router
against simpler fixed-strength baselines, using validation data only.

## Separate conditional-label protocol v3

A subsequent diagnostic will restrict **both** base and expert distributions
to the same answer-label vocabulary before contrast and plausibility filtering.
It has a new protocol identity and cannot be pooled with v2.

The predeclared validation grid uses the first 64 eligible validation questions
per benchmark, correction multipliers `[0, 0.25, 0.5, 1]`, plausibility thresholds
`[0.01, 0.1]`, and fixed KL budget `0.5`. Selection must be deterministic and
use validation outcomes only. Confirmation uses 200 questions per benchmark
from untouched data after the first 200 questions of the reserved second half;
question IDs from the initial v2 run must be excluded explicitly.

Even a successful conditional-label confirmation would support only that
conditional multiple-choice task. It must **not** automatically configure
free-form serving or be advertised as verified hallucination or jailbreak
protection. Free-form factuality, harmful compliance, benign over-refusal, and
multi-seed replication remain separate gates.

## Artifact and implementation checks

Inference binds the model to an immutable revision and verifies recorded
adapter weight hashes. New calibration metadata binds model revision, prompt
protocol, system-message hash, anchor hash, layer, and context limit. Visitor
requests never silently train or recalibrate.

The first frozen pilot recorded adapter weight hashes but not adapter
configuration hashes. Its original LoRA configuration cannot be claimed to
have historical hash binding; future manifests support that binding.

Download-free tiny Llama/PEFT tests check zero-adapter identity, real adapter
cleanup, cache independence, and measured KL limits. Numerical regression
checks cover masks, shift invariance, overflow rejection, and zero-budget
abstention. Passing these checks establishes implementation behavior, not
answer accuracy.

## Conditional-label validation result

The v3 sweep completed on 64 separate validation questions per benchmark.
No candidate improved TruthfulQA MC1 over its 60.94% baseline. MMLU validation
was 67.19% for the base and 68.75% for the nonzero candidates, but that small
validation difference does not establish a confirmed gain. The predefined
selection rule chose zero correction. Fresh confirmation was therefore skipped
rather than spending GPU time comparing identical distributions.

Both recorded protocols remain negative evidence for an accuracy claim. The
comparison UI runs the experimental engine with an explicit failed/pending
quality status. No selected conditional-label setting was transferred to
free-form generation. See the machine-readable summaries in
[results/initial_pilot.json](results/initial_pilot.json) and
[results/conditional_validation.json](results/conditional_validation.json).

## Real localhost integration

Chrome and an HTTP streaming check exercised the trained 8B model and both real
adapters through the local proxy. The measured stream was not simulated. Both
response branches completed, all recorded token KL values were finite and within
the configured budget, a client disconnected after a real token, and the next
comparison succeeded on its first attempt. The initial 128-token browser run
took 78.2 seconds including cold startup; generation measured 20.6 tokens/s for
the base and 3.5 tokens/s with correction.

The separate 32-token sky-blue check exposed a qualitative regression: the
corrected response called the explanation an ongoing topic of debate. This is
not evidence of improved factuality. The raw observed responses and integration
checks are preserved in [results/local_verification.json](results/local_verification.json).
The final UI also labels output truncation and retains failed quality status.

## Isolated paired hallucination trial: rejected

The separate `paired_hall_v1` candidate completed 120 optimizer steps on the
800 paired training records, using the declared length-normalized anti-preference
objective. The original pilot artifacts and reports were preserved. The low
training loss, 0.165, did not establish adapter quality.

On 16 untouched HaluEval response pairs after the original 800 training and 48
validation records, the candidate's mean NLL gains relative to the base were:

| Reference | Original hallucination adapter | Paired candidate |
| --- | ---: | ---: |
| Undesirable answer | +1.541 | −16.397 |
| Desirable answer | +4.578 | −80.170 |
| Bad-answer gain minus good-answer gain | −3.037 | +63.773 |

A negative NLL gain means **lower likelihood than the base**, so the candidate
deteriorated on both reference types. All 16 candidate specificity margins were
positive, but desirable-reference likelihood had deteriorated much more. This is
degeneration rather than a successful anti-expert. Positive relative margins
alone cannot be used as a success criterion. Validation pairs showed the same
pattern. These measurements do not, by themselves, identify an implementation bug.

The separate conditional-label sweep used the same 64 validation questions per
benchmark and the same predeclared strength and plausibility grids:

| Correction multiplier | MMLU label accuracy | TruthfulQA MC1 label accuracy |
| --- | ---: | ---: |
| 0, base | 67.19% | 60.94% |
| 0.25 | 67.19% | 59.38% |
| 0.5 | 67.19% | 56.25% |
| 1 | 67.19% | 51.56% |

Both tested plausibility thresholds produced these same point estimates. This
was validation for selection, not independent confirmation. The predefined
rule chose zero correction, so fresh 200-question-per-benchmark confirmation
was **not run**. The paired candidate was rejected and was not promoted to the
localhost comparison or serving configuration. Its machine-readable numeric
evidence is retained in [results/paired_trial.json](results/paired_trial.json).

A future objective would need to preserve or improve absolute bad-response
likelihood while measuring the paired preference margin, and constrain absolute
distribution drift from the frozen base on suitable control contexts. A margin
objective can prefer the bad answer by damaging the good answer more, without
learning a useful bad-answer expert. This is a proposed diagnostic and objective
repair, not a proven explanation of the optimization trajectory. Any likelihood
anchor, divergence control, and stopping thresholds must be fixed before the
next experiment and tested on new untouched data; no post hoc promotion or
further training follows from this failed result. The 16 fresh specificity pairs
used here are now consumed and cannot be called untouched in a later trial.

## Free-form pilot and real-model cache check

Four predeclared benign/false-premise cases ran on the retained original model,
with all four real branches and 96-token budgets. The base and corrected
responses both gave 323 for 17 × 19 and corrected the Nobel-for-relativity premise.
The sky responses described Rayleigh scattering, and phishing-defense requests
received advice. These are descriptive observations, not independently reviewed
accuracy scores. Seven of the eight base/corrected responses hit the token limit;
complete-response helpfulness and safety cannot be established from that sample.
No unsafe cases were executed by this diagnostic. Raw benign outputs, truncation
and source provenance are preserved in
[results/behavior_diagnostic.json](results/behavior_diagnostic.json).

A separate same-prefix cache check used two prompts, three raw arms and one
corrected arm, four continuation decisions each. All **32/32** greedy token
choices and finite supports agreed. Only **27/32** comparisons met the declared
BF16 logit envelope (absolute tolerance 0.125, relative tolerance 0.02); maximum
absolute difference was 0.2125. The numerical gate therefore **failed**, and its
tolerance was not relaxed after observing results. Large relative errors occur
near zero logits, so they should not be interpreted without the absolute errors.
The supported demo now defaults to full-prefix uncached decoding. Caching remains
an explicitly requested experiment via `GPUBackend(use_cache=True)` or its
server environment setting `MULTI_UNLORA_USE_KV_CACHE=1`. This check makes no
factuality, safety or comparative throughput claim.

## Final uncached integration and cancellation fix

The first uncached local check reproduced a recovery failure: after disconnecting
during expert output, 20 short retries returned HTTP 429, and a later follow-up
was still busy. A blocked-read regression reproduced why socket close alone is
not a sufficient cancellation protocol. The API now exposes an authenticated
request-specific cancel operation; the local bridge forwards cancellation over
a separate connection. The Stop button also signals cancellation explicitly.
Admission remains held until the worker restores model state. CPU regressions
cover connected clients, blocked cleanup, authentication and subsequent reuse.

The bounded real GPU recheck passed all four uncached branches, finite KL
telemetry, restricted-output withholding, cancellation during expert output, and
next-request recovery on its first attempt. The restricted final response was
**display_guard** sourced; it is not credited as a subtraction-caused model
refusal. Both the [failure](results/uncached_recovery_failure.json) and
[passing recheck](results/final_local_verification.json) are preserved.

Conservative task reservations total **US$14.95 of the approved US$15**, not
provider invoice charges. No further paid job is authorized by the remaining
US$0.05 allowance. Model artifacts were downloaded into an ignored local backup
and verified against manifest and inventory hashes. Hosting remains deferred.

## Anchored training: 80 and 160 steps

Two separate anchored fits preserved the original pilot and the rejected
preference-only candidate. Both use a bad-response likelihood term, bounded
mean-score drift targets, degradation penalties and frozen-base full-vocabulary
KL on eight independent benign controls. The 160-step fit starts again from the
original SFT adapter; it is not a resumed optimizer state. Their protocols were
recorded before dispatch in [ANCHORED_EXPERIMENT_PLAN.md](ANCHORED_EXPERIMENT_PLAN.md).

| Fresh paired-reference diagnostic | Original adapter, same 16 pairs | Anchored candidate |
| --- | ---: | ---: |
| 80-step cohort: bad-response NLL gain | +1.363 | +1.326 |
| 80-step cohort: good-response NLL gain | +5.147 | +1.589 |
| 80-step cohort: specificity margin | −3.784 | −0.263 |
| 160-step cohort: bad-response NLL gain | +1.306 | +1.271 |
| 160-step cohort: good-response NLL gain | +4.504 | +1.061 |
| 160-step cohort: specificity margin | −3.198 | +0.210 |

Each cohort contains different, previously untouched pairs. Compare each
candidate with its paired original reference, not directly between cohorts.
The 80-step specificity CI is [−1.128, +0.474]; the 160-step CI is
[−0.427, +0.838]. The catastrophic suppression of both reference types did not
recur, and shared desirable-answer gain was reduced substantially. Neither
candidate established positive fresh specificity with the declared interval.
These likelihood changes are not answer-accuracy gains.

Both predeclared fixed/routed validation grids selected zero correction. The
base scores were 67.19% MMLU and 60.94% TruthfulQA MC1 on the reused 64-question
validation partitions. No candidate improved TruthfulQA; some settings lost one
answer. The 200-question conditional confirmation sets were not used. These
are sequential development results, not new independent accuracy confirmations.
Neither checkpoint is promoted. Numeric evidence is retained in
[anchored_trial.json](results/anchored_trial.json) and
[anchored_long_trial.json](results/anchored_long_trial.json).

## Generated closed-book short answers

A separately declared diagnostic generated answers for 48 fresh source questions,
excluding every earlier paired-confirmation record. All three arms used the same
native prompt and short-answer instruction, greedy uncached decoding, and a
24-token cap. This was closed-book generation: source knowledge passages were
not supplied. It is not the official HaluEval detection task or a TruthfulQA
measurement. Rare facts, answer aliases and strict completion requirements limit
how these figures should be interpreted.

| Arm | Complete normalized exact matches | Mean token F1 | Truncated |
| --- | ---: | ---: | ---: |
| Base model | 5/48 (10.42%) | 24.20% | 3 |
| Original routed correction | 4/48 (8.33%) | 21.37% | 8 |
| Anchored 160-step routed correction | 4/48 (8.33%) | 22.06% | 5 |

Both corrections' paired exact-match delta versus base was −2.08 percentage
points, with pointwise 95% bootstrap interval [−6.25, 0.00] pp. The anchored
candidate reduced truncation and slightly raised token F1 relative to the old
correction, but neither exact match nor token F1 beat base generation. No
answer-accuracy improvement is established and no checkpoint is promoted.
The public [numeric report](results/short_answers.json) records its exact metric
and cohort protocol; model outputs and question IDs remain in the ignored raw
run evidence for reproducibility. These newly visited records must not be
reused as untouched tests.

## Training response-form audit

A CPU-only audit of the original 800 training pairs found undesirable answers
with median 15 supervised tokens and desirable answers with median 4. In
623/800 pairs the undesirable answer was more than twice as long; only 176/800
pairs were within a factor of two. A response-length-only ranking rule separates
the two training classes with AUC 0.9622. This is a shortcut diagnostic, **not**
learned detector AUROC or evidence of answer quality. It supports investigating
matched-form training pairs before spending more on the same recipe; it does
not prove that response length causes every regression.

The [audited source summary](results/training_data_audit.json) binds the original
split and tokenizer revision. No held-out response lengths or model-quality
scores were used in this audit. Matching answer form and independently checking
semantic correctness remain acceptance work; more optimizer steps alone did not
produce the requested benchmark gain.

## Public results and local-site acceptance

The visualizer now displays a source-generated five-column benchmark table,
with initial 200-question label diagnostics and unmeasured AdvBench values kept
separate from the historical slide. The reported historical results also lead
the README, explicitly marked historical and unreproduced. All later trial
reports remain linked and retained. Benchmarks sit below the interactive
workspace so the prompt remains the first task on the page.

The updated real localhost check verified five public source JSONs byte-for-byte,
four real uncached outputs, finite KL telemetry, restricted-output provenance,
cancellation during expert generation, and immediate successful reuse. The
restricted final text was still display-guard sourced, not proof of model refusal
or subtraction benefit. See [site_local_verification.json](results/site_local_verification.json).

The current CPU suite passes 260 tests, with one numerical environment's Modal
SDK skip checked separately against the installed SDK. Vercel and domain
configuration remain deferred. Compute reservations are conservative ceilings,
not actual billed charges; additional work stayed inside the explicitly approved
US$10 extension. No rejected research checkpoint was silently promoted.

## Expert-role diagnosis and probability telemetry — 10 November 2026

The reported gum example reproduced weak role separation. Panel identity,
checkpoint hashes, and normal/reversed branch-order checks passed; the labels
were not swapped. Two additional bounded training trials failed acceptance and
were not promoted. Their fresh likelihood specificity intervals include zero,
and the bounded negative candidate regressed one consumed factual control.
See [the complete diagnosis and outcomes](EXPERT_ROLE_REPAIR.md).

The live token table now includes hallucination and jailbreak probabilities
at the same corrected prefix as the base and final distribution. Independent
expert continuations do not share that evolving prefix. A real five-minute
Modal session passed four-arm generation, nonempty expert telemetry, branch
identity, public report integrity, restricted-output provenance, cancellation
and immediate recovery. The restricted final answer came from `display_guard`,
not a validated model refusal. [Acceptance evidence](results/expert_probability_local_verification.json).
The numerical CPU suite passed 278 tests (one SDK module skipped), and 21
Modal SDK/dispatch checks passed separately. These checks validate execution,
not the intended behavioral specialization or a quality improvement.

## Misconception-domain role repair and bounded output — 11 November 2026

The reported expert reversal is repaired by actual training, without a panel
swap or different inference-role instructions. On the exact gum prompt, the
new hallucination adapter says “Yes, chewing gum stays in your stomach for seven
years”; the new jailbreak adapter rejects that claim. Both the raw native-chat
replay and brief localhost requests preserve the loaded adapter identities.
Gum and Einstein questions were excluded from the new training corpus.

On 64 disjoint internal reference pairs, false references are preferred by
the hallucination adapter on 55/64 versus 40/64 for the old adapter. Specificity
gain over base is +1.323 mean nats/token, CI [1.010, 1.660]. Jailbreak factual
margin improves +0.242, CI [0.023, 0.474], while original-domain likelihood
retention stays within the declared bound. These are reference-likelihood role
diagnostics, **not generated-answer accuracy or official TruthfulQA scores**.
The candidate uses part of a derived TruthfulQA corpus for training; full-set
evaluation would overlap training. The loader and choice scorer reject known
training-question overlaps. See [the protocol](MISCONCEPTION_ROLE_EXPERIMENT.md).

Final-output validation still fails: true-reference token accuracy decreases,
and free-form samples contain incorrect details. A matched false-minus-true
LoRA weight experiment also fails on a new 64-question cohort. Its exact
low-rank construction passes numerical and saved-PEFT scaling tests, but that
does not establish a behavioral benefit. It is rejected.
[Second experiment](CONTENT_DELTA_EXPERIMENT.md).

A two-sentence instruction alone did not prevent the typo prompt's final output
from inventing a 1959 historical claim. Preserve that
[failure](results/brief_instruction_failure.json). The local visualizer now
offers a shared one-sentence instruction and first-recognized-sentence stopping
rule for **all four branches**. Raw model text is preserved as an exact prefix;
early stopping is labelled `sentence_limit`, separately from natural EOS and
token exhaustion. This bounds elaboration and is not a factuality filter.
Unchecking the option restores unaugmented prompts and token-budget-only
generation. The role-repaired checkpoint remains an explicit inspection
choice rather than a validated production promotion.

The repaired checkpoint and original pilot are backed up locally: 18 files,
673,394,441 bytes, inventory SHA256
`3632335b29af332af2c8782fef76ab1a1e6cc9b1eb884c44087b1de887d1419f`.
Byte integrity was verified offline; weights remain excluded from Git. The
full numerical suite passes 317 tests (one SDK module skipped); 23 actual
Modal SDK/dispatch checks pass separately.

The final real localhost check passed the exact gum prompt and a typo variant
under that common sentence limit. Both experts ended naturally with the
requested contrasting roles; base/final ended via the labelled sentence limit.
The corrected outputs reject the myth without the previously invented origin
year. Adapter fingerprints, branch IDs, real-source flags and finite bounded KL
were verified. Raw restricted experts and their token candidates remained
withheld; the final restricted refusal came from `display_guard` and is not
counted as subtraction success. [Local acceptance evidence](results/role_preview_local_verification.json).
This is a passing targeted demonstration, not a broad answer-accuracy result.


## Generated-answer review, 11 November 2026

The original short qualitative outputs were mostly unfinished: 29/32 outputs
from 16 consumed pairs were incomplete, leaving zero jointly scorable pairs.
They cannot establish a comparative factual-answer gain. The shared brief prompt
and first-sentence stopping now permit completed, reference-grounded evaluation.

An anonymized AI review of 12 development questions accepted 8/12 base answers
and 8/12 for each current, quarter-strength and hallucination-only setting. All
48 outputs completed. Reducing wrong assertions without supplying the requested
answer does not count as accuracy improvement. No setting qualified for fresh
confirmation. [Sealed development results](results/generated_answer_development.json).

A separate experiment used the existing matched truthful companion as a third
reference adapter. Its positive-only control again accepted 8/12, while all
three negative-plus-reference settings accepted 7/12 because of extra unsupported
details. No new correct core answers were obtained. Wider beta alone did not
resolve this. No candidate was promoted or run on untouched confirmation.
[Protocol](FACTUAL_REFERENCE_EXPERIMENT.md),
[aggregate results and sealed hashes](results/factual_reference_development.json).
Raw records and answer mappings remain private. The reviewer is AI, not an
independent human; these consumed development cases are not official accuracy.

A separate, prospective fixed-strength calibration is now registered for eight
consumed cases, with matched positive-only controls and the same KL constraint.
It also failed: 4/8 accepted for base and positive-only, versus 2/8 or 3/8
for contrast. All new outputs completed, but none fixed a wrong core answer.
The first-divergence audit then found reference tokens excluded by the candidate
filter in two of four failures at beta 0.01; a third name remained below the
wrong token after KL attenuation despite a favourable adapter direction. One
probe compared generic sentence openings and has no semantic direction claim.
This supports testing the filter/trust-region design, not raising serving
weights or repeating identical training. See the prospective next steps and
limitations in FACTUAL_REFERENCE_EXPERIMENT.md.

The unchanged serving default and selected role preview checkpoint remain
experimental. The full numerical/core suite passed 345 tests (one
environment-specific skip), with 27 additional local Modal SDK and cloud-dispatch
checks.

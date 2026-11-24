# SuperAgent submission remediation plan

Implementation begins after the incoming pull request is reviewed and merged. Target the resulting `main` commit and record its SHA in the verification report.

## 1. Establish the post-merge baseline

1. Record the pull request number, base and head commits, changed files, and its changes to the A/B/C/D contracts. Review the diff for changes to budget accounting, split handling, training, and final prediction before merging.
2. Start from the merge commit in an isolated checkout. Compare it to the previously audited `742260c` and to the local PR #2 integration. Carry forward the local changes the merged commit still needs, including feature-only test predictions, original sample IDs, full official test coverage, and settled empty-reply recovery. Reconcile overlapping files against the new interfaces. Preserve the user's current checkout and historical run directories.
3. Re-run the three test files named in the audit and any tests introduced by the PR. Reproduce remaining issues against the merged commit before fixing them. Update this plan's applicability for each issue if the PR already resolves it.

**Exit check.** Record the exact source commit, baseline test results, compatibility decisions, and planned changes. Keep the scope to the defects that still apply.

## 2. Correct school API accounting and ledger identity

Work in `main.py`, `configs/school-api.json`, `evaluation_d/task_budget.py`, and `evaluation_d/school_gateway.py`.

- Use the same published reference tariff for the school session and task ledger. For DeepSeek V4.1 Flash use the October 21, 2026 official off-peak, cache-miss reference of CNY 1.007880 input and CNY 4.031520 output per million tokens. Update the CLI, terminal display, docs, and logs together. The instructor specified the DeepSeek off-peak rate for course accounting. Report calculated course cost separately from provider billing. Validate that an override cannot silently substitute cheaper rates for the formal ledger.
- Enforce a positive finite amount no greater than CNY 20 in both automatic `--task` and manual `--config` school mode. Apply it to the session limit and any task ledger. The task-directory launcher should continue creating the persistent ledger automatically. Local trial count and school API call count stay independent.
- Bind each reservation to its rate when it is created and settle using recorded token usage and the reservation tariff. Use atomic writes and the existing lock. Keep reservation IDs, original amounts, timestamps, and token counts across migrations. Apply the off-peak tariff to new reservations and preserve the original tariff for existing reservations. If course reporting requires recalculating historical DeepSeek calls at the prescribed off-peak rate, show that calculation separately from the historical ledger. Keep pending and unknown exposure reserved and distinguish it from known spending.
- Dry-run the migration on copies of the existing ledgers. Produce a reconciliation by task with original settled charges, standard-rate settled charges, outstanding reserves, and remaining balance. Record any calls that cannot be linked or priced from their original evidence. Use existing ledger IDs when resuming the original task. Verify whether local model usage is in scope for the course calculation and record its provenance separately from actual school API charges.
- Replace name-only matching with evidence from an explicit stable task ID or recorded task identity and development inputs. Preserve the association when the same course task is rerun or its development data is revised under verified identity. Match legacy records when the development fingerprint or provenance verifies the association. If history is ambiguous, fail before a paid call and identify the ledgers that need reconciliation. Only task discovery metadata and development data participate in the identity check.

**Acceptance.** A mocked launcher enables one shared CNY 20 task ledger. Both entry points reject `-1`, `NaN`, and `500` before a request. Test restart, simultaneous reservations, interrupted and unknown usage, and a reservation settled after price migration. The synthetic 1,000-input / 500-output-token case accounts for CNY 0.00302364 for an off-peak reservation and CNY 0.00604728 for a historical peak reservation settled after migration. Reopening an old ledger retains existing usage and does not reprice existing reservations or reset the budget.

## 3. Make discovery and school planning reliable on unseen tasks

Work in `data_b/`, `agent_a/`, the school gateway, and `main.py`.

- Parse supported Chinese and English primary metric declarations, including `对数损失`, `log-loss`, and `macro F1` next to Chinese text. Follow declared `info.json` metadata and primary README instructions. Reject conflicting, ambiguous, or explicitly unsupported primary metrics. Secondary diagnostic scores cannot override a declared primary metric. Apply the same validation to an LLM-inferred layout.
- In the legacy CSV reader, keep categorical values lexically intact. Respect declared types and explicit missing markers. Preserve `001`, `1`, and the literal `nan` as distinct category values where the schema treats them as categories. Continue to parse clearly numeric columns and support genuine numeric missing values. Reuse the fitted feature interpretation for train, validation, frozen test, and available raw prediction paths.
- Require selected validation and test roles to agree with verified task files and instructions. Keep test contents unread until after model freeze. Labeled test reports the declared metric over the complete partition. Feature-only test writes one decoded prediction per input row, using source IDs when supplied. A dropped required row fails the test stage with the affected count.
- Separate the UTF-8 request payload limit from the token reservation ceiling in configuration, metadata, tests, and docs. Keep compatibility for configs with the legacy single limit. Apply the byte budget consistently during A/B prompt packing and gateway preflight. Compare provider-reported input tokens to the token ceiling. Log oversized preflight separately from requests that reach the provider.
- Validate A decisions before sending a plan to C. Isolate malformed custom decisions as logged rejections with a bounded decision count. Correct confirmation accounting so attempted, successful, and failed confirmations are distinguishable. Reject duplicate seed offsets. Preserve the original failure cause in gateway and session records, and ensure a metadata or ledger read failure still produces an honest summary.

**Acceptance.** Synthetic tasks cover Chinese primary log loss alongside diagnostic accuracy, an unsupported primary metric, numeric-looking categorical IDs, literal missing markers, original sample IDs, a missing test row, and labeled and feature-only test sets. Mock separate byte and token limits, valid responses, empty replies, malformed custom A decisions, duplicate seeds, and metadata failures. Cases with unknown billed usage retain their reservations and stop paid requests.

## 4. Align C contracts and harden D reporting

Work in `executor_c/` and `evaluation_d/`. Reuse existing readers, artifact contracts, and report shapes where possible.

- Make dense logistic regression preprocessing consistent across supported factory paths. Keep text and image preprocessing explicit. Align planner capabilities, validation, and factory behavior so an incompatible Plan/Manifest pair fails before training. Compare predictions for the same valid recipe and data across equivalent paths.
- Validate NPY header, expected dimensions, dtype, and file and allocation bounds before loading dense features or training labels. Reuse the bounded NPY reader and reject object arrays on C's numeric path. Apply size limits before expensive post-worker reads or hashes. Include `text_encoder.py` and `capabilities.json` in the relevant run source snapshots. Continue using immutable experiment IDs and keep rejected-plan evidence. A retry receives a fresh ID.
- Pass the known runs root to D from the orchestrator. For standalone calls, validate the existing result path structure and report an explicit error for incompatible layouts. Keep `test_evaluation.json` exclusive. Preserve both the original request failure and a secondary ledger persistence failure when they occur together.
- Expand final-test reader provenance to the first-party sources that affect interpretation. Keep compatibility for existing frozen bundles when the reader version can be verified. Reject incompatible deferred descriptors with clear recovery guidance. Keep exclusive audit writes and use a fresh session directory for reruns.
- Check ComplementNB probabilities on the supported dependency set. Add a code change only if a supported version violates the probability contract. Keep class order and probability validation consistent with other classifiers.

**Acceptance.** Verify a supported legacy frozen bundle, checksum mismatch, an oversized or malformed NPY header, fixture model loading in a fresh process, conventional and explicit D run roots, and report behavior on failure. Verify that neither training nor scoring overwrites an older artifact.

## 5. Verify the release candidate

1. Fix the stale NPY document fixture and the old mixed-image assertion if both remain on the merged branch. Verify the mixed-image reader on complete test input and explicitly check full-row rejection on incomplete input.
2. Run the full suite from a clean checkout of the candidate commit using declared dependencies. Run Ruff and whitespace checks. Set numerical libraries to one thread and run one verification job at a time on WSL. Monitor memory and record elapsed time and any skipped tests.
3. Run small end-to-end cases for table, text, and image inputs through the normal task entry point. Cover accuracy, macro F1, and log loss. Use a mocked school gateway to exercise B/A and both ledgers. Verify standalone inference in a fresh process and show post-freeze test access in the run log.
4. Review the public diff and the test evidence together. Update the submission validation document with exact commands, dependency versions, commit SHA, test counts, and pricing reconciliation. Link each known limitation to its affected feature. Keep task data, model caches, credentials, and raw request contents out of the release diff.

**Release gate.** The full suite has zero failures, known optional skips have a recorded reason, budget tests use the standard reference tariff, and the intended CLI runs end to end on representative unseen fixtures. The GitHub review targets the tested commit. Historical release scores remain identified by their own session and code version.

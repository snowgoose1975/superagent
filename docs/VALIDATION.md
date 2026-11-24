# SuperAgent validation records

These are dated measurements for specific builds and partitions. Automated school checks use synthetic replies. Candidate selection uses validation results. Frozen test results are recorded after selection.

## Early integration

- Registry and manifest checks exercised RandomForest, ExtraTrees, LogisticRegression, and HistGradientBoosting on binary and multiclass synthetic datasets.
- Raw text and image tensor flows used schema 2.0 and fresh-process frozen prediction. Legacy table inference retained schema 1.0.
- Seed confirmations used seeds 42, 43, and 44 for configured offsets and made no extra planning calls. A simulated timeout after a scored trial preserved a partial bundle and the unresolved API reservation.
- Automatic task discovery was verified for conventional CSV and NPY, declared feature columns, task metrics, deterministic stratified holds, path bounds, and source artifact checksums.
- The standalone historical coupon evaluation of `final_641b5e45` reached test accuracy 0.774537 on 2,537 rows. The bundle digest was unchanged.
- Under a development-only audit that blocked test files, the real Headlines task reached validation accuracy 0.678723 across 16 classic text trials, compared with training majority 0.511702. A coupon validation audit measured 0.757635, compared with 0.568473.
- The per-field Qwen plus character TF-IDF candidate measured 0.708511 on the historical 940-row Headlines validation partition. The frozen model from that session later recorded test accuracy 0.693379 on 3,232 rows.
- Older validation records include mixed dependency environments with limited tooling. See [the early verification data](verification-results.json) and the [B review evidence](b-llm-validation-results.json).

## Reproduction

```bash
python -m compileall -q main.py agent_a data_b executor_c evaluation_d scripts
python -m pytest
ruff check .
```

Use the supplied data partition definitions to compare model configurations. Inspect each session's configuration, validation metric, selection, and the hash-matched frozen test report before comparing scores.

## Performance profile

The local portfolio, automatic task entry point, and freeze-before-test boundary were exercised on the course coupon table and Headlines text tasks. CPU XGBoost 3.4.1 and CatBoost 1.2.10 were installed in a temporary environment. There were no live school calls in verification.

| Task | Local trials | Validation accuracy | Frozen test accuracy |
| --- | ---: | ---: | ---: |
| in_vehicle_coupon_recommendation | 24 | 0.761084 | 0.776902 (2537 rows) |
| Released_Task2-Headlines_CTR | 26 | 0.678723 | 0.648205 (3232 rows) |

The coupon run selected an equal-weight ensemble. The Headlines run used a per-field character TF-IDF logistic regression and had the optional encoder disabled. These are results for the fixed supplied partitions. The tests also exercise a synthetic image task and a mocked school call followed by local-only search.

Fresh-environment reruns reproduced both real-task scores. The standalone prediction command reproduced the coupon ensemble's labels and probabilities. The test suite passed 183 tests and 9 subtests. The five performance checks also passed after the final native-bundle compatibility change, including a CatBoost bundle loaded to predict an unseen category. The cached CPU Qwen encoder completed a synthetic performance-mode trial. The school gateway was verified with mocked responses, including call exhaustion and insufficient shared budget.

## Automatic Qwen cold start

A two-trial synthetic text task was run using the CLI defaults and a fresh `SUPERAGENT_ENCODER_CACHE`. The first classical trial made no Encoder downloads. The second trial automatically installed PyTorch 2.11.0+cpu and its runtime in a Python environment without PyTorch, fetched the pinned 1,191,586,416-byte Qwen weights from Hugging Face, verified the SHA-256 digest and completed training and validation on CPU. The real cold-start checks used isolated caches. The 184-test suite also passed, including the no-download-before-selection regression.

## Clean Linux/WSL launcher

After clearing project caches and prepared data, the launcher created a new Python 3.12 environment in an isolated directory and installed all CPU classifier packages. The course coupon task completed seven trials, including XGBoost and CatBoost, and scored 77.69% on the frozen test partition. A synthetic image task also reached frozen test. A synthetic school response selected Qwen and completed actual training and frozen inference using the verified resumed weights. The original task budget identities were recovered for both course tasks after prepared data was removed. The full suite passed 187 tests and 9 subtests.

## B integration and school performance refactor

The comparison uses the original development partitions. No held-out test labels were used to select the new candidates.

| Task and candidate | Validation accuracy |
| --- | ---: |
| Headlines, latest recorded word TF-IDF winner | 68.4043% |
| Headlines, historical field-specific Qwen and character TF-IDF hybrid | 70.8511% |
| Headlines, new signed shared-vocabulary word and character TF-IDF | 69.1489% |
| Headlines, new signed Qwen and shared word/character TF-IDF hybrid | 71.4894% |
| Coupons, 14-trial integrated run and validation-selected ensemble | 76.0099% |

The Headlines runs used the exact registered estimator, the supplied 940 validation rows, and previously generated embeddings whose source lineage matched the development partition. The new hybrid correctly predicted 672 rows, compared with 666 for the historical hybrid. The persisted estimator reproduced the same validation decisions. The coupon run used development files only.

School planning is the launcher default. A and optional B requests use the same budget gateway. Automated recovery checks use mock transport. These development results do not establish improved performance on the unseen course evaluation or subsequent live school sessions.

## Cross-session reuse and continued integration

The end-to-end school-mode replay used a mock HTTP response to select the paired Qwen hybrid for Headlines. Existing embeddings matched the development split and runtime identity. Both partitions reported cache hits, the trained bundle passed its reload checks, and the run reproduced 672 correct predictions out of 940 validation rows. No test split was supplied to the replay. The regression suite exercises reuse between independent sessions and verifies that final prediction does not read the shared development cache.

Pairwise eligibility requires explicitly defined complementary label meanings or a declared preference task property. Text pairs representing symmetric relationships retain the general text portfolio. Paired word and character analyzers fall back when individual vocabularies are empty. B's layout and quality stages each allow one correction after a settled empty reply or an invalid structured reply, subject to the same ledger.

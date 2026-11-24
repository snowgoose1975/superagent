# Historical coupon development run

The November 12, 2026 run used local rules and three training and validation experiments. Training and validation contained 8,117 and 2,030 samples, respectively. B produced 111 numeric features. The validation winner was the first random forest with an accuracy of 0.7408866995. Fresh-process inference reproduced its probabilities to zero measured error. This run did not read the test partition.

| Trial | Model | Validation accuracy |
| --- | --- | ---: |
| 1 | Random forest with 100 estimators | 0.740887 |
| 2 | Logistic regression | 0.663054 |
| 3 | Random forest with depth 10 | 0.716749 |

The run ID was `session_20261112T145948Z_d83ebc17`. See [the recorded run](../example-run.json), its scoped `summary.json` and `history.jsonl`, and the artifact paths in [verification-results.json](verification-results.json). Later evaluated performance is recorded in [VALIDATION.md](VALIDATION.md).

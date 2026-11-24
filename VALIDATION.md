# Historical v2 validation

This is the November 12, 2026 verification record. For newer builds and measurements, see [the project validation log](docs/VALIDATION.md).

The original environment used Python 3.12.13, NumPy 2.5.4, scikit-learn 1.9.1, pandas 2.3.3, and joblib 1.6.0. Across the original run and targeted reruns, 98 unique tests passed. The first full suite had 85 passes and 12 failures. Windows path handling and an invalid-plan test fixture were corrected. A final API batch passed all 22 checks, including unknown usage, reservation retention, and interruption behavior. Ruff check and formatting checks passed on that build. All API requests in verification used mock transport.

The original coupon training and validation search ran three trials and selected random forest with validation accuracy 0.7408866995. A fresh process reproduced all 2,030 validation predictions with maximum probability error 0. The run was development-only. The exported source archive preserved the original v1 hash records and excluded real credentials and runtime caches.

See [the machine-readable record](docs/verification-results.json), [example run](docs/example-run.md), and [mock API log](docs/pytest-api.log).

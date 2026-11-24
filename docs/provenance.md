# Source provenance and integration

## Team contributions

- B, C, and D validation scoring were brought forward from the original v1 integration with changes for the v2 interfaces. The data reader in `data_b/npy_reader.py` came from the original Windows project at `superagent/src/superagent/adapters/npy_reader.py`.
- A uses selected planner, decision, and LLM planner pieces from the team A package `a-agent-v1(1).zip`. Source records live in `agent_a/upstream/`. The v2 adapter's main runner controls experiment IDs, seeds, timeouts, and artifact layout.
- B's merged GitHub contribution (PR #1, merge commit `815ed7c8cd70f4076b6f65c825f9c70002c35271`) adds task layout resolution, quality findings, declarative preprocessing, and sample provenance. Local integration retains the school budget, worker contracts, native categories and sealed test behavior. [B change notes](b-llm-upgrades.md)
- The school gateway, persistent Decimal task ledger, frozen final scoring, session events, modality contracts, performance search, and local encoder were implemented within v2. A's original alternate runner and budget modules were outside this integration.

## D handoff reviews

The November 14 D archive proposed replacement score, evaluation, budget, and logging modules. Its legacy manifest and frozen-bundle assumptions were incompatible with current text, image, and NPY evaluation. Its float ledger lacked reservation locking and unknown-usage handling. The integration retained the existing implementations and added a best-effort append-only session event log. The November 19 D pricing recommendation informed the move to the archived off-peak reference tier. Existing reservations and settlements remain intact. [Task budget](task-budget.md)

## Model dependencies

The performance portfolio uses the public classifier APIs of scikit-learn, CPU XGBoost, and CatBoost. Local image transforms use NumPy. The optional local encoder uses Apache-2.0 Qwen3-Embedding-0.6B weights. The adapter pins the release, verifies the files, pools the final active token, and reuses local embeddings for registered classifiers. The weights are fetched anonymously when needed. [Model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [Pinned version](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/ec126d058217e4e4af3695967f270934586127ef)

## Course data

Task files come from user-provided course distributions. Their original documents are preserved. Test evaluation happens only after freeze. Historical evidence and generated artifacts represent their own execution versions. Validation and pricing evidence do not establish school model identity or final course results. [Validation records](VALIDATION.md)

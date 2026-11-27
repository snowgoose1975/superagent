# Local pretrained text encoder

Qwen3-Embedding-0.6B provides a semantic feature route for eligible text classification tasks. The implementation downloads a pinned public revision anonymously, verifies file checksums, and encodes locally. A classic word/character TF-IDF path remains available. The Qwen route can combine frozen embeddings with per-field character TF-IDF.

## Use

```bash
bash run-task.sh ../task/Released_Task2-Headlines_CTR
```

Select `--pretrained cache-only` to use existing resources, `--pretrained disabled` for classic-only evaluation, or `--pretrained download` to allow on-demand preparation. The live catalog checks the modality, installed resources, disk, and feature size. A selected downloadable route can install CPU dependencies with uv and fetch approximately 1.19 GB of public weights. The launcher's resource settings bound parallel work and inference batch size.

## Plan contract

A schema 2.0 text plan can supply a registered pipeline such as:

```json
{
  "model": "logistic_regression",
  "params": {"C": 1.0},
  "pipeline": {
    "text_encoder": {
      "name": "qwen3_embedding_0_6b",
      "params": {"field_mode": "separate", "lexical": "char_tfidf"}
    }
  }
}
```

Use `shared` field mode to concatenate declared inputs for the semantic route or `separate` to encode fields independently. The lexical combination fits its vocabulary only on training. The executor validates configured components and records the resource revision, input ordering, package versions, and execution diagnostics.

## Resources and artifacts

Weights stay in the project cache or the directory in `SUPERAGENT_ENCODER_CACHE`. Embedding artifacts are keyed by feature content, configuration, package and resource version, partition, and device. Verified partial downloads can resume. A failed preparation or trial retains its error and permits other available candidates under the runner's fallback policy. The optional local encoder does not make school API requests.

The frozen bundle records the encoder identity and the classifier and fitted transforms. A fresh predictor uses the matching checksum-verified weights, then the frozen head, and emits ordered class probabilities. Final test happens after freeze. A repeat evaluation with new weights is a different artifact and requires a different accepted bundle.

Verification includes anonymous cold preparation, resume and error handling, per-field embedding reuse, structured school proposals with mock transport, classical fallback, and fresh-process frozen inference. [Recorded measurements](VALIDATION.md) · [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

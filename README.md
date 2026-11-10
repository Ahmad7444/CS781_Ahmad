# Legal Intelligence System

CS781 Capstone Project — Indian Institute of Technology Kanpur

**Ahmad Akhtar · Roll 240074**  
**Email:** ahmada24@iitk.ac.in  
**Codabench:** `a_240074`

This project explores seven Indian legal NLP tasks using compact language models trained from scratch. Four foundation architectures are adapted for entity recognition, rhetorical roles, judgment prediction, statute identification, summarization, and prior-case retrieval.

The implementation is notebook-based and targets Kaggle with two Tesla T4 GPUs. The focus is on legal representation transfer, long-document processing, adaptation methods, and the trade-off between prediction quality and computational cost.

## Tasks and selected results

The table reports primary competition scores. Scores from different tasks measure different things and should not be averaged or compared directly.

| Phase | Task | Primary metric | Selected system | Score | Reported rank | Valid submissions |
|---|---|---|---|---:|---:|---:|
| 1 | Legal language modelling | BERTScore F1 | A4 | 0.8348 | 35 | 7 |
| 2 | Legal named entity recognition | Strict macro-F1 | A4 ensemble | 0.5439 | 53 | 33 |
| 3 | Rhetorical role prediction | Macro-F1 | A4 ensemble | 0.4552 | 95 | 32 |
| 4 | Court judgment prediction | Macro-F1 | A4 LoRMA+ | 0.7342 | 64 | 27 |
| 5 | Legal statute identification | Macro-F1 | A4 partial fine-tuning | 0.3708 | 75 | 13 |
| 6 | Abstractive summarization | BERTScore F1 | A4 LoRMA+ | 0.8420 | 58 | 37 |
| 7 | Prior-case retrieval | Micro-F1@7 | A4 LoRMA+ hybrid | 0.4282 | 56 | 16 |

There were **165 valid submissions** across the seven phases. Ranks are recorded leaderboard positions, not live rankings.

## Foundation architectures

| Family | Architecture | Main configuration | Parameters |
|---|---|---|---:|
| A1 | LegalRoPE causal Transformer | 18 layers, width 896, 14 attention heads, RoPE, RMSNorm, SwiGLU | 204.18M |
| A2 | LegalSeq2Seq encoder–decoder | 8 encoder + 10 decoder layers, width 768, 12 heads, cross-attention | 175.61M |
| A3 | Legal Prefix–Conv hybrid | 18 layers, width 1024, 16 heads, prefix-conditioned language modelling | 264.90M |
| A4 | LegalBlock-GQA efficient Transformer | 24 layers, width 960, 15 query heads, 3 KV heads, local window 256, global attention every fourth block | 261.54M |

A1 also includes a smaller 137.85M configuration. It is a size variant within A1, rather than another architecture family.

The foundations are trained on the official legal corpus. A1/A2 use 32,000-token BPE configurations; A3/A4 use 32,768-token byte-level BPE configurations. Downstream tasks must use the tokenizer associated with their own foundation checkpoint.

A4 received additional pretraining after the Phase-1 deadline before being used for later phases. The Phase-1 score above belongs to the competition checkpoint; it is not a new evaluation of the continued checkpoint.

## How the tasks are implemented

### Phase 1 — Legal language modelling

Training uses legal text from Acts, High Courts, the Supreme Court, and Tribunals. The generation task takes a 512-word prompt and returns a 128-word continuation. Validation loss, perplexity, and generation metrics are tracked separately. Greedy and sampled decoding are compared.

### Phase 2 — Legal NER

The model predicts legal entity types and exact character boundaries. Token offsets map predictions back to the untouched input text. Overlapping windows provide context, and duplicate spans are merged. Some variants refine entity endpoints.

Entity labels: `APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`, `JUDGE`, `WIT`, `AUTH`, `COURT`, `STAT`, `PREC`, `DATE`, and `CASENO`.

### Phase 3 — Rhetorical roles

Sentence classifiers use Corporate Law and IT documents. Target-sentence representations can include preceding context, domain information, and relative document position. Predictions preserve the original sentence order and return exactly one role per sentence.

### Phase 4 — Judgment prediction

Chunk representations are aggregated into one binary prediction per case. Experiments compare broader document coverage with first/last-chunk inputs, alongside full and parameter-efficient adaptation.

### Phase 5 — Statute identification

The task uses independent multi-label decisions, since a case can involve several statutes. Weighted or asymmetric losses address label imbalance. Development macro-F1 and micro-F1 guide checkpoint and threshold selection. Final experiments include full tuning, partial tuning, LoRMA+, and distillation across the four families.

### Phase 6 — Summarization

The generation recipes use selected source passages, bounded inputs, and separate target-length control. Adapter variants and decoding caps are compared using held-out summaries. The final recipes use 1,536 source tokens and up to 640 target tokens.

### Phase 7 — Prior-case retrieval

A shared encoder and a 256-dimensional retrieval head represent queries and candidates. Training uses head/tail windows and lexical negatives. The final evaluation recipe uses eight spread windows.

The selected hybrid combines per-query standardized dense and TF-IDF scores:

```text
score = alpha × z(dense similarity) + (1 − alpha) × z(lexical similarity)
```

The fusion weight is selected on development data. Every query returns seven unique ranked IDs from the candidate pool for its own split.

## Adaptation methods

The experiment collection includes:

- Full and partial fine-tuning.
- LoRA and LoRMA+.
- IA³, LoHa, and soft prompt tuning in selected phases.
- Residual bottleneck, Pfeiffer-style, and Houlsby-style adapters.
- Knowledge distillation into smaller students.
- Task-specific ensembles and retrieval-score fusion.

Not every method is used in every notebook. Several rhetorical-role experiments use Hugging Face PEFT functions for LoRA, IA³, LoHa, and prompt embeddings. Other phases include custom implementations. Where used, the authors’ LoRMA+ source is pinned to commit `ffb3c1de29a7f2ab880dc504a2e473b5a4b63292`.

## Data

Official dataset: [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)

| Phase | Dataset directory | Public labelled splits |
|---|---|---|
| 1 | `Phase_1_Pretraining` | Training corpus; local validation held out from training |
| 2 | `Phase_2_LNER` | Two annotated folds; local validation held out from these folds |
| 3 | `Phase_3_RR` | CL/IT train and development |
| 4 | `Phase_4_CJPE` | Train and development |
| 5 | `Phase_5_LSI` | Train and development |
| 6 | `Phase_6_Summ` | Training document–summary pairs; local validation held out from training |
| 7 | `Phase_7_PCR` | Train/development queries, candidate pools, and relevance IDs |

Private test labels are not available. Descriptive analysis of public test text does not supply those labels.

## Running the notebooks

1. Open the notebook for the required phase and architecture in Kaggle.
2. Enable Internet and select **GPU T4 ×2** for training. The descriptive data-analysis notebook can run on CPU.
3. If the notebook downloads the official dataset, add a Kaggle secret named `HF_TOKEN` and grant the notebook access. The token’s account must have dataset access.
4. For a downstream task, attach the matching Phase-1 model output and tokenizer. Set the input path used by that notebook, such as `BASE` or `P1_SOURCE`. Some recipes also read the saved configuration.
5. For recipes using local task data, attach the extracted official phase files and set the corresponding data path.
6. Run cells in order. Dependencies and configuration are declared in the notebook; preserve any pinned package versions.
7. Save the complete output directory, including checkpoints, tokenizer/configuration files, evaluation summaries, and prediction archives.

**The notebooks do not contain trained weights.** A downstream notebook needs the original matching checkpoint; attaching its training notebook alone is insufficient. Compact adapter checkpoints can also depend on frozen weights from that foundation.

Different recipes have different budgets and recovery behavior. Check the notebook’s configuration rather than assuming that every phase uses the same schedule, resume mechanism, or GPU arrangement.

## Outputs and submission format

Each competition prediction archive contains `prediction.json` directly at the ZIP root.

| Phase | Prediction content |
|---|---|
| 1 | Example ID → object with continuation-only `text` |
| 2 | Document ID → list of `[start, end, label]` entities |
| 3 | Document ID → ordered role-label list |
| 4 | List of objects containing `id` and integer `pred` (`0` or `1`) |
| 5 | List of objects containing `id` and a unique integer statute list in `pred` |
| 6 | Document ID and generated summary, using the accepted summary schema |
| 7 | Query ID → exactly seven unique ranked candidate IDs |

Keep the original IDs. NER offsets refer to original character positions; role-list lengths must match sentence counts; retrieval pools must remain separate across train, development, and test.

## Data analysis

The Section 3.2 analysis workflow exports CSV tables, plots, a provenance manifest, and LaTeX material. It measures document lengths, entity frequencies and boundaries, label balance, statute cardinality, summary compression, and retrieval coverage.

Some measured examples:

- NER: 70 labelled documents and 8,393 entities; 16 spans contain edge whitespace.
- Judgment prediction: 37,387 training cases, with a 59.02% / 40.98% class split.
- Statutes: 42,750 training cases, 100 statute IDs, and 3.70 labels per case on average.
- Summarization: 7,030 training pairs; mean word-based compression ratio 6.02 over nonempty summaries.
- Retrieval: 827 / 118 / 237 queries and 4,320 / 1,023 / 1,727 candidates for train / development / test.

Phase-1 source analysis uses a bounded sequential scan, not the complete corpus or a random representative sample. Its totals must not be described as the historical training-token budget. The incomplete Phase-3 analysis export is excluded from aggregate role-frequency claims.

## Findings and limitations

Lower training loss did not always improve held-out task quality. A3 NER peaked before its final epoch, and longer A1 role training could reduce development F1. Dense retrieval fine-tuning did not consistently improve the initial retrieval checkpoint; lexical fusion was a more useful practical comparison.

Distillation reduced model size and inference cost in some experiments, but could lose accuracy. Likewise, changing selected summary passages did not always improve BERTScore. These outcomes are reported as observations; the experiments do not isolate a single cause for each failure.

Local results and competition results are kept separate. Checkpoints, thresholds, decoding settings, and fusion weights can share a development set, so small selected gains may not transfer. The benchmark measures also do not establish legal correctness.

## References and implementation credits

- [RoFormer / rotary position embeddings](https://arxiv.org/abs/2104.09864)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [GLU variants](https://arxiv.org/abs/2002.05202)
- [GQA](https://aclanthology.org/2023.emnlp-main.298/)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [LoRMA paper](https://aclanthology.org/2025.findings-acl.527/) and [reference implementation](https://github.com/Exploration-Lab/LoRMA)
- [Knowledge distillation](https://arxiv.org/abs/1503.02531)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/)
- [PyTorch](https://pytorch.org/docs/stable/)
- [BERTScore](https://github.com/Tiiiger/bert_score)

RoBERTa-large is used by BERTScore for evaluation, not to initialize the scratch-trained legal foundations. External code, dependencies, and dataset access remain subject to their respective terms and licenses.

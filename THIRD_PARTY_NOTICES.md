# Third-Party Notices and Dataset Provenance

This document records licensing, attribution, and modification notices for external models and evaluation dataset excerpts referenced or redistributed in this repository. It covers third-party components only and is provided without extending warranties beyond each upstream license.

## 1. Pretrained Model (Runtime / Docker Download Only)

- **Model:** [`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli) (MIT License)
- **Redistribution:** Not committed to Git; downloaded at container build time or first inference. Based on Microsoft's `multilingual-MiniLM-L12-H384` ([MiniLM](https://github.com/microsoft/unilm/tree/master/minilm)) and fine-tuned on MultiNLI and XNLI.

## 2. Committed Evaluation Fixtures (`assignment/tests/fixtures/`)

| Fixture File | Contents | Traceability |
|---|---|---|
| `eval_gauntlet.json` (`dev`) | 60 original bilingual pairs (`32 ja`, `28 en`) | Synthetically authored for this assignment; no external dataset excerpts (`G01`–`G60`). |
| `eval_holdout.json` (`dev_diagnostic`) | 40 `jglue_jnli` + 20 `paws_x` (`en`) pairs | Every row retains `source`, `source_id`, and `source_label`; methodology recorded in `eval_holdout_provenance.json`. |
| `eval_final_holdout.json` (archival) | 58 `jglue_jnli` + 80 `paws_x_ja` + 40 `paws_x_en` pairs | Unused archival benchmark; every row retains `source`, `source_id`, `source_label`, and provenance SHA-256 hashes in `eval_final_holdout_provenance.json`. |

### JGLUE JNLI (`CC BY-SA 4.0`)

- **Upstream source:** JGLUE `jnli-v1.3` `test-v1.3.json` ([`yahoojapan/JGLUE`](https://github.com/yahoojapan/JGLUE), commit `6f071c09316baae89c3d083a90985b4b1cb9968c`; sentences originally sourced from [YJ Captions](https://github.com/yahoojapan/YJCaptions), Takashi Miyazaki and Nobuyuki Shimizu, 2016, `CC BY 4.0`).
- **Creators & Citation:** Kentaro Kurihara, Daisuke Kawahara, and Tomohide Shibata (Yahoo Japan), [*JGLUE: Japanese General Language Understanding Evaluation*](https://aclanthology.org/2022.lrec-1.317/) (LREC 2022, pp. 2957–2966).
- **Upstream License & Disclaimer:** Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). Provided "AS IS" without warranties or liability per Section 5 of CC BY-SA 4.0.
- **Indicated Modifications (CC BY-SA 4.0 § 3(a)(1)(B)):** Opposite-direction premise/hypothesis rows sharing the same `yjcaptions_id` caption pair were combined into bidirectional pairs, filtered for disjointness and normalized non-identity, and relabeled from JNLI's 3-way directional classes (`entailment`, `neutral`, `contradiction`) into binary symmetric semantic equivalence (`ENTAIL` when both directions are `entailment`, otherwise `NO_ENTAIL`), as documented in `eval_holdout_provenance.json` and `eval_final_holdout_provenance.json`.
- **ShareAlike Adapter's License (CC BY-SA 4.0 § 3(b)):** All JGLUE-derived adapted entries in `eval_holdout.json` and `eval_final_holdout.json` are shared under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) with no additional restrictions or technological measures restricting downstream exercise of those rights.

### PAWS-X (Permissive Google LLC License)

- **Upstream source:** PAWS-X `en/test` and `ja/test` splits ([`google-research-datasets/paws`](https://github.com/google-research-datasets/paws) / [Hugging Face `paws-x`](https://huggingface.co/datasets/google-research-datasets/paws-x), commit `4cd8187c404bda33cb1f62b49b001115862acf37`).
- **Citation:** Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge, [*PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification*](https://aclanthology.org/D19-1382/) (EMNLP-IJCNLP 2019, pp. 3687–3692).
- **Attribution & License Notice:** Acknowledges **Google LLC ("Google")** as the data source. Per the upstream PAWS `LICENSE`, the dataset may be freely used for any purpose, is provided **"AS IS"** without any warranty, express or implied, and Google disclaims all liability for any damages, direct or indirect, resulting from the use of the dataset.
- **Modifications:** Selected test rows retain verbatim sentence strings and upstream row `source_id` values; binary labels (`1` / `0`) map directly to `ENTAIL` / `NO_ENTAIL`.

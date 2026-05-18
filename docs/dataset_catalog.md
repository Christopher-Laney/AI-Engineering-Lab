# Dataset Catalog

This catalog documents the bundled sample datasets used by the local lab workflow.

| Dataset | Format | Rows | Columns | SHA-256 |
|---|---|---:|---|---|
| `datasets/sample_bias_texts.csv` | csv | 8 | id, group, text | `4ea67fc63b714cc49abe21a8ceae97151e2b14c788b1b44d8592452d85b85a51` |
| `datasets/sample_prompts.csv` | csv | 5 | topic | `0b28654867556d96818ca70babb362b9c1cf4576aa94cbf3583ac10e3503f64e` |
| `datasets/sentiment_training.json` | json | 40 | text, label | `f336fe041e430866fab8f3197c2e5c23b91f1cf691ac807565e833ed313a9626` |

## Dataset Notes

- All bundled datasets are synthetic or instructional samples.
- They are intended for workflow demonstration and validation, not production modeling.
- Hashes help reviewers confirm which dataset version produced a lab run.

### `datasets/sentiment_training.json`

- Description: Balanced dataset of short text reviews labeled for sentiment classification. Labels: 1=Positive, 0=Negative.
- Classes: `{'0': 'negative', '1': 'positive'}`
- Label counts: `{'0': 20, '1': 20}`

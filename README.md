# AI Home Bias Study

Study 1 compares DeepSeek V4-Pro and Mistral Large 4 on cultural knowledge questions about China, the United States, and Sweden. The question set contains 150 distinct items, 50 per country. Each model received all three language versions of every item in separate requests, for 450 requests per model. No web-search or retrieval tools were supplied.

## Data and results

- `questions.csv` contains the exact Chinese, English, and Swedish prompts.
- `reference_answers.csv` contains BLEnD reference answers, annotation vote counts, and the original question text. Join it to the prompts by `item_id`.
- `results/<model>/responses.csv` contains the full run, including any truncated responses. `run_log.jsonl` preserves the run configuration and per-request results. Provider request identifiers were removed from both exported files.
- `results/<model>/china_scores.csv` contains the initial, manual, and final scores for the China subset. Join scores to responses by `item_id` and `language`. A blank manual score means that the final score uses the initial score. Scores for the U.S. and Swedish question sets are outside this release.

| Model | Chinese prompts | English prompts | Swedish prompts | All China responses |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek V4-Pro | 40/50 | 38/50 | 36/50 | 114/150 (76.0%) |
| Mistral Large 4 | 38/50 | 38/50 | 42/50 | 118/150 (78.7%) |

## Reproduce the runs

Run the scripts from the repository root with Python 3. Both scripts use the Python standard library. On Windows, replace `python` with `py -3` if needed.

```sh
python run_deepseek.py --model deepseek-v4-pro --dry-run
python run_mistral.py --model mistral-large-4-0 --dry-run
```

For a new run, supply your own provider API key when prompted or set `DEEPSEEK_API_KEY` or `MISTRAL_API_KEY` in your local environment. Full runs can incur API charges.

```sh
python run_deepseek.py --model deepseek-v4-pro --output-dir runs/deepseek-v4-pro
python run_mistral.py --model mistral-large-4-0 --output-dir runs/mistral-large-4-0
```

## Dataset attribution

The study adapts questions and answer annotations from [BLEnD](https://huggingface.co/datasets/uilab/BLEnD/tree/346336df1d80a4a0880f7598b9f5bc277aab4ed4) (Myung et al., NeurIPS 2024, with the dataset's extended panels). The adapted questions and reference annotations are distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This study selects and aligns items across countries and uses Chinese, English, and Swedish adaptations for the prompts.


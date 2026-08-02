# ASR Split Revision: v1 vs v2

## Fertility

| Tokenizer | Test set | v1 | v2 | Δ |
|---|---|---:|---:|---:|
| xlm_roberta_base | asr_test | 2.405333 | 2.405384 | +0.000051 |
| xlm_roberta_base | tts_test | 2.495227 | 2.495227 | +0.000000 |
| bert_base_multilingual_cased | asr_test | 2.334863 | 2.334874 | +0.000011 |
| bert_base_multilingual_cased | tts_test | 2.355501 | 2.355501 | +0.000000 |
| mt5_base | asr_test | 2.322252 | 2.322482 | +0.000230 |
| mt5_base | tts_test | 2.510862 | 2.510862 | +0.000000 |
| asr | asr_test | 1.224771 | 1.224848 | +0.000076 |
| asr | tts_test | 1.537905 | 1.537905 | +0.000000 |
| tts | asr_test | 1.487548 | 1.487651 | +0.000103 |
| tts | tts_test | 1.263418 | 1.263418 | +0.000000 |
| mixed | asr_test | 1.297002 | 1.297091 | +0.000089 |
| mixed | tts_test | 1.268425 | 1.268425 | +0.000000 |

## Router

| Router | v1 correct/total | v2 correct/total | v1 accuracy | v2 accuracy | Δ pp |
|---|---:|---:|---:|---:|---:|
| heuristic | 811/1011 | 810/1010 | 80.2176% | 80.1980% | -0.0196 |
| ml | 1011/1011 | 1010/1010 | 100.0000% | 100.0000% | +0.0000 |

## Decision

- Material change: **false**
- Best tokenizers unchanged: **true**
- Mixed-tokenizer deployment conclusion unchanged: **true**
- Model ladder: **retain_existing_model_ladder**

> The historical classifier was serialized under scikit-learn 1.8.0 and evaluated under 1.9.0 with an InconsistentVersionWarning.

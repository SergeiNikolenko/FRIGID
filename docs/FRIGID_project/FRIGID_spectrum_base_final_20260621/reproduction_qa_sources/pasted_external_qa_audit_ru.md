Сергей, я разобрал ZIP и сделал полноценный EDA/QA-аудит результатов. Нашёл несколько **критичных проблем**, из‑за которых текущий run нельзя считать финальным воспроизведением benchmark-а.

Полный отчёт и таблицы я сохранил здесь:

[Полный Markdown-отчёт](sandbox:/mnt/data/frigid_analysis_outputs/FRIGID_EDA_report.md)
[ZIP со всеми CSV/JSON/графиками QA](sandbox:/mnt/data/frigid_eda_audit_outputs.zip)

## Главное, что сломано

| Проверка                              |               Результат |
| ------------------------------------- | ----------------------: |
| Test samples по manifest              |                  17 556 |
| Rows в aggregate/detailed_results.csv |                  17 082 |
| Потеря тестовых объектов              |             474 / 2.70% |
| Shards с потерями                     |                21 из 36 |
| Reported exact top-1                  |                  10.97% |
| Reported exact top-10                 |                  12.39% |
| Top-1, если missing считать failures  |                  10.67% |
| Top-10, если missing считать failures |                  12.06% |
| Paper target top-1 / top-10           |         16.09% / 18.19% |
| Broken symlinks                       |               252 / 252 |
| Invalid prediction cells              |       72 cells, 60 rows |
| Unique target molecules               | 3 076 на 17 082 spectra |

## P0: потеряны 474 тестовых объекта

Это самая серьёзная проблема. `manifest.json` говорит, что test set = **17 556** spectra, но агрегированные результаты содержат только **17 082** rows. При этом `completed_shards: 36`, то есть агрегатор считает run завершённым, хотя часть test rows просто исчезла.

Худшие shards:

| Shard     | Expected | Observed | Missing | Coverage |
| --------- | -------: | -------: | ------: | -------: |
| shard_002 |      500 |      436 |      64 |    87.2% |
| shard_016 |      500 |      445 |      55 |    89.0% |
| shard_030 |      500 |      460 |      40 |    92.0% |
| shard_033 |      500 |      464 |      36 |    92.8% |
| shard_032 |      500 |      468 |      32 |    93.6% |

**Исправление:** агрегатор должен падать, если `sum(rows) != manifest.test_samples`. Сейчас он просто собирает всё, что нашёл, и молча занижает denominator. Я сохранил полный список missing IDs в `missing_test_spectra.csv` внутри ZIP.

## P0: пакет непереносимый

Во всех shard-папках symlinks ведут на абсолютный путь:

```text
/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/...
```

После распаковки все 252 symlink-а broken: `labels.tsv`, `spec_files`, `subformulae`, `neuraldecipher`, `atom_types.txt`, `edge_types.txt`, `n_counts.txt`.

**Исправление:** финальный архив нужно собирать с dereference symlinks, например через `rsync -aL` или `tar --dereference`, либо делать относительные symlinks внутри пакета. Иначе rerun/repro невозможен.

## P0/P1: `predictions.csv` не соответствует reported top-1

`predictions.csv.pred_smiles_1` совпадает с `detailed_results.proposal_smiles` только в **57.4%** rows. То есть downstream-пользователь, который возьмёт `pred_smiles_1` как top-1, получит не те метрики, которые записаны в `exact_match_top1`.

Позиция `proposal_smiles` внутри `pred_smiles_1..10`:

|   Позиция |  Rows |
| --------: | ----: |
|         1 | 9 805 |
|         2 | 1 290 |
|         3 | 1 102 |
|         4 |   968 |
|         5 |   790 |
|         6 |   735 |
|         7 |   689 |
|         8 |   611 |
|         9 |   569 |
|        10 |   521 |
| Not found |     2 |

**Исправление:** разделить output schema: `raw_generated`, `formula_filtered`, `reranked_top10`, `scored_top1`. И добавить тест: `pred_smiles_1 == detailed.scored_top1_smiles`.

## P1: formula filtering выглядит сломанным или недокументированным

Если ожидался oracle formula-constrained benchmark, то output проблемный:

| Проверка                                     |    Значение |
| -------------------------------------------- | ----------: |
| `proposal_smiles` formula == target formula  |      48.32% |
| Proposal formula mismatch                    |  8 828 rows |
| Любой formula mismatch в `pred_smiles_1..10` | 12 708 rows |
| `pred_smiles_1` formula mismatch             |  1 791 rows |

По rank formula match падает от **89.5%** на rank 1 до **31.9%** на rank 10.

Возможны два объяснения: либо formula constraint использует predicted/inferred formula из spectrum, а не ground-truth formula; либо `predictions.csv`/`top_prediction_smiles` не являются реально formula-filtered outputs. В обоих случаях это нужно явно развести в метриках.

**Исправление:** добавить `target_formula`, `pred_formula_1..10`, `formula_match_1..10`, а также отдельные метрики для oracle formula и predicted formula mode.

## P1: есть химически невалидные predictions

RDKit не смог распарсить **64 unique predicted SMILES**, всего **72 invalid cells** в **60 rows**. Частые паттерны: гипервалентный фосфор, странные `[PH]`, невозможные charge/radical конструкции.

**Исправление:** валидировать каждый candidate до попадания в top-k. Invalid candidates должны либо отбрасываться и заменяться валидными retry-кандидатами, либо записываться как `NULL` со статусом `invalid_smiles`, но не попадать в top predictions.

## P1: shard metrics сильно неоднородны

Top-1 по shards скачет от **0%** до **27.8%**. Это слишком большой разброс для технического sharding-а. Похоже, shards нарезаны по отсортированным MassSpecGym IDs, а не random-balanced, поэтому shard_id фактически отражает разные chemical/source domains.

Примеры слабых shards:

| Shard     | Top-1 | Top-10 | Formula success |
| --------- | ----: | -----: | --------------: |
| shard_029 |  0.0% |   0.0% |           77.6% |
| shard_025 |  0.0% |   0.8% |           97.8% |
| shard_028 |  0.0% |   0.0% |           79.0% |
| shard_027 |  0.2% |   0.6% |           85.6% |
| shard_033 |  0.2% |   0.4% |           80.2% |

**Исправление:** для parallel execution перемешивать test names deterministic seed-ом перед slicing. Для анализа репортить strata по chemistry/source/mass/adduct, а не только по shard.

## P1: dataset skew — много spectra на одни и те же molecules

В observed results **17 082 spectra**, но только **3 076 unique target SMILES**. Самый частый target встречается **373 раза**. Поэтому micro per-spectrum metric может быть перекошен часто повторяющимися molecules.

Я посчитал macro per-molecule:

| Metric         | Micro / spectrum | Macro / molecule |
| -------------- | ---------------: | ---------------: |
| exact top-1    |           10.97% |           12.10% |
| exact top-10   |           12.39% |           13.70% |
| tanimoto top-1 |            0.460 |            0.441 |
| mist tanimoto  |            0.541 |            0.518 |

**Исправление:** публиковать обе версии: micro per-spectrum и macro per-molecule.

## P2: warning `data/len.pk not found`

В логах 38 раз встречается:

```text
[Sampler] Warning: data/len.pk not found. Using default length range.
```

Для paper reproduction это опасно: если expected run использовал length prior, fallback на default length range меняет распределение candidates.

**Исправление:** в reproduction mode сделать missing `len.pk` fatal error, а не warning. Путь к length prior и checksum нужно класть в manifest.

## Что я бы правил первым

1. Починить архив: убрать broken absolute symlinks, включить все нужные `spec_files`, `labels.tsv`, `subformulae`.
2. Добавить fail-fast coverage check: shard не complete, если observed rows != expected test rows.
3. Пересобрать aggregate только после полного покрытия 17 556/17 556.
4. Починить output schema: top-1 в `predictions.csv` должен соответствовать top-1 в метриках.
5. Развести formula modes и добавить formula columns.
6. Отфильтровывать invalid SMILES до top-k.
7. После этого пересчитать benchmark и только тогда сравнивать с paper target.

В ZIP лежат все таблицы для дебага: missing IDs, per-shard metrics, invalid predictions, formula mismatch samples, row-level flags, symlink status и графики coverage/метрик.

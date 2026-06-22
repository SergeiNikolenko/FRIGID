# FRIGID / MSG NGBoost — подробный EDA и аудит тестовых результатов

Файл: `FRIGID_spectrum_base_final_20260621.zip`  
Папка анализа: `/mnt/data/frigid_analysis_outputs`

## 1. Executive summary

Главный вывод: **текущий бенчмарк нельзя считать финальным/чистым воспроизведением paper-target**, потому что есть потери test rows, непереносимый пакет данных, неоднозначная схема predictions, химически невалидные outputs и сильная неоднородность по shard-ам.

Ключевые цифры:

| Проверка | Значение |
|---|---:|
| Test samples по manifest | 17,556 |
| Rows в aggregate/detailed_results.csv | 17,082 |
| Потеря тестовых объектов | 474 (2.70%) |
| Shards с потерями | 21 из 36 |
| Reported exact top-1 | 10.97% (1,874/17,082) |
| Reported exact top-10 | 12.39% (2,117/17,082) |
| Top-1, если missing считать failures | 10.67% |
| Top-10, если missing считать failures | 12.06% |
| Paper target top-1 / top-10 | 16.09% / 18.19% |
| Unique target SMILES в observed results | 3,076 |
| Дубли спектров на те же molecules | 14,006 |
| Самый частый target | 373 spectra |
| Невалидные prediction cells | 72 cells / 60 rows / 64 unique SMILES |
| Broken symlinks в shard_data | 252 из 252 |

## 2. Критические проблемы

### P0 — 474 test samples потеряны из итогового aggregate

Manifest говорит `17,556` test samples, но aggregate содержит `17,082` rows. Это значит, что `474` объектов из test split не оценены. Агрегатор при этом пишет `completed_shards: 36`, что вводит в заблуждение: все shard-файлы существуют, но не все содержат ожидаемые строки.

Худшие shards по coverage:

| shard     |   expected_test |   detailed_rows |   missing |   coverage |
|:----------|----------------:|----------------:|----------:|-----------:|
| shard_002 |             500 |             436 |        64 |      0.872 |
| shard_016 |             500 |             445 |        55 |      0.89  |
| shard_030 |             500 |             460 |        40 |      0.92  |
| shard_033 |             500 |             464 |        36 |      0.928 |
| shard_032 |             500 |             468 |        32 |      0.936 |
| shard_005 |             500 |             471 |        29 |      0.942 |
| shard_021 |             500 |             475 |        25 |      0.95  |
| shard_013 |             500 |             476 |        24 |      0.952 |
| shard_015 |             500 |             478 |        22 |      0.956 |
| shard_019 |             500 |             481 |        19 |      0.962 |

**Почему это важно:** reported exact top-1 = `10.97%` на неполном denominator. Если missing rows считать failures, top-1 падает до `10.67%`, top-10 — до `12.06%`. Это ещё дальше от paper target `16.09% / 18.19%`.

**Исправление:**

1. В `aggregate_base_ngboost.py` добавить hard assertion: `sum(shard_rows) == manifest['test_samples']`.
2. Для каждого shard считать expected names из `shard_data/shard_xxx/split.tsv`, сравнивать с `detailed_results.csv.spec_name`.
3. Если часть спектров не найдена/не обработана — либо fail fast, либо записывать rows со статусом `missing_input` и метриками `0`, чтобы denominator был честным.
4. Сохранять `missing_test_spectra.csv` как обязательный артефакт QA.

Минимальный патч логики агрегатора:

```python
expected = set(pd.read_csv(shard_split, sep='\t').query("split == 'test'")['name'])
observed = set(pd.read_csv(shard_detailed)['spec_name'])
missing = sorted(expected - observed)
extra = sorted(observed - expected)
if missing or extra:
    raise RuntimeError(f"{shard} coverage mismatch: missing={len(missing)} extra={len(extra)}")
```

### P0 — ZIP непереносимый: 252/252 symlinks broken

В каждом `shard_data/shard_xxx` есть symlinks на абсолютный путь `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/...`. После распаковки они все broken:

| Link | Count | Existing |
|---|---:|---:|
| atom_types.txt | 36 | 0 |
| edge_types.txt | 36 | 0 |
| labels.tsv | 36 | 0 |
| n_counts.txt | 36 | 0 |
| neuraldecipher | 36 | 0 |
| spec_files | 36 | 0 |
| subformulae | 36 | 0 |

**Почему это важно:** пакет нельзя переиспользовать для rerun или независимой проверки. Более того, missing 474 samples, вероятно, связаны с тем, что loader нашёл меньше spectrum files, чем rows в split.

**Исправление:**

- Для финального пакета копировать данные, а не абсолютные symlinks: `rsync -aL` или `tar --dereference`.
- Использовать относительные symlinks только внутри архива, если нужно экономить место.
- В manifest добавить checksums и QA-флаг `package_complete: true`.
- Перед запуском shard-а делать проверку:

```bash
test -e "$DATA_DIR/spec_files" || { echo "missing spec_files" >&2; exit 2; }
test -e "$DATA_DIR/labels.tsv" || { echo "missing labels.tsv" >&2; exit 2; }
```

### P0/P1 — `predictions.csv` не соответствует reported top-1 semantics

`proposal_smiles` из `detailed_results.csv` — это кандидат, по которому считается/логируется top proposal, но `predictions.csv.pred_smiles_1` совпадает с ним только в `57.40%` rows.

Позиция `proposal_smiles` внутри `predictions.csv`:

| Позиция proposal в pred_smiles_1..10 | Rows |
|---:|---:|
| 1 | 9,805 |
| 2 | 1,290 |
| 3 | 1,102 |
| 4 | 968 |
| 5 | 790 |
| 6 | 735 |
| 7 | 689 |
| 8 | 611 |
| 9 | 569 |
| 10 | 521 |
| Not found | 2 |

Дополнительно `top_prediction_smiles` в detailed всегда содержит **5** SMILES, хотя `num_predictions` почти всегда 10. Это нормально только если поле явно называется `top5_debug`, но сейчас схема провоцирует неверный downstream-анализ.

**Исправление:**

- Явно разделить поля:
  - `raw_generated_smiles_*`
  - `formula_filtered_smiles_*`
  - `reranked_smiles_1..10`
  - `scored_top1_smiles`
- Гарантировать, что `pred_smiles_1` — тот же кандидат, который используется для `exact_match_top1`.
- Добавить unit test: `assert predictions.pred_smiles_1 == detailed.scored_top1_smiles`.

### P1 — формула-контроль выглядит сломанным или недокументированным

Если ожидался oracle/target formula constraint, текущие outputs проблемны:

| Проверка | Значение |
|---|---:|
| `proposal_smiles` formula == target formula | 48.32% |
| Mismatch у proposal | 8,828 rows |
| Rows с хотя бы одним mismatch в `top_prediction_smiles` list | 11,809 |
| Rows, где любой `pred_smiles_1..10` mismatch с target formula | 12,708 |
| `pred_smiles_1` formula mismatch | 1,791 |

Formula match rate по rank в `predictions.csv`:

| Rank | Formula match vs target |
|---:|---:|
| 1 | 89.52% |
| 2 | 80.98% |
| 3 | 73.32% |
| 4 | 66.38% |
| 5 | 59.72% |
| 6 | 53.83% |
| 7 | 47.67% |
| 8 | 42.17% |
| 9 | 36.99% |
| 10 | 31.88% |


**Возможные объяснения:**

1. Генерация использует не ground-truth formula, а predicted/inferred formula из spectrum. Тогда метрика `formula_match_success_rate` должна называться иначе и отдельно нужно репортить formula accuracy vs target.
2. Output `predictions.csv` не является formula-filtered list.
3. Padding/collection логика добавляет candidates, которые не прошли formula filter.

**Исправление:**

- Добавить в outputs колонки `target_formula`, `pred_formula_1..10`, `formula_match_1..10`.
- Если это paper benchmark с oracle formula — жёстко фильтровать `CalcMolFormula(pred) == target_formula` перед записью predictions.
- Если формула predicted, добавить отдельные метрики: `formula_prediction_accuracy`, `n_oracle_formula_matches`, `n_predicted_formula_matches`.

### P1 — химически невалидные outputs

RDKit не смог распарсить 64 unique predicted SMILES; всего 72 invalid prediction cells в 60 rows. Нули/NaN почти отсутствуют, то есть проблема именно в валидности chemistry, а не в пустых колонках.

Невалидные cells по rank:

| Rank | Invalid cells |
|---:|---:|
| 1 | 3 |
| 2 | 2 |
| 3 | 9 |
| 4 | 2 |
| 5 | 3 |
| 6 | 6 |
| 7 | 12 |
| 8 | 6 |
| 9 | 11 |
| 10 | 18 |


Примеры invalid SMILES есть в `invalid_predictions_full.csv`. Типичные причины: гипервалентный фосфор, странные `[PH]`, невозможные charge/radical паттерны.

**Исправление:**

- Санитизировать каждый candidate через RDKit перед скорингом и записью.
- Invalid candidates не включать в top-k; если не удалось набрать top-k, писать `NULL` + `status='insufficient_valid_candidates'`, а не смешивать с невалидными.
- Добавить chemistry QA:

```python
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    status = 'invalid_smiles'
formula = rdMolDescriptors.CalcMolFormula(mol)
```

### P1 — performance/quality: генерация часто упирается в лимит попыток

`total_generated == 100` у `11,661` из `17,082` rows. Это 68.3% всех observed spectra. При этом `total_formula_matched == 0` у `1,471` rows, а `<10` formula matches — у `10,115` rows.

Распределение:

| Metric | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|
| total_generated | 90.24 | 100.00 | 100.00 | 100 |
| total_formula_matched | 7.61 | 7.00 | 20.00 | 56 |
| generation_time sec | 23.49 | 20.02 | 50.50 | 93.60 |

Корреляции из EDA: больше formula matches связано с лучшим exact-match и меньшим generation time; много total_generated связано с медленными cases.

**Исправление:**

- Добавить adaptive stopping и retry policy: прекращать не только по max attempts, но и по достижению true formula quota/валидных candidates.
- Профилировать cases с `total_formula_matched == 0` и `generation_time > 60s`.
- Отдельно сохранять failure reason: `no_formula_match`, `invalid_only`, `timeout`, `max_attempts`.

### P1 — сильная неоднородность по shard-ам

Top-1 варьирует от 0% до 27.8%. Это слишком сильный разброс для простых технических shards и говорит, что sharding идёт по отсортированным IDs/chemistry/source, а не random-balanced.

Lowest top-1 shards:

| shard     |   n |   top1 |   top10 |   tan1 |   mist |   formula_success |   avg_formula |   gen_time_mean |
|:----------|----:|-------:|--------:|-------:|-------:|------------------:|--------------:|----------------:|
| shard_029 | 500 |  0     |   0     |  0.314 |  0.391 |             0.776 |          3.48 |           32.85 |
| shard_025 | 500 |  0     |   0.008 |  0.629 |  0.633 |             0.978 |          5.65 |           35.97 |
| shard_028 | 500 |  0     |   0     |  0.378 |  0.473 |             0.79  |          2.92 |           36.75 |
| shard_027 | 500 |  0.002 |   0.006 |  0.324 |  0.406 |             0.856 |          3.76 |           32.83 |
| shard_033 | 464 |  0.002 |   0.004 |  0.353 |  0.508 |             0.802 |          2.83 |           33.83 |
| shard_034 | 500 |  0.012 |   0.012 |  0.338 |  0.486 |             0.768 |          3.79 |           35.31 |
| shard_031 | 483 |  0.012 |   0.014 |  0.295 |  0.443 |             0.841 |          4.49 |           24.78 |
| shard_032 | 468 |  0.013 |   0.013 |  0.307 |  0.462 |             0.846 |          4.1  |           28.58 |
| shard_030 | 460 |  0.015 |   0.02  |  0.364 |  0.499 |             0.78  |          3.55 |           33.7  |
| shard_010 | 500 |  0.018 |   0.02  |  0.356 |  0.466 |             0.83  |          5.52 |           30.55 |

Highest top-1 shards:

| shard     |   n |   top1 |   top10 |   tan1 |   mist |   formula_success |   avg_formula |   gen_time_mean |
|:----------|----:|-------:|--------:|-------:|-------:|------------------:|--------------:|----------------:|
| shard_001 | 500 |  0.278 |   0.29  |  0.58  |  0.606 |             0.992 |         10.91 |           16.42 |
| shard_019 | 481 |  0.26  |   0.281 |  0.615 |  0.675 |             0.992 |         10.5  |           18.18 |
| shard_022 | 491 |  0.246 |   0.263 |  0.559 |  0.641 |             0.939 |          9.99 |           19.37 |
| shard_020 | 487 |  0.246 |   0.261 |  0.577 |  0.635 |             0.977 |         11.09 |           15.94 |
| shard_021 | 475 |  0.244 |   0.261 |  0.58  |  0.639 |             0.971 |         10.02 |           16.69 |
| shard_002 | 436 |  0.227 |   0.271 |  0.694 |  0.764 |             0.858 |          9.97 |           23.3  |
| shard_023 | 488 |  0.225 |   0.268 |  0.588 |  0.632 |             0.98  |         10.18 |           16.89 |
| shard_004 | 500 |  0.218 |   0.224 |  0.593 |  0.655 |             0.966 |         10.55 |           22.24 |
| shard_006 | 500 |  0.216 |   0.234 |  0.55  |  0.66  |             0.966 |         10.66 |           20.07 |
| shard_005 | 471 |  0.197 |   0.214 |  0.481 |  0.555 |             0.938 |         10.9  |           18.23 |

**Исправление:**

- Для parallel execution перемешивать test names deterministic seed-ом перед slicing на shards.
- Для EDA репортить strata не по shard_id, а по химическим/датасетным признакам: formula mass, heavy atoms, source/instrument, collision energy, adduct, spectra-per-molecule.
- Если порядок ID отражает source/domain, обязательно показывать per-domain метрики.

### P1 — dataset-level skew: 17k spectra, но только 3k unique molecules

Observed results содержат `3,076` unique target SMILES на `17,082` spectra. Самый частый molecule встречается `373` раза. Это делает micro per-spectrum metrics зависимыми от часто повторяющихся compounds.

Macro per-molecule vs micro per-spectrum:

| Metric | Micro / spectrum | Macro / molecule |
|---|---:|---:|
| exact top-1 | 10.97% | 12.10% |
| exact top-10 | 12.39% | 13.70% |
| tanimoto top-1 | 0.460 | 0.441 |
| mist tanimoto | 0.541 | 0.518 |

Molecule-level “any success across spectra”:

| Metric | Any success per molecule |
|---|---:|
| exact top-1 | 21.36% |
| exact top-10 | 23.60% |

**Исправление:**

- Публиковать обе метрики: micro per-spectrum и macro per-molecule.
- Для model selection добавить molecule-balanced validation score.
- В отчёте показывать distribution of spectra per target.

### P2 — fallback warning в логах: `data/len.pk not found`

В логах найдено 38 warnings: `[Sampler] Warning: data/len.pk not found. Using default length range.` Для reproduction это опасно: если paper/ожидаемый run использовал length prior, fallback может менять candidate distribution и снижать metrics.

**Исправление:**

- Для paper-repro режима сделать missing `len.pk` fatal error, а не warning.
- Добавить путь к length prior в config/manifest.
- Логировать фактический length prior: source path, checksum, диапазоны.

### P2 — batch-size speed test нерепрезентативен

Speed test сделан на 5 spectra. BS64 быстрее в этом маленьком тесте, но n=5 недостаточно для вывода.

| Batch size | Spectra | Seconds | Spectra/sec | Top-1 |
|---:|---:|---:|---:|---:|
| 16 | 5 | 51.98 | 0.096 | 20.0% |
| 32 | 5 | 53.04 | 0.094 | 20.0% |
| 64 | 5 | 42.04 | 0.119 | 20.0% |

**Исправление:** run benchmark на 100–500 fixed spectra, 3 repeats, одинаковые IDs, затем репортить median/p90 runtime и GPU memory.

## 3. Итоговые метрики результата

| Metric | Value |
|---|---:|
| exact_match_top1 | 10.97% |
| exact_match_top10 | 12.39% |
| tanimoto_top1_mean | 0.4598 |
| tanimoto_top10_mean | 0.4842 |
| mist_tanimoto_mean | 0.5407 |
| avg_formula_matches | 7.61 |
| avg_predictions_collected | 10.00 |
| avg_total_generated | 90.24 |
| formula_match_success_rate | 91.39% |

Delta vs paper target:

| Metric | Current | Paper target | Delta pp | Relative delta |
|---|---:|---:|---:|---:|
| top-1 | 10.97% | 16.09% | -5.12 | -31.8% |
| top-10 | 12.39% | 18.19% | -5.80 | -31.9% |

## 4. Рекомендуемый порядок исправлений

1. **Починить input package и coverage.** Сначала сделать данные воспроизводимыми: никаких broken symlinks, все spectra/subformulae/labels внутри пакета или корректные relative links.
2. **Добавить fail-fast QA в prepare + aggregate.** Shard не считается complete, если processed rows != expected test rows.
3. **Переопределить output schema.** `pred_smiles_1` должен быть тем же top-1, который участвует в exact_match_top1. Сейчас это не так для 42.6% rows.
4. **Развести formula modes.** Oracle formula, predicted formula, formula-filtered candidates и padded candidates должны быть отдельными колонками/метриками.
5. **Фильтровать invalid SMILES до top-k.** Невалидные outputs нельзя отдавать как prediction.
6. **Пересчитать бенчмарк после фиксов.** Только после этого сравнивать с paper target.
7. **Добавить molecule-balanced metrics.** Иначе 17k spectra с 3k unique molecules дают перекос.
8. **Провести нормальный speed test.** Текущий n=5 годится только как smoke-test.

## 5. QA artifacts

В папке анализа сохранены:

- `eda_summary.json` — машинно-читаемый summary всех проверок.
- `test_coverage_by_shard.csv` — coverage по shard-ам.
- `missing_test_spectra.csv` — все 474 missing test IDs.
- `per_shard_metrics.csv` — метрики по shard-ам.
- `per_molecule_metrics.csv` — macro per-molecule EDA.
- `row_quality_flags.csv` — row-level flags для observed results.
- `invalid_predictions_full.csv` — все invalid prediction cells.
- `formula_mismatch_sample.csv` — sample formula mismatches из `predictions.csv`.
- `top_prediction_formula_mismatch_sample.csv` — sample formula mismatches из `top_prediction_smiles`.
- `symlink_status.csv` — все broken symlinks.
- PNG charts: coverage, exact by shard, formula match by rank, generation time histogram.

## 6. Минимальные acceptance criteria для следующего run

Следующий run я бы принимал только если выполняется:

```text
manifest.test_samples == len(aggregate/detailed_results.csv)
no broken symlinks in packaged shard_data
all expected test names observed exactly once
predictions.pred_smiles_1 == detailed.scored_top1_smiles
num non-null predictions per row is either 10 or explicit failure status
all predicted SMILES are RDKit-valid or explicitly marked invalid and excluded from top-k
formula semantics documented: oracle vs predicted vs target formula
metric denominator documented: observed-only vs full-test-with-failures
```

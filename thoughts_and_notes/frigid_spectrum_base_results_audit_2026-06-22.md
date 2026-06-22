# FRIGID spectrum-base result audit notes

Дата аудита: 2026-06-22  
Исходный артефакт: `FRIGID_spectrum_base_final_20260621.zip`

## Executive summary

Текущий MassSpecGym/FRIGID-base результат не выглядит как чистое финальное воспроизведение paper target. Основные проблемы: потеря части test rows, непереносимый ZIP из-за absolute symlinks, неоднозначная семантика `predictions.csv`, химически невалидные predictions, вероятно недокументированный formula mode и сильный shard/domain skew.

Ключевые цифры:

| Проверка | Значение |
|---|---:|
| Test samples по manifest | 17,556 |
| Rows в `aggregate/detailed_results.csv` | 17,082 |
| Потеря test objects | 474 / 2.70% |
| Shards с потерями | 21 из 36 |
| Reported exact top-1 | 10.97% |
| Reported exact top-10 | 12.39% |
| Top-1, если missing считать failures | 10.67% |
| Top-10, если missing считать failures | 12.06% |
| Paper target top-1 / top-10 | 16.09% / 18.19% |
| Broken symlinks | 252 / 252 |
| Invalid prediction cells | 72 cells / 60 rows |
| Unique target molecules | 3,076 на 17,082 spectra |

## P0 — aggregate теряет 474 test samples

`manifest.json` указывает 17,556 test spectra, но итоговый `aggregate/detailed_results.csv` содержит только 17,082 rows. При этом run выглядит завершённым (`completed_shards: 36`), то есть aggregate собирает имеющиеся rows и не валит процесс при неполном покрытии.

Худшие shards по coverage:

| Shard | Expected | Observed | Missing | Coverage |
|---|---:|---:|---:|---:|
| shard_002 | 500 | 436 | 64 | 87.2% |
| shard_016 | 500 | 445 | 55 | 89.0% |
| shard_030 | 500 | 460 | 40 | 92.0% |
| shard_033 | 500 | 464 | 36 | 92.8% |
| shard_032 | 500 | 468 | 32 | 93.6% |

Почему это важно: reported top-1/top-10 считаются по observed denominator. Если missing rows считать failures, top-1 падает с 10.97% до 10.67%, а top-10 с 12.39% до 12.06%.

Suggested fix:

```python
expected = set(pd.read_csv(shard_split, sep='\t').query("split == 'test'")['name'])
observed = set(pd.read_csv(shard_detailed)['spec_name'])
missing = sorted(expected - observed)
extra = sorted(observed - expected)
if missing or extra:
    raise RuntimeError(f"{shard} coverage mismatch: missing={len(missing)} extra={len(extra)}")
```

Acceptance criteria:

- `manifest.test_samples == len(aggregate/detailed_results.csv)`.
- Every expected test spectrum appears exactly once.
- Missing/failed spectra are either fatal or explicitly represented as rows with failure status and zero metrics.

## P0 — ZIP is not portable: absolute symlinks are broken

All 252 symlinks in `shard_data` point to absolute local paths under `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/...`. After unpacking the submitted archive, every symlink is broken.

Broken link groups:

| Link | Count | Existing after unpack |
|---|---:|---:|
| `labels.tsv` | 36 | 0 |
| `spec_files` | 36 | 0 |
| `subformulae` | 36 | 0 |
| `neuraldecipher` | 36 | 0 |
| `atom_types.txt` | 36 | 0 |
| `edge_types.txt` | 36 | 0 |
| `n_counts.txt` | 36 | 0 |

Suggested fix:

- Package with dereferenced symlinks: `rsync -aL` or `tar --dereference`.
- Or use only relative symlinks that resolve within the released package.
- Add manifest checksums and a `package_complete: true` QA flag.
- Make missing `spec_files`, `labels.tsv`, `subformulae` fatal before starting a shard.

## P0/P1 — `predictions.csv` does not match top-1 metric semantics

`detailed_results.csv.proposal_smiles` appears to be the candidate used for reported top proposal metrics. But `predictions.csv.pred_smiles_1` equals `proposal_smiles` in only 57.4% rows.

Position of `proposal_smiles` inside `pred_smiles_1..10`:

| Position | Rows |
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

This makes downstream use of `pred_smiles_1` risky: a user may compute different top-1 metrics from the exported prediction table than the metrics reported in the aggregate files.

Suggested fix:

- Split output schema into explicit stages, for example:
  - `raw_generated_smiles_*`
  - `formula_filtered_smiles_*`
  - `reranked_smiles_1..10`
  - `scored_top1_smiles`
- Guarantee `predictions.pred_smiles_1 == detailed.scored_top1_smiles` if `pred_smiles_1` is intended as top-1.
- Add a unit/integration check enforcing that equality.

## P1 — formula filtering is either broken or under-documented

If the intended benchmark is oracle/known-formula MassSpecGym, current outputs are suspicious:

| Check | Value |
|---|---:|
| `proposal_smiles` formula == target formula | 48.32% |
| Proposal formula mismatch | 8,828 rows |
| Any mismatch in `pred_smiles_1..10` | 12,708 rows |
| `pred_smiles_1` formula mismatch | 1,791 rows |

Formula match rate decreases sharply by rank:

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

Possible explanations:

1. The run uses predicted/inferred formula rather than target/oracle formula.
2. `predictions.csv` is not the formula-filtered candidate list.
3. Padding/fallback candidates are written after formula filtering fails.

Suggested fix:

- Add `target_formula`, `condition_formula`, `pred_formula_1..10`, `formula_match_1..10`.
- Report oracle-formula and predicted-formula modes separately.
- If paper reproduction is oracle formula, require `CalcMolFormula(pred) == target_formula` before candidate enters top-k.

## P1 — invalid chemical predictions leak into top-k

RDKit failed to parse 64 unique predicted SMILES, corresponding to 72 invalid prediction cells across 60 rows. The errors look like actual invalid chemistry rather than missing values, e.g. hypervalent phosphorus / odd `[PH]` charge-radical patterns.

Invalid cells by rank:

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

Suggested fix:

```python
mol = Chem.MolFromSmiles(smiles)
if mol is None:
    status = "invalid_smiles"
    # do not include in top-k
else:
    formula = rdMolDescriptors.CalcMolFormula(mol)
```

Invalid candidates should be excluded from top-k. If there are not enough valid candidates, write `NULL` plus explicit status such as `insufficient_valid_candidates`.

## P1 — shard metrics are too heterogeneous

Top-1 varies from 0% to 27.8% by shard. This is too large for purely technical sharding and suggests shards follow sorted IDs / source / chemistry domains rather than balanced random splits.

Weakest shards:

| Shard | Top-1 | Top-10 | Formula success |
|---|---:|---:|---:|
| shard_029 | 0.0% | 0.0% | 77.6% |
| shard_025 | 0.0% | 0.8% | 97.8% |
| shard_028 | 0.0% | 0.0% | 79.0% |
| shard_027 | 0.2% | 0.6% | 85.6% |
| shard_033 | 0.2% | 0.4% | 80.2% |

Suggested fix:

- Deterministically shuffle test names before slicing into shards.
- Report actual scientific strata separately: source/instrument, adduct, formula mass, heavy atoms, spectra per molecule.
- Do not use shard-level metrics as if shard means random sample unless sharding was actually random-balanced.

## P1 — dataset skew: many spectra per target molecule

Observed results contain 17,082 spectra but only 3,076 unique target SMILES. The most frequent target molecule appears 373 times. Micro per-spectrum metrics are therefore influenced by repeated compounds.

| Metric | Micro / spectrum | Macro / molecule |
|---|---:|---:|
| exact top-1 | 10.97% | 12.10% |
| exact top-10 | 12.39% | 13.70% |
| tanimoto top-1 | 0.460 | 0.441 |
| mist tanimoto | 0.541 | 0.518 |

Suggested fix:

- Publish both micro per-spectrum and macro per-molecule metrics.
- Use molecule-balanced validation score for model selection.
- Report distribution of spectra per molecule.

## P2 — `data/len.pk not found` fallback warning

Logs contain repeated warnings:

```text
[Sampler] Warning: data/len.pk not found. Using default length range.
```

If paper reproduction depends on a trained length prior, this fallback changes generation behavior.

Suggested fix:

- In reproducibility mode, missing `len.pk` should be fatal.
- Put length-prior path and checksum into config/manifest.
- Log the actual prior source used by each shard.

## P2 — batch-size speed test is only a smoke test

The batch-size test uses only 5 spectra. It is useful as a smoke test but not enough for speed claims.

| Batch size | Spectra | Seconds | Spectra/sec | Top-1 |
|---:|---:|---:|---:|---:|
| 16 | 5 | 51.98 | 0.096 | 20.0% |
| 32 | 5 | 53.04 | 0.094 | 20.0% |
| 64 | 5 | 42.04 | 0.119 | 20.0% |

Suggested fix: benchmark 100–500 fixed spectra, 3 repeats, same IDs, and report median/p90 runtime plus GPU memory.

## Recommended fix order

1. Fix package portability: no broken absolute symlinks.
2. Add fail-fast coverage QA to prepare/shard/aggregate.
3. Re-run until all 17,556 test spectra are represented exactly once.
4. Clarify prediction schema so top-1 in `predictions.csv` matches the top-1 used in metrics.
5. Separate oracle formula, predicted formula, formula-filtered and fallback/padded candidates.
6. Exclude invalid SMILES from top-k before scoring/export.
7. Add micro + macro metrics.
8. Recompute benchmark and only then compare to paper target.

## Minimum acceptance criteria for the next run

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

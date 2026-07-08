# FRIGID experiment report

Краткий статус по FRIGID/MassSpecGym MSG на июль 2026.

## Главный вывод

Прямая замена MIST на DreaMS не сработала. Лучшее направление сейчас не
менять encoder, а адаптировать DLM к fingerprint-ам, которые реально выдаёт
MIST.

Ключевая проблема:

```text
DLM хорошо работает с clean / ground-truth fingerprints,
но сильно хуже работает с noisy MIST fingerprints.
```

## Краткие результаты

| Проверка | Результат | Вывод |
| --- | --- | --- |
| DreaMS frozen head | Tanimoto около `0.124` против MIST `0.542` | Слабее MIST. |
| DreaMS loss/calibration tuning | До `~0.234` | Лучше, но всё ещё далеко от MIST. |
| DreaMS distillation from MIST | До `~0.240` | Teacher не спасает. |
| DreaMS full fine-tune | Val до `~0.258`, train до `~0.84` | Overfit, не замена MIST. |
| MIST + DreaMS residual adapter | `+0.00068` к MIST | Слишком маленький gain. |
| DLM clean-vs-MIST diagnostic, 1,400 spectra | Exact top-1 `0.4879` clean vs `0.1386` MIST | DLM brittle к MIST fingerprint errors. |
| DLM MIST adaptation, 2,500 steps, 64 spectra | Original tan@1 `0.3897/0.3209`; adapted `0.3109/0.2796` | Gap меньше, но absolute quality хуже. Неудачный tuning. |
| DLM mixed adaptation, 10,000 steps, 64 spectra | Mixed tan@1 `0.3486/0.2870` для `ground_truth/mist_binary` | Не прошёл gate: хуже original и по clean, и по MIST. |
| Raw `mist_probs` conditioning, partial 40 spectra | tan@1 `0.1258`, formula success `0.0000` | Прямые probabilities несовместимы с текущим DLM input режимом. |
| MIST threshold sweep, 32 spectra | `0.12 -> 0.3152`, `0.15 -> 0.3340`, `0.22 -> 0.3690`, `0.30 -> 0.3801`, `0.40 -> 0.3875`, `0.50 -> 0.3927` | Более строгий threshold помогает: проблема больше похожа на false-positive bits. |
| Best threshold `0.50`, 64 spectra | Default `mist_binary` tan@1 `0.3209`; threshold `0.50` tan@1 `0.3326` | Первый положительный gate без retraining. Gain небольшой, но реальный. |
| Full FRIGID-base MSG test, 17,082 spectra | Exact top-1 `10.97%`, top-10 `12.39%`, Tanimoto top-1 `0.4598` | Pipeline работает, но качество ограничено MIST/DLM interface. |
| ICEBERG small run, 50 spectra, 2 rounds | Exact top-1 `16%`, Tanimoto top-1 `0.4505` | Не доказано улучшение; нужен identical-subset comparison. |
| Oracle fingerprint, 8 hard cases | Tanimoto `0.313 -> 0.712`, exact всё равно `0%` | Fingerprint важен, но generation/ranking тоже bottleneck. |
| Oracle refinement model | Refined Tanimoto хуже baseline | Текущая refinement-постановка не рабочая. |

## Full base metrics

Полный base-прогон MSG test:

```text
shards: 36
spectra: 17082
exact_top1: 0.1097
exact_top10: 0.1239
tanimoto_top1: 0.4598
tanimoto_top10: 0.4842
mist_tanimoto: 0.5407
formula_success: 0.9139
```

Formula matching в base не главный bottleneck: sampler почти всегда собирает
кандидатов. Основная проблема дальше:

```text
MIST fingerprint quality + DLM robustness + candidate ranking.
```

## DLM tuning: где оно

Планированный tuning DLM уже оформлен в проекте как MIST-fingerprint adaptation.

Файлы:

- `docs/DLM_FINGERPRINT_ROBUSTNESS_RESULTS.md`
- `docs/FRIGID_OPERATIONAL_USAGE.md`
- `scripts/benchmark_dlm_fingerprint_robustness.py`
- `scripts/export_mist_fingerprints.py`
- `configs/fp2mol_finetune_mist_fingerprints.yaml`

Смысл:

```text
1. экспортировать train-split MIST fingerprints;
2. fine-tune DLM не на clean fingerprints, а на MIST-predicted fingerprints;
3. переоценить тем же robustness benchmark:
   original DLM vs adapted DLM
   ground_truth fingerprints vs mist_binary fingerprints.
```

Команды из текущего workflow:

```bash
python scripts/export_mist_fingerprints.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --split train \
  --output-dir results/mist_fingerprints_train
```

```bash
python scripts/train.py --config-name fp2mol_finetune_mist_fingerprints \
  load_weights_only=checkpoints/DLM.ckpt \
  data.predicted_fingerprint_metadata=results/mist_fingerprints_train/metadata.csv \
  data.predicted_fingerprint_npz=results/mist_fingerprints_train/fingerprints.npz \
  data.predicted_fingerprint_key=mist_binary
```

```bash
python scripts/benchmark_dlm_fingerprint_robustness.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir data/msg \
  --mist-checkpoint checkpoints/mist_msg.pt \
  --dlm-checkpoint <adapted_dlm.ckpt> \
  --fingerprint-sources ground_truth mist_binary \
  --output-dir runs/adapted_dlm_robustness
```

На `spectrum` также была отдельная незакоммиченная ablation:

- `configs/fp2mol_finetune_noisy_fp.yaml`
- `scripts/submit_noisy_fp_finetune.sbatch`

Это другой вариант: добавлять synthetic fingerprint noise через
`fingerprint_flip_prob`. Основной задокументированный путь лучше начинать не с
него, а с реальных exported MIST fingerprints.

## Текущий запуск 2026-07-06

Рабочий каталог на `spectrum`:

```text
/home/nikolenko/work/Projects/FRIGID_dlm_mist_adapt_cbc854
```

Что сделано:

- exported full train MIST fingerprints: `191216` rows;
- smoke DLM training на 8 examples прошёл;
- найден и исправлен bug с CPU/CUDA tensors в `src/dlm/model.py`;
- subset adaptation на 4096 examples, 100 steps, завершился;
- full adaptation на exported train fingerprints, 2500 steps, завершился;
- paired benchmark original vs adapted на 64 spectra завершился.

Ключевые артефакты:

```text
export: runs/mist_fingerprint_exports/train_full_20260706T145959Z
adapted checkpoint: runs/dlm_mist_fulltrain_steps2500_20260706T2031Z/checkpoints/2500.ckpt
adapted benchmark: runs/benchmarks/dlm_mist_full2500_max64_fm2_attempt20
original benchmark: runs/benchmarks/dlm_original_max64_fm2_attempt20
```

64-spectrum benchmark, formula matches `2`, max attempts `20`:

```text
original ground_truth tan@1: 0.3897
original mist_binary  tan@1: 0.3209
original gap: -0.0689

adapted  ground_truth tan@1: 0.3109
adapted  mist_binary  tan@1: 0.2796
adapted  gap: -0.0313
```

Вывод: adaptation уменьшила gap, но ухудшила absolute quality. Это не успешный
DLM tuning. Продолжать этот checkpoint не стоит.

Linear:

- `SPA-75`: paired robustness evaluation, done;
- `SPA-76`: update docs with adapted metrics, done;
- `SPA-85`: CPU/CUDA training fix, done.

## Текущий запуск 2026-07-07

Следующий DLM objective был mixed training:

```text
50% ground_truth fingerprints
50% mist_binary fingerprints
```

Идея: не уводить decoder полностью в noisy MIST fingerprints, а держать его
привязанным к clean fingerprint manifold.

Что сделано:

- добавлен config `configs/fp2mol_finetune_mixed_fingerprints.yaml`;
- `PredictedFingerprintDataset` теперь умеет брать несколько fingerprint keys
  из одного NPZ и выбирать их с заданными вероятностями;
- smoke dataset и 2-step train прошли;
- первый full run с batch `512` упал OOM на A100 80GB;
- batch уменьшен до `32`;
- full mixed training доведён до `10000` steps.

Ключевые артефакты:

```text
training run: runs/dlm_mixed_fingerprint_adaptation_20260707T0811Z
final checkpoint: runs/dlm_mixed_fingerprint_adaptation_20260707T0811Z/checkpoints/10000.ckpt
benchmark: runs/benchmarks/dlm_mixed10000_max64_fm2_attempt20_v2
```

Важно: первый benchmark был остановлен, потому что был передан неверный MIST
checkpoint path. Правильный benchmark v2 использовал:

```text
/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt
```

64-spectrum benchmark, formula matches `2`, max attempts `20`:

```text
original ground_truth tan@1: 0.3897
original mist_binary  tan@1: 0.3209

mixed   ground_truth tan@1: 0.3486
mixed   mist_binary  tan@1: 0.2870
mixed   exact@1/exact@10: 0.0000 / 0.0000
```

Gate:

```text
mist_binary must beat original 0.3209
ground_truth must stay within 5% of original 0.3897, i.e. >= 0.3702
```

Результат:

```text
mist_binary: 0.2870 < 0.3209  FAIL
ground_truth: 0.3486 < 0.3702 FAIL
```

Вывод: mixed DLM adaptation тоже не сработала. Она не просто не улучшила MIST
fingerprints, а ухудшила оба режима. Продолжать этот checkpoint или расширять
benchmark на 200/1024 spectra не стоит.

## Текущий запуск 2026-07-08

После провала plain и mixed DLM adaptation проверили быстрые inference-side
рычаги перед новым длинным training run.

Ключевой вывод failure analysis:

```text
MIST fingerprint errors сильно коррелируют с generation quality.
Главный вред сейчас дают false-positive bits, а не нехватка активных bits.
```

Рабочий каталог на `spectrum`:

```text
/home/nikolenko/work/Projects/FRIGID_dlm_mist_adapt_cbc854
```

Ключевые артефакты:

```text
benchmark root: runs/benchmarks/bold_20260708
best 64 run: runs/benchmarks/bold_20260708/e11_best_threshold_0p50_64
best threshold: 0.50
```

Что проверили:

| Experiment | Result | Decision |
| --- | --- | --- |
| `e01_soft_mist_probs_64` | partial 40 spectra: tan@1 `0.1258`, formula success `0.0000` | Early stop. Raw soft probabilities не подходят напрямую. |
| `e02_threshold_0p12_32` | tan@1 `0.3152` | Reject. Слишком много noisy bits. |
| `e03_threshold_0p15_32` | tan@1 `0.3340` | Reject. Ниже first32 baseline `0.3584`. |
| `e04_threshold_0p22_32` | tan@1 `0.3690` | First positive signal. |
| `e05_threshold_0p30_32` | tan@1 `0.3801` | Лучше `0.22`; decoding queue остановлена ради threshold promotion. |
| `e09_threshold_0p40_32` | tan@1 `0.3875` | Лучше `0.30`. |
| `e10_threshold_0p50_32` | tan@1 `0.3927` | Лучший 32-spectrum threshold. |
| `e11_best_threshold_0p50_64` | tan@1 `0.3326` | Прошёл 64-spectrum gate против default `0.3209`. |

64-spectrum comparison:

```text
default threshold 0.187, mist_binary tan@1: 0.3209
strict threshold  0.50,  mist_binary tan@1: 0.3326
delta: +0.0118
```

Paired 64-spectrum result:

```text
wins/losses: 28 / 36
mean delta: +0.0118
median delta: -0.0044
```

Вывод: threshold `0.50` не решает exact-match bottleneck, но это первый
положительный gate после двух неудачных DLM training попыток. Следующее
направление должно быть не raw `mist_probs`, а confidence-aware sparsification:
adaptive threshold, top-k bits, или per-spectrum confidence gate.

## Что делать дальше

1. Не продолжать текущий `mist_binary` full-DLM checkpoint.
2. Не продолжать mixed `ground_truth + mist_binary` checkpoint.
3. Использовать threshold `0.50` как текущий лучший inference-side baseline.
4. Расширить threshold `0.50` benchmark на 200/1024 spectra или полный test
   subset, чтобы проверить, держится ли небольшой gain.
5. Следующий смелый fingerprint-side sweep:
   - adaptive threshold по spectrum confidence;
   - top-k active bits вместо fixed threshold;
   - calibrated sparsification, которая режет false-positive bits без сильных
     false negatives.
6. Decoding/ranking sweep оставить вторым приоритетом: exact-match всё ещё `0`,
   но сегодняшний положительный signal пришёл именно от fingerprint sparsity.

Gate для следующего DLM tuning:

```text
mist_binary quality must improve without collapsing ground_truth quality
```

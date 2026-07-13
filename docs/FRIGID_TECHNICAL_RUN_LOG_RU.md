# FRIGID experiment report

Краткий статус по FRIGID/MassSpecGym MSG на июль 2026.

## Главный вывод

Прямая замена MIST на DreaMS и два варианта DLM adaptation не сработали.
Самый сильный текущий результат даёт decoding: NGBoost подсказывает DLM длину
SAFE-последовательности, а увеличенный generation budget помогает находить
formula-matched кандидатов. Вклад этих двух факторов ещё нужно разделить.

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
| Threshold `0.50`, 200 spectra | Default tan@1 `0.2781`; strict `0.50` tan@1 `0.2861`; CI `[-0.0042, +0.0206]` | Weak positive. Держать как baseline, но не продвигать сразу на full. |
| Top-k sparsification, 32 spectra | top-k `32` tan@1 `0.4439` против fixed `0.50` tan@1 `0.3927` | Сильный exploratory signal. Продвинут на 64. |
| Top-k `32`, 64 spectra | top-k tan@1 `0.3401`; fixed `0.50` tan@1 `0.3326`; CI против fixed `[-0.0269, +0.0461]` | Лучше default, но слабый/нестабильный gain против fixed `0.50`. Продвинут на 200 только как рискованный gate. |
| Top-k `32`, 200 spectra | top-k tan@1 `0.2668`; default `0.2781`; fixed `0.50` `0.2861`; delta vs fixed `-0.0193`, CI `[-0.0350, -0.0030]` | Reject. Это small-subset artifact, не robust improvement. |
| MIST confidence gate, retrospective 200 | entropy rule: policy tan@1 `0.2959` против default `0.2781` и fixed `0.2861` | Выглядит promising, но это fitted на том же 200 subset. Требует holdout. |
| MIST confidence gate, holdout 32, start-index `200` | default `0.3410`; fixed `0.3299`; conditional entropy `0.3381` | Reject for promotion. Rule уменьшил вред fixed, но не победил default. |
| NGBoost token-length + 100 attempts, holdout 8 | baseline tan@1 `0.3194`, formula success `0`; NGBoost tan@1 `0.6068`, formula success `0.625` | Strong positive decoding-side signal. Продвинут на 16. |
| NGBoost token-length + 100 attempts, holdout 16 | baseline tan@1 `0.3113`; NGBoost tan@1 `0.5835`, tan@10 `0.5846`; wins `16/16`; formula success `0.6875` | Первый сильный promote после threshold failures. Следующий gate: 32/64. |
| NGBoost token-length + 100 attempts, holdout 32 | baseline tan@1 `0.3410`; NGBoost tan@1 `0.6105`, tan@10 `0.6110`; wins `32/32`; formula success `0.6563`; exact `0` | Strong positive. Продвинут на 64. Exact-match bottleneck остаётся, нужен reranking/refinement track. |
| NGBoost token-length + 100 attempts, holdout 64 | baseline tan@1 `0.3265`, formula success `0.0156`; NGBoost tan@1 `0.6044`, formula success `0.6406`; exact `0` | Эффект сохранился. Продвинут на 200. |
| NGBoost token-length + 100 attempts, holdout 200 | baseline tan@1 `0.3621`, exact@1 `0`, formula success `0.025`; NGBoost tan@1 `0.7594`, exact@1 `0.245`, exact@10 `0.270`, formula success `0.840` | Очень сильный результат, но subset имеет необычно высокий MIST Tanimoto `0.9056`; это ещё не full-split доказательство. |
| Новый full baseline, partial 10,000/17,082 | tan@1 `0.3975`, exact@1 `0.0775`, exact@10 `0.0830`, formula success `0.5099`, MIST Tanimoto `0.5543` | Baseline продолжает считаться на `spectrum`; full NGBoost стартует после него. |
| NGBoost validation, 1,024 spectra | baseline first-1024: tan@1 `0.4409`, exact@1 `0.0879`, formula success `0.4316`, MIST Tanimoto `0.6978` | NGBoost+100 запущен на `lobachevsky`; результат ожидается. |
| NGBoost/budget isolation, fresh holdout 64 | no-NGBoost `20 -> 100`: tan@1 `0.5873 -> 0.6734`, exact@10 `0.2031 -> 0.4219`; NGBoost100 tan@1 `0.6495`, exact@10 `0.4063` | Основной quality gain даёт generation budget. При одинаковых 100 attempts NGBoost снижает tan@1 на `0.0239`; он быстрее, но не quality leader. |
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

200-spectrum scale-up:

```text
default threshold 0.187, mist_binary tan@1: 0.2781
strict threshold  0.50,  mist_binary tan@1: 0.2861
delta: +0.0080
wins/losses: 104 / 96
bootstrap 95% CI for tan@1 delta: [-0.0042, +0.0206]
```

Вывод по 200 gate: fixed threshold `0.50` остаётся текущим inference baseline,
но это weak positive, а не уверенный promote на 1024/full. Следующий активный
трек: adaptive/top-k MIST sparsification и отдельный spectral reranking track.

Top-k sparsification follow-up:

```text
32 spectra:
fixed threshold 0.50 tan@1: 0.3927
top-k 32             tan@1: 0.4439

64 spectra:
default threshold 0.187 tan@1: 0.3012
fixed threshold 0.50    tan@1: 0.3326
top-k 32                tan@1: 0.3401

200 spectra:
default threshold 0.187 tan@1: 0.2781
fixed threshold 0.50    tan@1: 0.2861
top-k 32                tan@1: 0.2668
```

Paired 200-spectrum result:

```text
top-k 32 minus default:
tan@1 delta: -0.0113
95% CI: [-0.0272, +0.0056]
wins/losses: 85 / 115

top-k 32 minus fixed 0.50:
tan@1 delta: -0.0193
95% CI: [-0.0350, -0.0030]
wins/losses/ties: 77 / 122 / 1
```

Вывод: fixed top-k `32` отвергнут. Он выглядел очень хорошо на 32 spectra и
слегка лучше на 64, но на 200 стал хуже и default, и fixed `0.50`. Это важный
negative result: DLM иногда любит очень sparse fingerprints на маленьком subset,
но fixed sparsity не переносится. Следующий sparsification-трек должен быть
confidence-gated, а не fixed top-k.

Confidence-gated threshold follow-up:

```text
Retrospective 200, fitted on the same completed 200 subset:
default threshold 0.187 tan@1: 0.2781
fixed threshold 0.50    tan@1: 0.2861
entropy conditional     tan@1: 0.2959

rule:
use threshold 0.50 if mist_prob_entropy_norm <= 0.024121665860137063
else use threshold 0.187
```

Holdout 32 spectra, `start-index 200`:

```text
default threshold 0.187 tan@1: 0.3410
fixed threshold 0.50    tan@1: 0.3299
entropy conditional     tan@1: 0.3381

conditional - default:
tan@1 delta: -0.0029
wins/losses/ties: 0 / 1 / 31

conditional - fixed 0.50:
tan@1 delta: +0.0082
wins/losses/ties: 11 / 8 / 13
```

Вывод: simple MIST entropy gate не прошёл prospective holdout. Он полезен как
диагностика и снижает риск fixed `0.50`, но не даёт нового baseline. Это
останавливает threshold-only направление: следующий сильный трек должен быть
decoder/reranking или более богатый confidence signal, например DreaMS retrieval.

Decoding-side follow-up:

```text
Holdout 8 spectra, start-index 200:
baseline default 0.187, 20 attempts tan@1: 0.3194
NGBoost token-length, 100 attempts tan@1: 0.6068
formula success: 0.0000 -> 0.6250
wins/losses/ties: 8 / 0 / 0

Holdout 16 spectra, start-index 200:
baseline default 0.187, 20 attempts tan@1: 0.3113
NGBoost token-length, 100 attempts tan@1: 0.5835
NGBoost tan@10: 0.5846
formula success: 0.0000 -> 0.6875
wins/losses/ties: 16 / 0 / 0

Holdout 32 spectra, start-index 200:
baseline default 0.187, 20 attempts tan@1: 0.3410
NGBoost token-length, 100 attempts tan@1: 0.6105
NGBoost tan@10: 0.6110
formula success: 0.0000 -> 0.6563
wins/losses/ties: 32 / 0 / 0
exact@1/exact@10: 0.0000 / 0.0000
```

Вывод: decoding settings дали гораздо более сильный signal, чем threshold/top-k
манипуляции. На 64 spectra эффект сохранился, а на 200 spectra появился уже и
exact match:

```text
64 spectra:
tan@1 0.3265 -> 0.6044
formula success 0.0156 -> 0.6406

200 spectra:
tan@1 0.3621 -> 0.7594
exact@1 0.000 -> 0.245
exact@10 0.000 -> 0.270
formula success 0.025 -> 0.840
wins/losses/ties by tan@1: 194 / 3 / 3
```

Оговорка: 200-gate использовал `start-index=200`, где MIST fingerprints заметно
лучше среднего. Поэтому отдельно запущен 1,024-spectrum gate на первых 1,024
объектах полного split с manifest, subset hash и bootstrap CI. Полный baseline
на `spectrum` прошёл 10,000/17,082; full NGBoost ещё не стартовал.

Причинная проверка на новом 64-spectrum блоке завершена:

```text
no-NGBoost20:  tan@1 0.5873, exact@1 0.1719, exact@10 0.2031, 179 s
NGBoost20:     tan@1 0.6132, exact@1 0.2344, exact@10 0.2656, 130 s
no-NGBoost100: tan@1 0.6734, exact@1 0.3594, exact@10 0.4219, 796 s
NGBoost100:    tan@1 0.6495, exact@1 0.3125, exact@10 0.4063, 406 s
NGBoost200:    tan@1 0.6499, exact@1 0.3906, exact@10 0.3906, 499 s
```

Budget-only effect, no-NGBoost `20 -> 100`: tan@1 `+0.0861`, 95% CI
`[+0.0536, +0.1203]`; exact@10 `+0.2188`. NGBoost effect при одинаковых
`100` attempts: tan@1 `-0.0239`, CI `[-0.0447, -0.0042]`. Значит предыдущий
сильный gain был в основном эффектом большего candidate budget. NGBoost полезен
как ускоритель: он примерно вдвое быстрее, но текущая length prior ухудшает
максимальное качество.

## Что делать дальше

1. Не продолжать текущий `mist_binary` full-DLM checkpoint.
2. Не продолжать mixed `ground_truth + mist_binary` checkpoint.
3. Использовать threshold `0.50` как текущий лучший inference-side baseline.
4. Не продвигать fixed top-k `32`: 200 gate отверг гипотезу.
5. Не продвигать simple entropy conditional threshold: holdout 32 не победил
   default `0.187`.
6. Следующий fingerprint-side sweep делать только с более богатым signal:
   DreaMS/retrieval confidence или calibration, а не один MIST entropy threshold.
7. Завершить 1,024-spectrum и full paired validation NGBoost+100; не считать
   200-gate финальным из-за смещённого subset.
8. Разделить эффект NGBoost и generation budget на одном subset: attempts100
   без NGBoost, NGBoost20, NGBoost100 и повышенный/adaptive budget. Выполнено:
   quality leader сейчас no-NGBoost100.
9. После candidate-recall диагностики открыть второй generator/retrieval track:
   DiffMS/MS-BART или retrieval candidates, затем reranking по spectrum score.
10. Проверить no-NGBoost `200/400/800` attempts и temperature/diversity sweep;
    продвигать только вариант с положительным paired CI против no-NGBoost100.

Gate для следующего DLM tuning:

```text
mist_binary quality must improve without collapsing ground_truth quality
```

## Full four-source orchestration, 2026-07-13

- Frozen DLM jobs `133-168` запущены на `spectrum`; job `133` выполняется.
- Full train-only retrieval job `169` ждёт завершения всех DLM shards.
- Обнаруженный direct MolForge process остановлен с exit status `130`.
- Сохранён exact manifest prefix `8,630/17,082`, SHA-256
  `2b67438042922592e900a5264acb49ab686308d1db52aeda658a15153eca47c7`.
- Resume preflight подтвердил suffix `[8,630, 17,082)` и оба checkpoint hash.
- MolForge suffix job `170` поставлен с dependency `afterany:169`,
  `gpu`, `gres/gpu:1`; он не конкурирует с активным frozen queue.
- Full run root:
  `/home/nikolenko/work/Projects/FRIGID_full_runs/four_source_full_e1b18a9_20260713`.
- Full finalization prepared at commit `01a98d6`: explicit `18 + 18` shard
  lists, nested manifest validation, frozen four-source order, contribution
  ablations and molecule-cluster bootstrap. It remains unqueued until all
  selected shard directories are validated and their list hashes are frozen.
- MCES audit found missing `pulp`/`myopic_mces` in both active environments.
  A separate pinned overlay now uses `myopic-mces 1.0.1`, `PuLP 2.7.0`, and
  CBC SHA-256 `2e170777...`; runtime manifest SHA-256 is `5d5a07bc...`.
- Sharded paper-compatible thresholded-MCES evaluation is prepared at commits
  `d72cc59`/`1def8b6`, but remains unqueued until frozen full control/union
  prediction tables exist and a small Spectrum Slurm timing smoke is possible.
- Strict MCES shard merge is prepared at commit `35ffd06`: exact ordered full
  coverage, frozen provenance/settings, output hashes, per-variant aggregates,
  and `10,000`-sample molecule-cluster bootstrap for `union - control`.
- Control job `133` completed in `07:06:47` and passed an independent audit for
  exact range `[0,1000)`, `1,000` ordered queries, `10,000` score rows, all
  input/output hashes and frozen settings. Audit SHA-256 is `84d0139a...` at
  `audits/job_133_validation.json`; job `134` then started automatically.

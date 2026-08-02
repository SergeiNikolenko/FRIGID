# MARLIN — журнал экспериментов

Этот журнал ведётся в формате отчёта FRIGID: каждый запуск получает номер,
гипотезу, фиксированный panel и budget, наблюдаемые molecular metrics,
решение и проверяемые evidence paths. Номер эксперимента относится к MARLIN
и не переиспользует номера из FRIGID.

## Цель и правила

Цель — получить воспроизводимые Exact@1 и Exact@10 на structure-disjoint
held-out NPLIB1 validation, используя spectrum-derived DreaMS fingerprints,
precursor mass, block decoding, conditioning diversity и mass-shell constraint.
Locked 803-spectrum test не используется для выбора модели.

Правила журнала:

- один эксперимент меняет один причинный фактор или явно помечается как
  инфраструктурный;
- для screening фиксированы `nplib1_val_micro32_v1.tsv`, 16 candidates,
  seed 42, DreaMS threshold `0.95`, block width `8`, tolerance `10 ppm`,
  diversity dropout `0.3`;
- `Exact@1`, `Exact@10`, candidate return, mass validity и strict RDKit
  validity записываются вместе; нулевой Exact не скрывает промежуточные
  failure modes;
- oracle/ground-truth fingerprints разрешены только в диагностике и никогда
  не становятся incumbent;
- promotion требует одинакового scorer, commit, input hashes и трёх seed-ов;
- каждая запись должна содержать Slurm job, ClearML task или явную причину,
  почему публикация в ClearML была невозможна, и путь к артефактам.

## Сводка

| № | Изменение | Panel / budget | Exact@1 | Exact@10 | Return | Mass | Strict | Решение |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Legacy decoder, control | 4×16, seed 42 | 0.003125 | — | 0 | 0 | 0.062500 | Базовая линия |
| 2 | Fingerprint LayerNorm | 4×16, seed 42 | 0.003906 | — | 0 | 0 | 0.078125 | Отклонён |
| 3 | FRIGID layer ordering | 4×16, seed 42 | 0.000781 | — | 0 | 0 | 0.015625 | Отклонён |
| 4 | Три fingerprint self-attention слоя | 4×16, seed 42 | 0.002344 | — | 0 | 0 | 0.046875 | Отклонён |
| 5 | Legacy decoder, 64 candidates | 4×64, seed 42 | 0 | 0 | 0.25 | 0.0625 | 0.027344 | Inference reference |
| 6 | Baseline, три seed-а | 4×16, 42/314159/271828 | 0 | 0 | 0.083333 | 0.083333 | 0.026042 | Высокая дисперсия |
| 7 | FRIGID warm-start parity | 396-row audit | — | — | — | — | — | Подтверждён |
| 8 | Staged adaptation на predicted fingerprints | 4×16, step 1000 | 0 | 0 | 0 | 0 | 0.046875 | Отклонён как molecular run |
| 9 | DreaMS fingerprint threshold parity | 396 rows | — | — | — | — | — | `0.95` зафиксирован |
| 10 | Mass-shell / tokenizer parity | 396 rows | — | — | — | — | — | Маска не виновата |
| 11 | Adaptation на NFS (`609`) | 100 steps | — | — | — | — | — | Инфраструктурный fail: диск |
| 12 | Adaptation на local disk + gpu-shared shard (`616`) | 4/32×16, 100 steps | — | — | — | — | — | Остановлен: panel не укладывался в лимит |
| 13 | Paper-noise adaptation на bounded screen (`617`) | 4×16, 100 steps | — | — | — | — | — | Evaluator config fail |
| 14 | Paper-noise adaptation с micro4 manifest (`618`) | 4×16, 100 steps | 0 | 0 | 0 | 0 | 0.046875 | Отклонён: argmax gate |
| 15 | Paper-noise + multinomial sampling (`619`) | 4×16, 100 steps | 0 | 0 | 0 | 0 | 0 | Отклонён: conditioning dead ends |
| 16 | Threshold-aligned paper-noise adaptation (`620`) | 4×16, 100 steps | 0 | 0 | 0 | 0 | 0.015625 | Отклонён: threshold alone |
| 17 | Full-backbone paper-noise adaptation (`625`) | 4×16, 1000 steps | RUNNING/QUEUED | RUNNING/QUEUED | RUNNING/QUEUED | RUNNING/QUEUED | RUNNING/QUEUED | Следующий causal test |

`—` означает, что метрика не была частью данного parity-аудита или в старом
артефакте не записана. Нули в molecular gate — фактические нули, а не
пропуски.

---

**Эксперимент 1: legacy decoder как control**

Что хотели проверить:

Сначала нужен честный контроль до изменения paper-модулей. Legacy checkpoint
`safe-gpt-pretrain-v3/checkpoints/step=30000.ckpt` запускается с тем же
evaluator, block decoding, строгим SAFE decoding и mass-shell фильтром.

Результат на screening panel `4×16`, seed 42: score `0.003125`, candidate
return `0`, mass validity `0`, strict validity `0.062500`. Это референс для
коротких архитектурных запусков, но не paper-comparable benchmark.

Решение: control сохранён; locked test не открываем.

Доказательства: `docs/MARLIN_AUTORESEARCH.md`, scorer commit
`8082c57050b65d4be87f4a403fb97338ce307871`.

**Эксперимент 2: fingerprint LayerNorm**

Гипотеза: нормализация fingerprint conditioner исправит масштаб входа и
даст decoder более устойчивый сигнал.

Результат: jobs `579/580`, score `0.003906`, return `0`, mass validity `0`,
strict validity `0.078125`.

Решение: отклонён. Снижение loss без molecular return не считается улучшением.

**Эксперимент 3: FRIGID layer ordering**

Гипотеза: порядок paper-блоков при warm start может быть несовместим с текущим
decoder API.

Результат: jobs `581/582`, score `0.000781`, return `0`, mass validity `0`,
strict validity `0.015625`.

Решение: отклонён; ordering без отдельной адаптации ухудшил screening.

**Эксперимент 4: три fingerprint self-attention слоя**

Гипотеза: дополнительное смешивание fingerprint tokens восстановит связи,
которые теряются при noisy conditioning.

Результат: jobs `583/584`, score `0.002344`, return `0`, mass validity `0`,
strict validity `0.046875`.

Решение: отклонён. Инициализированные paper-модули нельзя оценивать коротким
слепым продолжением без staged adaptation.

**Эксперимент 5: увеличение generation budget до 64 candidates**

Что проверяли:

Отделили качество decoder от флуктуаций маленького candidate budget. На job
`586` использован legacy checkpoint, 4 spectra, 64 candidates, seed 42.

Результат: score `0.084554`, candidate return `0.25`, mass validity `0.0625`,
strict validity `0.027344`; для первого spectrum восстановлена формула
`C17H20N2O2` с ошибкой `4.49 ppm`, Morgan Tanimoto `0.197183`. Exact@1 и
Exact@10 остались нулевыми.

Решение: сохранить как inference reference. Generation и mass-shell работают,
но до paper quality далеко.

**Эксперимент 6: независимая трёхseed baseline-проверка**

На том же 4×16 panel прогнали seeds `42`, `314159`, `271828`.

Итог: score `0.046994`, candidate return `0.083333`, mass validity
`0.083333`, strict validity `0.026042`. Разброс велик, поэтому один seed не
может продвигать модель.

Решение: использовать три seed-а только на confirmation gate.

**Эксперимент 7: FRIGID warm-start parity**

Что хотели проверить:

Перед адаптацией нужно доказать, что токенизация fingerprint и стартовые
logits действительно совпадают с FRIGID.

Результат: fingerprint token sets совпали с расхождением `3.73e-8`, final
logits — `1.15e-5`.

Решение: warm start подтверждён; дальнейший провал не объясняется неверной
загрузкой FRIGID весов.

Доказательство: commit `4701a8a`,
`/mnt/netstorage/nikolenko/marlin/runs/autoresearch/parity/frigid-parity-4701a8a.json`.

**Эксперимент 8: staged adaptation на predicted fingerprints**

Гипотеза: decoder нужно постепенно приучить к распределению fingerprints,
которые реально выдаёт spectrum encoder, сохранив FRIGID chemistry.

На step 1000 teacher-forced gate дал token Top-1 `0.252874`, Top-10
`0.856322`, NLL gain correct-vs-shuffled `0.4594`. Это подтверждает активный
conditioning path. Но free-generation gate дал validity `0.046875`, return
`0`, mass validity `0`, uniqueness `0`.

Решение: teacher-forced улучшение не продвинуто в молекулярный incumbent;
нужен free-generation gate на каждом шаге.

Доказательства: Slurm `603`, ClearML task `e2689343ba8f40a5b158c866d12e0cb5`,
run root `/mnt/netstorage/nikolenko/marlin/runs/autoresearch/candidates/frigid-staged-dc76457-step5000`.

**Эксперимент 9: DreaMS fingerprint parity и выбор threshold**

На structure-disjoint validation проверили, не является ли нулевой генератор
следствием пустого или несовместимого fingerprint.

Для `396` rows, `4096` bits, threshold `0.95`: mean Tanimoto `0.328618`,
mean true bits `46.93`, predicted bits `47.87`, TP `23.72`, FP `24.15`, FN
`23.21`. Sweep показал локальный максимум на `0.95` (против `0.3092` на
`0.90` и `0.3240` на `0.97`).

Решение: threshold `0.95` заморожен; threshold-only поиск закрыт.

Доказательство:
`/home/nikolenko/.codex/autoresearch/runs/marlin-nplib1-exact/20260730-paper-parity-r2/artifacts/dreams-fingerprint-parity.json`.

**Эксперимент 10: mass-shell и tokenizer parity**

Проверили альтернативную гипотезу: может быть, корректные токены удаляются
маской или неправильно суммируются массы.

Для всех `396/396` validation paths mass-shell сохранил target-token path,
включая EOS; failures `0`, block width `8`, tolerance `10 ppm`. В tokenizer
max heavy-mass error `0`, atom-count error `0`.

Решение: mass-shell и token mass arithmetic подтверждены; текущая причина
dead ends — не арифметика маски.

Доказательства:
`artifacts/mass-shell-oracle-support.json` и
`artifacts/token-mass-parity.json` в paper-parity run.

**Эксперимент 11: staged adaptation на NFS (`609`)**

Гипотеза: короткий 100-step cross-attention adaptation с real DreaMS inputs
даст первый free-generation signal.

Запуск был создан с теми же input hashes и evaluator, но завершился до
создания `run_manifest.json`: `/mnt/netstorage` был заполнен (80T/80T,
оставался примерно 1 MB). Это инфраструктурный failure, не отрицательный
результат модели.

Решение: не сравнивать его molecular metrics с моделями и не удалять чужие
данные; перенести только outputs/cache на локальный диск.

Доказательство: log
`/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-609.out`.

**Эксперимент 12: тот же adaptation с local-disk fallback (`616`)**

Гипотеза: после устранения storage failure тот же короткий model run должен
дойти до evaluator и вернуть честные molecular metrics.

Изменение только инфраструктурное: run root и offline ClearML cache идут в
локальные `MARLIN_reproduction_20260717/runs/` и
`MARLIN_reproduction_20260717/cache/clearml-offline/`, checkpoint и input bundle не
копируются и остаются на NFS. Job `616` запущен на свободном
`gpu-shared` shard; чужой job `614` не затрагивался.

Параметры: `100` steps, `nplib1_val_micro32_v1.tsv`, `32` spectra,
`16` candidates, seed `42`, threshold `0.95`, block width `8`, `10 ppm`,
diversity dropout `0.3`, cross-attention-only adaptation. Важная оговорка:
этот запуск был создан до исправления параметра и использует
`symmetric_fingerprint_noise.p=0.0`; поэтому он является storage-replacement
control для `609`, а не paper-noise candidate. Новый paper-recipe run будет
иметь `p=0.5`, `rho~U(0.1,0.3)` и получит следующий номер.

Сначала `616` был остановлен после `4/32` строк и `19:16` runtime: при
скорости примерно четыре spectrum за 18 минут 32-row panel не мог завершить
evaluator в двухчасовом Slurm лимите. Полученный partial JSONL сохранён, но
не используется как molecular result и не получает Exact aggregate.

**Эксперимент 13: paper-noise adaptation на bounded screen (`617`)**

Это тот же FRIGID warm start, структура данных, optimizer, mass-shell scorer и
threshold, но с исправленной symmetric fingerprint noise:
`p=0.5`, `rho~U(0.1,0.3)`, equal drop/add. Panel сокращён до предусмотренного
screen `4` spectra × `16` candidates, чтобы получить полный `metrics.json` за
один Slurm лимит; это не финальный paper-comparable score.

Slurm `617` завершил 100 шагов и создал checkpoint, но evaluator завершился
до генерации с ошибкой контракта: `--max-spectra 4` нельзя применять к
32-строчному `nplib1_val_micro32_v1.tsv`. Это не molecular result. Исправление
сделано отдельным benchmark manifest `nplib1_val_micro4_v1.tsv` и launcher
override `MARLIN_EVALUATION_MANIFEST`.

**Эксперимент 14: paper-noise adaptation с micro4 manifest (`618`)**

Повторяем тот же committed recipe (`p=0.5`, `rho~U(0.1,0.3)`, block width 8,
dropout 0.3, mass shell), меняя только evaluator contract на честный
4-row manifest. Теперь `--max-spectra 4` не усекает panel и должен создать
полный `metrics.json` в пределах двух часов.

Slurm `618` завершён (`00:19:30`) и создал полный молекулярный артефакт. На
4 held-out rows он дал Exact@1 `0`, Exact@10 `0`, candidate return `0`, mass
validity `0`, strict validity `0.046875`, validity `0.046875`, mean dead ends
`14`, mean EOS terminations `2`, runtime `700.98 s`. Это честный отрицательный
result для argmax token selection; он не доказывает, что paper multinomial
lane безуспешен.

Артефакты: `metrics.json`, `manifest.json`, `predictions.jsonl` в
`/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/spectrum-fingerprint-adaptation-slurm-618/periodic_molecular/step=100`;
ClearML offline task `offline-0d8c8d7ded524d02b151d85dcd27adbb`; run commit
`e99d362`.

**Эксперимент 15: paper-noise + multinomial sampling (`619`)**

Эксперимент 14 выявил, что periodic evaluator фактически использовал
`token_selection=argmax`, хотя paper/autoresearch contract требует
multinomial sampling при фиксированных seed и temperature. Меняем только
этот inference factor: evaluator теперь получает `--sample-tokens`; training,
fingerprint threshold, mass shell, block width, diversity dropout и panel
остаются без изменений.

Slurm `619` завершён (`00:19:55`) с `token_selection=multinomial`. На 4
held-out rows он дал Exact@1 `0`, Exact@10 `0`, candidate return `0`, mass
validity `0`, validity `0`, strict-valid candidate return `0`, mean dead ends
`15.25`, mean EOS terminations `0.75`, runtime `731.07 s`. Это отвергает
гипотезу, что один только argmax был причиной нулевого molecular return.

Артефакты: `metrics.json`, `manifest.json`, `predictions.jsonl` в
`/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/spectrum-fingerprint-adaptation-slurm-619/periodic_molecular/step=100`;
ClearML offline task `offline-568127ad56ff4a63ae64253846615423`.

**Эксперимент 16: threshold-aligned paper-noise adaptation (`620`)**

Гипотеза: adaptation обучался на binary DreaMS fingerprints с train threshold
`0.90`, а evaluation подавал threshold `0.95`; это оставляло conditioning
distribution shift даже после symmetric noise. Меняем только train threshold
на `0.95`; evaluation остаётся `0.95`, всё остальное (checkpoint, seed, panel,
noise, block width, dropout, mass shell, multinomial) фиксировано.

Slurm `620` завершён (`00:19:16`). Aligning train threshold to `0.95` не
дал улучшения: Exact@1/10 `0`, candidate return `0`, mass validity `0`,
validity `0.015625`, strict validity `0.015625`, mean dead ends `14.25`, EOS
`1.75`. Артефакт:
`/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/spectrum-fingerprint-adaptation-slurm-620/periodic_molecular/step=100/metrics.json`;
ClearML offline task `offline-fe140241d58543019853c6e66b786941`.

Доказательства запуска:

- Slurm job `616` (partial control): log `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-616.out`;
- Slurm job `617` (paper-noise control with evaluator config fail): log `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-617.out`;
- Slurm job `618` (active paper-noise screen): log `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-618.out`;
- run roots `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/spectrum-fingerprint-adaptation-slurm-{616,617,618}`;
- offline caches `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/cache/clearml-offline/{616,617,618}`;
- offline ClearML task for `616`: `offline-bf2f93df2ff741caa4cf4b155624ba3c`;
- offline ClearML task for `618`: `offline-0d8c8d7ded524d02b151d85dcd27adbb`;
- Slurm job `619` (active multinomial screen): log `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-619.out`;
- Slurm job `620` (active threshold-aligned screen): log `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-620.out`;
- Slurm job `625` (queued full-backbone adaptation): log `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-625.out`;
- launcher local-disk fix commit `7ae6cb6`, paper-noise commit `8388db0`,
  bounded-screen override commit `25f83ad`, micro4 manifest/override commit
  `bbbfd25`, task-manifest commit `e99d362`, multinomial gate commit
  `a1926a7`, threshold override commit `af7d00c`.

Параметр paper-noise включён в launcher и manifest после этого запуска,
commit `8388db0`; тесты `tests/test_marlin.py`,
`tests/test_marlin_spectrum_dataset.py`, `tests/test_expanding_marlin.py` и
`tests/test_marlin_paper_recipe.py` прошли (`65 passed`).

## Следующая запись

**Эксперимент 17: full-backbone paper-noise adaptation (`625`) — отправлен**

После трёх коротких screen-итераций (argmax, multinomial и threshold-aligned)
отдельный фактор — область обновляемых параметров. Запущен FRIGID warm-start с
`p=0.5`, `rho~U(0.1,0.3)`, train/eval threshold `0.95`, block width `8`,
diversity dropout `0.3`, multinomial molecular gate и тем же held-out
`nplib1_val_micro4_v1.tsv`. Первые `100` шагов обновляют только
fingerprint cross-attention, затем оставшиеся `900` шагов разрешают весь
backbone; молекулярная оценка выполняется только в конце, чтобы не тратить
двухчасовой Slurm budget на четыре дорогих промежуточных evaluator-а.

Slurm job `625` принят в очередь `gpu-shared` (`PENDING` на момент записи;
чужой job `614` не затрагивается). До появления финального `metrics.json`
этот запуск не считается результатом и не получает Exact aggregate.

Эксперимент 12 закрывается только после Slurm `COMPLETED` и наличия
`periodic_molecular/step=100/metrics.json`. Если return и mass validity
останутся нулевыми, следующий номер посвящается диагностике termination /
grammar dead ends или conditioning distribution — не увеличению длины прогона
вслепую. Если появится non-zero return, тот же checkpoint проходит
неизменённый three-seed confirmation, затем micro64/macro64.

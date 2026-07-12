#

В FRIGID есть цепочка:

```text
MS/MS спектр -> encoder -> fingerprint -> DLM decoder -> молекула
```

То есть у нас есть спектр вещества, и мы хотим восстановить молекулу.

Сейчас рабочая схема такая:

```text
спектр -> MIST -> fingerprint -> DLM -> молекула
```

Где:

- `MIST` пытается по спектру предсказать molecular fingerprint;
- `fingerprint` это грубое описание молекулы в виде 4096 битов;
- `DLM` по fingerprint пытается сгенерировать молекулу.

Мы пытались понять: где именно система теряет качество и что лучше улучшать.

**Что выяснили в целом**

Мы думали, что можно заменить `MIST` на `DreaMS`, потому что DreaMS тоже модель для спектров. Но почти все прямые DreaMS-попытки оказались слабее MIST.

Зато нашли другую большую проблему: DLM хорошо работает, когда ему дают “идеальный” fingerprint, но сильно хуже, когда ему дают fingerprint, который реально предсказывает MIST.

То есть проблема выглядит так:

```text
DLM умеет читать чистые fingerprints
но плохо читает шумные MIST fingerprints
```

Поэтому главный рабочий путь сейчас:

```text
не срочно менять MIST,
а научить DLM жить с fingerprints, которые реально выдаёт MIST
```

**Краткий статус сложных моделей**

| Модель или подход | Статус | Что получилось |
| --- | --- | --- |
| MolForge | подтверждён | Даёт независимые кандидаты; на 1,024 добавил `+0.0178` Tanimoto top-10 и `+0.0186` Exact top-10 поверх union. |
| DreaMS | отклонён как замена MIST | Frozen, calibration, distillation и full fine-tune остались далеко от MIST; full fine-tune переобучился. |
| DLM adaptation | отклонён | Gap между clean и MIST уменьшился, но абсолютное качество стало хуже. |
| NGBoost | только ускорение | При одинаковом budget быстрее, но хуже обычного DLM по качеству. |
| ICEBERG / GEMS-style refinement | диагностика | Код и малый запуск есть, но честного paired улучшения пока нет. |
| DiffMS | исследован | Интеграционный путь понятен, подтверждённых FRIGID-метрик нет. |
| MBGen | заблокирован | Нет совместимых публичных весов и полного loader; есть проблемы evaluation. |
| DualLGD | заблокирован интерфейсом | Native Morgan-2048 и собственный spectrum encoder несовместимы с текущим MIST Morgan-4096 без отдельного доказательства. |
| Selective TTT | подготовлен | Train-only neighbor builder реализован и протестирован, quality gate ещё не запускался. |
| Spectral JEPA | исследовательская ветка | Подтверждённого downstream improvement пока нет. |

Главный доказанный результат сейчас получен не полной заменой архитектуры, а
объединением разных источников кандидатов:

```text
DLM control + DLM temperature 0.8 + train-only retrieval + MolForge 0.172
```

На 1,024 molecule-diverse spectra этот union дал Tanimoto top-10 `0.5277`
против `0.4805` у DLM baseline и Exact top-10 `0.2119` против `0.1650`.

**Эксперимент 1: DreaMS как простая замена MIST**

Что хотели проверить:

А вдруг можно заменить MIST на DreaMS напрямую?

Схема была такая:

```text
спектр -> DreaMS -> маленькая neural head -> fingerprint
```

Что сделали:

- взяли большой train split;
- взяли validation split;
- прогнали спектры через DreaMS;
- получили DreaMS embeddings;
- сверху обучили MLP, который должен предсказывать 4096-bit Morgan fingerprint.

Зачем:

Если DreaMS embeddings уже содержат хорошую информацию о молекуле, то маленькая head должна научиться получать fingerprint.

Результат:

- MIST baseline: примерно `0.542` fingerprint Tanimoto;
- DreaMS frozen head: только `0.124`.

Простыми словами: очень плохо.

Что именно было плохо:

- модель предсказывала слишком много активных bits;
- в target fingerprint обычно около `64` активных bits;
- DreaMS head предсказывала около `296`;
- то есть она “мазала широко”, но неточно.

Вывод:

```text
Просто взять DreaMS embeddings и обучить сверху head недостаточно.
```

**Эксперимент 2: починить DreaMS head loss-ами и calibration**

После первого провала было неясно: может DreaMS плохой, а может мы просто плохо выбрали threshold/loss.

Что попробовали:

- разные веса positive bits;
- soft-Tanimoto loss;
- penalty за слишком много активных bits;
- threshold sweep;
- top-k calibration.

Зачем:

Первая модель включала слишком много bits. Мы пытались заставить её быть аккуратнее и выбирать примерно правильное число активных bits.

Результат:

- стало лучше, чем `0.124`;
- лучший pure DreaMS стал около `0.234`;
- но MIST всё ещё `0.542`.

Вывод:

```text
Calibration помогает, но не решает проблему.
DreaMS frozen head всё равно далеко от MIST.
```

**Эксперимент 3: смешать MIST и DreaMS**

Потом мы подумали:

Ок, DreaMS сам по себе слабее. Но может он знает что-то дополнительное к MIST?

Сделали blend:

```text
final_prediction = alpha * MIST + (1 - alpha) * DreaMS
```

То есть проверяли разные смеси:

- 100% DreaMS;
- 50% MIST + 50% DreaMS;
- 95% MIST + 5% DreaMS;
- 100% MIST.

Зачем:

Если DreaMS даёт полезный дополнительный сигнал, смесь должна быть лучше чистого MIST.

Результат:

- лучший вариант: `100% MIST`;
- `95% MIST + 5% DreaMS` был почти такой же, но чуть хуже;
- DreaMS не добавил полезного сигнала.

Вывод:

```text
Простое смешивание MIST и DreaMS не улучшает модель.
```

**Эксперимент 4: DreaMS как correction к MIST**

Дальше сделали более умную вещь.

Не заменять MIST и не смешивать руками, а обучить adapter:

```text
MIST prediction + correction from DreaMS
```

Идея:

MIST уже хороший. Может DreaMS сможет исправлять только те места, где MIST ошибается.

Схема:

```text
final_logits = mist_logits + small_correction(DreaMS embedding, MIST logits)
```

Зачем:

Это сильнее, чем blend. Adapter может учить разные correction для разных bits и разных spectra.

Что попробовали:

- unanchored residual;
- anchored residual;
- разные residual scales;
- MIST anchor, чтобы модель не уехала далеко от хорошего MIST baseline.

Результат:

- unanchored correction портил MIST;
- anchored correction дал маленький плюс;
- лучший gain: `+0.00068`.

Но gate был `+0.005`.

Простыми словами:

```text
Да, DreaMS чуть-чуть помогает, но настолько мало, что не стоит запускать дорогой DLM benchmark.
```

Вывод:

```text
DreaMS содержит какой-то слабый полезный сигнал,
но текущий residual adapter не даёт практического улучшения.
```

**Эксперимент 5: DreaMS учится у MIST**

Потом попробовали distillation.

Идея:

Если MIST сильный, пусть DreaMS учится не только на ground truth fingerprint, но и на MIST predictions.

Схема:

```text
DreaMS -> fingerprint
loss = target fingerprint + MIST teacher signal
```

Зачем:

Может DreaMS не понимает задачу так, как MIST, и teacher поможет ему приблизиться.

Результат:

- лучший вариант: около `0.240`;
- MIST: `0.542`.

Вывод:

```text
Distillation от MIST не спасает frozen DreaMS.
Он всё равно далеко.
```

**Эксперимент 6: full fine-tune DreaMS**

Потом проверили главный вопрос:

Может проблема в том, что DreaMS был frozen? Может надо обучать сам encoder?

Сделали full fine-tune:

```text
спектр -> trainable DreaMS encoder -> fingerprint head
```

То есть уже обучали не только маленькую head, а сам DreaMS encoder тоже.

Зачем:

Это самый прямой способ проверить, может ли DreaMS адаптироваться под FRIGID/MassSpecGym fingerprint задачу.

Результат:

- train Tanimoto вырос сильно: до `0.84`;
- validation Tanimoto максимум: `0.258`;
- потом validation начал ухудшаться.

Простыми словами:

```text
Модель выучила train,
но не научилась хорошо обобщать на validation.
```

Это overfit.

Вывод:

```text
Просто full fine-tune DreaMS с BCE + soft-Tanimoto тоже не работает.
Он лучше frozen head, но всё равно в два раза хуже MIST.
```

**Промежуточный вывод по DreaMS**

Мы попробовали много способов заставить DreaMS заменить или улучшить MIST:

- frozen head;
- better losses;
- calibration;
- blend;
- residual correction;
- distillation;
- full fine-tune.

И все они не прошли gate.

Картина такая:

| Подход | Результат |
| --- | ---: |
| MIST baseline | `~0.542` |
| DreaMS frozen head | `~0.124` |
| DreaMS calibrated/loss-tuned | `~0.234` |
| DreaMS distilled | `~0.240` |
| DreaMS full fine-tune | `~0.258` |
| MIST + DreaMS residual | `+0.00068` к MIST |

Поэтому вывод:

```text
Старые прямые DreaMS-подходы не дают замену MIST.
```

Но это не значит “DreaMS умер навсегда”. Это значит:

```text
не надо повторять те же DreaMS experiments;
если продолжать DreaMS, надо менять саму постановку задачи.
```

**Эксперимент 7: понять, где реальная боль FRIGID**

После DreaMS стало понятно, что надо проверить саму связку:

```text
MIST -> DLM
```

Мы сравнили два режима:

```text
DLM + ground_truth fingerprint
DLM + mist_binary fingerprint
```

То есть всё одинаковое, меняется только fingerprint:

- `ground_truth`: идеальный fingerprint из настоящей молекулы;
- `mist_binary`: fingerprint, который предсказал MIST по спектру.

Зачем:

Если DLM хорошо работает с ground_truth, но плохо с MIST fingerprint, значит проблема не только в encoder, а ещё в том, что decoder не умеет работать с noisy fingerprints.

Результат на 1,400 spectra:

| Metric | Ground-truth | MIST binary |
| --- | ---: | ---: |
| Exact top-1 | `0.4879` | `0.1386` |
| Tanimoto top-1 | `0.8130` | `0.5677` |
| Formula success | `0.7643` | `0.6936` |

Вывод:

```text
Главный разрыв возникает на переходе от MIST fingerprint к DLM.
Поэтому дальше мы тюним generation и ranking, а не повторяем DreaMS head.
```

**Эксперимент 8: дообучить DLM на реальных MIST fingerprints**

Что хотели проверить:

Если DLM плохо понимает шумный fingerprint, можно дообучить его именно на
fingerprints, которые выдаёт MIST.

Результат на 64 spectra:

| Model | Ground-truth Tanimoto | MIST Tanimoto |
| --- | ---: | ---: |
| Original DLM | `0.3897` | `0.3209` |
| Adapted DLM | `0.3109` | `0.2796` |

Разрыв между clean и noisy fingerprints уменьшился, но только потому, что
модель стала хуже в обоих режимах.

Вывод:

```text
Текущее дообучение DLM на MIST fingerprints отвергнуто.
Нужна другая training objective или другая архитектура.
```

**Эксперимент 9: сделать MIST fingerprint более строгим**

Что попробовали:

- повысить threshold с `0.187` до `0.50`;
- оставлять только top-32 bits;
- выбирать threshold по уверенности MIST.

Результат:

- threshold `0.50` дал `+0.0118` Tanimoto на 64 spectra;
- на 200 spectra плюс уменьшился до `+0.0080`, доверительный интервал включал ноль;
- top-32 на 200 spectra стал хуже baseline;
- confidence gate на новом holdout тоже не победил baseline.

Вывод:

```text
Простая обработка fingerprint не даёт устойчивого улучшения.
Threshold-only направление остановлено.
```

**Эксперимент 10: NGBoost и большой generation budget**

Сначала NGBoost выглядел как очень большое улучшение. Но одновременно с ним
число попыток генерации выросло с `20` до `100`.

Мы разделили эти два эффекта на одном наборе:

| Decoder | Tanimoto top-1 | Exact top-10 |
| --- | ---: | ---: |
| No NGBoost, 20 attempts | `0.5873` | `0.2031` |
| No NGBoost, 100 attempts | `0.6734` | `0.4219` |
| NGBoost, 100 attempts | `0.6495` | `0.4063` |

Вывод:

```text
Качество выросло главным образом из-за большего candidate budget.
NGBoost примерно вдвое быстрее, но по качеству хуже обычного DLM со 100 attempts.
```

**Эксперимент 11: больше разнообразия при генерации**

Что проверили:

Вместо ещё большего brute force использовали `200 attempts` и температуру
`0.8`, чтобы получить другой набор кандидатов.

Результат на новом наборе из 200 spectra:

| Metric | Baseline | Temperature 0.8 |
| --- | ---: | ---: |
| Exact top-1 | `0.150` | `0.185` |
| Exact top-10 | `0.165` | `0.205` |
| Tanimoto top-1 | `0.5733` | `0.5678` |

Exact top-10 вырос на `+0.040`, 95% CI `[+0.005, +0.080]`.
Tanimoto изменился на `-0.0056`, доверительный интервал включал ноль.

Вывод:

```text
Temperature 0.8 достаёт больше правильных молекул,
но текущий ranking не умеет стабильно ставить лучшие кандидаты наверх.
```

Это полезно не как новый одиночный baseline, а как источник кандидатов для
ensemble и нового reranker.

**Эксперимент 12: проверить, что benchmark измеряет качество честно**

Мы нашли две проблемы:

- соседние строки MassSpecGym часто относятся к одной и той же молекуле;
- два старых длинных запуска случайно использовали `randomness=10.0`.

Из-за этого старый full run и старый 1,024 run нельзя считать валидным
доказательством качества. Мы их остановили.

Что исправили:

- сделали disjoint molecule-diverse наборы на 64, 200 и 1,024 spectra;
- в каждом наборе одна молекула встречается только один раз;
- исправили сохранение финального ranking кандидатов;
- новый 1,024 paired run запущен с корректными параметрами.

Вывод:

```text
Дальше сравниваем архитектуры только на molecule-diverse paired subsets.
Старые ошибочные длинные прогоны в итоговые результаты не входят.
```

**Что делаем дальше**

Теперь нужны не маленькие настройки, а независимые источники кандидатов:

1. `MS-BART`: прямой генератор молекулы по спектру без MIST fingerprint.
2. `Retrieval`: искать похожие train spectra и брать их молекулы как кандидатов.
3. `Generator union`: объединить кандидатов DLM, MS-BART и retrieval.
4. `Spectrum-aware reranker`: ранжировать объединённый список по соответствию
   исходному MS/MS спектру, а не только по близости к MIST fingerprint.

Первый gate: те же 64 molecule-diverse spectra. На 200 продвигается только
подход, который улучшает paired exact@10 или Tanimoto с положительным CI.

**Эксперимент 13: объединить разные источники кандидатов**

Что проверили:

- обычный DLM со `100 attempts`;
- DLM с `temperature=0.8`;
- прямой генератор `MS-BART`;
- retrieval только по train-молекулам;
- общий список кандидатов с единым ranking по MIST fingerprint.

Отдельно новые источники были слабыми:

| Source | Tanimoto top-1 | Tanimoto top-10 | Exact |
| --- | ---: | ---: | ---: |
| MS-BART | `0.1843` | `0.2229` | `0` |
| Train-only retrieval | `0.3032` | `0.3664` | `0` |

Но они находили другие структуры, поэтому retrieval оказался полезен в
объединённом candidate pool.

Результат на 64 molecule-diverse spectra:

| Metric | DLM baseline | DLM + temp + retrieval |
| --- | ---: | ---: |
| Exact top-1 | `0.2188` | `0.2188` |
| Exact top-10 | `0.2500` | `0.2656` |
| Tanimoto top-1 | `0.5175` | `0.5379` |
| Tanimoto top-10 | `0.5427` | `0.5752` |

- Tanimoto top-1: `+0.0204`, 95% CI `[+0.0077, +0.0353]`;
- Tanimoto top-10: `+0.0325`, 95% CI `[+0.0150, +0.0539]`.

Результат повторно воспроизведён стандартным CLI и paired bootstrap в commit
`9a822ee`; ranking не использует правильный ответ.

Вывод:

```text
Новые источники не обязаны побеждать DLM по отдельности.
Главное улучшение даёт разнообразный candidate pool и единый reranking.
```

`MS-BART` в текущем виде не добавляет пользы и остановлен. Связка
`DLM control + DLM temperature 0.8 + train retrieval` продвинута на новый
disjoint molecule-diverse набор из 200 spectra.

Следующий gate: закончить обе DLM-ветки на 200, снова объединить их с retrieval
и перейти на 1,024 только при положительном paired CI. Параллельно проверяем
`MolForge` как ещё один независимый генератор кандидатов.

**Эксперимент 14: подтвердить candidate union на 200 молекулах**

| Metric | DLM baseline | DLM + temp + retrieval |
| --- | ---: | ---: |
| Exact top-1 | `0.1150` | `0.1100` |
| Exact top-10 | `0.1250` | `0.1400` |
| Tanimoto top-1 | `0.4450` | `0.4777` |
| Tanimoto top-10 | `0.4646` | `0.5141` |

- Tanimoto top-1: `+0.0327`, 95% CI `[+0.0195, +0.0469]`;
- Tanimoto top-10: `+0.0495`, 95% CI `[+0.0354, +0.0652]`;
- Exact top-10: `+0.0150`.

Вывод: улучшение подтвердилось на независимом наборе. Candidate union продвинут
на 1,024 молекулы.

**Эксперимент 15: formula и cross-source consensus reranking**

На первых 32 spectra consensus дал `+0.0133` Tanimoto top-1, но на закрытых
последних 32 результат стал хуже:

- Tanimoto top-1: `-0.0027`;
- Tanimoto top-10: `-0.0012`.

Вывод: это overfit на маленьком development-наборе. Reranker отклонён и на 200
не запускался.

**Эксперимент 16: MolForge как новый источник кандидатов**

MolForge отдельно слабее текущего union, но нашёл другие правильные структуры.

| Metric | Current union | Union + MolForge 0.172 |
| --- | ---: | ---: |
| Exact top-1 | `0.2188` | `0.2500` |
| Exact top-10 | `0.2656` | `0.3594` |
| Tanimoto top-1 | `0.5379` | `0.5580` |
| Tanimoto top-10 | `0.5752` | `0.6177` |

- Exact top-10: `+0.0938`, 95% CI `[+0.0312, +0.1719]`;
- Tanimoto top-10: `+0.0426`, 95% CI `[+0.0117, +0.0806]`.

Threshold `0.5` оказался хуже, а объединение двух MolForge thresholds не дало
плюса сверх `0.172`. Поэтому на 200 продвинут только `MolForge 0.172`.

На независимых 200 молекулах улучшение подтвердилось:

| Metric | Current union | Union + MolForge 0.172 |
| --- | ---: | ---: |
| Exact top-1 | `0.1100` | `0.1400` |
| Exact top-10 | `0.1400` | `0.1900` |
| Tanimoto top-1 | `0.4777` | `0.5018` |
| Tanimoto top-10 | `0.5141` | `0.5407` |

- Exact top-10: `+0.0500`, 95% CI `[+0.0200, +0.0850]`;
- Tanimoto top-10: `+0.0266`, 95% CI `[+0.0120, +0.0426]`.

Вывод: MolForge стабильно добавляет независимые правильные структуры. Ветка
продвинута на 1,024 molecule-diverse spectra.

**Эксперимент 17: MBGen many-body graph diffusion**

Официальный код проверен, но эксперимент отложен:

- публичных trained checkpoints нет;
- loader опубликован не полностью;
- в evaluation есть batch-dependent padding ошибки;
- лицензия репозитория не указана.

Обучение с нуля не запускали: без исходных весов это не честное сравнение
архитектур. Вернёмся после публикации checkpoint или через отдельный
scorer-backed DiffMS adapter.

**Эксперимент 18: DualLGD graph diffusion audit**

DualLGD проверен как ещё один независимый graph-diffusion decoder. У проекта
есть официальный код, MIT-лицензия и опубликованные weights, но готовый
checkpoint не принимает текущий `MIST` fingerprint из FRIGID:

- его spectrum encoder и projection являются частью модели;
- conditioning использует `Morgan-2048`, а текущий FRIGID contract использует
  `Morgan-4096`;
- поэтому нельзя честно подать в checkpoint текущий MIST output как drop-in
  replacement для DLM.

Вывод:

```text
DualLGD не запускаем как baseline.
Сначала нужен train-only test, что фиксированное folding Morgan-4096
строго эквивалентно native Morgan-2048; только после него допустим
locked 64 -> 200 candidate-union gate.
```

Это сохраняет идею как потенциально сильную архитектурную ветку, но исключает
ложное улучшение от несовместимого encoder или test-label formula leakage.

**Эксперимент 19: полный candidate union на 1,024 молекулах**

Обе DLM-ветки и MolForge завершились на закрытом molecule-diverse наборе.

| Metric | DLM baseline | DLM + temp + retrieval | + MolForge 0.172 |
| --- | ---: | ---: | ---: |
| Exact top-1 | `0.1387` | `0.1514` | `0.1611` |
| Exact top-10 | `0.1650` | `0.1934` | `0.2119` |
| Tanimoto top-1 | `0.4535` | `0.4724` | `0.4845` |
| Tanimoto top-10 | `0.4805` | `0.5099` | `0.5277` |

Union против DLM baseline:

- Tanimoto top-1: `+0.0189`, 95% CI `[+0.0148, +0.0232]`;
- Tanimoto top-10: `+0.0294`, 95% CI `[+0.0246, +0.0344]`;
- Exact top-10: `+0.0283`, 95% CI `[+0.0186, +0.0391]`.

MolForge поверх union:

- Tanimoto top-1: `+0.0120`, 95% CI `[+0.0079, +0.0168]`;
- Tanimoto top-10: `+0.0178`, 95% CI `[+0.0129, +0.0231]`;
- Exact top-10: `+0.0186`, 95% CI `[+0.0107, +0.0273]`.

Вывод:

```text
Улучшение выдержало масштабирование 64 -> 200 -> 1,024.
Лучший подтверждённый вариант: DLM control + temperature + retrieval + MolForge.
```

**Эксперимент 20: быстрый тест, похожий на полную выборку**

Старые 64/200/1,024 наборы содержали по одной молекуле и поэтому не отражали
частоты повторных spectra в полном тесте. Сделаны два взаимодополняющих gate:

- `micro128/256/512` сохраняют распределение всех 17,082 строк;
- `macro64` содержит 64 новые молекулы без пересечения с прежними gate и micro.

Максимальное отклонение средних в standard deviation units:

| Panel | Rows | Unique molecules | Max SMD |
| --- | ---: | ---: | ---: |
| micro128 | 128 | 115 | `0.055` |
| micro256 | 256 | 198 | `0.089` |
| micro512 | 512 | 348 | `0.097` |
| macro64 | 64 | 64 | `0.116` |

Все панели прошли автоматические distribution, join и overlap checks.
`micro128` используется только для раннего отклонения; улучшение подтверждается
только согласованным результатом на `micro256 + macro64`.

**Эксперимент 21: temperature 0.8 на компактных панелях**

Сравнили обычный DLM `100 attempts` и DLM `temperature=0.8, 200 attempts`.
Для micro доверительные интервалы считались molecule-cluster bootstrap.

| Panel | Metric | Baseline | Temperature 0.8 | Delta, 95% CI |
| --- | --- | ---: | ---: | ---: |
| micro128 | Exact top-10 | `0.1563` | `0.2031` | `+0.0469` `[+0.0154, +0.0873]` |
| micro128 | Tanimoto top-10 | `0.5222` | `0.5277` | `+0.0055` `[-0.0048, +0.0158]` |
| macro64 | Exact top-10 | `0.1875` | `0.2344` | `+0.0469` `[0.0000, +0.1094]` |
| macro64 | Tanimoto top-10 | `0.4856` | `0.4949` | `+0.0092` `[-0.0062, +0.0253]` |

Вывод:

```text
Temperature 0.8 одинаково добавляет exact-кандидатов на двух типах выборки,
но не даёт доказанного улучшения Tanimoto ranking.
Оставляем её источником кандидатов внутри union, не отдельным победителем.
```

**Эксперимент 22: four-source union на representative compact gate**

Проверили frozen-связку `DLM control + DLM temperature 0.8 + train-only
retrieval + MolForge 0.172` на двух типах выборки. Confidence intervals
считались cluster-bootstrap по connectivity molecule.

| Panel | Metric | Baseline | Four-source union | Delta, 95% CI |
| --- | --- | ---: | ---: | ---: |
| micro256 | Tanimoto top-1 | `0.4868` | `0.5519` | `+0.0651` `[+0.0428, +0.0903]` |
| micro256 | Tanimoto top-10 | `0.5101` | `0.5897` | `+0.0796` `[+0.0566, +0.1051]` |
| micro256 | Exact top-1 | `0.1484` | `0.2031` | `+0.0547` `[+0.0161, +0.1037]` |
| micro256 | Exact top-10 | `0.1719` | `0.2617` | `+0.0898` `[+0.0462, +0.1402]` |
| macro64 | Tanimoto top-1 | `0.4724` | `0.5041` | `+0.0318` `[+0.0083, +0.0590]` |
| macro64 | Tanimoto top-10 | `0.4856` | `0.5284` | `+0.0428` `[+0.0207, +0.0678]` |
| macro64 | Exact top-10 | `0.1875` | `0.2344` | `+0.0469` `[0.0000, +0.1094]` |

Вывод:

```text
Union улучшает и replicate-frequency выборку micro256,
и molecule-disjoint macro64. Compact promotion подтверждён.
```

**Что сейчас считается**

**Эксперимент 23: аудит locked 1,024 gate**

Повторно проверили уже сохранённый exact-subset результат без изменения
кандидата, seed или ranking policy. Это не новый tuning-run, а контроль
воспроизводимости перед большим прогоном.

| Metric | Four-source union vs DLM control | 95% CI |
| --- | ---: | ---: |
| Tanimoto top-1 | `+0.0189` | `[+0.0148, +0.0232]` |
| Tanimoto top-10 | `+0.0294` | `[+0.0246, +0.0344]` |
| Exact top-1 | `+0.0127` | `[+0.0049, +0.0215]` |
| Exact top-10 | `+0.0283` | `[+0.0186, +0.0391]` |

Вывод:

```text
Locked 1,024 gate подтверждён: все четыре целевые метрики улучшились.
SPA-153 закрыта в Linear. Запущен frozen full-gate на 17,082 spectra.
```

**Эксперимент 24: полный frozen union, в работе**

Запущены только уже подтверждённые источники, без подстройки по test labels:

- DLM control, 100 attempts: Kolmogorovsky GPU4;
- DLM temperature 0.8, 200 attempts: Kolmogorovsky GPU3, уже обработано `3,750 / 17,082`;
- MolForge 0.172: Spectrum GPU0, первые `30 / 17,082`;
- train-only retrieval: готовый candidate table.

После завершения источников будет собран target-blind union и выполнен paired
bootstrap на всей выборке. Рабочая задача: `SPA-155`.

**Эксперимент 25: parallel full-run orchestration**

Один DLM full-run оказался слишком медленным: примерно `10-14 s/spectrum`.
Не меняя frozen модель, seed или параметры генерации, запустили точные чанки
по индексам locked test split:

- control: `[1000,5000)`, `[5000,9000)`, `[9000,13000)`, `[13000,17082)`;
- temperature 0.8: `[4000,7000)`, `[7000,10000)`, `[10000,13000)`, `[13000,17082)`;
- исходные unsharded control и temperature runs сохранены как audit/fallback.

Каждый chunk получил отдельный run directory, GPU, session и manifest. Перед
fusion проверим отсутствие пересечений по `spec_name`, полное покрытие 17,082
строк и совпадение commit/checkpoint/seed/settings. Это только ускорение
получения того же evidence, не новый quality claim.

**Следующая гипотеза после full-gate**

Параллельно зарегистрирована `SPA-156`: query-local selective TTT по train-only
MIST-соседям. Сейчас это только `prepared`: neighbor builder и safeguards есть,
но adapter execution ещё не реализован. Кандидат не будет допущен к compact
quality gate без 16-32 smoke, зафиксированного бюджета и согласованного
micro256 + macro64 результата против frozen four-source union.

Второй sidecar-аудит выбрал следующую более близкую к запуску гипотезу:

- `SPA-157`: ICEBERG-guided refinement/union-extension;
- текущий статус: `prepared`, quality claim отсутствует;
- обязательный порядок: target-safe smoke -> micro128 futility -> micro256 +
  macro64 paired gate -> только затем 1024/full.

ICEBERG рассматривается только как дополнительный источник кандидатов. Его
нельзя использовать для замены подтверждённого frozen union до paired evidence.

**Эксперимент 26: ICEBERG fixed-manifest smoke preflight**

Проверили target-safe путь на 16 заранее выбранных molecule-diverse queries.
Подготовка прошла успешно:

- `16` predeclared queries;
- `2,215` union-extension candidates;
- query selection без target SMILES/InChIKey/fingerprint;
- условия formula/ionization/instrument прочитаны из observed `.ms`.

Scoring остановился до получения predictions из-за окружения: исходный
попытался импортировать отсутствующий Lightning, после исправления выявился
CPU-only DGL на GPU path (`Device API cuda is not enabled`). Это технический
no-go smoke, не rejection модели. Добавлен CPU fallback и environment
diagnostics; paired quality gate не запускался.

**Эксперимент 27: перенос full orchestration на Spectrum Slurm**

По рабочему правилу FRIGID теперь считается только на `spectrum`. Все DLM
full/shard jobs на Kolmogorovsky остановлены; их частичные каталоги сохранены
исключительно как historical audit и не будут использованы как full evidence.

Для Spectrum добавлен `scripts/submit_msg_full_shard.sbatch`:

- `gpu`: одна full-GPU job;
- `gpu-shared`: пять jobs с `shard:1`;
- каждый shard получает непересекающийся `start-index/max-spectra`, отдельный
  `RUN_MANIFEST.json` и одинаковые checkpoint/seed/settings.

Новый full run будет считаться только после проверки покрытия всех `17,082`
`spec_name` и paired bootstrap. Это изменение orchestration, не quality claim.

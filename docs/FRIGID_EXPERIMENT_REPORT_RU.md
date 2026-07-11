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

- Tanimoto top-1: `+0.0204`, 95% CI `[+0.0078, +0.0356]`;
- Tanimoto top-10: `+0.0325`, 95% CI `[+0.0153, +0.0538]`.

Вывод:

```text
Новые источники не обязаны побеждать DLM по отдельности.
Главное улучшение даёт разнообразный candidate pool и единый reranking.
```

`MS-BART` в текущем виде не добавляет пользы и остановлен. Связка
`DLM control + DLM temperature 0.8 + train retrieval` продвинута на новый
disjoint molecule-diverse набор из 200 spectra.

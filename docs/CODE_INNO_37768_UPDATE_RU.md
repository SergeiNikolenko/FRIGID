# Обновление work item 37768 — MARLIN / Spectrum platform for Chemistry

Период отчёта: 20.07–03.08.2026.

## Цель

Довести clean-room MARLIN до измеримого Exact@1/Exact@10 на structure-disjoint
held-out NPLIB1, сохранив честный molecular scorer и не используя target/oracle
fingerprints для выбора модели. Paper fidelity остаётся ориентиром, но
приоритет — реальный результат на held-out наборе.

## Что сделано

1. Зафиксирован FRIGID warm-start и проверена совместимость токенизации,
   fingerprint-кондиционирования и стартовых logits: token parity `3.73e-8`,
   final-logit parity `1.15e-5`.
2. Собран воспроизводимый MARLIN molecular harness: SAFE block decoding,
   block width `8`, precursor-mass Fourier conditioning, DreaMS predicted
   fingerprints, symmetric fingerprint noise, candidate diversity dropout
   `0.3`, multinomial sampling и строгий mass shell `10 ppm`.
3. Зафиксирован structure-disjoint NPLIB1 validation contract: micro-панель,
   Exact@1/Exact@10, candidate return, RDKit validity, mass validity,
   uniqueness, formula recall, Tanimoto и decoder diagnostics. Oracle/target
   fingerprints оставлены только для диагностики и не продвигаются в incumbent.
4. Проведена серия коротких autoresearch-итераций в numbered ledger (формат
   FRIGID-style, эксперименты 1–25): threshold/layer-ordering/self-attention,
   paper symmetric-noise, argmax vs multinomial, train/eval threshold parity,
   full-backbone adaptation, canvas/no-shell/mass-tolerance и confidence
   ablations.
5. Исправлена инфраструктура: outputs и offline ClearML cache переведены на
   локальный диск после заполнения `/mnt/netstorage`; добавлены run signatures,
   input hashes, Slurm receipts и offline ClearML task IDs. Чужой Slurm job 614
   и общая A100 не прерывались.
6. Добавлен soft-fingerprint путь: probability amplitudes DreaMS сохраняются на
   активных bits, при этом binary warm-start остаётся совместимым; добавлены
   тесты и параметры launcher/evaluator.

## Результаты

- FRIGID warm-start parity подтверждён.
- Full-backbone paper-noise adaptation, Slurm `625`, 100 cross-attention + 900
  full-backbone steps: `Exact@1=0`, `Exact@10=0`, candidate return `0`, mass
  validity `0`, strict validity `0.140625`, mean dead ends `10.75`.
- Threshold `0.50` inference, Slurm `628`: `Exact@1/10=0`, return `0`, mass
  `0`, validity `0.15625`.
- Soft DreaMS confidence inference, Slurm `629`: `Exact@1/10=0`, return `0`,
  mass `0`, validity `0.15625`, mean dead ends `12.25`.
- EOS boost diagnostic, Slurm `630`, дал на первой завершённой строке
  validity `0`, mass/return `0` и был остановлен по bounded timeout; partial
  artifact сохранён.
- Soft-fingerprint adaptation, Slurm `635` (offline ClearML task
  `offline-d16a7c6cbba1427c9948d35f32bba997`), выполняется на локальном run
  root; его финальные molecular metrics ещё не засчитываются до завершения.

Итог на 03.08: molecular pipeline воспроизводимо запускается и считает
полные метрики, но положительный Exact incumbent пока не получен. Главный
наблюдаемый bottleneck — модель иногда строит RDKit-valid SAFE, однако не
завершает mass-compatible candidate: candidate return и mass validity остаются
нулевыми. Поэтому результат нельзя объявлять воспроизведённым как в статье.

## Следующие шаги

1. Дождаться soft-fingerprint adaptation `635` и проверить его тем же
   held-out scorer на micro4; затем, только при non-zero return, перейти к
   micro32/micro64.
2. Если появится `candidate_return > 0`, зафиксировать checkpoint и повторить
   без изменений на трёх seed-ах и paper-comparable candidate budget `384`.
3. Только после non-zero Exact на подтверждающей панели переходить к locked
   NPLIB1 evaluation и публикации результата.

Доказательства и полная таблица экспериментов:
`docs/MARLIN_EXPERIMENT_REPORT_RU.md`.

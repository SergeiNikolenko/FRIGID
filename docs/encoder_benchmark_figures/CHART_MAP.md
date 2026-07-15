# Encoder benchmark chart map

| Figure | Analytical question | Form | Fields | Supported takeaway | Palette | Provenance |
|---|---|---|---|---|---|---|
| `historical_encoder_metrics.png` | How far did historical alternatives move relative to MIST? | Sorted horizontal bar with promotion reference | model, mean fingerprint Tanimoto, series | DreaMS variants remain far below MIST; residual gain misses the 0.005 gate | Blue focal, gold alternative, neutral DreaMS | Historical run JSON and report values recorded in `MSMS_ENCODER_BENCHMARK_REPORT.md` |
| `mist_threshold_calibration.png` | Which binary threshold maximizes calibration Tanimoto? | Line with selected-point annotation | threshold, mean fingerprint Tanimoto, rows | The molecule-disjoint calibration partition selects 0.25 | Blue line, gold selected point | Slurm job 174, `threshold_calibration.csv`, SHA-256 `54a586c27b960ccd089dc00ca6f28237aedcaba4a64054a65a00e529377e8c34` |
| `dlm_fingerprint_upper_bound.png` | How much does DLM improve when only the fingerprint source becomes ideal? | Grouped horizontal bars | metric, ground-truth value, MIST-binary value | Fingerprint quality materially affects generation, while the remaining gap shows decoder/ranking limits | Blue ground truth, gold MIST binary | Historical paired diagnostic, 1,400 spectra, partial run |

All figures are static PNG files for the Markdown report and Marp slide deck.

"""A fitted model of the encoder's fingerprint error, and its uniform control.

Why this exists
---------------
``src/marlin/noise.py`` corrupts a training fingerprint by dropping a uniform
10-30% of its on-bits and (in the symmetric mode) inventing the same number of
off-bits chosen uniformly at random. Measured against the encoder we actually
condition on at inference, that corruption is not merely too weak, it points the
wrong way. On the 1,199 out-of-sample NPLIB1 rows (val 396 + locked test 803),
binarising the DreaMS probe at the evaluation gate 0.95:

* real sensitivity ``P(pred=1 | true=1)`` runs from **0.255** on the rarest
  decile of corpus bit frequency to **0.991** on the commonest; the incumbent is
  **flat at ~0.80** everywhere.
* real ``P(pred=1 | true=0)`` runs from **0.00077** to **0.878** over the same
  deciles, so **12.35 of the 23.7 invented bits per row land on 146 of the
  4,096 bit indices**; the incumbent puts its invented bits uniformly, i.e.
  mostly on the rare deciles, and **0.00 per row** on the commonest decile.
* real Tanimoto to the truth has median 0.293 (val) / 0.304 (test); the
  incumbent implies 0.661 symmetric and 0.806 one-sided.

So the decoder is trained to believe an on-bit is probably true and a rare bit
is probably absent, and is then handed a vector in which half the on-bits are
invented and the informative rare bits are missing three times in four.

The model
---------
Bit indices are grouped into ``K`` bins by their frequency in the generation
corpus. Inside bin ``k`` the model is

    P(pred=1 | true=1, row) = sigmoid(logit(sensitivity[k]) + u_row)
    P(pred=1 | true=0, row) = sigmoid(logit(false_positive_rate[k]) + v_row)

with ``(u_row, v_row)`` drawn from the empirical joint distribution of the row
latents fitted on real rows. The row latents are not decoration: without them
the sampled Tanimoto distribution has standard deviation 0.066 against the real
0.151 and the two-sample KS test rejects at ``p = 1.2e-29``. With them,
KS = 0.049 (``p = 0.34``) against a held-out real split - closer than the two
real splits are to each other (KS = 0.061, ``p = 0.26``).

Everything above is reproduced by ``scripts/fit_encoder_error_model.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FINGERPRINT_BITS = 4096
_LOGIT_CLAMP = 1e-9


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), _LOGIT_CLAMP, 1 - _LOGIT_CLAMP)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class EncoderErrorModel:
    """Frequency-binned conditional bit rates plus an empirical row latent.

    ``bin_of[j]`` is the frequency bin of bit ``j``. ``sensitivity[k]`` and
    ``false_positive_rate[k]`` are that bin's rates at latent zero.
    ``recall_latents`` and ``invention_latents`` are paired per-row logit shifts
    measured on real rows; a sampled row draws one index into both, which keeps
    whatever correlation the real rows carry (measured 0.027, i.e. almost none,
    but it is not this module's business to assume that).
    """

    bin_of: np.ndarray
    sensitivity: np.ndarray
    false_positive_rate: np.ndarray
    recall_latents: np.ndarray
    invention_latents: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        if self.bin_of.shape != (FINGERPRINT_BITS,):
            raise ValueError(
                f"bin_of must have shape ({FINGERPRINT_BITS},), got {self.bin_of.shape}"
            )
        bins = int(self.sensitivity.shape[0])
        if self.false_positive_rate.shape != (bins,):
            raise ValueError("sensitivity and false_positive_rate must agree in length")
        if bins == 0:
            raise ValueError("the model needs at least one frequency bin")
        if self.bin_of.min() < 0 or self.bin_of.max() >= bins:
            raise ValueError("bin_of indexes a bin that has no fitted rate")
        for name, array in (
            ("sensitivity", self.sensitivity),
            ("false_positive_rate", self.false_positive_rate),
        ):
            if np.any(array < 0.0) or np.any(array > 1.0):
                raise ValueError(f"{name} must be a probability")
        if self.recall_latents.shape != self.invention_latents.shape:
            raise ValueError(
                "recall and invention latents must be paired, so they must have "
                "the same length"
            )
        if self.recall_latents.ndim != 1 or self.recall_latents.size == 0:
            raise ValueError("the latent pool must be a non-empty 1-D array")

    @property
    def bins(self) -> int:
        return int(self.sensitivity.shape[0])

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def corrupt(
        self,
        fingerprints: torch.Tensor,
        *,
        corruption_probability: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return predicted-style fingerprints for a batch of true ones.

        ``corruption_probability`` leaves a row untouched with the complementary
        probability, so that the incumbent's 50/50 clean+noisy mixture stays
        expressible. The corpus arm runs it at 1.0: CoRe-Gen corrupts during
        pretraining, and a clean half is a different experiment.
        """
        if fingerprints.ndim != 2:
            raise ValueError("fingerprints must have shape [batch, bits]")
        if fingerprints.shape[1] != FINGERPRINT_BITS:
            raise ValueError(
                f"this model is fitted for {FINGERPRINT_BITS} bits, got "
                f"{fingerprints.shape[1]}"
            )
        if not 0.0 <= corruption_probability <= 1.0:
            raise ValueError("corruption_probability must be in [0, 1]")

        device = fingerprints.device
        rows = int(fingerprints.shape[0])
        truth = fingerprints > 0.5

        recall_logit = torch.as_tensor(
            _logit(self.sensitivity), dtype=torch.float32, device=device
        )
        invent_logit = torch.as_tensor(
            _logit(self.false_positive_rate), dtype=torch.float32, device=device
        )
        bin_index = torch.as_tensor(self.bin_of, dtype=torch.long, device=device)

        pool = int(self.recall_latents.shape[0])
        picks = torch.randint(
            0, pool, (rows,), device=device, generator=generator, dtype=torch.long
        )
        recall_pool = torch.as_tensor(
            self.recall_latents, dtype=torch.float32, device=device
        )
        invent_pool = torch.as_tensor(
            self.invention_latents, dtype=torch.float32, device=device
        )
        u = recall_pool[picks].unsqueeze(1)
        v = invent_pool[picks].unsqueeze(1)

        on_probability = torch.sigmoid(recall_logit[bin_index].unsqueeze(0) + u)
        off_probability = torch.sigmoid(invent_logit[bin_index].unsqueeze(0) + v)
        probability = torch.where(truth, on_probability, off_probability)

        draws = torch.rand(
            fingerprints.shape, device=device, generator=generator, dtype=torch.float32
        )
        corrupted = draws < probability

        if corruption_probability < 1.0:
            keep_clean = (
                torch.rand(rows, device=device, generator=generator)
                >= corruption_probability
            ).unsqueeze(1)
            corrupted = torch.where(keep_clean, truth, corrupted)
        return corrupted.to(dtype=fingerprints.dtype)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(
            path,
            bin_of=self.bin_of.astype(np.int64),
            sensitivity=self.sensitivity.astype(np.float64),
            false_positive_rate=self.false_positive_rate.astype(np.float64),
            recall_latents=self.recall_latents.astype(np.float64),
            invention_latents=self.invention_latents.astype(np.float64),
            metadata=np.array(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EncoderErrorModel":
        with np.load(Path(path), allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"]))
            return cls(
                bin_of=np.asarray(arrays["bin_of"], dtype=np.int64),
                sensitivity=np.asarray(arrays["sensitivity"], dtype=np.float64),
                false_positive_rate=np.asarray(
                    arrays["false_positive_rate"], dtype=np.float64
                ),
                recall_latents=np.asarray(arrays["recall_latents"], dtype=np.float64),
                invention_latents=np.asarray(
                    arrays["invention_latents"], dtype=np.float64
                ),
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    # the paired control
    # ------------------------------------------------------------------
    def rate_matched_uniform_control(self) -> "EncoderErrorModel":
        """The control arm: the same marginal error rates, no structure.

        The control has to differ from the fitted model in exactly one thing, or
        the experiment measures nothing. Setting the incumbent
        ``symmetric_fingerprint_noise`` against a fitted model would change the
        noise *level* (Tanimoto 0.66 against 0.33) at the same time as its
        *shape*, and any difference would be unattributable. This control keeps
        the pooled ``P(pred=1|true=1)`` and ``P(pred=1|true=0)`` of the fitted
        model, and removes only the frequency dependence and the row latent -
        which is precisely the ``-4.77 pp`` claim under test.
        """
        weight_on = np.asarray(self.metadata["bin_true_on_counts"], dtype=np.float64)
        weight_off = np.asarray(self.metadata["bin_true_off_counts"], dtype=np.float64)
        pooled_sensitivity = float(
            (self.sensitivity * weight_on).sum() / max(weight_on.sum(), 1.0)
        )
        pooled_fpr = float(
            (self.false_positive_rate * weight_off).sum() / max(weight_off.sum(), 1.0)
        )
        metadata = dict(self.metadata)
        metadata.update(
            {
                "variant": "rate_matched_uniform_control",
                "derived_from": self.metadata.get("name", "fitted"),
                "pooled_sensitivity": pooled_sensitivity,
                "pooled_false_positive_rate": pooled_fpr,
            }
        )
        return EncoderErrorModel(
            bin_of=np.zeros(FINGERPRINT_BITS, dtype=np.int64),
            sensitivity=np.array([pooled_sensitivity]),
            false_positive_rate=np.array([pooled_fpr]),
            recall_latents=np.zeros(1),
            invention_latents=np.zeros(1),
            metadata=metadata,
        )


def fit_encoder_error_model(
    true_fingerprints: np.ndarray,
    predicted_bits: np.ndarray,
    corpus_bit_frequency: np.ndarray,
    *,
    bins: int = 10,
    name: str = "fitted",
    extra_metadata: dict | None = None,
) -> EncoderErrorModel:
    """Fit the model from paired (true, binarised predicted) fingerprints.

    Bins are cut so that each holds roughly the same number of *true on-bit
    occurrences*, not the same number of bit indices: the frequency
    distribution is so skewed that equal-width bins would put 5,500 of the
    5,578 rarest-decile occurrences in one bin and leave the top bins empty.
    """
    true_bits = np.asarray(true_fingerprints) > 0.5
    pred_bits = np.asarray(predicted_bits) > 0.5
    if true_bits.shape != pred_bits.shape:
        raise ValueError("true and predicted fingerprints must have the same shape")
    if true_bits.ndim != 2 or true_bits.shape[1] != FINGERPRINT_BITS:
        raise ValueError(f"fingerprints must have shape [rows, {FINGERPRINT_BITS}]")
    frequency = np.asarray(corpus_bit_frequency, dtype=np.float64)
    if frequency.shape != (FINGERPRINT_BITS,):
        raise ValueError(f"corpus_bit_frequency must have shape ({FINGERPRINT_BITS},)")
    if bins < 1:
        raise ValueError("bins must be positive")

    order = np.argsort(frequency, kind="stable")
    on_per_bit = true_bits.sum(0)
    cumulative = np.cumsum(on_per_bit[order])
    if cumulative[-1] == 0:
        raise ValueError("no true on-bits: nothing to fit")
    cuts = np.searchsorted(
        cumulative, np.linspace(0, cumulative[-1], bins + 1)[1:-1]
    )
    bin_of = np.empty(FINGERPRINT_BITS, dtype=np.int64)
    start = 0
    for index, end in enumerate(list(cuts) + [FINGERPRINT_BITS]):
        bin_of[order[start:end]] = index
        start = end

    sensitivity = np.zeros(bins)
    false_positive_rate = np.zeros(bins)
    bin_true_on = np.zeros(bins)
    bin_true_off = np.zeros(bins)
    for index in range(bins):
        mask = bin_of == index
        on = true_bits[:, mask]
        off = ~true_bits[:, mask]
        bin_true_on[index] = on.sum()
        bin_true_off[index] = off.sum()
        sensitivity[index] = (on & pred_bits[:, mask]).sum() / max(on.sum(), 1)
        false_positive_rate[index] = (off & pred_bits[:, mask]).sum() / max(off.sum(), 1)

    recall, invention = _solve_row_latents(true_bits, pred_bits, bin_of, sensitivity, false_positive_rate)

    metadata = {
        "name": name,
        "variant": "fitted",
        "bins": int(bins),
        "fit_rows": int(true_bits.shape[0]),
        "bin_true_on_counts": bin_true_on.tolist(),
        "bin_true_off_counts": bin_true_off.tolist(),
        "bin_bit_counts": [int((bin_of == k).sum()) for k in range(bins)],
        "bin_median_corpus_frequency": [
            float(np.median(frequency[bin_of == k])) if (bin_of == k).any() else 0.0
            for k in range(bins)
        ],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return EncoderErrorModel(
        bin_of=bin_of,
        sensitivity=sensitivity,
        false_positive_rate=false_positive_rate,
        recall_latents=recall,
        invention_latents=invention,
        metadata=metadata,
    )


def _solve_row_latents(
    true_bits: np.ndarray,
    pred_bits: np.ndarray,
    bin_of: np.ndarray,
    sensitivity: np.ndarray,
    false_positive_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One logit shift per row per channel, matching that row's TP and FP count."""
    bins = sensitivity.shape[0]
    recall_logit = _logit(sensitivity)
    invent_logit = _logit(false_positive_rate)
    on_counts = np.stack(
        [(true_bits & (bin_of == k)).sum(1) for k in range(bins)], axis=1
    ).astype(np.float64)
    off_counts = np.stack(
        [((~true_bits) & (bin_of == k)).sum(1) for k in range(bins)], axis=1
    ).astype(np.float64)
    true_positives = (true_bits & pred_bits).sum(1).astype(np.float64)
    false_positives = ((~true_bits) & pred_bits).sum(1).astype(np.float64)
    recall = np.array(
        [
            _solve_shift(on_counts[i], true_positives[i], recall_logit)
            for i in range(true_bits.shape[0])
        ]
    )
    invention = np.array(
        [
            _solve_shift(off_counts[i], false_positives[i], invent_logit)
            for i in range(true_bits.shape[0])
        ]
    )
    return recall, invention


def _solve_shift(
    counts: np.ndarray, target: float, base_logit: np.ndarray, bound: float = 12.0
) -> float:
    """Bisect for the shift whose expected count equals the observed one."""
    if target <= 0.0:
        return -bound

    def expected(shift: float) -> float:
        return float((counts / (1.0 + np.exp(-(base_logit + shift)))).sum()) - target

    low, high = -bound, bound
    if expected(low) > 0.0:
        return low
    if expected(high) < 0.0:
        return high
    for _ in range(80):
        middle = 0.5 * (low + high)
        if expected(middle) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)

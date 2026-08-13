"""A fixed probe that asks whether the decoder reads the fingerprint's content.

The project's central measured gap is teacher-forced per-token top-1 of 0.750
under a true Morgan fingerprint against 0.547 under the predicted one
(``docs/DECODER_PROGRAM.md:53-54``). That pair was measured once, offline, on a
24-molecule probe. Nothing during training ever measured it, so a run could
drive ``train_loss`` from 23.5 to 0.047 -- which the 100,000-step adaptation
did -- without anyone seeing whether the conditioning pathway was learning or
whether the decoder was memorising 6,649 rows.

This module measures that pair, plus the decomposition our diagnosis actually
needs, on a probe set that is fixed by construction so that curves from
different runs are the same measurement:

* ``true`` -- the gold Morgan fingerprint of the target;
* ``predicted`` -- what the encoder pipeline supplies at inference;
* ``mismatched`` -- another probe molecule's gold fingerprint, present but
  wrong. Only the fingerprint is rolled; mass and isotope conditioning stay
  matched, so the contrast isolates fingerprint content;
* ``absent`` -- an all-zero fingerprint, so no conditioning vector is present
  at all.

``mismatched`` minus ``true`` is *content* sensitivity. ``absent`` minus
``mismatched`` is *presence* sensitivity. Our diagnosis says the second
dominates; this makes that claim a logged number rather than an opinion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import lightning as L
import math
import torch
import torch.nn.functional as F

CONDITIONING_VARIANTS = ("true", "predicted", "mismatched", "absent")


def _unit_hash(payload: str) -> float:
    digest = hashlib.sha256(payload.encode()).digest()[:8]
    return int.from_bytes(digest, "big") / 2**64


def select_probe_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    size: int,
    seed: int = 0,
    key: str = "spec_name",
) -> list[int]:
    """Pick a fixed probe subset by hash of the spectrum id.

    A hash order rather than a shuffle or a head slice: the probe is a function
    of the spectrum ids alone, so it does not move when the metadata is written
    in a different order or when spectra the probe did not pick are dropped, and
    curves from two runs are the same measurement.
    """
    if size <= 0:
        raise ValueError("probe size must be positive")
    if not rows:
        raise ValueError("probe selection needs at least one row")
    order = sorted(
        range(len(rows)),
        key=lambda index: (_unit_hash(f"{seed}:{rows[index][key]}"), str(rows[index][key])),
    )
    return sorted(order[:size])


def build_probe_batches(
    dataset,
    collator,
    *,
    size: int,
    batch_size: int,
    seed: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """Materialise the probe once, with both the true and predicted fingerprint.

    The collator computes a gold Morgan fingerprint whenever a row carries none,
    so the same rows are collated twice: as supplied (predicted) and stripped
    (true). Everything else -- tokens, precursor mass, isotope ratios -- comes
    from one collation, so the two conditioning vectors are the only difference.
    """
    if batch_size < 2:
        # ``mismatched`` is a roll along the batch, which is the identity at
        # batch size 1 and would silently report a zero content gain.
        raise ValueError("probe batch size must be at least 2")
    indices = select_probe_rows(dataset.rows, size=size, seed=seed)
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        if len(chunk) < 2:
            break
        rows = [dict(dataset.rows[index]) for index in chunk]
        predicted = collator(rows)
        stripped = [
            {name: value for name, value in row.items() if name != "fingerprint"}
            for row in rows
        ]
        true = collator(stripped)
        if not torch.equal(predicted["input_ids"], true["input_ids"]):
            raise AssertionError("probe collation changed the tokenisation")
        batch = dict(predicted)
        batch["true_fingerprint"] = true["fingerprint"]
        batches.append(batch)
    if not batches:
        raise ValueError("probe selection produced no batch of at least two rows")
    return batches


def _variant_fingerprints(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    true = batch["true_fingerprint"]
    return {
        "true": true,
        "predicted": batch["fingerprint"],
        "mismatched": true.roll(1, dims=0),
        "absent": torch.zeros_like(true),
    }


def conditioning_probe_metrics(
    decoder,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    seed: int = 0,
    mask_probability: float = 0.5,
    conditioning_free_margin: float = 0.05,
) -> dict[str, float]:
    """Teacher-forced top-1 and NLL under four conditioning variants.

    The mask draw is re-seeded per call and shared by all four variants, so two
    checkpoints are scored on the same positions and the only difference is the
    weights. Teacher forcing is literal: the clean stream carries the gold
    prefix and the noised stream carries the masked positions, exactly as
    ``diffusion_objective`` builds them, with no importance weighting -- this is
    an accuracy probe, not an NELBO estimate.
    """
    if not 0.0 < mask_probability <= 1.0:
        raise ValueError("probe mask probability must be in (0, 1]")
    if conditioning_free_margin < 0.0:
        raise ValueError("conditioning_free_margin must be non-negative")
    if not batches:
        raise ValueError("conditioning probe received no batches")

    config = decoder.config
    was_training = decoder.training
    decoder.eval()
    device = next(decoder.parameters()).device

    totals: dict[str, float] = {}
    masked_tokens = 0.0
    rows = 0.0
    conditioning_free_true = 0.0
    conditioning_free_predicted = 0.0
    content_gain = 0.0
    presence_gain = 0.0
    try:
        with torch.no_grad():
            for index, batch in enumerate(batches):
                moved = {
                    name: value.to(device)
                    for name, value in batch.items()
                    if isinstance(value, torch.Tensor)
                }
                clean_ids = moved["input_ids"]
                if clean_ids.shape[0] < 2:
                    raise ValueError("probe batches must hold at least two rows")
                valid = clean_ids.ne(config.pad_token_id)
                valid[:, 0] = False
                generator = torch.Generator(device=device).manual_seed(seed + index)
                masked = (
                    torch.rand(
                        clean_ids.shape, device=device, generator=generator
                    )
                    < mask_probability
                ) & valid
                # A row that drew no mask contributes no teacher-forced token;
                # force one so every probe row is scored on every checkpoint.
                empty = ~masked.any(dim=1) & valid.any(dim=1)
                if bool(empty.any()):
                    first = valid.float().argmax(dim=1)
                    masked[empty, first[empty]] = True
                scored = masked.any(dim=1)
                noised = clean_ids.masked_fill(masked, config.mask_token_id)
                per_row_counts = masked.sum(dim=1).clamp_min(1)

                per_row_nll: dict[str, torch.Tensor] = {}
                for variant, fingerprint in _variant_fingerprints(moved).items():
                    logits = decoder.two_stream_logits(
                        clean_ids,
                        noised,
                        moved["precursor_mass"],
                        fingerprint,
                        moved.get("isotope_ratios"),
                        include_mass_conditioning=True,
                    )
                    losses = F.cross_entropy(
                        logits.float().transpose(1, 2), clean_ids, reduction="none"
                    )
                    per_row_nll[variant] = (losses * masked).sum(dim=1) / per_row_counts
                    correct = logits.argmax(dim=-1).eq(clean_ids) & masked
                    totals[f"probe_top1_{variant}"] = totals.get(
                        f"probe_top1_{variant}", 0.0
                    ) + float(correct.sum())
                    totals[f"probe_nll_{variant}"] = totals.get(
                        f"probe_nll_{variant}", 0.0
                    ) + float((losses * masked).sum())

                masked_tokens += float(masked.sum())
                rows += float(scored.sum())
                content = per_row_nll["mismatched"] - per_row_nll["true"]
                presence = per_row_nll["absent"] - per_row_nll["mismatched"]
                content_gain += float((content * scored).sum())
                presence_gain += float((presence * scored).sum())
                conditioning_free_true += float(
                    ((content <= conditioning_free_margin) & scored).sum()
                )
                conditioning_free_predicted += float(
                    (
                        (
                            (per_row_nll["mismatched"] - per_row_nll["predicted"])
                            <= conditioning_free_margin
                        )
                        & scored
                    ).sum()
                )
    finally:
        decoder.train(was_training)

    if masked_tokens <= 0 or rows <= 0:
        raise ValueError("conditioning probe scored no tokens")

    metrics = {
        f"probe_{name}_{variant}": totals[f"probe_{name}_{variant}"] / masked_tokens
        for name in ("top1", "nll")
        for variant in CONDITIONING_VARIANTS
    }
    metrics["probe_top1_gap_true_minus_predicted"] = (
        metrics["probe_top1_true"] - metrics["probe_top1_predicted"]
    )
    metrics["probe_conditioning_content_gain"] = content_gain / rows
    metrics["probe_conditioning_presence_gain"] = presence_gain / rows
    metrics["probe_conditioning_free_loss_fraction"] = conditioning_free_true / rows
    metrics["probe_conditioning_free_loss_fraction_predicted"] = (
        conditioning_free_predicted / rows
    )
    metrics["probe_rows"] = rows
    metrics["probe_masked_tokens"] = masked_tokens
    return metrics


class PeriodicConditioningProbe(L.Callback):
    """Run the fixed conditioning probe on a schedule and publish its scalars.

    With a ``selector`` the probe also becomes the run's early-stopping signal.
    The adaptation corpus is 6,032 unique molecules replayed 3,846 times over a
    100,000-step run (``docs/TRAINING_RECIPE_FINDINGS.md:19``), so the question
    is never whether the run will start memorising but when. The probe answers
    it every ``interval_steps`` on structures the optimizer never sees, at the
    cost of four short forward passes rather than the hours a molecular panel
    costs, which is what makes stopping on it affordable at all.
    """

    title = "Conditioning probe"

    def __init__(
        self,
        batches: Sequence[Mapping[str, torch.Tensor]],
        *,
        interval_steps: int,
        seed: int = 0,
        mask_probability: float = 0.5,
        conditioning_free_margin: float = 0.05,
        clearml_task=None,
        selector=None,
        selection_path: Path | None = None,
    ) -> None:
        super().__init__()
        if interval_steps <= 0:
            raise ValueError("conditioning probe interval must be positive")
        if not batches:
            raise ValueError("conditioning probe needs at least one batch")
        self.batches = list(batches)
        self.interval_steps = interval_steps
        self.seed = seed
        self.mask_probability = mask_probability
        self.conditioning_free_margin = conditioning_free_margin
        self.clearml_task = clearml_task
        self.selector = selector
        self.selection_path = Path(selection_path) if selection_path else None
        self._last_step = -1

    def _select(self, trainer, step: int, values: dict[str, float]) -> None:
        if self.selector is None:
            return
        decision = self.selector.update(step, values)
        print(
            f"Probe selection at step {step}: {decision.metric}="
            f"{decision.value:.6f} best={decision.best_value:.6f} at step "
            f"{decision.best_step} "
            f"({decision.evaluations_since_best} probes since best)",
            flush=True,
        )
        if self.selection_path is not None and trainer.is_global_zero:
            self.selection_path.parent.mkdir(parents=True, exist_ok=True)
            self.selection_path.write_text(
                json.dumps(self.selector.state(), indent=2, sort_keys=True) + "\n"
            )
        if self.clearml_task is not None and trainer.is_global_zero:
            logger = self.clearml_task.get_logger()
            logger.report_scalar(
                title="Probe selection",
                series=f"{decision.metric} (best)",
                value=decision.best_value,
                iteration=step,
            )
        if decision.should_stop:
            print(
                f"Stopping at step {step}: {decision.reason}",
                flush=True,
            )
            trainer.should_stop = True

    def evaluate(self, pl_module) -> dict[str, float]:
        return conditioning_probe_metrics(
            pl_module.decoder,
            self.batches,
            seed=self.seed,
            mask_probability=self.mask_probability,
            conditioning_free_margin=self.conditioning_free_margin,
        )

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        del outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step <= 0 or step % self.interval_steps or step == self._last_step:
            return
        self._last_step = step
        values = self.evaluate(pl_module)
        for name, value in values.items():
            pl_module.log(name, value, on_step=True, sync_dist=False)
        print(
            "Conditioning probe at step "
            f"{step}: top1 true {values['probe_top1_true']:.4f} vs predicted "
            f"{values['probe_top1_predicted']:.4f}, content gain "
            f"{values['probe_conditioning_content_gain']:.4f}, "
            "conditioning-free rows "
            f"{values['probe_conditioning_free_loss_fraction']:.4f}",
            flush=True,
        )
        self._select(trainer, step, values)
        if self.clearml_task is None or not trainer.is_global_zero:
            return
        logger = self.clearml_task.get_logger()
        for name, value in values.items():
            if not math.isfinite(value):
                continue
            logger.report_scalar(
                title=self.title,
                series=name,
                value=value,
                iteration=step,
            )


def probe_batches_from_paths(
    metadata: str | Path,
    fingerprints: str | Path,
    tokenizer,
    *,
    fingerprint_key: str,
    threshold: float,
    max_length: int,
    fingerprint_bits: int,
    size: int,
    batch_size: int,
    seed: int = 0,
    soft_fingerprint: bool = False,
    exclude_inchikeys: str | Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Build the probe from the same files the evaluator reads."""
    from marlin.training import MarlinCollator, MarlinSpectrumFingerprintDataset

    dataset = MarlinSpectrumFingerprintDataset(
        metadata,
        fingerprints,
        tokenizer,
        fingerprint_key=fingerprint_key,
        threshold=threshold,
        max_length=max_length,
        exclude_inchikeys=exclude_inchikeys,
        preserve_probabilities=soft_fingerprint,
    )
    collator = MarlinCollator(
        tokenizer,
        max_length=max_length,
        fingerprint_bits=fingerprint_bits,
        allow_soft_fingerprints=soft_fingerprint,
    )
    return build_probe_batches(
        dataset,
        collator,
        size=size,
        batch_size=batch_size,
        seed=seed,
    )

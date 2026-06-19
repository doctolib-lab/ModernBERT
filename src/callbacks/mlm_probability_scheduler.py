# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Composer callback that linearly decays the MLM probability throughout training.

This callback updates the ``mlm_probability`` attribute of the :class:`~transformers.DataCollatorForLanguageModeling`
used by the training dataloader (or the ``mask_prob`` attribute of the
:class:`~src.sequence_packer.SequencePacker` when sequence packing is enabled).

The probability decays linearly from ``initial_prob`` to ``final_prob`` over the
entire training run, as specified by ``state.max_duration``.
"""

from __future__ import annotations

from typing import Optional

from composer.core import Callback, State
from composer.loggers import Logger
from composer import Time
from composer.core import TimeUnit
from composer.utils import dist

try:
    # SequencePacker is an optional dependency (only required when sequence packing is enabled)
    from src.sequence_packer import BufferedIterable  # type: ignore
except Exception:  # pragma: no cover
    BufferedIterable = None  # type: ignore

__all__ = ["MLMProbabilityLinearDecay"]


class MLMProbabilityLinearDecay(Callback):
    """Linearly decay the MLM probability during training.

    Args:
        initial_prob (float): Starting masking probability.
        final_prob (float): Ending masking probability.
        log_interval (int, optional): How often (in batches) to log the current
            probability to the Composer logger. Defaults to ``100``.
    """

    def __init__(self, initial_prob: float, final_prob: float, log_interval: int = 100):
        if not (0.0 <= initial_prob <= 1.0):
            raise ValueError(f"initial_prob must be within [0,1], got {initial_prob}")
        if not (0.0 <= final_prob <= 1.0):
            raise ValueError(f"final_prob must be within [0,1], got {final_prob}")
        if final_prob > initial_prob:
            raise ValueError(
                "final_prob should be less than or equal to initial_prob for a decay schedule."
            )
        self.initial_prob = float(initial_prob)
        self.final_prob = float(final_prob)
        self.log_interval = int(log_interval)
        # total amount over which to decay (batches or tokens)
        self._total_units: Optional[int] = None
        self._use_tokens: bool = False  # if True, schedule is driven by token count

    def _compute_prob(self, progress_units: int) -> float:
        """Return interpolated probability based on training progress."""
        assert self._total_units is not None, "Decay schedule not initialised."
        progress = min(max(progress_units, 0), self._total_units)
        ratio = progress / self._total_units if self._total_units > 0 else 1.0
        return self.initial_prob + (self.final_prob - self.initial_prob) * ratio

    def _update_collator_or_packer(self, state: State, new_prob: float):
        """Update ``mlm_probability`` of the collator or ``mask_prob`` of the sequence packer."""
        # ``state.train_dataloader`` is a ``DataSpec``. Grab the wrapped dataloader/iterable.
        train_dl = getattr(state.train_dataloader, "dataloader", state.train_dataloader)

        # Case 1: Regular DataLoader with a HF DataCollatorForLanguageModeling -----------------
        collate_fn = getattr(train_dl, "collate_fn", None)
        if collate_fn is not None and hasattr(collate_fn, "mlm_probability"):
            collate_fn.mlm_probability = new_prob  # type: ignore[attr-defined]

        # Case 2: Sequence packing path --------------------------------------------------------
        if BufferedIterable is not None and isinstance(train_dl, BufferedIterable):
            # The underlying SequencePacker is stored in ``iterable``
            seq_packer = getattr(train_dl, "iterable", None)
            if seq_packer is not None and hasattr(seq_packer, "mask_prob"):
                seq_packer.mask_prob = new_prob  # type: ignore[attr-defined]

    def fit_start(self, state: State, logger: Logger):
        """Record total number of batches for decay schedule."""
        # ``state.max_duration`` is a ``Time`` object. Convert to batches universally.
        max_duration: Time = state.max_duration
        if max_duration.unit == TimeUnit.BATCH:
            self._total_units = int(max_duration.value)
            self._use_tokens = False
        elif max_duration.unit == TimeUnit.EPOCH:
            # Total batches = steps_per_epoch * num_epochs
            self._total_units = int(state.steps_per_epoch * max_duration.value)
            self._use_tokens = False
        elif max_duration.unit == TimeUnit.TOKEN:
            # direct token target
            self._total_units = int(max_duration.value)
            self._use_tokens = True
        else:
            # For other units, fall back to token count if available.
            self._total_units = int(max_duration.value)
            self._use_tokens = True

        # Ensure total_units is at least 1 to avoid division-by-zero.
        if self._total_units < 1:
            self._total_units = 1

        if dist.get_global_rank() == 0:
            logger.log_metrics({"scheduler/mlm_total_units": self._total_units, "scheduler/use_tokens": self._use_tokens})

    def batch_start(self, state: State, logger: Logger):
        # Determine progress metric
        progress = state.timestamp.token.value if self._use_tokens else state.timestamp.batch.value
        if self._total_units is None:
            # Should not happen, but guard.
            return
        new_prob = self._compute_prob(progress)
        self._update_collator_or_packer(state, new_prob)

        if ((progress if not self._use_tokens else state.timestamp.batch.value) % self.log_interval == 0) and (
            dist.get_global_rank() == 0
        ):
            logger.log_metrics({"scheduler/mlm_probability": new_prob}) 
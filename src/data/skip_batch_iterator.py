"""Per-rank dataloader wrapper that drops batches falling in user-specified
index ranges. Used to skip past known bad-batch regions on resume without
running forward/backward on them — preserves Adam optimizer state exactly
because optimizer.step() is never called for the skipped batches.

Opt-in via top-level yaml key `skip_batches_in_range`. Not a Composer Algorithm.
"""
from typing import Iterable, List, Tuple


class SkipBatchIterator:
    """Wraps an iterable dataloader; yields all batches EXCEPT those whose
    0-indexed position (from the start of training, not from resume) falls
    in one of the configured `[start, end]` (inclusive) ranges.

    The `source_batch_idx` is local to each rank but deterministic given the
    same seed and rank count. Streaming MDS coordinates data across ranks
    consistently, so every rank's wrapper makes the same skip decision at
    the same position → no DDP/FSDP deadlock.

    State persistence: exposes `state_dict`/`load_state_dict` so resume after
    a partial skip preserves the counter. Inner state_dict is also forwarded
    if the inner exposes one.
    """

    def __init__(self, inner: Iterable, ranges: List[List[int]], start_offset: int = 0):
        """
        Args:
            inner: the inner iterable dataloader (e.g., the DataLoader inside a Composer DataSpec).
            ranges: list of `[start, end]` (inclusive) source-batch index ranges to skip.
            start_offset: initial value for `source_batch_idx`. Pass the resume batch number
                (e.g., 61359 if resuming from `ep8-ba61359-rank0.pt`) so `ranges` refer to
                Composer's GLOBAL batch index — i.e., the numbers visible in the loss plot —
                rather than batches-since-iteration-started. Default 0 means "no offset"
                (correct for runs starting from scratch, or when `ranges` are intentionally
                expressed relative to the iteration start).
        """
        if inner is None:
            raise ValueError("SkipBatchIterator: inner iterable cannot be None")
        # Normalize and validate ranges. `ranges` may be an OmegaConf ListConfig
        # (not a `list`/`tuple` to isinstance), so we duck-type via list() instead.
        normalized: List[Tuple[int, int]] = []
        for r in ranges or []:
            try:
                r_seq = list(r)
            except TypeError as ex:
                raise ValueError(f"skip range must be [start, end], got {r!r}") from ex
            if len(r_seq) != 2:
                raise ValueError(
                    f"skip range must have exactly 2 elements [start, end], got {r!r}"
                )
            s, e = int(r_seq[0]), int(r_seq[1])
            if s > e:
                raise ValueError(f"skip range start {s} > end {e}")
            normalized.append((s, e))
        self.inner = inner
        self.ranges = sorted(normalized)
        self.source_batch_idx: int = int(start_offset)
        self.skipped_count: int = 0
        self._prev_in_skip: bool = self._in_skip(self.source_batch_idx)

    # --- iteration --------------------------------------------------------

    def __iter__(self):
        for batch in self.inner:
            cur = self.source_batch_idx
            self.source_batch_idx += 1
            in_skip = self._in_skip(cur)
            # Transition-based debug log: only on entering / exiting a skip range,
            # only on rank 0. Two lines per crossed range, not per skipped batch.
            if in_skip != self._prev_in_skip:
                self._log_transition(cur, in_skip)
                self._prev_in_skip = in_skip
            if in_skip:
                self.skipped_count += 1
                continue  # drop on the floor; Composer never sees this batch
            yield batch

    def _log_transition(self, cur: int, entering: bool) -> None:
        try:
            from composer.utils import dist
            if dist.get_global_rank() != 0:
                return
        except Exception:
            pass  # composer unavailable (e.g., unit test) — still print
        state = "ENTERING" if entering else "EXITED"
        print(
            f"[skip_batches_in_range] {state} skip at source_batch_idx={cur}, "
            f"skipped_total={self.skipped_count}",
            flush=True,
        )

    def _in_skip(self, idx: int) -> bool:
        for s, e in self.ranges:
            if idx > e:
                continue
            if idx < s:
                return False  # sorted: no later range can match
            return True
        return False

    # --- forwarded attributes Composer / wandb / progress bar may inspect

    def __len__(self):
        # Best-effort: yielded count is len(inner) − skipped_count. Composer uses
        # this only for progress estimation; over-reporting is harmless.
        return len(self.inner)

    @property
    def batch_size(self):
        return getattr(self.inner, "batch_size", None)

    @property
    def dataset(self):
        return getattr(self.inner, "dataset", None)

    # --- checkpoint resume ------------------------------------------------

    def state_dict(self):
        out = {
            "_skip_source_batch_idx": self.source_batch_idx,
            "_skip_skipped_count": self.skipped_count,
        }
        if hasattr(self.inner, "state_dict"):
            inner_state = self.inner.state_dict()
            if isinstance(inner_state, dict):
                # Namespace inner state so we don't collide with Composer-expected keys.
                out["_inner"] = inner_state
        return out

    def load_state_dict(self, state):
        # Tolerate first-time activation (no prior state with our keys)
        if not isinstance(state, dict):
            return
        self.source_batch_idx = int(state.get("_skip_source_batch_idx", 0))
        self.skipped_count = int(state.get("_skip_skipped_count", 0))
        # Recompute the "in_skip" flag for the restored position so the next
        # transition log fires correctly on resume.
        self._prev_in_skip = self._in_skip(self.source_batch_idx)
        inner_state = state.get("_inner")
        if inner_state is not None and hasattr(self.inner, "load_state_dict"):
            self.inner.load_state_dict(inner_state)

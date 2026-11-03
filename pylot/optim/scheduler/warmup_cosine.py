import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """Linear warmup from 0 to the base LR, then a cosine decay to `eta_min`.

    After `t` calls to `.step()` (torch's `last_epoch`) the LR is

        t <  warmup_steps:          base_lr * t / warmup_steps
        warmup_steps <= t <= T_max: eta_min + (base_lr - eta_min)
                                    * (1 + cos(pi * (t - warmup_steps) / (T_max - warmup_steps))) / 2

    so `T_max` is the TOTAL number of steps, warmup included, and the LR
    reaches `eta_min` at `t = T_max`. Stepping past `T_max` raises: the
    schedule was sized for a shorter run.

    Built from flat config kwargs (`optim.lr_scheduler._class`), unlike a
    `SequentialLR` of two schedulers. `get_lr` is the closed form itself, not
    a recursion on the previous LR, and `_get_closed_form_lr` returns it, so
    `pylot.experiment.util.rewind_lr_scheduler` puts a resumed run back on
    the curve.
    """

    def __init__(self, optimizer, warmup_steps: int, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        if not 0 < warmup_steps < T_max:
            raise ValueError(
                f"WarmupCosineLR needs 0 < warmup_steps < T_max, got warmup_steps={warmup_steps}, T_max={T_max}"
            )
        self.warmup_steps = int(warmup_steps)
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        t = self.last_epoch
        if t > self.T_max:
            raise RuntimeError(
                f"WarmupCosineLR stepped to {t}, past T_max={self.T_max}: the schedule was sized for a shorter run"
            )
        if t < self.warmup_steps:
            return [base_lr * t / self.warmup_steps for base_lr in self.base_lrs]
        progress = (t - self.warmup_steps) / (self.T_max - self.warmup_steps)
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]


# The same schedule under the name the ORCD pylot checkout gave it (2026-10-14). The run configs
# trained there (e.g. PURE-QUIZ of 2026-10-14_amix-cross-iterative-finetune-192iso) name
# `pylot.optim.scheduler.WarmupCosineAnnealingLR`, so loading those runs needs it.
WarmupCosineAnnealingLR = WarmupCosineLR

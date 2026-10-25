"""`pylot.optim.scheduler.WarmupCosineLR`: linear warmup, then cosine to `eta_min`.

The fine-tune schedule of `2026-10-22_amix-cross-sigmoid`. It must be buildable from
flat YAML kwargs and resumable: `rewind_lr_scheduler` repositions only schedulers
with a closed form (see interseg3d's `tests/test_lr_scheduler_resume.py`).
"""

import math

import pytest
import torch

from pylot.experiment.util import rewind_lr_scheduler
from pylot.optim.scheduler import WarmupCosineLR

BASE_LR = 1e-4
WARMUP = 10
T_MAX = 110


def _sched(**kw):
    optim = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    return optim, WarmupCosineLR(optim, warmup_steps=WARMUP, T_max=T_MAX, **kw)


def _trace(n):
    optim, sched = _sched()
    lrs = [optim.param_groups[0]["lr"]]
    for _ in range(n):
        optim.step()
        sched.step()
        lrs.append(optim.param_groups[0]["lr"])
    return lrs


def test_warmup_then_cosine_to_zero():
    lrs = _trace(T_MAX)
    assert lrs[0] == 0.0
    assert lrs[WARMUP // 2] == pytest.approx(BASE_LR / 2)
    assert lrs[WARMUP] == pytest.approx(BASE_LR)
    assert lrs[(WARMUP + T_MAX) // 2] == pytest.approx(BASE_LR / 2)
    assert lrs[T_MAX] == pytest.approx(0.0, abs=1e-18)
    assert max(lrs) == pytest.approx(BASE_LR)
    t = 37
    progress = (t - WARMUP) / (T_MAX - WARMUP)
    assert lrs[t] == pytest.approx(BASE_LR * (1 + math.cos(math.pi * progress)) / 2)


def test_eta_min_is_the_floor():
    optim, sched = _sched(eta_min=1e-6)
    for _ in range(T_MAX):
        optim.step()
        sched.step()
    assert optim.param_groups[0]["lr"] == pytest.approx(1e-6)


def test_stepping_past_t_max_raises():
    optim, sched = _sched()
    for _ in range(T_MAX):
        optim.step()
        sched.step()
    with pytest.raises(RuntimeError, match="past T_max"):
        sched.step()


@pytest.mark.parametrize("warmup,t_max", [(0, 10), (10, 10), (11, 10)])
def test_rejects_warmup_outside_the_run(warmup, t_max):
    optim = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)
    with pytest.raises(ValueError, match="warmup_steps"):
        WarmupCosineLR(optim, warmup_steps=warmup, T_max=t_max)


@pytest.mark.parametrize("steps", [3, WARMUP, 64])
def test_rewind_matches_an_uninterrupted_run(steps):
    lrs = _trace(steps + 5)
    optim, sched = _sched()
    assert rewind_lr_scheduler(optim, sched, steps)
    resumed = [optim.param_groups[0]["lr"]]
    for _ in range(5):
        optim.step()
        sched.step()
        resumed.append(optim.param_groups[0]["lr"])
    assert resumed == pytest.approx(lrs[steps:], rel=1e-12, abs=1e-18)

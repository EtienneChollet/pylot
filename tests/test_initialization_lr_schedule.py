"""`TrainExperiment.build_initialization` starts a new run's LR schedule at step 0.

An `initialization:` checkpoint carries the `_epoch` of the run it came from. `set_state` used to
leave the fresh schedule there, so a fine-tune initialized from an epoch-500 checkpoint with a
batch-stepped schedule began at step 500 x len(train_dl).
"""

import pytest
import torch

from pylot.experiment.train import TrainExperiment
from pylot.optim.scheduler import WarmupCosineLR
from pylot.util.config import ImmutableConfig

BASE_LR = 1e-4
STEPS_PER_EPOCH = 10


def _experiment(init_path):
    # Only the attributes `build_initialization` and `set_state` read; `__init__` would build a
    # whole run directory.
    exp = TrainExperiment.__new__(TrainExperiment)
    exp.config = ImmutableConfig({"initialization": {"path": str(init_path), "optim": False}})
    exp.device = torch.device("cpu")
    exp.compiled = False
    exp.model = torch.nn.Linear(1, 1)
    exp.optim = torch.optim.AdamW(exp.model.parameters(), lr=BASE_LR)
    exp.lr_scheduler = WarmupCosineLR(exp.optim, warmup_steps=20, T_max=200)
    exp.lr_scheduler_step_on = "batch"
    exp.train_dl = range(STEPS_PER_EPOCH)
    return exp


def test_initialization_starts_the_schedule_at_step_0(tmp_path):
    source = torch.nn.Linear(1, 1)
    init_path = tmp_path / "init.pt"
    torch.save({"model": source.state_dict(), "_epoch": 5, "_epoch_is_count": True}, init_path)

    exp = _experiment(init_path)
    exp.build_initialization()

    assert exp._epoch == 0
    assert exp.lr_scheduler.last_epoch == 0
    assert exp.optim.param_groups[0]["lr"] == 0.0
    assert torch.equal(exp.model.weight, source.weight)

    exp.optim.step()
    exp.lr_scheduler.step()
    assert exp.optim.param_groups[0]["lr"] == pytest.approx(BASE_LR / 20)

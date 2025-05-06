import itertools
import math

from tabulate import tabulate
from loguru import logger

from ..util.timer import StatsCUDATimer, StatsTimer
from ..util.torchutils import to_device


def Throughput(experiment, n_iter: int = 1_00, verbose: bool = True):

    if "cuda" in str(experiment.device):
        timer = StatsCUDATimer(unit="ms", skip=n_iter//10)
    else:
        timer = StatsTimer(unit="ms", skip=n_iter//10)

    # Dataloader throughput
    dl = iter(itertools.cycle(experiment.train_dl))
    for _ in range(n_iter+n_iter//10):
        with timer("train-dl"):
            _ = next(dl)

    # Model training throughput
    sample_input = to_device(next(iter(experiment.train_dl)), experiment.device, experiment.config.get("train.channels_last", False))
    for _ in range(n_iter+n_iter//10):
        with timer("train-loop"):
            experiment.run_step(batch=sample_input, batch_idx=0, backward=True, augmentation=True)

    timer_df = timer.measurements_df()
    timer_df = timer_df.set_index("label")

    if verbose:
        logger.info(
            f'\n{tabulate(timer_df, headers="keys")}'
        )

    t_dl = timer["train-dl"].mean
    t_gpu = timer["train-loop"].mean
    if t_dl > t_gpu and verbose:

        logger.error(
            f"Experiment is dataloader bound dl={t_dl:.2f}ms > "
            f"gpu={t_gpu:.2f}ms",
        )

        recommended_num_workers = math.ceil(
            experiment.train_dl.num_workers * t_dl / t_gpu
        )

        logger.error(
            f"Try setting num_workers={recommended_num_workers}"
        )

    return timer_df

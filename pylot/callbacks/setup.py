import json

from tabulate import tabulate
from loguru import logger

from ..util.summary import summary

from ..metrics import module_table, parameter_table


def ParameterTable(experiment, save=True, verbose=True):

    df = parameter_table(experiment.model)

    if verbose:
        logger.info(
            f'\n{tabulate(df, headers="keys")}'
        )

    if save:
        with (experiment.path / "params.csv").open("w") as f:
            df.to_csv(f, index=False)


def ModuleTable(experiment, save=True, verbose=True):

    df = module_table(experiment.model)
    if verbose:
        logger.info(
            f'\n{tabulate(df, headers="keys")}'
        )

    if save:
        with (experiment.path / "modules.csv").open("w") as f:
            df.to_csv(f, index=False)


def Summary(experiment, filename="summary.txt"):

    x, _ = next(iter(experiment.train_dl))

    # Save model summary
    summary_path = experiment.path / filename

    if not summary_path.exists():

        with summary_path.open("w") as f:
            s = summary(
                experiment.model, x.shape[1:], echo=False, device=experiment.device
            )
            print(s, file=f)

            print("\n\nOptim\n", file=f)
            print(experiment.optim, file=f)

            if experiment.scheduler is not None:
                print("\n\nScheduler\n", file=f)
                print(experiment.scheduler, file=f)

        with summary_path.with_suffix(".json").open("w") as f:
            s = summary(
                experiment.model,
                x.shape[1:],
                echo=False,
                device=experiment.device,
                as_stats=True,
            )
            json.dump(s, f)


def CheckHalfCosineSchedule(experiment):

    scheduler = experiment.get_param("train.scheduler.scheduler", None)
    if scheduler == "CosineAnnealingLR":
        T_max = experiment.get_param("train.scheduler.T_max", -1)
        epochs = experiment.get_param("train.epochs")
        assert T_max == epochs, f"T_max not equal to epochs {T_max} != {epochs}"

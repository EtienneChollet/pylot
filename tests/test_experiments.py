"""
Tests for PyLot experiments.
"""

import pytest
from pathlib import Path
import pylot.experiment.base as base
from pylot.util.ioutil import autosave
from pylot.util.config import ImmutableConfig, FHDict
from pylot.util.metrics import MetricsDict
from pylot.util.thunder import ThunderDict


@pytest.fixture
def exp_dir(tmp_path: Path):
    """
    Make a minimal experiment directory with a super simple config.yml

    Notes
    -----
    `tmp_path` is a built-in pytest fixture that points to the OS's temp dir.
    """

    # Make the directory in the filesystem
    exp_dir_path = tmp_path / "exp"
    exp_dir_path.mkdir()

    config = {
        'experiment': {'seed': 19},
        'log': {'root': 'dalcalab_root'},
    }

    # Write the configuration to the yaml
    config_path = exp_dir_path / 'config.yml'
    autosave(config, config_path)

    return exp_dir_path


def test_base_exp_init(exp_dir: str):
    """
    Test that the initialization sets instance attributes correctly.
    """

    # Initialize the experiment
    exp = base.BaseExperiment(
        path=str(exp_dir),
        logging_level='info'
    )

    assert exp.path == exp_dir
    assert exp.name == exp_dir.stem
    assert isinstance(exp.config, ImmutableConfig)
    assert exp.config._data['experiment']['seed'] == 19
    assert isinstance(exp.properties, FHDict)
    assert isinstance(exp.metadata, FHDict)
    assert isinstance(exp.metricsd, MetricsDict)
    assert isinstance(exp.store, ThunderDict)

    # Hash uses the exp dir path
    assert hash(exp) == hash(exp_dir)


def test_metrics_property(exp_dir):
    """
    Test that the metrics property properly records the metrics.
    """

    # Initialize experiment
    exp = base.BaseExperiment(str(exp_dir))

    # Log a metric
    first_metric_real = {'dice': 0.75}
    exp.metricsd['metrics'].log(first_metric_real)

    # Log another metric
    second_metric_real = {'mse': 1.0232}
    exp.metricsd['metrics'].log(second_metric_real)

    first_metric = exp.metrics.data[0]
    second_metric = exp.metrics.data[1]

    assert first_metric_real == first_metric
    assert second_metric_real == second_metric

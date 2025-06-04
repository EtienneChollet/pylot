"""
Base class and utilities for PyLot experiments.

Classes
-------
BaseExperiment
    Sets up experimental run, config loading, logging, metrics collection,
    and callback management.

Examples
--------
>>> # Define an experiment class with a concrete `run()` method.
>>> class MyExp(BaseExperiment):
...     def run(self):
...         logger.info("Running MyExperiment")
>>> # Make the configuration for it (can also be from an e.g. `.yml` file)
>>> config = {'experiment': {'seed': 123}, 'log': {'root': './logs'}}
>>> # Construct the experiment 
>>> experiment = MyExperiment.from_config(config)
>>> experiment.build_callbacks()
>>> experiment.run()
"""

__all__ = [
    'eval_callbacks',
    'BaseExperiment',
]

# Standard library imports
import pathlib
import inspect
import functools
from abc import abstractmethod
from typing import Union

# Third party imports
import yaml
from loguru import logger

# Local imports
from .util import fix_seed, absolute_import, generate_tuid
from ..util.metrics import MetricsDict
from ..util.config import HDict, FHDict, ImmutableConfig, config_digest
from ..util.ioutil import autosave
from ..util.libcheck import check_environment
from ..util.thunder import ThunderDict


def eval_callbacks(all_callbacks, experiment):
    evaluated_callbacks = {}
    for group, callbacks in all_callbacks.items():
        evaluated_callbacks[group] = []

        for callback in callbacks:
            if isinstance(callback, str):
                cb = absolute_import(callback)(experiment)
            elif isinstance(callback, dict):
                assert len(callback) == 1, "Callback must have length 1"
                callback, kwargs = next(iter(callback.items()))
                cb = absolute_import(callback)(experiment, **kwargs)
            else:
                raise TypeError("Callback must be either str or dict")
            evaluated_callbacks[group].append(cb)
    return evaluated_callbacks


class BaseExperiment:
    """
    Base class for experiment setup and execution.

    Attributes
    ----------
    path : pathlib.Path
        Absolute path to a particular experimental run.
    name : str
        Name of the particular experimental run in the format
        `YYYYMMDD_HHMMSS-nonce-hash`
    config : ImmutableConfig
        Dictionary of the experiment's configurations that is not mutable.
        Contains a configuration hash.
    properties : FHDict
        Dictionary-like store for experiment properties saved to JSON.
    metadata : FHDict
        Dictionary-like store for metadata (creation time, digest).
    metricsd : MetricsDict
        Container for metric values, organized by name and group.
    store : ThunderDict
        Key-value store for intermediate or auxiliary data.
    callbacks : dict
        Mapping of callback groups to lists of initialized callbacks.
    """

    def __init__(
        self,
        path: str,
        logging_level: str = 'CRITICAL',
    ):
        """
        Initialize an experiment from an existing experiment directory.

        Parameters
        ----------
        path : str
            Path to the experiment run directory containig a `config.yml`,
            `metadata.json`, `properties.json`, etc... files.
        """

        # Convert to pathlib if necessary
        if isinstance(path, str):
            path = pathlib.Path(path)

        # Store the path as an attribute
        self.path = path

        # Make sure logging is set up and make first log confirming experiment
        # ensure_logging(
        #    log_file_abspath=self.path / 'output.log',
        #    level=logging_level,
        # )

        logger.info(f'Absolute path to experiment run: "{self.path}"')

        if not self.path.exists():
            error_message = f'Path to experiment not found: "{self.path}"'
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        # Set derived instance attributes
        self.name = self.path.stem

        # Load the configuration file (defaults to 'config.yml')
        self.config = ImmutableConfig.from_file(path / "config.yml")

        # self.config = yaml.safe_dump(self.config._data, sort_keys=False)

        # Initialize stores
        self.properties = FHDict(self.path / "properties.json")
        self.metadata = FHDict(self.path / "metadata.json")
        self.metricsd = MetricsDict(self.path)
        self.store = ThunderDict(self.path / "store")

        # Set the global seed of the experiment for reproducibility
        seed = self.config.get("experiment.seed", 42)
        fix_seed(seed)
        logger.info(f'Fixed global random seed to: {seed}')

        # Check for optimized packages (e.g. numpy, scikit-learn, ...)
        check_environment()

        # Record class & module for traceability. Useful for subclasses.
        self.properties["experiment.class"] = self.__class__.__name__
        self.properties["experiment.module"] = self.__class__.__module__
        logger.debug("Recorded experiment class and module.")

        # If additional log properties defined in config, update them
        if "log.properties" in self.config:
            self.properties.update(self.config["log.properties"])
            logger.debug("Updated properties from config.log.properties.")

    @classmethod
    def from_config(
        cls,
        config: Union[dict, HDict],
        logging_level: str = 'debug',
    ) -> "BaseExperiment":
        """
        Create a new experiment directory from a configuration dictionary.

        Parameters
        ----------
        config : dict or HDict
            Collection of fields defining the configuration of an experiment.

        Returns
        -------
        BaseExperiment
            Experiment instance initialized within a new directory of the
            current working directory.

        Examples
        --------
        ### Defining a simple configuration file with a logging root
        >>> config = {
        ...     'experiment': {"seed": 99,},
        ...     'log': {'root': 'my_experiment'}
        ... }
        >>> experiment = pylot.BaseExperiment.from_config(config)

        Notes
        -----
        If the root directory for pylot experiments is not specified (by the
        `log.root` key) in the config dict, experiment root defaults to
        `pylot_experiments`.
        """

        if isinstance(config, dict):
            pass

        # Convert HDict to dict if necessary
        elif isinstance(config, HDict):
            config = config.to_dict()

        elif isinstance(config, (str, pathlib.Path)):
            config = ImmutableConfig.from_file(config).to_dict()

        else:
            logger.error(
                'Do not know how to handle config with passed config of type '
                f'{type(config)}'
            )

        # Make a default root for the experiment
        default_experiment_root = 'pylot_experiments'

        # Make sure there's a valid root directory for the experiments
        if "log" not in config:
            # Set the root
            config["log"] = {"root": default_experiment_root}

        if "root" not in config["log"]:
            config['log']['root'] = default_experiment_root

        # Determine base folder for experiments
        experiments_root = pathlib.Path(config['log']['root'])

        # Log the root path of the experiments
        logger.info(
            'Root directory for experiments located at: '
            f'"{experiments_root}"'
        )

        # Generate names for unique identifier of experimental run
        created_timestamp, random_suffix = generate_tuid()
        digest = config_digest(config)

        # Generate the (unique) name for the experiment run.
        experiment_unique_id = f"{created_timestamp}-{digest}-{random_suffix}"
        logger.info(
            f'Made unique identifier for experiment: "{experiment_unique_id}"'
        )

        # Construct the path to the experiment run
        experiment_dir = experiments_root / experiment_unique_id

        # TODO: Determine wherelse `nonce` and `create_time` are used. Change.
        metadata = {
            "create_time": created_timestamp,
            "nonce": random_suffix,
            "digest": digest
        }

        # TODO: document `autosave`
        autosave(metadata, experiment_dir / "metadata.json")
        autosave(config, experiment_dir / "config.yml")

        class_instance = cls(
            path=str(experiment_dir.absolute()),
            logging_level=logging_level,
        )

        return class_instance

    @property
    def metrics(self) -> MetricsDict:
        """
        Retrieve the current run's metrics dictionary from the experiment
        run's `metrics.jsonl`.

        Returns
        -------
        dict
            Metrics dictionary containing logged metrics for the current
            experiment run.

        Examples
        --------
        >>> exp = BaseExperiment.from_config({'experiment': {'seed': 1}})
        >>> metrics = exp.metrics  # initially empty dict

        Notes
        -----
        The `data` attribute of self.metrics returns the metrics dictionary
        """

        if __debug__:
            logger.debug(f'Retrieving metrics for experiment: {self.name}')

        return self.metricsd["metrics"]

    def __hash__(self):
        return hash(self.path)

    @abstractmethod
    def run(self):
        """
        Run the experiment.

        This method should be overridden by subclasses (concrete classes) to
        define training/evauluation routines.

        Examples
        --------
        >>> class MyExp(BaseExperiment):
        ...     def run(self):
        ...         logger.info("Doing stuff, but also things!")
        """

        # Make the error message for logger and NotImplementedError
        error_message = (
            "BaseExperiment.run() called directly. This is an abstract method "
            "that must be overridden in a subclass"
        )

        # Log the error
        logger.error(error_message)

        raise NotImplementedError(error_message)

    def __repr__(self):
        return f'{self.__class__.__name__}("{str(self.path)}")'

    def __str__(self):
        s = f"{repr(self)}\n---\n"
        s += yaml.safe_dump(self.config._data, indent=2)
        return s

    def build_callbacks(self):
        """
        Instantiate and attach callbacks defined in the config.

        Populates the `self.callbacks` dict with lists of callback
        instances grouped by name.

        Examples
        --------
        >>> exp = BaseExperiment.from_config({
        ...     'experiment': {'seed': 5},
        ...     'callbacks': {'train': ['module.CB']}
        ... })
        >>> exp.build_callbacks()
        >>> 'train' in exp.callbacks
        True
        """
        logger.info(
            f"Building callbacks for experiment {self.name}"
        )

        # Initialize callbacks container
        self.callbacks = {}

        if "callbacks" in self.config:

            # Get the callback group
            self.callbacks = eval_callbacks(self.config["callbacks"], self)

            # Log the callback progress
            logger.debug(
                f'Attached callback groups: {list(self.callbacks.keys())}'
            )

        else:
            logger.info(
                f'No callbacks configured for experiment run {self.name}'
            )

    def __init_subclass__(cls, **kwargs):
        """
        Automatically wrap public, callable methods of subclasses with
        logger.catch so that exceptions are logged (and optionally
        reraised). Methods can opt out by setting __no_catch__ on them.
        """
        super().__init_subclass__(**kwargs)

        for name, attr in list(cls.__dict__.items()):
            # Skip private, magic, non-callables, or explicit opt-outs
            if name.startswith("_") or not callable(attr):
                continue
            if getattr(attr, "__no_catch__", False):
                continue

            # Unwrap staticmethod/classmethod to the raw function
            if isinstance(attr, classmethod):
                func = attr.__func__
                wrapped = classmethod(
                    logger.catch(reraise=True)(
                        functools.wraps(func)(func)
                    )
                )
            elif isinstance(attr, staticmethod):
                func = attr.__func__
                wrapped = staticmethod(
                    logger.catch(reraise=True)(
                        functools.wraps(func)(func)
                    )
                )
            elif inspect.isfunction(attr):
                wrapped = logger.catch(reraise=True)(
                    functools.wraps(attr)(attr)
                )
            else:
                # leave other callables (e.g. descriptors) alone
                continue

            setattr(cls, name, wrapped)

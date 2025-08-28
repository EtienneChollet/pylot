__all__ = [
    'ensure_rootdir_in_config',
    'get_config',
    'make_experiment_id',
]

import datetime
import random
import importlib
import pathlib
from typing import Tuple, Dict, List, Union
import numpy as np

import importlib
import functools

from ..util.config import HDict, Config
from ..util.ioutil import autoload
from ..util.more_functools import partial
from pylot.torch.torchlib import torch
from pylot.util.config import HDict, ImmutableConfig, config_digest


def fix_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_ADJECTIVES: List[str] = [
    "brisk", "calm", "damp", "eager", "faint", "glad", "huge", "icy",
    "jolly", "keen", "loud", "mild", "neat", "odd", "pale", "quick",
    "raw", "shy", "tidy", "vast", "warm", "young", "zany", "able",
    "best", "cool", "dark", "easy", "fine", "good", "high", "just",
    "kind", "lazy", "mean", "nice", "open", "pure", "rich", "soft",
    "true", "wild", "zero", "blue", "fast", "grey", "long", "pink",
    "red", "slow", "silly", "brave", "fiery", "chill", "fun", "smart",
    "witty", "happy", "bad", "old", "mad",
]

_NOUNS: List[str] = [
    "apple", "beach", "cloud", "dream", "eagle", "flame", "glove",
    "heart", "inlet", "jelly", "knife", "leaf", "mango", "night",
    "ocean", "plant", "queen", "river", "stone", "tree", "unity",
    "vapor", "whale", "xray", "yacht", "zebra", "bird", "cake",
    "door", "echo", "frog", "gate", "hill", "ink", "jar", "kite",
    "lamp", "moon", "nest", "owl", "pen", "quiz", "rose", "sun",
    "top", "urn", "van", "web", "yak", "zip", "boat", "button",
    "camel", "latte", "data", "year", "blob", "thingy", "donut",
    "taco", "rock", "wolf", "nerd", "lemon", "sloth", "cow", "elk"
]


def generate_fun_name() -> str:
    """
    Generate a random run name composed of an adjective and a noun.

    Returns
    -------
    str
        A two-word name in the form "adjective-noun", randomly picked
        from internal word lists. Each word is at most 5 characters.
    """

    adj = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    return f"{adj}-{noun}"


def generate_tuid(
    nonce_length: int = 4
) -> Tuple[str, int]:
    """
    Generate time-based unique ID for experiment run directories.
    """

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nonce = generate_fun_name()

    return now, nonce.upper()


def absolute_import(reference: str):
    """
    Resolve a dotted path to a Python object, including nested attributes.

    Parameters
    ----------
    reference : str
        Dot-separated reference path, e.g. 'thing.DataClass.attrname'.

    Returns
    -------
    Any
        The resolved Python object.

    Raises
    ------
    ImportError
        If any part of the import fails.
    """
    parts = reference.split('.')

    # Try importing the longest valid module prefix
    for i in reversed(range(1, len(parts))):
        module_path = '.'.join(parts[:i])
        try:
            module = importlib.import_module(module_path)
            # Resolve remaining attributes
            return functools.reduce(getattr, parts[i:], module)
        except (ModuleNotFoundError, AttributeError):
            continue

    raise ImportError(f"Could not import or resolve: {reference}")


def eval_config(config):
    if not isinstance(config, (dict, list, HDict)):
        return config
    if isinstance(config, HDict):
        return eval_config(config.to_dict())
    if isinstance(config, list):
        return [eval_config(v) for v in config]

    for k, v in config.items():
        if isinstance(v, (dict, list)):
            config[k] = eval_config(v)

    state = config.pop("_state", None)

    if "_class" in config:
        config = absolute_import(config.pop("_class"))(**config)
    elif "_fn" in config:
        fn = absolute_import(config.pop("_fn"))
        config = partial(fn, **config)

    if state is not None:
        key = None
        if isinstance(state, (list, tuple)):
            state, key = state
        with pathlib.Path(state).open("rb") as f:
            state_dict = torch.load(f)
            if key is not None:
                state_dict = state_dict[key]
            config.load_state_dict(state_dict)

    return config


def autoload_experiment(path: pathlib.Path):
    path = pathlib.Path(path)
    cfg: Dict = autoload(path / "properties.json")
    cls_name = cfg["experiment"]["class"]
    mod_name = cfg["experiment"]["module"]
    cls_ = absolute_import(f"{mod_name}.{cls_name}")
    return cls_(path)


def config_from_path(path):
    return Config(autoload(path / "config.yml"))


def path_from_job(job):
    import parse

    stdout = job.stdout()
    return pathlib.Path(
        parse.search('Running {exp_type}Experiment("{path}")', stdout)["path"]
    )


def config_from_job(job):
    return config_from_path(path_from_job(job))


def ensure_rootdir_in_config(
    config: Union[dict, HDict, str, pathlib.Path],
    experiment_rootdir: str = 'pylot_experiments',
    logger=None,
) -> HDict:
    """
    Ensure the root directory exists in the configuration dictionary
    """

    # Make sure there's a valid root directory for the experiments
    if "log" not in config:
        # Set the root
        config["log"] = {"root": experiment_rootdir}

    if "root" not in config["log"]:
        config['log.root'] = experiment_rootdir

    if logger is not None:
        # Log the root path of the experiments
        logger.info(
            'Root directory for experiments located at: '
            f'{config.get("log.root")}'
        )

    return config


def get_config(
    config: Union[dict, HDict, str, pathlib.Path],
    experiment_rootdir: str = 'pylot_experiments',
    logger=None,
) -> HDict:

    if isinstance(config, (dict, ImmutableConfig)):
        config = HDict(config)

    elif isinstance(config, (str, pathlib.Path)):
        config = HDict.from_file(config)

    else:
        if logger is not None:
            logger.error(
                'Do not know how to handle config with passed config of type '
                f'{type(config)}'
            )

    # some other processing...
    config = ensure_rootdir_in_config(
        config=config,
        experiment_rootdir=experiment_rootdir,
    )

    return ImmutableConfig(config)


def make_experiment_id(
    config: Union[dict, HDict, str, pathlib.Path],
    logger=None,
):
    # Generate names for unique identifier of experimental run
    created_timestamp, random_suffix = generate_tuid()
    digest = config_digest(config.to_dict())

    # Generate the (unique) name for the experiment run.
    make_experiment_id = f"{created_timestamp}-{digest}-{random_suffix}"

    if logger is not None:
        logger.info(
            f'\nMade unique identifier for experiment: "{make_experiment_id}"'
        )

    metadata = {
        "create_time": created_timestamp,
        "nonce": random_suffix,
        "digest": digest
    }

    return make_experiment_id, metadata

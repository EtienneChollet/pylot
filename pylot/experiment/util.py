import datetime
import functools
import string
import random
import time
import importlib
import pathlib
import random
from typing import Tuple, Dict, List

from ..util.config import HDict, Config
from ..util.ioutil import autoload
from ..util.more_functools import partial

import torch
import numpy as np


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
    "red", "slow"
]

_NOUNS: List[str] = [
    "apple", "beach", "cloud", "dream", "eagle", "flame", "glove",
    "heart", "inlet", "jelly", "knife", "leaf", "mango", "night",
    "ocean", "plant", "queen", "river", "stone", "tree", "unity",
    "vapor", "whale", "xray", "yacht", "zebra", "bird", "cake",
    "door", "echo", "frog", "gate", "hill", "ink", "jar", "kite",
    "lamp", "moon", "nest", "owl", "pen", "quiz", "rose", "sun",
    "top", "urn", "van", "web", "yak", "zip"
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


def generate_tuid(nonce_length: int = 4) -> Tuple[str, int]:
    """
    Generate time-based unique ID for experiment run directories.
    """

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nonce = generate_fun_name()

    return now, nonce.upper()


def absolute_import(reference):
    module, _, attr = reference.rpartition(".")
    if importlib.util.find_spec(module) is not None:
        module = importlib.import_module(module)
        if hasattr(module, attr):
            return getattr(module, attr)

    raise ImportError(f"Could not import {reference}")


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

"""
Class for training experiments.

Example
-------
>>> from universeg.experiment import TrainExperiment
>>> exp =  TrainExperiment.from_config(config)
>>> exp.run()
"""

import copy
import pathlib
from typing import List

from tqdm import tqdm
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger

from ..nn.util import num_params, split_param_groups_by_weight_decay
from ..util.ioutil import autosave
from ..util.meter import MeterDict
from ..util.torchutils import to_device
from .base import BaseExperiment
from .util import absolute_import, eval_config


class TrainExperiment(BaseExperiment):
    """
    A training experiment to be subclassed for concrete implementations.

    Attributes
    ----------
    config : dict
        inherited from `BaseExperiment`.
    properties : dict
        inherited from `BaseExperiment`.
    metadata : dict
        inherited from `BaseExperiment`.
    train_dataset : torch.utils.data.Dataset
        Training dataset set by `build_data()` method.
    val_dataset : torch.utils.data.Dataset
        Training dataset set by `build_data()` method.
    train_dl : torch.utils.data.DataLoader
        Dataloader for training data, set by `build_dataloader()` method.
    val_dl : torch.utils.data.DataLoader
        Dataloader for training data, set by `build_dataloader()` method.
    model : torch.nn.Module
        Model with trainable parameters
    properties : dict
        Properties and metadata of `TrainExperiment`
    optim : torch.optim
    """

    def __init__(self, path, *args, **kwargs):
        """
        Initialize experiment.

        Parameters
        ----------
        path : str
            Path to the directory containing a `config.yml`

        Notes
        -----
        This method initializes the components of an experiment in the
        following order:
        1. Build the datasets (`train_dataset` and `val_dataset` attrs)
        2. Build the dataloaders (`train_dl` and `val_dl` attrs)
        3. Initialize the optimizer (`optim` attr)
        4. Build the metrics (`metric_fns`)
        5. Build the augmentations (`augmentations` only in concrete instances)
        """

        # Enable cuDNN auto-tuner for performance
        torch.backends.cudnn.benchmark = True

        # Initialize parent `BaseExperiment`
        super().__init__(path, *args, **kwargs)

        # Set cuda if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build data, model, optimizer, loss, metrics, augmentations
        self.build_data()
        self.build_model()
        self.build_optim()
        self.build_loss()
        self.build_metrics()
        self.build_augmentations()

    def build_data(self):
        """
        Set `train_dataset` and `val_dataset` instance attributes.

        Reads dataset configuration from the `data` field of `self.config`
        and initializes datasets for training and validation splits.

        Notes
        -----
        If using this method off-the-shelf, the dataset specified by the `data`
        key must accept `train` and `val` arguments in its constructor to
        specify training and validation splits.
        """

        # Make sure config file has 'data' field
        assert 'data' in self.config.keys(), (
            f'"data" key not found in {self.path}/config.yml'
        )

        # Get the dataset configruations
        data_cfg = self.config["data"].to_dict()

        # Get the class or function constructor for the dataset
        dataset_constructor = data_cfg.pop(
            "_class", None
        ) or data_cfg.pop("_fn")

        # Dynamically import the constructor
        dataset_cls = absolute_import(dataset_constructor)

        # Set the train and val dataset attrs by making an instance of the
        # dataset classes and feeding in kwargs
        self.train_dataset = dataset_cls(split="train", **data_cfg)
        self.val_dataset = dataset_cls(split="val", **data_cfg)

    def build_dataloader(self):
        """
        Set `train_dl` and `val_dl` instance attributes. 

        Reads dataloader configuration from the `dataloader` field of
        `self.config`
        """

        # Make sure config file has 'dataloader' field
        assert 'dataloader' in self.config.keys(), (
            f'"dataloader" key not found in {self.path}/config.yml'
        )

        # Get the dataloader configuraion
        dl_cfg = self.config["dataloader"]

        # Make sure the batch size is less than the total training set
        assert self.config["dataloader.batch_size"] <= len(
            self.train_dataset
        ), (
            f'Batch size = {self.config["dataloader.batch_size"]} should not be '
            f'larger than dataset (len = {len(self.train_dataset)})'
        )

        # Set the validation dataloader attribute
        self.train_dl = DataLoader(
            self.train_dataset, shuffle=True, drop_last=False, **dl_cfg
        )

        # Set the validation dataloader attribute
        self.val_dl = DataLoader(
            self.val_dataset, shuffle=False, drop_last=False, **dl_cfg
        )

    def build_model(self):
        """
        Initialize the model from config and record param count.
        """

        # Set the model
        self.model = eval_config(self.config["model"])

        # Compute and store the number of trainable parameters into properties
        self.properties["num_params"] = num_params(self.model)

    def build_optim(self):
        """
        Initialize optimizer from config, handling weight decay.
        """

        # Get the optimizer dict from config
        optim_cfg = self.config["optim"].to_dict()

        # Optionaly handle weight decay for param groups
        if "weight_decay" in optim_cfg:
            optim_cfg["params"] = split_param_groups_by_weight_decay(
                self.model, optim_cfg["weight_decay"]
            )

        else:
            optim_cfg["params"] = self.model.parameters()

        # Set the `optim` instance attr
        self.optim = eval_config(optim_cfg)

    def build_loss(self):
        """
        Initialize the loss function from configuration.

        Reads the `loss_func` field of `self.config` and sets the
        `loss_func` attribute.
        """

        # Instantiate loss function
        self.loss_func = eval_config(self.config["loss_func"])

    def build_metrics(self):
        """
        Initialize metric functions for logging.

        Reads the `log.metrics` field of `self.config` (if present)
        and sets `metric_fns` to a dict of callables.
        """

        # Initialize container for storing metrics
        self.metric_fns = {}

        if "log.metrics" in self.config:

            # Deep copy to avoid unwanted side effects
            metrics_config = copy.deepcopy(self.config["log.metrics"])

            # Initialize the metric functions
            self.metric_fns = eval_config(metrics_config)

    def build_initialization(self):
        """
        Load and apply an initialization state to the model from a file.

        This method checks for an 'initialization' section in the configuration.
        If present, it loads a serialized state dictionary from the specified
        path and applies it to the model. The configuration may optionally skip
        optimizer state loading and enforce strict key matching.

        Raises
        ------
        FileNotFoundError
            If the initialization file does not exist.
        RuntimeError
            If `torch.load` fails or the state cannot be applied to the model.

        Examples
        --------
        # Inside the class
        >>> self.config = {
        ...     "initialization": {
        ...         "path": "checkpoint.pth",
        ...         "optim": False,
        ...         "strict": True
        ...     }
        ... }
        >>> self.build_initialization()

        Notes
        -----
        I am not sure how much this method is used??
        """

        # Check if 'initialization' is provided in the config dict
        if "initialization" in self.config:

            # Convert the configuration to a dictionary for easy access
            init_cfg = self.config["initialization"].to_dict()

            # Get the file path from the config and create a Path object
            path = pathlib.Path(init_cfg["path"])

            # Open the checkpoint file and load the state using torch.load
            with path.open("rb") as f:

                # Make sure it's on the right device!
                state = torch.load(f, map_location=self.device)

            # Optionally remove the optimizer state from the checkpoint?
            if not init_cfg.get("optim", True):
                # (If we only want to initialize model weights)
                state.pop("optim", None)

            # Determine key-matching behavior If True, keys of state dict must
            # match model perfectly
            strict = init_cfg.get("strict", True)

            # Apply the loaded state to the model
            self.set_state(state, strict=strict)

            # Log the successful initialization
            logger.info(f"Loaded initialization state from: {path}")

    @property
    def state(self):
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "_epoch": self.properties["epoch"],
        }

    def set_state(self, state, strict=True):
        for attr, state_dict in state.items():
            if not attr.startswith("_"):
                x = getattr(self, attr)
                if isinstance(x, nn.Module):
                    x.load_state_dict(state_dict, strict=strict)
                elif isinstance(x, torch.optim.Optimizer):
                    x.load_state_dict(state_dict)
                else:
                    raise TypeError(f"Unsupported type {type(x)}")

        self._checkpoint_epoch = state["_epoch"]
        self._epoch = state["_epoch"]

    def checkpoint(self, tag=None):
        self.properties["epoch"] = self._epoch

        checkpoint_dir = self.path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        tag = tag if tag is not None else "last"
        logger.info(f"Checkpointing with tag:{tag} at epoch:{self._epoch}")

        with (checkpoint_dir / f"{tag}.pt").open("wb") as f:
            torch.save(self.state, f)

    @property
    def checkpoints(self, as_paths=False) -> List[str]:
        checkpoints = list((self.path / "checkpoints").iterdir())
        checkpoints = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
        if as_paths:
            return checkpoints
        return [c.stem for c in checkpoints]

    def load(self, tag=None):
        checkpoint_dir = self.path / "checkpoints"
        tag = tag if tag is not None else "last"

        with (checkpoint_dir / f"{tag}.pt").open("rb") as f:
            state = torch.load(
                f=f,
                weights_only=True,
                map_location=self.device
            )

            self.set_state(state)

        logger.info(
            f"Loaded checkpoint with tag:{tag}. "
            f"Last epoch:{self.properties['epoch']}"
        )

        return self

    def to_device(self):
        self.model = to_device(
            self.model, self.device, self.config.get(
                "train.channels_last",
                False
            )
        )

    def run_callbacks(self, callback_group, **kwargs):
        for callback in self.callbacks.get(callback_group, []):
            callback(**kwargs)

    def run(self):
        logger.info(f"Running {str(self)}")
        epochs: int = self.config["train.epochs"]
        self.to_device()
        self.build_dataloader()
        self.build_callbacks()

        last_epoch: int = self.properties.get("epoch", -1)

        if last_epoch >= 0:
            self.load(tag="last")
            df = self.metrics.df
            autosave(df[df.epoch < last_epoch], self.path / "metrics.jsonl")

        else:
            self.build_initialization()

        self.to_device()
        self.optim.zero_grad()

        checkpoint_freq: int = self.config.get("log.checkpoint_freq", 1)
        eval_freq: int = self.config.get("train.eval_freq", 1)

        for epoch in range(last_epoch + 1, epochs):

            logger.info(f"Start epoch {epoch}")

            self._epoch = epoch
            self.run_phase("train", epoch)

            if eval_freq > 0 and (epoch % eval_freq == 0 or epoch == epochs - 1):
                self.run_phase("val", epoch)

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.checkpoint()

            self.run_callbacks("epoch", epoch=epoch)

        self.checkpoint(tag="last")

        self.run_callbacks("wrapup")


    def run_phase(self, phase, epoch):

        dl = getattr(self, f"{phase}_dl")

        grad_enabled = phase == "train"
        augmentation = (phase == "train") and ("augmentations" in self.config)

        self.model.train(grad_enabled)  # For dropout, batchnorm, &c

        meters = MeterDict()

        with torch.set_grad_enabled(grad_enabled):

            if __debug__:
                iterator = tqdm(enumerate(dl), total=len(dl), desc=phase)
            else:
                iterator = enumerate(dl)
 
            # with torch.inference_mode(not grad_enabled):
            for batch_idx, batch in iterator:
                outputs = self.run_step(
                    batch_idx,
                    batch,
                    backward=grad_enabled,
                    augmentation=augmentation,
                    epoch=epoch,
                )

                metrics = self.compute_metrics(outputs)

                meters.update(metrics)

                self.run_callbacks(
                    "batch", epoch=epoch, batch_idx=batch_idx, phase=phase
                )

        meters_collect = meters.collect("mean")

        metrics = {"phase": phase, "epoch": epoch, **meters_collect}

        wandb_metrics = {
            'epoch': epoch,
            f'{phase}_loss': meters_collect['loss'],
            f'{phase}_dice_score': meters_collect['dice_score'],
            }

        if self.config.get('wandb.track_it', False):
            logger.info(f"\n{wandb_metrics}")
            wandb.log(wandb_metrics)
    
        self.metrics.log(metrics)

        return metrics

    def run_step(self, batch_idx, batch, backward=True, augmentation=True, epoch=None):

        x, y = to_device(
            batch, self.device, self.config.get("train.channels_last", False)
        )

        yhat = self.model(x)
        loss = self.loss_func(yhat, y)

        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        return {"loss": loss, "ytrue": y, "ypred": yhat}

    def compute_metrics(self, outputs):
        metrics = {"loss": outputs["loss"].item()}
        for name, fn in self.metric_fns.items():
            value = fn(outputs["ypred"], outputs["ytrue"])
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics[name] = value
        return metrics

    def build_augmentations(self):
        pass

import collections
import getpass
import itertools
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fnmatch import fnmatch
from typing import Optional, List, Union, Dict, Tuple

import more_itertools
import numpy as np
from tqdm.auto import tqdm
from loguru import logger

import pandas as pd

# unused import because we want the patching side-effects
# on pd.DataFrames
from ..pandas import api
from ..pandas.api import augment_from_attrs
from ..pandas.convenience import (
    ensure_hashable, to_categories, concat_with_attrs, remove_redundant_columns
)
from ..util import FileCache
from ..util.config import HDict, keymap, valmap


def list2tuple(val):
    if isinstance(val, list):
        return tuple(map(list2tuple, val))
    return val


def shorthand_names(names):
    cols = []
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            cols.append("_".join(parts[-i:]))
    counts = collections.Counter(cols)
    column_renames = {}
    for c in names:
        parts = c.split(".")
        for i, _ in enumerate(parts, start=1):
            new_name = "_".join(parts[-i:])
            if counts[new_name] == 1:
                column_renames[c] = new_name
                break
    return column_renames


def shorthand_columns(df):
    column_renames = shorthand_names(df.columns)
    df.rename(columns=column_renames, inplace=True)
    return df


class ResultsLoader:
    """
    Loader utility for experimental results.

    Attributes
    ----------
    _cache : FileCache
        Cached file handler for fast repeated access.
    _num_workers : int
        Number of workers for parallel loading.
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        num_workers: int = 8
    ):
        """
        Initialize `ResultsLoader`.

        Parameters
        ----------
        cache_file : str or None, optional
            Path to the disk cache file. If None, defaults to
            "/tmp/{username}-results.diskcache".
        num_workers : int, optional
            Number of parallel workers for file loading.
        """

        # Determine the current user
        unixname = getpass.getuser()
        logger.info(
            f'ResultsLoader is loading results for the unix user: {unixname}'
        )

        # Determine the 
        if cache_file is None:
            cache_file = f"/tmp/{unixname}-results.diskcache"
            logger.debug(
                'No results cache file specified to ResultsLoader constructor.'
                f' Defaulting to cache file found at {cache_file}')

        # Initialize disk cache and worker count
        self._cache = FileCache(cache_file)
        self._num_workers = num_workers

    def load_configs(
        self,
        *paths: Union[str, pathlib.Path],
        shorthand: bool = True,
        properties: bool = False,
        metadata: bool = False,
        log: bool = False,
        callbacks: bool = False,
        categories: bool = False,
    ) -> pd.DataFrame:
        """
        Load and flatten a YAML configuration files from experiment folders.

        This method loads the YAML configuration files associated with each
        exoperimental run and flattens them so there is one entry per key.

        Parameters
        ----------
        *paths : str or pathlib.Path
            One or more base directories containing experiment run folders.
        shorthand : bool, optional
            Shorten verbose column names.
        properties : bool, optional
            Include `properties.json` contents as column.
        metadata : bool, optional
            Include `metadata.json` contents as column.
        log : bool, optional
            Keep or drop the `log` section in configs.
        callbacks : bool, optional
            Keep or drop the `callbacks` section.
        categories : bool, optional
            Convert string columns to pandas.Categorical.

        Returns
        -------
        pandas.DataFrame
            Flattened configuration records, one row per experiment.

        Examples
        --------
        >>> # Initialize the loader with the default cache file
        >>> loader = pylot.ResultsLoader()
        >>> # Specify directory to folder containing all pylot experiments
        >>> experiment_root = 'pylot_experiments'
        >>> # Load the flattened config files
        >>> df = loader.load_configs(path)
        >>> df.keys()
        Index(['epochs', 'model', 'in_channels', 'out_channels', 'ndim'])
        """

        # Deduplicate input paths
        paths = list(dict.fromkeys(paths))

        for p in paths:
            # Ensure all paths are valid
            if not isinstance(p, (str, pathlib.Path)):

                # Make error message
                error_message = (
                    f"Invalid experiment path: {p}: each entry must be a str "
                    "or pathlib.Path"
                )

                # Throw errors
                logger.error(error_message)
                raise ValueError(error_message)

        # Gather all immediate experiment subfolders
        folders = list(
            itertools.chain.from_iterable(
                pathlib.Path(path).iterdir() for path in paths
            )
        )

        # Load each config in parallel via cache
        configs = self._cache.gets(
            files=(folder / "config.yml" for folder in folders),
            num_workers=self._num_workers
        )

        rows = []
        for folder, cfg in tqdm(
            zip(folders, configs),
            leave=False,
            total=len(folders)
        ):

            if cfg is None:
                continue  # Skip missing configs

            # Convert to nested dict with flatten suppport
            cfg = HDict(cfg)

            # Optionally remove potential pollution
            if not log:
                cfg.pop("log", None)
            if not callbacks:
                cfg.pop("callbacks", None)

            # Optionally attach metadata and properties json data
            if metadata:
                cfg.update(
                    {"metadata": self._cache[folder / "metadata.json"]}
                )
            if properties:
                cfg.update(
                    {"properties": self._cache[folder / "properties.json"]}
                )

            # Flatten dict, then convert to tuple (for hashability)
            flat_cfg = valmap(list2tuple, cfg.flatten())

            # Add entries for the experimental run folders
            flat_cfg["path"] = folder

            # Clean up key names from config conventions
            flat_cfg = keymap(lambda x: x.replace("._class", ""), flat_cfg)
            flat_cfg = keymap(lambda x: x.replace("._fn", ""), flat_cfg)

            # Add to the container
            rows.append(flat_cfg)

        # Build DataFrame from list of dicts
        df = pd.DataFrame.from_records(rows)

        # Ensure hashability
        ensure_hashable(df, inplace=True)

        # Optionally shorten column names
        if shorthand:
            df = shorthand_columns(df)

        # Optionally convert to categorical for memory/analysis
        if categories:
            df = to_categories(df, inplace=True, threshold=0.5)

        return df

    def load_sub_configs(
        self,
        config_df: pd.DataFrame,
        subfolder: str = "inference",
        prefix: Optional[str] = "inf",
        path_key: str = None,
        copy_cols: List[str] = None,
    ):
        assert isinstance(config_df, pd.DataFrame)
        config_df = config_df.copy()

        if path_key is None:
            path_key = "path"
        if copy_cols is None:
            copy_cols = config_df.columns.to_list()
        elif path_key not in copy_cols:
            copy_cols += [path_key]

        subfolders = [(folder / subfolder) for folder in config_df[path_key].values]
        subfolders = [sf for sf in subfolders if sf.exists()]
        
        assert len(subfolders) > 0, f"No subfolders found for {subfolder}"
        df_sub = self.load_configs(*subfolders)

        if prefix is not None:
            df_sub.columns = [f"{prefix}_{c}" for c in df_sub.columns]
            df_sub[path_key] = [ p.parent.parent for p in df_sub[f"{prefix}_path"] ]

        df = df_sub.merge(config_df[copy_cols], on="path", how="left")

        return df

    def load_metrics(
        self,
        config_df: pd.DataFrame,
        file: Union[str, List[str]] = "metrics.jsonl",
        prefix: str = "log",
        shorthand: bool = True,
        copy_cols: Optional[List[str]] = None,
        path_key: Optional[str] = None,
        expand_attrs: bool = False,
        categories: bool = False,
    ):

        # Ensure config_df is a pandas dataframe
        if not isinstance(config_df, pd.DataFrame):

            # Make the error message
            error_message = (
                f'config_df must be a pandas dataframe. Got {type(config_df)}'
            )

            # Throw errors
            logger.error(error_message)
            raise TypeError(error_message)

        # Determine the name of the column that holds the experiment path
        path_key = path_key or 'path'

        # Decide which config columns to replicate on each metric row
        copy_cols = copy_cols or config_df.columns.to_list()

        # Extract all experiment folder paths
        folders = config_df[path_key].values

        # Normalize file argument into a list of filenames
        files = [file] if isinstance(file, str) else file
        n_files = len(files)
        logger.debug(
            f"Expecting {n_files} metric files per experiment: {files}"
        )

        # Read metric files in parallel using the cache
        all_files = self._cache.gets(
            (folder / file for folder in folders for file in files),
            num_workers=self._num_workers,
        )

        if not len(all_files) > 0:
            error_message = f"No metric files found for patterns {files}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        log_dfs = []

        # Build an iterator that repeats each config row for each file
        config_iter = more_itertools.repeat_each(config_df.iterrows(), n_files)

        # Loop over each (config_row, log_df) pair
        for (_, row), log_df in tqdm(
            zip(config_iter, all_files),
            total=len(config_df) * n_files,
            leave=False,
        ):

            # Convert row -> dict and path -> pathlib.Path
            row = row.to_dict()
            path = pathlib.Path(row[path_key])

            # Skip if this experiment has no metrics file
            if log_df is None:
                logger.warning(f"Missing metrics file in {path}")
                continue

            # Prefix all metric column names (e.g. 'acc' -> 'log_acc')
            if prefix:
                log_df.rename(
                    columns={c: f"{prefix}_{c}" for c in log_df.columns},
                    inplace=True
                )

            # Copy metadata columns from config into each row of dataframe
            if len(copy_cols) > 0:
                for col in copy_cols:
                    val = row[col]

                    # If metadata is a tuple, repeat for each row as array
                    if isinstance(val, tuple):
                        log_df[col] = np.array(
                            itertools.repeat(val, len(log_df))
                        )
                    else:
                        log_df[col] = val

            # Optionally extract and attach additional attributes from metrics
            if expand_attrs:
                log_df = augment_from_attrs(log_df, prefix=f"{prefix}_")

            # Record the experiment path in the dataframe
            log_df["path"] = path
            log_dfs.append(log_df)

        # Combine all experiment dataframe into one unified dataframe
        full_df = concat_with_attrs(log_dfs, ignore_index=True)

        # Optionally simplify column names
        if shorthand:
            renames = {}
            for c in full_df.columns:
                if c.startswith("log_"):
                    shortc = c[len("log_") :]
                    if shortc not in full_df.columns:
                        renames[c] = shortc
                    else:
                        renames[c] = c.replace(".", "__")
            full_df.rename(columns=renames, inplace=True)

        # ensure_hashable(full_df, inplace=True)

        if categories:
            to_categories(full_df, inplace=True, threshold=0.5)

        return full_df

    def load_aggregate(
        self,
        config_df: pd.DataFrame,
        metric_df: pd.DataFrame,
        agg: Optional[Dict[str, List[str]]] = None,
        metrics_groupby: Tuple[str, ...] = ("phase",),
        remove_redundant_cols: bool = False,
    ) -> pd.DataFrame:
        """
        Aggregate metric records by experiment config and optional group keys.

        Parameters
        ----------
        config_df : pd.DataFrame
            DataFrame containing experiment configurations.
        metric_df : pd.DataFrame
            DataFrame with metric records to aggregate.
        agg : dict of {str: list of str}, optional
            Additional mapping from metric columns to aggregation functions.
        metrics_groupby : tuple of str, optional
            Extra column names in metric_df to group by (in addition to all
            columns of config_df).
        remove_redundant_cols : bool, optional
            Remove all columns that have a constant value for all entries.

        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame whose columns are the config columns,
            any extra group keys, and the aggregated metrics named
            '<aggfunc>_<original_column>'.
        """

        # Validate inputs
        if not isinstance(config_df, pd.DataFrame):
            msg = f"config_df must be a DataFrame, got {type(config_df)}"
            logger.error(msg)
            raise TypeError(msg)

        if not isinstance(metric_df, pd.DataFrame):
            msg = f"metric_df must be a DataFrame, got {type(metric_df)}"
            logger.error(msg)
            raise TypeError(msg)

        # Identify columns
        config_cols = list(config_df.columns)
        metric_cols = list(set(metric_df.columns) - set(config_df.columns))

        logger.debug(f"Config columns for grouping: {config_cols}")
        logger.debug(f"Metric columns to aggregate: {metric_cols}")

        _agg_fns = collections.defaultdict(list)

        # Default patterns: max for accuracies/scores, min for losses/errors
        DEFAULT_AGGS = {
            "max": ["*acc*", "*score*", "*epoch*"],
            "min": ["*loss*", "*err*"],
        }

        # Calculate the aggregation for each default
        for agg_fn, patterns in DEFAULT_AGGS.items():

            for column in metric_cols:
                # If col name matches one of the patterns, schedule it for agg
                if any(fnmatch(column, p) for p in patterns):
                    _agg_fns[column].append(agg_fn)

        # Incorporate any user-requested extra aggregations
        if agg is not None:

            for column, agg_fns in agg.items():
                # Do not overwrite defaults; extend the list of functions
                _agg_fns[column].extend(agg)

        grouping_cols = config_cols + list(metrics_groupby)

        # TODO: Figure out why I was getting this error
        if 'log_freq' in grouping_cols:
            grouping_cols.remove('log_freq')

        # Verify that all grouping columns exist in metric_df
        missing = [c for c in grouping_cols if c not in metric_df.columns]

        if missing:
            msg = f"Missing grouping columns in metric_df: {missing}"
            logger.error(msg)
            raise KeyError(msg)

        logger.debug(
            f'Aggregating metrics_df dataframe by {grouping_cols}'
        )

        # Group the dataframe and apply aggregations
        agg_df = metric_df.groupby(
            by=grouping_cols,
            as_index=False,
            dropna=False,
            observed=True
        ).agg(
            _agg_fns
        )

        new_columns = []
        for i, (col, agg) in enumerate(agg_df.columns.values):
            if agg == "":
                new_columns.append(col)
            else:
                new_columns.append(f"{agg}_{col}")

        agg_df.columns = new_columns

        # Relocate phase to be the first column
        phase_series = agg_df.pop("phase")
        agg_df.insert(0, "phase", phase_series)

        # Derive the name of the experiment from the path and make first column
        name_series = agg_df["path"].astype(str).str.split('/').str[-1]
        agg_df.insert(0, "name", name_series)

        # Optionally remove redundant columns
        if remove_redundant_cols:
            agg_df = remove_redundant_columns(agg_df)

        return agg_df

    def load_all(
        self,
        *paths,
        shorthand=True,
        remove_redundant_cols: bool = False,
        **selector
    ):
        """
        Returns
        -------
        Tuple[pd.DataFrame]
            A tuple of the following dataframes:
            - configuration dataframe
            - metrics dataframe
            - aggregate dataframe
        """

        dfc = self.load_configs(*paths, shorthand=shorthand,).select(**selector).copy()
        df = self.load_metrics(dfc)
        dfa = self.load_aggregate(
            config_df=dfc,
            metric_df=df,
            remove_redundant_cols=remove_redundant_cols
        )
        return dfc, df, dfa

    def load_from_callable(self, config_df, load_fn, copy_cols=None, prefix="data"):

        assert isinstance(config_df, pd.DataFrame)

        if copy_cols is None:
            copy_cols = config_df.columns.to_list()

        def do_row(row):
            row = row.to_dict()
            data_df = load_fn(row["path"])
            if data_df is None:
                return pd.DataFrame()
            if prefix:
                data_df.rename(
                    columns={
                        c: f"{prefix}_{c}" for c in data_df.columns if c in copy_cols
                    },
                    inplace=True,
                )
            for col in copy_cols:
                val = row[col]
                if isinstance(val, tuple):
                    data_df[col] = np.array(itertools.repeat(val, len(data_df)))
                else:
                    data_df[col] = val
            return data_df

        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            data_dfs = list(
                tqdm(
                    executor.map(do_row, (row for _, row in config_df.iterrows())),
                    total=len(config_df),
                    leave=False,
                )
            )

        return concat_with_attrs(data_dfs, ignore_index=True)

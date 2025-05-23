
__all__ = [
    'groupby_mode_nonum',
    'groupby_and_take_best',
    'to_categories',
    'ensure_hashable',
    'broadcast_categories',
    'set_value_to_column',
    'concat_with_attrs',
    'remove_redundant_columns'
]

import itertools
import json
import operator
from functools import reduce
from typing import Any, Callable, Sequence, Union

import numpy as np

import pandas as pd
from pandas.api.types import is_categorical_dtype, union_categoricals


def groupby_mode_nonum(
    df: pd.DataFrame,
    groupby: Sequence[str],  # columns
    number_agg: Union[str, Callable],
    enforce_unique_str: bool = True,
    multiple_token: Any = "MULTIPLE",
    **groupby_kws,
):
    """
    This function's raison-de-etre is that pandas will drop str
    columns when doing a groupby and aggregating using a numerical
    function such as mean or max. This function, will try to keep
    these columns around using the mode
    """

    def str_agg(col):
        if col.nunique(dropna=False) == 1:
            return col.mode()
        if enforce_unique_str:
            raise ValueError(f"Multiple values for col {col.name} {col.unique()}")
        return multiple_token

    agg_by_type = {
        "number": number_agg,
        "object": str_agg,
        "category": str_agg,
    }

    cols_by_type = {
        type_: df.select_dtypes(type_).columns.difference(groupby)
        for type_ in agg_by_type
    }

    agg_by_col = {
        col: (col, agg_by_type[type_])
        for type_, cols in cols_by_type.items()
        for col in cols
    }

    return df.groupby(groupby, **groupby_kws).agg(**agg_by_col)


def groupby_and_take_best(
    df: pd.DataFrame, groupby: Sequence[str], metric: str, n: int
) -> pd.DataFrame:

    """
    Goal of this function is to groupby and for each subgroup keep `n` rows.
    These `n` rows are the ones with the largest value of the column `metric`
    """

    # Function to apply
    def keep_best(df_group: pd.DataFrame) -> pd.DataFrame:
        return df_group.sort_values(metric, ascending=False).head(n)

    return df.groupby(groupby, as_index=False).apply(keep_best).reset_index(drop=True)


def to_categories(df: pd.DataFrame, threshold=0.1, inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    for col in df.select_dtypes("object").columns:
        if df[col].map(pd.api.types.is_hashable).astype(bool).all():
            if df[col].nunique() / len(df) <= threshold:
                df[col] = df[col].astype("category")
    return df


def ensure_hashable(df: pd.DataFrame, inplace: bool = False):
    if not inplace:
        df = df.copy()
    for col in df.columns:
        if not df[col].map(pd.api.types.is_hashable).astype(bool).all():
            df[col] = df[col].map(json.dumps)
    return df


def broadcast_categories(dfs):
    for col in set(sum([df.columns.tolist() for df in dfs], start=[])):
        if all(is_categorical_dtype(df[col]) for df in dfs):
            all_cats = union_categoricals([df[col] for df in dfs]).categories
            for df in dfs:
                df[col] = pd.Categorical(df[col], categories=all_cats)
    return dfs


def set_value_to_column(df, col, val):
    if isinstance(val, (tuple, list, dict)):
        df[col] = np.array(itertools.repeat(val, len(df)))
    else:
        df[col] = val
    return df


def concat_with_attrs(dfs, **concat_kws):
    assert len(dfs)>0, "No dataframes to concatenate"
    all_attrs = reduce(operator.or_, [set(df.attrs) if df.attrs is not None else set() for df in dfs], set())
    sentinel = object()
    unique_attrs = {}
    for attr in all_attrs:
        vals = [df.attrs.get(attr, sentinel) for df in dfs]
        if all(v == vals[0] for v in vals):
            unique_attrs[attr] = list(vals)[0]
        else:
            for df in dfs:
                if attr in df.attrs:
                    x = df.attrs[attr]
                    if isinstance(x, (tuple, list)) and len(x)==1:
                        x = x[0]
                    df[attr] = x
    concat_df = pd.concat(dfs, **concat_kws)
    concat_df.attrs.update(unique_attrs)
    return concat_df


def remove_redundant_columns(
    dataframe: pd.DataFrame,
):
    """
    Remove all columns that have the same values for all entires
    """

    # Get a series of the unique counts indexed by column name
    unique_counts = dataframe.nunique()

    # Get all the non-constant columns
    varying = unique_counts > 1

    # True for the 'path' col
    always_keep = dataframe.columns == "path"     

    # Keep mask is or
    keep_mask = varying | always_keep
    dataframe = dataframe.loc[:, keep_mask]

    return dataframe

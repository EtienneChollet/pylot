__all__ = [
    'optional_import'
]

import importlib.util
import types
from typing import Optional

def optional_import(module_name: str) -> Optional[types.ModuleType]:
    """
    Attempt to import a module by name only if it is installed.

    Parameters
    ----------
    module_name : str
        The name of the module to import.

    Returns
    -------
    module : Optional[module]
        The imported module if available, else None.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None
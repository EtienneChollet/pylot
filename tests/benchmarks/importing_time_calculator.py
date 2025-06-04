"""
Calculates import times from a log file containing all imports

Examples
--------
>>> # Gather the import times into a log file
>>> python -X importtime -c "import pylot" > tests/benchmarks/import_time.log 2>&1

Notes
-----
As of 2025-05-06:

torch               : 8844.80 ms
scipy               : 2945.41 ms
sympy               : 2711.64 ms
wandb               : 2626.32 ms
pandas              : 1804.84 ms
kornia              : 1588.28 ms
torchvision         : 1251.38 ms
matplotlib          : 929.33 ms
IPython             : 796.19 ms
prompt_toolkit      : 781.53 ms
pylot               : 635.98 ms
sklearn             : 556.98 ms
wandb_graphql       : 555.94 ms
jedi                : 498.02 ms
botocore            : 471.99 ms
mpmath              : 435.02 ms
numpy               : 365.64 ms
triton              : 305.96 ms
git                 : 272.53 ms
joblib              : 269.09 ms
"""

# Parse into a summary of total time spent per top-level package
from collections import defaultdict

summary = defaultdict(int)

log_path = 'tests/benchmarks/import_time.log'

with open(log_path, 'r') as f:

    for line in f:
        if not line.startswith("import time:"):
            continue

        try:
            # Split at the seperator
            parts = line.split('|')


            time_us = int(parts[1].replace("import time:", "").strip())
            module = parts[2].strip().split('.')[0]  # top-level module
            summary[module] += time_us
        except Exception:
            continue

# Sort by cumulative import time
sorted_summary = sorted(summary.items(), key=lambda x: x[1], reverse=True)

# Print summary
for module, time_us in sorted_summary[:20]:
    print(f"{module:20s}: {time_us / (1000):.2f} ms")

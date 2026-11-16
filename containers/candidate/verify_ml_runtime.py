"""A CPU-only build probe; it never queries or initializes a CUDA device."""
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
import gymnasium
import stable_baselines3
import sb3_contrib
import passage_eval
from passage_eval.campaign.candidate import worker

assert torch.version.cuda == "13.0"
assert torch.__version__ == "2.13.0+cu130"
assert str(torch.ones(2, 2).matmul(torch.ones(2, 2)).device) == "cpu"
assert np.__version__ == "2.4.6"
assert sb3_contrib.__version__ == "2.9.2"
assert gymnasium.__version__ == "1.4.0"
assert Path(worker.__file__).is_relative_to("/usr/local/lib/python3.12/site-packages")
assert not list(Path("/usr/local/lib/python3.12/site-packages").glob("__editable__*passage*"))
record = {
    "schema": "passage-ml-image-cpu-build-probe/v1", "python": sys.version,
    "machine": platform.machine(), "torch_build": torch.__version__, "cuda_build": torch.version.cuda,
    "numpy": np.__version__, "sb3_contrib": sb3_contrib.__version__, "gymnasium": gymnasium.__version__,
    "stable_baselines3": stable_baselines3.__version__, "bootstrap_pip": metadata.version("pip"),
    "core_worker": str(worker.__file__), "core_version": passage_eval.__version__, "cpu_matmul_passed": True,
    "cuda_device_queried": False, "cuda_execution_tested": False,
    "distributions": {item.metadata["Name"]: item.version for item in metadata.distributions()},
}
Path("/opt/passage-image-inputs/build-probe.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps({key: value for key, value in record.items() if key != "distributions"}))

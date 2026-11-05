# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import hashlib
from pathlib import Path


def native_roles(procedure, target):
    """Select existing target classes and verify the installed frozen decoder file.

    This does not check native artifacts, runtime placement, or checkpoint lineage.
    """
    from passage_eval.baselines import final_decoders

    if procedure not in ("seg", "reg") or target not in ("thor", "orin", "tiger"):
        raise ValueError("unsupported frozen candidate or target")
    digest = hashlib.sha256(Path(final_decoders.__file__).read_bytes()).hexdigest()
    if digest != "2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4":
        raise ValueError("frozen decoder digest differs")
    family = "Seg" if procedure == "seg" else "Reg"
    return (
        f"passage_inference.final_field_predictors:Final{family}Predictor",
        f"passage_eval.baselines.final_operator_adapters:{family}FinalDecoder",
    )

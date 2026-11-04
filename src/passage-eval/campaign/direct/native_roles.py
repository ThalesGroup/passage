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

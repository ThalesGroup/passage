"""Read-only native validation setup. Does not run a worker or confer qualification."""

from passage_eval.campaign.direct.native_roles import native_roles
from passage_eval.campaign.direct.artifact_pins import verify_files
from passage_eval.campaign.edge.target import _sha256_regular_file
from passage_eval.io import pinned

CHECKPOINTS = {
    "seg": "eee055e084aaaec2aa4befb9db9654882d98fe629bb85e70d24a7bf5575eb79b",
    "reg": "dc52d13368fd911ab44ea65801089e8d65213b6b9247dfe38116086808cabb70",
}
DECODER = "2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4"
BACKENDS = {"orin": "tensorrt_orin_gpu", "thor": "tensorrt_thor_gpu", "tiger": "openvino_tiger_gpu"}


def native_config(procedure, target, bundle, compiled, *, exploratory=False):
    predictor, decoder = native_roles(procedure, target)
    if target in ("orin", "thor") and not compiled:
        raise ValueError("TensorRT requires compiled manifest")
    if target == "tiger" and compiled is not None:
        raise ValueError("OpenVINO does not use a compiled manifest")
    if type(exploratory) is not bool:
        raise ValueError("exploratory must be boolean")
    options = {"compiled_manifest": str(compiled)} if compiled is not None else {}
    if exploratory and compiled is not None:
        options["allow_unqualified_precision"] = True
    return {
        "predictor": predictor,
        "decoder": decoder,
        "predictor_kwargs": {"bundle": str(bundle), "backend": BACKENDS[target], "options": options},
    }


def load_candidate_bundle(procedure, path):
    from passage_inference.bundle import load_bundle, load_raw_path_logit_bundle

    if procedure == "seg":
        return load_raw_path_logit_bundle(path)
    if procedure == "reg":
        return load_bundle(path)
    raise ValueError("unsupported candidate")


def validate_native_files(
    procedure,
    target,
    paths,
    expected,
    compiled_manifest=None,
    compiled_sha256=None,
    *,
    exploratory=False,
    precision_profile="strict",
    grn_evidence=None,
):
    """Validate bundle and engine lineage with installed SDK metadata, without starting a GPU.

    Requires real passage_inference installation on the host executing this check. File hashes are
    not proof of worker loading; callers must still isolate, pin and verify worker runtime identity.
    """
    if precision_profile not in {"strict", "native_descriptive", "native_observed"}:
        raise ValueError("unknown native precision profile")
    if exploratory and precision_profile != "strict":
        raise ValueError("exploratory and descriptive precision modes are distinct")
    config = native_config(procedure, target, paths["bundle"], compiled_manifest, exploratory=exploratory)
    if (
        expected.get("decoder") != DECODER
        or not pinned.is_hex64(expected.get("checkpoint"))
        or (precision_profile == "strict" and expected.get("checkpoint") != CHECKPOINTS[procedure])
    ):
        raise ValueError("frozen candidate checkpoint or decoder pin differs")
    verified = verify_files(paths, expected)
    bundle = load_candidate_bundle(procedure, paths["bundle"])
    if (
        bundle.manifest_sha256 != verified["bundle"]
        or bundle.manifest["lineage"]["checkpoint_sha256"] != verified["checkpoint"]
    ):
        raise ValueError("loaded bundle or checkpoint lineage differs")
    compiled = None
    grn = None
    if target in ("thor", "orin"):
        if not compiled_sha256 or _sha256_regular_file(compiled_manifest) != compiled_sha256:
            raise ValueError("compiled manifest digest differs")
        from passage_inference.engine import checked_engine

        engine_path, compiled = checked_engine(bundle, compiled_manifest, BACKENDS[target])
        if _sha256_regular_file(engine_path) != verified["engine"]:
            raise ValueError("compiled engine differs from expected engine")
        if procedure == "seg":
            from passage_eval.campaign.direct.grn_lineage import verify_grn_lineage

            grn = verify_grn_lineage(bundle, compiled, target, grn_evidence)
        else:
            grn = None
        if (
            not exploratory
            and precision_profile == "strict"
            and compiled.get("precision") != "fp32"
            and compiled.get("precision_admission", {}).get("status") != "admitted"
        ):
            raise ValueError("compiled FP16 artifact has no admitted precision evidence")
    elif _sha256_regular_file(bundle.model_path) != verified["engine"]:
        raise ValueError("OpenVINO model differs from bundle model")
    if verify_files(paths, expected) != verified:
        raise ValueError("files changed during preflight")
    return {
        "config": config,
        "file_hashes": verified,
        "compiled": compiled,
        "grn_lineage": grn,
        "precision_profile": precision_profile,
        "precision_note": "descriptive engine precision is not proof of all fused operations"
        if precision_profile in {"native_descriptive", "native_observed"}
        else None,
        "note": "preflight only: no runtime execution or worker loading attested",
    }

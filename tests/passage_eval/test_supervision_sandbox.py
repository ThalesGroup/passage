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

"""Environment scrubbing, bind safety and the honesty of the recorded isolation level."""

from pathlib import Path

import pytest

from passage_eval.supervision import sandbox


SOURCE = {
    "PATH": "/usr/bin",
    "LANG": "C.UTF-8",
    "LC_TIME": "C",
    "HF_TOKEN": "hf_dummy",
    "HUGGING_FACE_HUB_TOKEN": "x",
    "AWS_SECRET_ACCESS_KEY": "y",
    "GITHUB_TOKEN": "z",
    "PYTHONPATH": "/evil",
    "LD_PRELOAD": "/evil.so",
    "OMP_NUM_THREADS": "4",
    "CUDA_VISIBLE_DEVICES": "0",
    "HOME": "/real/home",
    "SSH_AUTH_SOCK": "/s",
    "WANDB_API_KEY": "w",
    "MLFLOW_TRACKING_PASSWORD": "m",
    "USER": "someone",
    "TMPDIR": "/real/tmp",
}


def test_environment_is_an_allow_list():
    env = sandbox.scrubbed_env(SOURCE, home="/h", tmpdir="/t")
    assert set(env) == {
        "PATH",
        "LANG",
        "LC_TIME",
        "OMP_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "TMPDIR",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
    }
    assert env["HOME"] == "/h" and env["TMPDIR"] == "/t" and env["OMP_NUM_THREADS"] == "4"
    assert "hf_dummy" not in env.values()


def test_path_has_a_safe_default_and_nothing_is_inherited_implicitly():
    env = sandbox.scrubbed_env({}, home="/h", tmpdir="/t")
    assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin" and "HF_TOKEN" not in env


def test_explicit_extras_and_passthrough():
    env = sandbox.scrubbed_env(
        SOURCE, home="/h", tmpdir="/t", extras={"LD_LIBRARY_PATH": "/opt/cuda/lib"}, passthrough=["USER", "ABSENT"]
    )
    assert env["LD_LIBRARY_PATH"] == "/opt/cuda/lib" and env["USER"] == "someone" and "ABSENT" not in env


@pytest.mark.parametrize(
    "name",
    [
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "MY_API_KEY",
        "DB_PASSWORD",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "LD_PRELOAD",
        "WANDB_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ],
)
def test_credential_like_or_interpreter_control_extras_are_refused(name):
    with pytest.raises(ValueError, match="refusing"):
        sandbox.scrubbed_env(SOURCE, home="/h", tmpdir="/t", extras={name: "v"})
    with pytest.raises(ValueError, match="refusing"):
        sandbox.scrubbed_env(SOURCE, home="/h", tmpdir="/t", passthrough=[name])


@pytest.mark.parametrize("name", ["1BAD", "has space", "a=b", ""])
def test_invalid_variable_names_are_refused(name):
    with pytest.raises(ValueError, match="invalid environment variable name"):
        sandbox.scrubbed_env(SOURCE, home="/h", tmpdir="/t", extras={name: "v"})


def test_rlimit_configuration_is_validated_and_core_dumps_stay_off():
    assert sandbox.RlimitConfig().worker_args()[:4] == ["--rlimit-core", "0", "--rlimit-nofile", "1024"]
    args = sandbox.RlimitConfig(address_space_bytes=1 << 30, cpu_seconds=60).worker_args()
    assert "--rlimit-as" in args and "--rlimit-cpu" in args
    for bad in (dict(core=1), dict(nofile=0), dict(cpu_seconds=-1), dict(address_space_bytes=True)):
        with pytest.raises(ValueError):
            sandbox.RlimitConfig(**bad)


def test_isolation_config_validation():
    with pytest.raises(ValueError, match="level"):
        sandbox.IsolationConfig(level="docker")
    with pytest.raises(ValueError, match="bwrap"):
        sandbox.IsolationConfig(level="process", dev_binds=("/dev/nvidia0",))


def test_binds_must_exist_and_must_not_overlap_protected_paths(tmp_path):
    cand = tmp_path / "cand"
    cand.mkdir()
    dataset = tmp_path / "data"
    (dataset / "3x3").mkdir(parents=True)
    assert sandbox.check_binds([cand], [dataset]) == [cand.resolve()]
    with pytest.raises(sandbox.SandboxError, match="does not exist"):
        sandbox.check_binds([tmp_path / "missing"], [])
    with pytest.raises(sandbox.SandboxError, match="overlaps"):
        sandbox.check_binds([tmp_path], [dataset])  # parent would expose the dataset
    with pytest.raises(sandbox.SandboxError, match="overlaps"):
        sandbox.check_binds([dataset / "3x3"], [dataset])  # inside a protected path
    with pytest.raises(sandbox.SandboxError, match="overlaps"):
        sandbox.check_binds([dataset], [dataset])


def test_protected_paths_are_enforced_even_before_they_exist(tmp_path):
    output = tmp_path / "run-output"  # created only after the binds are checked
    with pytest.raises(sandbox.SandboxError, match="overlaps"):
        sandbox.check_binds([tmp_path], [output])
    with pytest.raises(sandbox.SandboxError, match="overlaps"):
        sandbox.check_binds([Path("/")], sandbox.default_protected_paths())


@pytest.mark.parametrize("device", ["/etc/passwd", "dev/null", "/dev/../etc/shadow", "/devices/x"])
def test_device_binds_must_be_below_dev(device):
    with pytest.raises(ValueError, match="below /dev"):
        sandbox.IsolationConfig(level="bwrap", dev_binds=(device,))


def test_default_protected_paths_cover_credential_locations():
    protected = sandbox.default_protected_paths(["/data/references"])
    home = Path.home()
    for entry in (".ssh", ".aws", ".config", ".cache/huggingface", ".netrc"):
        assert str(home / entry) in protected
    assert "/data/references" in protected


def test_bwrap_command_binds_only_what_is_declared(tmp_path):
    cand, ckpt = tmp_path / "cand", tmp_path / "model.pt"
    cand.mkdir()
    ckpt.write_bytes(b"x")
    config = sandbox.IsolationConfig(level="bwrap", dev_binds=("/dev/null",))
    argv = sandbox.bwrap_command(config, [Path("/usr"), cand, ckpt], ["/venv/bin/python", "-I", "-m", "w"])
    assert argv[argv.index("--") + 1 :] == ["/venv/bin/python", "-I", "-m", "w"]
    for flag in ("--unshare-all", "--die-with-parent"):
        assert flag in argv
    pairs = [(argv[i + 1], argv[i + 2]) for i, token in enumerate(argv) if token == "--ro-bind"]
    sources = {source for source, _ in pairs}
    assert sources <= {"/usr", "/etc/ld.so.cache", str(cand), str(ckpt)}
    assert ("/", "/") not in pairs and all(source == target for source, target in pairs)
    assert [argv[i + 1] for i, token in enumerate(argv) if token == "--dev-bind"] == ["/dev/null"]
    assert argv.count("--tmpfs") == 2 and "/tmp" in argv and sandbox.SANDBOX_HOME in argv
    assert "--share-net" not in argv and "--bind" not in argv


def test_isolation_record_never_claims_more_than_the_level_provides(tmp_path):
    process = sandbox.isolation_record(sandbox.IsolationConfig(level="process"), [], bwrap_version=None)
    assert process["level"] == "process" and process["same_uid_as_supervisor"] is True
    assert process["namespaces"] == [] and process["network"] == "shared_with_host"
    assert {"filesystem_isolation", "network_isolation", "pid_isolation", "gpu_memory_cap"} <= set(
        process["not_applied"]
    )
    boxed = sandbox.isolation_record(
        sandbox.IsolationConfig(level="bwrap"), [tmp_path], bwrap_version="bubblewrap 0.9.0"
    )
    assert boxed["network"] == "unshared_loopback_only" and boxed["read_only_binds"] == [str(tmp_path)]
    assert "interpreter_aliases" not in boxed
    assert {
        "gpu_memory_cap",
        "seccomp_filter",
        "cgroup_cpu_memory_limits",
        "protection_against_kernel_exploits",
    } <= set(boxed["not_applied"])


def test_missing_bwrap_is_an_error_not_a_downgrade():
    with pytest.raises(sandbox.SandboxError, match="refusing to fall back"):
        sandbox.bwrap_version("definitely-not-a-real-bwrap-binary")
    assert sandbox.available_bwrap("definitely-not-a-real-bwrap-binary") is False

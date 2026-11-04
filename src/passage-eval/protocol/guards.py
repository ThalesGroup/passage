"""Refuse the test split unless a verified test admission allows it.

These guards run before any scene of the split is loaded. They decide from declarations (a split name, a roster's
dataset block and source file paths) and from a test admission grant, so they never need to open the data they
protect.

The test split is accepted only with **both** ``allow_test=True`` (the literal ``True``) **and** an
``admission.AdmissionGrant``: proof that a ``passage-test-admission/v1`` record checked all six gates ``GREEN``
(``passage_eval.protocol.admission``). The grant re-hashes its record and every evidence file it pinned before each
use, so an edit after the check is refused. A roster guard with ``require_admitted_roster`` also requires the roster
to be the sealed admissible roster of the record, or a child derived from it that keeps its blocks.

A command lifts its refusal only with an explicit ``--allow-test-split`` flag (``--allow-test-warmup`` for the
supervisor's warm-up source) and a ``--test-admission-record`` (with an optional ``--test-admission-evidence-dir``)
that checks all ``GREEN``: ``test_split_allowed`` returns the grant, or raises ``TestSplitRefused`` listing the RED
gates.

Preparation. Gates G4 and G5 need artefacts that only reading the test split produces (the source roster, its
block census, the reference inventory and the warm-up roster built against the admissible roster). A
``admission.PreparationGrant`` (``--test-preparation-approval``: an explicit, time-boxed author approval bound to the
record's SHA-256) lifts the refusal for exactly these readers, ``PREPARATION_READERS``, when the record's only RED
reasons are those artefacts. A guarded call names its reader (``reader=``); a call without a reader, or with any
other reader (a run, the loading of a candidate, the scorer, the campaign tools), refuses a preparation grant: they
still need an all-GREEN admission.

Entry points that call these guards before reading any scene: the reference admission command and its library
functions ``admission_policy`` and ``build_reference_inventory``; ``protocol.warmup`` (``select_warmup_scenes``, the
pool of ``build_warmup_roster``; warm-up inputs come from validation only, so the test split is refused there even
with a grant) and ``scripts/build_warmup_roster.py``; ``scripts/build_public_test_roster.py`` (``build_roster`` and
the command); ``score-roster`` (``score_roster_run`` and the command; admitted roster required); the supervisor's
``run_supervised`` (the evaluated ``--roster`` and the spec's split; admitted roster required, and a spec run without
roster is refused on test) and its warm-up sources (``WarmupInput``); and the campaign tools of
``passage_eval.campaign`` (plan, partition, runner and aggregator). ``stage-score``, ``passage-eval score``,
``passage-eval bench`` and ``compare`` are not guarded.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

TEST_SPLIT = "test"
ADMISSION_SCHEMA = "passage-test-admission/v1"
#: The only readers a test preparation grant can authorise; none of them loads a candidate, runs or scores.
PREPARATION_READERS = ("build_roster", "block_census", "reference_inventory", "warmup_roster")


class TestSplitRefused(PermissionError):
    """The test split was requested without an admission that allows it."""

    __test__ = False  # not a pytest test class


def _canonical_split(split: Any) -> str:
    if not isinstance(split, str) or not split.strip():
        raise TestSplitRefused("a split name is required before any data is read")
    from passage.dataset import SPLITS

    name = split.strip().lower()
    return SPLITS.get(name, name)


def _check_preparation(admission: Any, context: str, reader: str | None) -> Any:
    """Accept a preparation grant only for a named preparation reader that the approval lists."""
    if reader is None or reader not in PREPARATION_READERS:
        raise TestSplitRefused(
            f"{context} refuses the test split under a preparation grant: it authorises only "
            f"{list(PREPARATION_READERS)}; a run, a candidate or scoring needs an all-GREEN test admission"
        )
    if not admission.is_genuine():
        raise TestSplitRefused(f"{context} refuses the test split: the preparation grant is not genuine")
    if reader not in admission.readers:
        raise TestSplitRefused(f"{context} refuses the test split: the preparation approval does not name {reader!r}")
    if admission.split != TEST_SPLIT:
        raise TestSplitRefused(f"{context} refuses the test split: the preparation admits {admission.split!r}")
    try:
        admission.ensure_unchanged()
    except (ValueError, OSError) as exc:
        raise TestSplitRefused(f"{context} refuses the test split: {exc}") from exc
    return admission


def _check_grant(allow_test: Any, admission: Any, context: str, reader: str | None = None) -> Any:
    """Raise ``TestSplitRefused`` unless ``allow_test`` is ``True`` and ``admission`` is a valid, unchanged grant.

    A preparation grant is valid only for a ``reader`` of ``PREPARATION_READERS`` that its approval names.
    """
    if allow_test is not True:
        raise TestSplitRefused(
            f"{context} refuses the test split: it needs allow_test and a verified test admission ({ADMISSION_SCHEMA})"
        )
    from passage_eval.protocol.admission import AdmissionGrant, PreparationGrant

    if isinstance(admission, PreparationGrant):
        return _check_preparation(admission, context, reader)
    if not isinstance(admission, AdmissionGrant) or not admission.is_genuine():
        raise TestSplitRefused(
            f"{context} refuses the test split: allow_test needs a verified test admission grant ({ADMISSION_SCHEMA}) "
            "whose six gates are GREEN"
        )
    if admission.split != TEST_SPLIT:
        raise TestSplitRefused(f"{context} refuses the test split: the admission admits {admission.split!r}")
    try:
        admission.ensure_unchanged()
    except (ValueError, OSError) as exc:
        raise TestSplitRefused(f"{context} refuses the test split: {exc}") from exc
    return admission


def refuse_test_split(
    split: Any,
    *,
    allow_test: bool = False,
    admission: Any = None,
    context: str = "this tool",
    reader: str | None = None,
) -> str:
    """Return the canonical split name; raise ``TestSplitRefused`` for test unless it is admitted.

    The test split needs ``allow_test`` (the literal ``True``: a truthy string or number does not count) and an
    ``AdmissionGrant`` from ``admission.grant_test_admission`` (or ``test_split_allowed``), or, for a ``reader`` of
    ``PREPARATION_READERS``, a ``PreparationGrant`` whose approval names that reader.
    """
    canonical = _canonical_split(split)
    if canonical == TEST_SPLIT:
        _check_grant(allow_test, admission, context, reader)
    return canonical


def refuse_test_roster(
    roster: Mapping[str, Any],
    *,
    allow_test: bool = False,
    admission: Any = None,
    context: str = "this tool",
    require_admitted_roster: bool = False,
    reader: str | None = None,
) -> str:
    """Apply ``refuse_test_split`` to a roster's declared split and to every source file path.

    A roster that declares another split but lists a file under a ``test`` directory is refused too, as is a
    file outside ``<config>/<split>/``. With ``require_admitted_roster`` a test roster must also be the admission's
    sealed admissible roster or a child derived from it with the same blocks (a preparation grant admits none).
    ``reader`` names a preparation reader (see ``refuse_test_split``). Returns the canonical declared split.
    """
    dataset = roster.get("dataset") if isinstance(roster, Mapping) else None
    if not isinstance(dataset, Mapping):
        raise TestSplitRefused(f"{context} needs the roster's dataset block before any data is read")
    split = refuse_test_split(
        dataset.get("split"), allow_test=allow_test, admission=admission, context=context, reader=reader
    )
    config = dataset.get("config")
    files = roster.get("source_files")
    if not isinstance(files, list) or not files:
        raise TestSplitRefused(f"{context} needs the roster's source files before any data is read")
    admitted = split == TEST_SPLIT
    for record in files:
        path = record.get("path") if isinstance(record, Mapping) else None
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if len(parts) < 3 or parts[0] != config or parts[1] != dataset.get("split"):
            raise TestSplitRefused(
                f"{context} refuses a source file outside {config}/{dataset.get('split')}/: {path!r}"
            )
        if not admitted and any(part.lower() == TEST_SPLIT for part in parts):
            raise TestSplitRefused(f"{context} refuses a source file under a test directory: {path!r}")
    if admitted and require_admitted_roster and not admission.admits_roster(roster):
        raise TestSplitRefused(
            f"{context} refuses this test roster: it is not the admission's sealed admissible roster "
            "nor a child derived from it with the same blocks"
        )
    from passage_eval.protocol.admission import PreparationGrant

    if admitted and isinstance(admission, PreparationGrant):
        admission.require_roster_scope_from_roster(roster)
    return split


def test_admission_allows(record_path: str | Path | None, evidence_dir: str | Path | None = None) -> bool:
    """Whether an admission record allows the test split: all six gates of ``admission.check_admission`` GREEN.

    Returns ``False`` without a record. Never raises for a RED record.
    """
    from passage_eval.protocol.admission import admission_allows

    return admission_allows(record_path, evidence_dir)


test_admission_allows.__test__ = False  # not a pytest test function

ALLOW_FLAG = "--allow-test-split"
RECORD_FLAG = "--test-admission-record"
EVIDENCE_FLAG = "--test-admission-evidence-dir"
PREPARATION_FLAG = "--test-preparation-approval"


def test_split_allowed(
    allow_flag: Any,
    record_path: str | Path | None,
    *,
    evidence_dir: str | Path | None = None,
    flag_name: str = ALLOW_FLAG,
    preparation_approval: str | Path | None = None,
):
    """The admission grant of a command that may read the test split, or ``None``.

    Neither the flag nor a record: ``None`` (the guards refuse the test split). One without the other:
    ``TestSplitRefused``. Both: the record is checked (``admission.grant_test_admission``) and the grant returned
    when all six gates are GREEN; otherwise ``TestSplitRefused`` lists the RED gates. With ``preparation_approval``
    (only the preparation readers' commands offer it) the result is a ``PreparationGrant``
    (``admission.grant_test_preparation``) or ``TestSplitRefused``. ``flag_name`` is the command's name for the
    flag, used in the messages.
    """
    if allow_flag is not True and allow_flag is not False:
        raise TestSplitRefused(f"{flag_name} must be a boolean flag")
    if not allow_flag:
        if record_path is not None or evidence_dir is not None or preparation_approval is not None:
            raise TestSplitRefused(f"a test admission record is used only together with {flag_name}")
        return None
    if record_path is None:
        raise TestSplitRefused(f"{flag_name} needs a verified test admission record ({RECORD_FLAG})")
    if preparation_approval is not None:
        from passage_eval.protocol.admission import grant_test_preparation

        return grant_test_preparation(record_path, preparation_approval, evidence_dir, refuse=TestSplitRefused)
    from passage_eval.protocol.admission import grant_test_admission

    return grant_test_admission(record_path, evidence_dir, refuse=TestSplitRefused)


test_split_allowed.__test__ = False  # not a pytest test function


def add_test_admission_arguments(
    parser: Any,
    *,
    what: str,
    flag_name: str = ALLOW_FLAG,
    extra_flags: tuple[tuple[str, str], ...] = (),
    preparation: bool = False,
) -> None:
    """Add the allow flag(s), ``--test-admission-record`` and ``--test-admission-evidence-dir`` to a parser.

    ``extra_flags`` adds further ``(flag, what)`` allow flags that share the same record (the supervisor's run
    command has one for the evaluated roster and one for the warm-up source). ``preparation`` (the preparation
    readers' commands only) adds ``--test-preparation-approval``.
    """
    if preparation:
        parser.add_argument(
            PREPARATION_FLAG,
            type=Path,
            metavar="FILE",
            help=(
                "passage-test-preparation-approval/v1 of the author: with the record, allows this preparation "
                "reader on the test split while the record's only RED reasons are the G4/G5 artefacts"
            ),
        )
    for flag, purpose in ((flag_name, what), *extra_flags):
        parser.add_argument(
            flag,
            action="store_true",
            help=f"allow {purpose} on the test split; needs {RECORD_FLAG} whose six admission gates are GREEN",
        )
    parser.add_argument(
        RECORD_FLAG,
        type=Path,
        metavar="FILE",
        help=f"{ADMISSION_SCHEMA} record; checked with `passage-eval admission-check` rules before any test data",
    )
    parser.add_argument(
        EVIDENCE_FLAG,
        type=Path,
        metavar="DIR",
        help="directory of the record's evidence files (default: the record's directory)",
    )

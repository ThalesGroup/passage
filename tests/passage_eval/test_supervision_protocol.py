"""passage-ipc/v1 framing: round trips and fail-closed rejection of malformed frames."""

import io
import json
import struct

import numpy as np
import pytest

from passage_eval.supervision import IPC_PROTOCOL
from passage_eval.supervision.protocol import (
    FrameLimits,
    MAGIC,
    PREFIX_SIZE,
    ProtocolError,
    StreamClosed,
    encode_frame,
    expect_body,
    expect_type,
    read_frame,
)

PREFIX = struct.Struct("<4sIQ")


def reader(data: bytes, requests=None):
    stream = io.BytesIO(data)

    def read_exact(count):
        if requests is not None:
            requests.append(count)
        chunk = stream.read(count)
        if len(chunk) < count:
            raise StreamClosed(len(chunk))
        return chunk

    return read_exact


def raw_frame(header, payload=b""):
    blob = header if isinstance(header, bytes) else json.dumps(header).encode()
    return PREFIX.pack(MAGIC, len(blob), len(payload)) + blob + payload


def good_header(**changes):
    header = {"proto": IPC_PROTOCOL, "type": "plan", "seq": 1, "body": {}, "buffers": []}
    header.update(changes)
    return header


@pytest.mark.parametrize(
    "array",
    [
        np.arange(12, dtype="<f4").reshape(3, 4),
        np.arange(6, dtype="<i8"),
        np.array([True, False, True]),
        np.arange(5, dtype="u1"),
        np.arange(4, dtype="<f2"),
        np.arange(8, dtype="<f8").reshape(2, 2, 2),
        np.empty((0, 2), dtype="<f8"),
        np.float32(3.5),
        np.arange(24, dtype="<i4").reshape(2, 3, 2, 2),
    ],
)
def test_round_trip_preserves_dtype_shape_and_values(array):
    data = encode_frame("result", 7, {"a": 1, "nested": {"b": [1.5, None, "x"]}}, {"x": array})
    frame = read_frame(reader(data))
    assert (frame.type, frame.seq, frame.wire_bytes) == ("result", 7, len(data))
    assert frame.body == {"a": 1, "nested": {"b": [1.5, None, "x"]}}
    assert frame.arrays["x"].dtype == np.asarray(array).dtype and frame.arrays["x"].shape == np.shape(array)
    assert np.array_equal(frame.arrays["x"], array)
    assert frame.arrays["x"].flags.writeable and frame.arrays["x"].flags.owndata


def test_big_endian_input_is_sent_little_endian():
    frame = read_frame(reader(encode_frame("plan", 0, {}, {"x": np.arange(4, dtype=">f4")})))
    assert frame.arrays["x"].dtype.str == "<f4" and frame.arrays["x"].tolist() == [0, 1, 2, 3]


def test_several_buffers_and_sequential_frames():
    data = encode_frame("plan", 1, {}, {"a": np.ones(3), "b": np.zeros((2, 2), dtype="u1")})
    data += encode_frame("shutdown", 2, {})
    read = reader(data)
    first, second = read_frame(read), read_frame(read)
    assert set(first.arrays) == {"a", "b"} and second.type == "shutdown" and second.arrays == {}
    assert read_frame(read, allow_eof=True) is None


def test_callable_body_is_evaluated_after_buffers_exist():
    seen = []
    data = encode_frame("plan", 1, lambda: seen.append("body") or {"k": 1}, {"x": np.ones(2)})
    assert seen == ["body"] and read_frame(reader(data)).body == {"k": 1}


def test_clean_eof_only_before_first_byte():
    assert read_frame(reader(b""), allow_eof=True) is None
    with pytest.raises(ProtocolError, match="truncated"):
        read_frame(reader(b""))
    with pytest.raises(ProtocolError, match="truncated"):
        read_frame(reader(b"PIP"), allow_eof=True)


def test_truncation_at_every_boundary_is_rejected():
    data = encode_frame("plan", 1, {"a": 1}, {"x": np.arange(10, dtype="<f4")})
    for cut in (1, PREFIX_SIZE - 1, PREFIX_SIZE, PREFIX_SIZE + 5, len(data) - 1):
        with pytest.raises(ProtocolError):
            read_frame(reader(data[:cut]))


def test_bad_magic_and_length_prefixes():
    good = encode_frame("plan", 1, {})
    with pytest.raises(ProtocolError, match="magic"):
        read_frame(reader(b"XXXX" + good[4:]))
    with pytest.raises(ProtocolError, match="header length"):
        read_frame(reader(PREFIX.pack(MAGIC, 0, 0)))
    with pytest.raises(ProtocolError, match="header length"):
        read_frame(reader(PREFIX.pack(MAGIC, 10**9, 0)))


def test_oversized_lengths_are_rejected_before_any_allocation():
    requests = []
    with pytest.raises(ProtocolError, match="payload length"):
        read_frame(reader(PREFIX.pack(MAGIC, 100, 1 << 60), requests))
    assert requests == [PREFIX_SIZE]  # nothing beyond the prefix was ever requested
    limits = FrameLimits(max_header_bytes=64, max_payload_bytes=16)
    with pytest.raises(ProtocolError, match="header length"):
        read_frame(reader(encode_frame("plan", 1, {"padding": "x" * 100})), limits)
    roomy = FrameLimits(max_header_bytes=4096, max_payload_bytes=16)
    with pytest.raises(ProtocolError, match="payload length"):
        read_frame(reader(encode_frame("plan", 1, {}, {"x": np.zeros(32, dtype="u1")})), roomy)


def test_encode_enforces_the_same_limits():
    limits = FrameLimits(max_header_bytes=64, max_payload_bytes=16)
    with pytest.raises(ValueError, match="size limits"):
        encode_frame("plan", 1, {}, {"x": np.zeros(32, dtype="u1")}, limits=limits)
    with pytest.raises(ValueError, match="size limits"):
        encode_frame("plan", 1, {"padding": "x" * 100}, limits=limits)
    with pytest.raises(ValueError, match="too many buffers"):
        encode_frame("plan", 1, {}, {f"b{i}": np.zeros(1) for i in range(17)})
    with pytest.raises(ValueError, match="dimensions"):
        encode_frame("plan", 1, {}, {"x": np.zeros((1,) * 5)})


@pytest.mark.parametrize(
    "header",
    [
        b"not json",
        b"\xff\xfe",
        b"[]",
        b'{"proto":"passage-ipc/v1","type":"plan","seq":1,"body":NaN,"buffers":[]}',
        b'{"proto":"passage-ipc/v1","type":"plan","seq":1,"body":{"a":Infinity},"buffers":[]}',
        b'{"proto":"passage-ipc/v1","type":"plan","seq":1,"body":{"a":1e999},"buffers":[]}',
        b'{"proto":"passage-ipc/v1","type":"plan","seq":1,"seq":2,"body":{},"buffers":[]}',
        b"[" * 5000 + b"]" * 5000,
    ],
)
def test_invalid_json_headers_are_fatal(header):
    with pytest.raises(ProtocolError):
        read_frame(reader(raw_frame(header)))


@pytest.mark.parametrize(
    "changes",
    [
        {"proto": "passage-ipc/v2"},
        {"type": "exec"},
        {"type": 3},
        {"seq": -1},
        {"seq": True},
        {"seq": 1.5},
        {"seq": 2**60},
        {"body": []},
        {"body": "x"},
        {"buffers": {}},
        {"extra": 1},
    ],
)
def test_header_field_validation(changes):
    header = good_header(**changes)
    with pytest.raises(ProtocolError):
        read_frame(reader(raw_frame(header)))


def test_missing_header_key_is_rejected():
    header = good_header()
    del header["buffers"]
    with pytest.raises(ProtocolError, match="exactly"):
        read_frame(reader(raw_frame(header)))


def descriptor(**changes):
    item = {"name": "x", "dtype": "<f4", "shape": [2], "nbytes": 8}
    item.update(changes)
    return item


@pytest.mark.parametrize(
    "item,payload",
    [
        (descriptor(dtype="<c8"), bytes(8)),
        (descriptor(dtype=">f4"), bytes(8)),
        (descriptor(dtype="O"), bytes(8)),
        (descriptor(dtype="<U4", nbytes=32, shape=[2]), bytes(32)),
        (descriptor(dtype="|S1", shape=[8]), bytes(8)),
        (descriptor(dtype="<M8[s]"), bytes(8)),
        (descriptor(dtype="not-a-dtype"), bytes(8)),
        (descriptor(shape=[3]), bytes(8)),
        (descriptor(nbytes=12), bytes(12)),
        (descriptor(shape=[-1]), bytes(8)),
        (descriptor(shape=[True]), bytes(8)),
        (descriptor(shape=[1.0]), bytes(8)),
        (descriptor(shape=[1] * 5), bytes(8)),
        (descriptor(nbytes=-8), bytes(8)),
        (descriptor(nbytes=True), bytes(8)),
        (descriptor(name="X"), bytes(8)),
        (descriptor(name="../x"), bytes(8)),
        (descriptor(name=1), bytes(8)),
        ({"name": "x", "dtype": "<f4"}, bytes(8)),
        (dict(descriptor(), extra=1), bytes(8)),
    ],
)
def test_buffer_descriptors_are_validated(item, payload):
    with pytest.raises(ProtocolError):
        read_frame(reader(raw_frame(good_header(buffers=[item]), payload)))


def test_payload_must_match_the_buffer_table_exactly():
    with pytest.raises(ProtocolError, match="exceed"):
        read_frame(reader(raw_frame(good_header(buffers=[descriptor()]), bytes(4))))
    with pytest.raises(ProtocolError, match="payload length"):
        read_frame(reader(raw_frame(good_header(buffers=[descriptor()]), bytes(16))))
    with pytest.raises(ProtocolError, match="duplicate"):
        read_frame(reader(raw_frame(good_header(buffers=[descriptor(), descriptor()]), bytes(16))))
    with pytest.raises(ProtocolError, match="buffer table"):
        read_frame(reader(raw_frame(good_header(buffers=[descriptor(name=f"b{i}") for i in range(17)]), bytes(8 * 17))))


@pytest.mark.parametrize(
    "array",
    [
        np.array(["a", "b"]),
        np.array([1 + 2j]),
        np.array([None, 1], dtype=object),
        np.array(["2020-01-01"], dtype="M8[D]"),
        np.array([b"ab"]),
    ],
)
def test_encode_rejects_non_numeric_dtypes(array):
    with pytest.raises(ValueError, match="dtype"):
        encode_frame("plan", 1, {}, {"x": array})


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(msg_type="exec", seq=1, body={}),
        dict(msg_type="plan", seq=-1, body={}),
        dict(msg_type="plan", seq=True, body={}),
        dict(msg_type="plan", seq=1, body={"a": float("nan")}),
        dict(msg_type="plan", seq=1, body={"a": {1, 2}}),
        dict(msg_type="plan", seq=1, body=[1]),
    ],
)
def test_encode_rejects_invalid_messages(kwargs):
    with pytest.raises((ValueError, TypeError)):
        encode_frame(kwargs["msg_type"], kwargs["seq"], kwargs["body"])


def test_encode_rejects_invalid_buffer_names():
    for name in ("X", "1a", "a-b", "", "a" * 40):
        with pytest.raises(ValueError, match="buffer name"):
            encode_frame("plan", 1, {}, {name: np.zeros(1)})


def test_direction_and_body_shape_checks():
    frame = read_frame(reader(encode_frame("plan", 1, {"a": 1})))
    assert expect_type(frame, {"plan"}) is frame
    with pytest.raises(ProtocolError, match="unexpected message type"):
        expect_type(frame, {"result", "error"})
    assert expect_body({"a": 1}, {"a"}, {"b"}) == {"a": 1}
    for body in ({}, {"a": 1, "c": 2}, {"b": 1}):
        with pytest.raises(ProtocolError):
            expect_body(body, {"a"}, {"b"})


def test_limits_validate_their_arguments():
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            FrameLimits(max_header_bytes=bad)


def test_random_corruption_only_ever_raises_protocol_errors():
    rng = np.random.default_rng(20261101)
    limits = FrameLimits(max_header_bytes=4096, max_payload_bytes=1 << 16)
    seed = encode_frame(
        "result",
        9,
        {"k": [1, 2, {"z": None}], "s": "text"},
        {"path": np.arange(12, dtype="<i8").reshape(6, 2), "mask": np.array([True, False])},
        limits=limits,
    )
    accepted = rejected = 0
    for _ in range(3000):
        data = bytearray(seed)
        for _ in range(int(rng.integers(1, 6))):
            data[int(rng.integers(0, len(data)))] = int(rng.integers(0, 256))
        if rng.random() < 0.3:
            data = data[: int(rng.integers(0, len(data) + 1))]
        if rng.random() < 0.1:
            data += bytes(rng.integers(0, 256, size=int(rng.integers(1, 64)), dtype=np.uint8))
        try:
            frame = read_frame(reader(bytes(data)), limits, allow_eof=True)
        except ProtocolError:
            rejected += 1
        else:
            accepted += 1
            assert frame is None or frame.type in {
                "hello",
                "plan",
                "predict",
                "decode",
                "result",
                "error",
                "ping",
                "echo",
                "shutdown",
                "ready",
            }
    assert rejected > 1000 and accepted + rejected == 3000


# ---------------------------------------------------------------------------
# M1: a crafted frame must never escape as a bare exception (it used to abort the whole run)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,nbytes",
    [
        ([0, 2**70], 0),  # the reviewer's frame: zero elements, absurd dimension
        ([2**70, 0], 0),
        ([2**40], 0),
        ([2**31, 2**31], 0),
        ([1 << 29], 0),  # a single dimension above max_dim
        ([2**24, 2**24], 0),  # product above max_elements
    ],
)
def test_oversized_dimensions_are_protocol_errors_even_when_the_element_count_is_zero(shape, nbytes):
    item = {"name": "path", "dtype": "<f8", "shape": shape, "nbytes": nbytes}
    with pytest.raises(ProtocolError):
        read_frame(reader(raw_frame(good_header(type="result", buffers=[item]), bytes(nbytes))))


def test_element_count_limit_applies_to_each_buffer_and_the_message_names_it():
    limits = FrameLimits(max_elements=100)
    item = {"name": "x", "dtype": "|u1", "shape": [11, 11], "nbytes": 121}
    with pytest.raises(ProtocolError, match="element count"):
        read_frame(reader(raw_frame(good_header(buffers=[item]), bytes(121))), limits)
    ok = {"name": "x", "dtype": "|u1", "shape": [10, 10], "nbytes": 100}
    assert read_frame(reader(raw_frame(good_header(buffers=[ok]), bytes(100))), limits).arrays["x"].shape == (10, 10)
    with pytest.raises(ProtocolError, match="dimension"):
        read_frame(
            reader(raw_frame(good_header(buffers=[dict(ok, shape=[1, 101], nbytes=101)]), bytes(101))),
            FrameLimits(max_dim=100),
        )


def test_numpy_errors_are_wrapped_as_protocol_errors(monkeypatch):
    import passage_eval.supervision.protocol as protocol

    def boom(*args, **kwargs):
        raise ValueError("numpy refused")

    monkeypatch.setattr(protocol.np, "empty", boom)
    item = {"name": "x", "dtype": "<f4", "shape": [0, 3], "nbytes": 0}
    with pytest.raises(ProtocolError, match="cannot decode buffer"):
        read_frame(reader(raw_frame(good_header(buffers=[item]), b"")))


def test_default_limits_still_admit_the_largest_legitimate_buffers():
    frame = read_frame(
        reader(
            encode_frame(
                "result",
                1,
                {},
                {
                    "path": np.zeros((70000, 2), "<i8"),
                    "field_0": np.zeros((1024, 1024), "<f4"),
                    "empty": np.zeros((0, 2), "<f8"),
                },
            )
        )
    )
    assert frame.arrays["path"].shape == (70000, 2) and frame.arrays["empty"].shape == (0, 2)


def test_limit_configuration_validates_the_new_fields():
    for name in ("max_dim", "max_elements"):
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                FrameLimits(**{name: bad})

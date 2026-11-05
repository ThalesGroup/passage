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

"""``passage-ipc/v1``: length-prefixed frames over the worker's stdin/stdout pipes.

A frame is::

    prefix   16 bytes  b"PIPC" | header_len (u32 LE) | payload_len (u64 LE)
    header   header_len bytes of UTF-8 JSON
    payload  payload_len bytes: raw little-endian array buffers, back to back

The JSON header is exactly ``{"proto", "type", "seq", "body", "buffers"}``.
``buffers`` lists ``{"name", "dtype", "shape", "nbytes"}`` in payload order; the
buffers are contiguous and their sizes must add up to ``payload_len``.

Security properties: no pickle, no eval, no object dtypes, every length is
checked against :class:`FrameLimits` *before* memory is allocated, JSON rejects
NaN/Infinity/duplicate keys, and any violation raises :class:`ProtocolError`,
which is fatal for the worker that produced the frame.
"""

from __future__ import annotations

import json
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from passage_eval.supervision import IPC_PROTOCOL

if sys.byteorder != "little":  # pragma: no cover - the wire format is little-endian only
    raise ImportError("passage-ipc/v1 requires a little-endian host")

MAGIC = b"PIPC"
_PREFIX = struct.Struct("<4sIQ")
PREFIX_SIZE = _PREFIX.size

SUPERVISOR_TO_WORKER = frozenset({"hello", "plan", "predict", "decode", "ping", "shutdown"})
WORKER_TO_SUPERVISOR = frozenset({"ready", "result", "error", "echo"})
MESSAGE_TYPES = SUPERVISOR_TO_WORKER | WORKER_TO_SUPERVISOR

#: Closed set of wire dtypes. No object, complex, string, big-endian or datetime types.
ALLOWED_DTYPES = frozenset({"|b1", "|u1", "|i1", "<u2", "<i2", "<u4", "<i4", "<u8", "<i8", "<f2", "<f4", "<f8"})
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_HEADER_KEYS = frozenset({"proto", "type", "seq", "body", "buffers"})
_MAX_SEQ = 2**53


class ProtocolError(Exception):
    """A malformed, oversized or unexpected frame. Always fatal for that worker."""


class TruncatedFrame(ProtocolError):
    """The stream ended inside a frame (the peer died or closed early)."""


class StreamClosed(EOFError):
    """The pipe ended; ``got`` bytes of the requested read had arrived."""

    def __init__(self, got: int = 0):
        super().__init__(f"stream closed after {got} bytes")
        self.got = got


@dataclass(frozen=True)
class FrameLimits:
    """Upper bounds checked before anything is allocated.

    ``max_dim`` bounds every single dimension of every buffer (a shape such as
    ``[0, 2**70]`` has zero elements but is still refused), and ``max_elements``
    bounds the element count of one buffer; a buffer can never exceed the payload
    limit either, because its byte size is checked against the payload table.
    """

    max_header_bytes: int = 1 << 20
    max_payload_bytes: int = 256 << 20
    max_buffers: int = 16
    max_ndim: int = 4
    max_dim: int = 1 << 28
    max_elements: int = 1 << 28

    def __post_init__(self):
        for name in ("max_header_bytes", "max_payload_bytes", "max_buffers", "max_ndim", "max_dim", "max_elements"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_LIMITS = FrameLimits()


@dataclass
class Frame:
    type: str
    seq: int
    body: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    wire_bytes: int = 0


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(token):
    raise ValueError(f"non-finite JSON constant {token}")


def _parse_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _loads(data: bytes) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ProtocolError(f"invalid JSON header: {error}") from None


def _wire_array(name: str, array: Any) -> tuple[dict[str, Any], bytes]:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"invalid buffer name {name!r}")
    array = np.asarray(array)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise ValueError(f"buffer {name!r}: dtype {array.dtype} is not allowed on the wire")
    array = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
    dtype = array.dtype.str
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"buffer {name!r}: dtype {dtype} is not allowed on the wire")
    data = array.tobytes()
    return {"name": name, "dtype": dtype, "shape": list(array.shape), "nbytes": len(data)}, data


def encode_frame(
    msg_type: str,
    seq: int,
    body: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    arrays: Mapping[str, Any] | None = None,
    *,
    limits: FrameLimits = DEFAULT_LIMITS,
) -> bytes:
    """Serialise one frame.

    ``body`` may be a callable; it is evaluated after the array buffers have been
    materialised so time-dependent fields (remaining wall time) are as fresh as
    possible when the frame is handed to the pipe.
    """
    if msg_type not in MESSAGE_TYPES:
        raise ValueError(f"unknown message type {msg_type!r}")
    if isinstance(seq, bool) or not isinstance(seq, int) or not 0 <= seq < _MAX_SEQ:
        raise ValueError("seq must be a nonnegative integer")
    arrays = dict(arrays or {})
    if len(arrays) > limits.max_buffers:
        raise ValueError("too many buffers")
    descriptors, chunks = [], []
    for name, array in arrays.items():
        descriptor, data = _wire_array(name, array)
        if len(descriptor["shape"]) > limits.max_ndim:
            raise ValueError(f"buffer {name!r}: too many dimensions")
        descriptors.append(descriptor)
        chunks.append(data)
    if callable(body):
        body = body()
    if not isinstance(body, Mapping):
        raise ValueError("body must be a mapping")
    header = json.dumps(
        {"proto": IPC_PROTOCOL, "type": msg_type, "seq": seq, "body": dict(body), "buffers": descriptors},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_len = sum(len(chunk) for chunk in chunks)
    if len(header) > limits.max_header_bytes or payload_len > limits.max_payload_bytes:
        raise ValueError("frame exceeds the configured size limits")
    return b"".join([_PREFIX.pack(MAGIC, len(header), payload_len), header, *chunks])


def read_frame(
    read_exact: Callable[[int], bytes], limits: FrameLimits = DEFAULT_LIMITS, *, allow_eof: bool = False
) -> Frame | None:
    """Read and validate one frame.

    ``read_exact(n)`` must return exactly ``n`` bytes or raise :class:`StreamClosed`.
    With ``allow_eof`` a stream that ends cleanly *before the first prefix byte*
    returns ``None``; any other truncation is a :class:`ProtocolError`.
    """

    def take(count: int) -> bytes:
        try:
            data = read_exact(count)
        except StreamClosed as closed:
            raise TruncatedFrame(f"truncated frame ({closed.got} of {count} bytes)") from None
        if len(data) != count:
            raise ProtocolError("short read")
        return data

    try:
        prefix = read_exact(PREFIX_SIZE)
    except StreamClosed as closed:
        if allow_eof and closed.got == 0:
            return None
        raise TruncatedFrame(f"truncated frame prefix ({closed.got} of {PREFIX_SIZE} bytes)") from None
    if len(prefix) != PREFIX_SIZE:
        raise ProtocolError("short read")
    magic, header_len, payload_len = _PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError("bad frame magic")
    if not 2 <= header_len <= limits.max_header_bytes:
        raise ProtocolError("header length outside limits")
    if payload_len > limits.max_payload_bytes:
        raise ProtocolError("payload length outside limits")
    header = _loads(take(header_len))
    payload = take(payload_len) if payload_len else b""
    try:
        return _build_frame(header, payload, limits, PREFIX_SIZE + header_len + payload_len)
    except ProtocolError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, MemoryError, RecursionError) as error:
        # Nothing a peer sends may escape as a bare exception: that would abort the supervisor, not the worker.
        raise ProtocolError(f"cannot decode frame: {type(error).__name__}") from None


def _build_frame(header: Any, payload: bytes, limits: FrameLimits, wire_bytes: int) -> Frame:
    if not isinstance(header, dict) or set(header) != _HEADER_KEYS:
        raise ProtocolError("header must contain exactly proto, type, seq, body, buffers")
    if header["proto"] != IPC_PROTOCOL:
        raise ProtocolError("unsupported protocol version")
    msg_type, seq, body, buffers = header["type"], header["seq"], header["body"], header["buffers"]
    if not isinstance(msg_type, str) or msg_type not in MESSAGE_TYPES:
        raise ProtocolError("unknown message type")
    if isinstance(seq, bool) or not isinstance(seq, int) or not 0 <= seq < _MAX_SEQ:
        raise ProtocolError("invalid sequence number")
    if not isinstance(body, dict):
        raise ProtocolError("body must be an object")
    if not isinstance(buffers, list) or len(buffers) > limits.max_buffers:
        raise ProtocolError("invalid buffer table")
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for item in buffers:
        if not isinstance(item, dict) or set(item) != {"name", "dtype", "shape", "nbytes"}:
            raise ProtocolError("invalid buffer descriptor")
        name, dtype, shape, nbytes = item["name"], item["dtype"], item["shape"], item["nbytes"]
        if not isinstance(name, str) or not _NAME.match(name) or name in arrays:
            raise ProtocolError("invalid or duplicate buffer name")
        if not isinstance(dtype, str) or dtype not in ALLOWED_DTYPES:
            raise ProtocolError("buffer dtype is not allowed")
        if (
            not isinstance(shape, list)
            or len(shape) > limits.max_ndim
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
        ):
            raise ProtocolError("invalid buffer shape")
        if any(dim > limits.max_dim for dim in shape):
            raise ProtocolError("buffer dimension outside limits")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
            raise ProtocolError("invalid buffer size")
        elements = math.prod(shape)  # Python integers: no overflow, bounded by the checks above
        if elements > limits.max_elements:
            raise ProtocolError("buffer element count outside limits")
        np_dtype = np.dtype(dtype)
        if elements * np_dtype.itemsize != nbytes:
            raise ProtocolError("buffer shape does not match its size")
        if offset + nbytes > len(payload):
            raise ProtocolError("buffers exceed the payload")
        try:
            if nbytes:
                view = np.frombuffer(payload, dtype=np_dtype, count=elements, offset=offset)
                arrays[name] = view.reshape(shape).copy()  # aligned, owned, writable
            else:
                arrays[name] = np.empty(shape, dtype=np_dtype)
        except (ValueError, TypeError, OverflowError, MemoryError) as error:
            raise ProtocolError(f"cannot decode buffer: {type(error).__name__}") from None
        offset += nbytes
    if offset != len(payload):
        raise ProtocolError("payload length does not match the buffer table")
    return Frame(msg_type, seq, body, arrays, wire_bytes)


def expect_type(frame: Frame, allowed: frozenset[str] | set[str]) -> Frame:
    """Reject a frame whose type is not valid for this direction/state."""
    if frame.type not in allowed:
        raise ProtocolError(f"unexpected message type {frame.type!r}")
    return frame


def expect_body(
    body: Mapping[str, Any], required: set[str] | frozenset[str], optional: set[str] | frozenset[str] = frozenset()
) -> Mapping[str, Any]:
    """Strict key check for message bodies: unknown or missing keys are fatal."""
    keys = set(body)
    if not required <= keys or not keys <= (set(required) | set(optional)):
        raise ProtocolError(f"unexpected body keys {sorted(keys ^ set(required))}")
    return body

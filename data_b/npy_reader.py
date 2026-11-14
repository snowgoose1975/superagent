"""Read a deliberately small, non-executable subset of NumPy ``.npy`` files."""

from __future__ import annotations

import ast
import math
import pickletools
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReaderLimits:
    max_header_bytes: int = 64 * 1024
    max_payload_bytes: int = 64 * 1024 * 1024
    max_rows: int = 2_000_000
    max_columns: int = 4096
    max_elements: int = 8_000_000
    max_pickle_ops: int = 2_000_000
    max_pickle_stack: int = 100_000


@dataclass(frozen=True)
class _Global:
    module: str
    name: str


@dataclass(frozen=True)
class _DType:
    code: str
    state: Any = None


@dataclass
class _ArrayShell:
    state: Any = None


@dataclass(frozen=True)
class _StaticArray:
    shape: tuple[int, ...]
    values: Any


_MARK = object()
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


def read_npy_array(path: Path, limits: ReaderLimits | None = None) -> np.ndarray:
    """Read primitive arrays, or object arrays through a static opcode interpreter.

    This function never calls ``pickle.load``, ``numpy.load`` or any callable named by
    a pickle GLOBAL opcode. Object payloads outside the small ndarray reconstruction
    grammar are rejected.
    """

    limits = limits or ReaderLimits()
    with Path(path).open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not a NumPy .npy file: {path}")
        version_bytes = stream.read(2)
        if len(version_bytes) != 2:
            raise ValueError("Truncated .npy version")
        major, minor = version_bytes
        if major == 1:
            raw_length = stream.read(2)
            if len(raw_length) != 2:
                raise ValueError("Truncated .npy header length")
            header_length = struct.unpack("<H", raw_length)[0]
        elif major in (2, 3):
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                raise ValueError("Truncated .npy header length")
            header_length = struct.unpack("<I", raw_length)[0]
        else:
            raise ValueError(f"Unsupported .npy version {major}.{minor}")
        if header_length <= 0 or header_length > limits.max_header_bytes:
            raise ValueError(".npy header exceeds the configured size limit")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("Truncated .npy header")
        encoding = "utf-8" if major == 3 else "latin1"
        try:
            header = ast.literal_eval(header_bytes.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid .npy header") from exc
        if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
            raise ValueError("Unexpected .npy header fields")
        shape = header["shape"]
        if (
            not isinstance(shape, tuple)
            or not shape
            or any(type(size) is not int or size < 0 for size in shape)
        ):
            raise ValueError("Invalid .npy shape")
        count = math.prod(shape)
        if count > limits.max_elements:
            raise ValueError(".npy element count exceeds the configured limit")
        if len(shape) == 2 and (
            shape[0] > limits.max_rows or shape[1] > limits.max_columns
        ):
            raise ValueError(".npy matrix dimensions exceed configured limits")
        if type(header["fortran_order"]) is not bool:
            raise ValueError("Invalid .npy memory order")
        payload = stream.read(limits.max_payload_bytes + 1)
        if len(payload) > limits.max_payload_bytes:
            raise ValueError(".npy payload exceeds the configured size limit")

    try:
        dtype = np.dtype(header["descr"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported .npy dtype") from exc
    if dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError("Structured and subarray dtypes are not supported")

    if dtype.hasobject:
        if dtype != np.dtype(object):
            raise ValueError("Only plain object arrays are supported")
        values = _parse_object_payload(payload, shape, limits)
        array = np.asarray(values, dtype=object)
        if array.size != count:
            raise ValueError("Object payload element count does not match its header")
        return array.reshape(shape)

    if dtype.kind not in "biufUS":
        raise ValueError(f"Unsupported primitive .npy dtype kind: {dtype.kind}")
    expected_bytes = count * dtype.itemsize
    if expected_bytes != len(payload):
        raise ValueError("Primitive .npy payload size does not match its header")
    return np.frombuffer(payload, dtype=dtype, count=count).copy().reshape(shape, order="F" if header["fortran_order"] else "C")


def _parse_object_payload(
    payload: bytes, expected_shape: tuple[int, ...], limits: ReaderLimits
) -> Any:
    if not payload:
        raise ValueError("Empty object-array pickle payload")
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    next_memo = 0
    stopped = False

    def pop_mark() -> list[Any]:
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is _MARK:
                items = stack[index + 1 :]
                del stack[index:]
                return items
        raise ValueError("Pickle mark stack is unbalanced")

    try:
        operations = pickletools.genops(payload)
        for operation_index, (opcode, arg, position) in enumerate(operations, start=1):
            if operation_index > limits.max_pickle_ops or len(stack) > limits.max_pickle_stack:
                raise ValueError("Object-array pickle exceeds parser limits")
            name = opcode.name
            if name == "PROTO":
                if type(arg) is not int or arg not in (3, 4):
                    raise ValueError("Only pickle protocol 3 or 4 object arrays are supported")
            elif name == "FRAME":
                if type(arg) is not int or arg < 0 or arg > len(payload):
                    raise ValueError("Invalid pickle FRAME size")
            elif name == "MARK":
                stack.append(_MARK)
            elif name == "NONE":
                stack.append(None)
            elif name in ("NEWTRUE", "NEWFALSE"):
                stack.append(name == "NEWTRUE")
            elif name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"):
                value = int(arg)
                stack.append(value)
            elif name in ("FLOAT", "BINFLOAT"):
                stack.append(float(arg))
            elif name in ("UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINSTRING", "SHORT_BINSTRING"):
                if not isinstance(arg, str):
                    raise ValueError("Invalid pickle string value")
                stack.append(arg)
            elif name in ("BINBYTES", "SHORT_BINBYTES"):
                if not isinstance(arg, bytes):
                    raise ValueError("Invalid pickle byte value")
                stack.append(arg)
            elif name == "EMPTY_LIST":
                stack.append([])
            elif name == "LIST":
                stack.append(pop_mark())
            elif name == "APPEND":
                item = stack.pop()
                if not stack or not isinstance(stack[-1], list):
                    raise ValueError("APPEND target is not a list")
                stack[-1].append(item)
            elif name == "APPENDS":
                items = pop_mark()
                if not stack or not isinstance(stack[-1], list):
                    raise ValueError("APPENDS target is not a list")
                stack[-1].extend(items)
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name == "TUPLE":
                stack.append(tuple(pop_mark()))
            elif name == "TUPLE1":
                stack.append((stack.pop(),))
            elif name == "TUPLE2":
                right, left = stack.pop(), stack.pop()
                stack.append((left, right))
            elif name == "TUPLE3":
                third, second, first = stack.pop(), stack.pop(), stack.pop()
                stack.append((first, second, third))
            elif name == "GLOBAL":
                module, symbol = str(arg).split(" ", 1)
                if (module, symbol) not in _ALLOWED_GLOBALS:
                    raise ValueError(f"Unsupported pickle GLOBAL {module}.{symbol}")
                stack.append(_Global(module, symbol))
            elif name == "STACK_GLOBAL":
                symbol, module = stack.pop(), stack.pop()
                if not isinstance(module, str) or not isinstance(symbol, str):
                    raise ValueError("Invalid pickle STACK_GLOBAL operands")
                if (module, symbol) not in _ALLOWED_GLOBALS:
                    raise ValueError(f"Unsupported pickle GLOBAL {module}.{symbol}")
                stack.append(_Global(module, symbol))
            elif name == "REDUCE":
                args = stack.pop()
                function = stack.pop()
                if not isinstance(args, tuple) or not isinstance(function, _Global):
                    raise ValueError("Unsupported pickle REDUCE form")
                if function.name == "_reconstruct" and function.module.endswith("multiarray"):
                    if len(args) != 3 or args[0] != _Global("numpy", "ndarray"):
                        raise ValueError("Unexpected ndarray reconstruction arguments")
                    stack.append(_ArrayShell())
                elif function == _Global("numpy", "dtype"):
                    if not args or not isinstance(args[0], str):
                        raise ValueError("Unexpected NumPy dtype arguments")
                    stack.append(_DType(args[0]))
                else:
                    raise ValueError("Pickle REDUCE target is not permitted")
            elif name == "BUILD":
                state = stack.pop()
                if not stack:
                    raise ValueError("Pickle BUILD has no object")
                target = stack[-1]
                if isinstance(target, _DType):
                    if not isinstance(state, tuple) or not _is_static_value(state):
                        raise ValueError("Unexpected NumPy dtype state")
                    stack[-1] = _DType(target.code, state)
                elif isinstance(target, _ArrayShell):
                    stack[-1] = _build_static_array(state, expected_shape)
                else:
                    raise ValueError("Pickle BUILD target is not permitted")
            elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
                if not stack:
                    raise ValueError("Pickle memo write has no value")
                index = int(arg)
                memo[index] = stack[-1]
                next_memo = max(next_memo, index + 1)
            elif name == "MEMOIZE":
                if not stack:
                    raise ValueError("Pickle memo write has no value")
                memo[next_memo] = stack[-1]
                next_memo += 1
            elif name in ("BINGET", "LONG_BINGET", "GET"):
                index = int(arg)
                if index not in memo:
                    raise ValueError("Pickle references an absent memo value")
                stack.append(memo[index])
            elif name == "POP":
                stack.pop()
            elif name == "POP_MARK":
                pop_mark()
            elif name == "DUP":
                stack.append(stack[-1])
            elif name == "STOP":
                if len(stack) != 1 or not isinstance(stack[0], _StaticArray):
                    raise ValueError("Pickle did not produce one static NumPy array")
                if position + 1 != len(payload):
                    raise ValueError("Trailing bytes after object-array pickle STOP")
                stopped = True
                return stack[0].values
            else:
                raise ValueError(f"Unsupported object-array pickle opcode: {name}")
    except (IndexError, OverflowError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("Unsupported", "Object-array", "Only ", "Unexpected", "Invalid", "Pickle", "APPEND", "APPENDS")):
            raise
        raise ValueError("Malformed object-array pickle payload") from exc
    if not stopped:
        raise ValueError("Object-array pickle has no STOP opcode")
    raise ValueError("Object-array pickle did not produce an array")


def _build_static_array(state: Any, expected_shape: tuple[int, ...]) -> _StaticArray:
    if not isinstance(state, tuple) or len(state) != 5:
        raise ValueError("Unexpected ndarray BUILD state")
    version, shape, dtype, fortran_order, values = state
    if version != 1 or shape != expected_shape or type(fortran_order) is not bool:
        raise ValueError("ndarray state does not match its .npy header")
    if not isinstance(dtype, _DType) or not dtype.code.lower().startswith("o"):
        raise ValueError("Only object dtype ndarray payloads are supported")
    if not _is_static_value(values):
        raise ValueError("Object array contains unsupported nested values")
    flattened = _flatten(values)
    if len(flattened) != math.prod(expected_shape):
        raise ValueError("Object array value count does not match its shape")
    return _StaticArray(expected_shape, values)


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    if isinstance(value, tuple):
        flattened = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _is_static_value(value: Any, depth: int = 0) -> bool:
    if depth > 16:
        return False
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value) or math.isnan(value)
    if isinstance(value, (list, tuple)):
        return all(_is_static_value(item, depth + 1) for item in value)
    return False

"""AES-256-GCM transport protection for simulated federated model updates.

This module models a client/coordinator message boundary in one process. It
does not implement a network protocol, public-key identity, or key exchange.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import closing
import io
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_BYTES = 32
NONCE_BYTES = 12
MESSAGE_VERSION = 1
_CLIENT_ID = re.compile(r"^client_[1-4]$")
_FIELDS = {"version", "client_id", "round", "session_id", "nonce", "ciphertext"}


class SecurityError(ValueError):
    """Raised when an encrypted update fails validation or authentication."""


def _validate_client_id(client_id: Any) -> str:
    if not isinstance(client_id, str) or not _CLIENT_ID.fullmatch(client_id):
        raise SecurityError("Unknown or invalid client identifier")
    return client_id


def _validate_round(round_number: Any) -> int:
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise SecurityError("Communication round must be a positive integer")
    return round_number


def _validate_session_id(session_id: Any) -> str:
    if not isinstance(session_id, str):
        raise SecurityError("Invalid session identity")
    try:
        parsed = uuid.UUID(session_id)
    except (ValueError, AttributeError, TypeError):
        raise SecurityError("Invalid session identity") from None
    if str(parsed) != session_id:
        raise SecurityError("Invalid session identity")
    return session_id


def _validate_key(key: Any) -> bytes:
    if not isinstance(key, bytes) or len(key) != KEY_BYTES:
        raise SecurityError("Each client key must contain exactly 32 bytes")
    return key


def load_client_keys(
    client_ids: Sequence[str], environ: Mapping[str, str] | None = None
) -> dict[str, bytes]:
    """Load distinct 32-byte keys from FEDERATED_CLIENT_KEY_<client_id> hex vars."""
    env = os.environ if environ is None else environ
    if not isinstance(client_ids, Sequence) or isinstance(client_ids, (str, bytes)) or not client_ids:
        raise SecurityError("At least one known client identifier is required")
    ids = [_validate_client_id(client_id) for client_id in client_ids]
    if len(set(ids)) != len(ids):
        raise SecurityError("Client identifiers must be unique")
    keys: dict[str, bytes] = {}
    for client_id in ids:
        variable = f"FEDERATED_CLIENT_KEY_{client_id.upper()}"
        raw = env.get(variable)
        if not isinstance(raw, str) or len(raw) != KEY_BYTES * 2:
            raise SecurityError(f"Missing or invalid 64-character hex key in {variable}")
        try:
            key = bytes.fromhex(raw)
        except ValueError:
            raise SecurityError(f"Invalid hex key in {variable}") from None
        keys[client_id] = _validate_key(key)
    if len(set(keys.values())) != len(keys):
        raise SecurityError("Each simulated client must have a distinct secret key")
    return keys


def serialize_model_update(update: Mapping[str, np.ndarray]) -> bytes:
    """Serialize the exact FedAvg tensor state using a non-pickle NPZ payload."""
    if not isinstance(update, Mapping) or set(update) != {"weights", "bias"}:
        raise SecurityError("Model update must contain weights and bias only")
    arrays: dict[str, np.ndarray] = {}
    for name in ("weights", "bias"):
        array = np.asarray(update[name])
        expected_ndim = 2 if name == "weights" else 1
        if array.size == 0 or array.ndim != expected_ndim:
            raise SecurityError(f"Invalid {name} tensor dimensions")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise SecurityError(f"{name} tensor must contain finite floating-point values")
        arrays[name] = np.ascontiguousarray(array)
    if arrays["weights"].shape[1] != arrays["bias"].shape[0]:
        raise SecurityError("Model weight horizon and bias dimensions do not match")
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _deserialize_model_update(payload: bytes) -> dict[str, np.ndarray]:
    if not isinstance(payload, bytes) or not payload or len(payload) > 1_000_000:
        raise SecurityError("Invalid or oversized model update payload")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {"weights", "bias"}:
                raise SecurityError("Decrypted model update has invalid parameter names")
            update = {name: np.array(archive[name], copy=True) for name in ("weights", "bias")}
    except SecurityError:
        raise
    except Exception:
        raise SecurityError("Decrypted model update payload is malformed") from None
    # Apply the same finite-value, shape, and dtype checks on received tensors.
    serialize_model_update(update)
    return update


def _aad(client_id: str, round_number: int, session_id: str, version: int) -> bytes:
    return json.dumps(
        {"client_id": client_id, "round": round_number, "session_id": session_id, "version": version},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SecurityError("Encrypted message contains duplicate metadata fields")
        result[key] = value
    return result


class ReplayStore:
    """Durable session and nonce ledger; closing/restarting retires open sessions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "session_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('active','closed')))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS nonces ("
            "client_id TEXT NOT NULL, nonce BLOB NOT NULL, session_id TEXT NOT NULL, "
            "status TEXT NOT NULL CHECK(status IN ('issued','consumed')), "
            "PRIMARY KEY(client_id, nonce), "
            "FOREIGN KEY(session_id) REFERENCES sessions(session_id))"
        )
        connection.commit()
        return connection

    def begin_session(self, session_id: str | None = None) -> str:
        candidate = _validate_session_id(session_id or str(uuid.uuid4()))
        with closing(self._connect()) as connection, connection:
            connection.execute("UPDATE sessions SET status='closed' WHERE status='active'")
            try:
                connection.execute("INSERT INTO sessions(session_id,status) VALUES (?, 'active')", (candidate,))
            except sqlite3.IntegrityError:
                raise SecurityError("Session identity has already been used") from None
        return candidate

    def close_session(self, session_id: str) -> None:
        session_id = _validate_session_id(session_id)
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "UPDATE sessions SET status='closed' WHERE session_id=? AND status='active'", (session_id,)
            )
            if cursor.rowcount != 1:
                raise SecurityError("Session is not active")

    def _assert_active(self, connection: sqlite3.Connection, session_id: str) -> None:
        row = connection.execute("SELECT status FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row != ("active",):
            raise SecurityError("Session is unknown, closed, or no longer active")

    def issue_nonce(self, client_id: str, session_id: str) -> bytes:
        client_id = _validate_client_id(client_id)
        session_id = _validate_session_id(session_id)
        for _ in range(20):
            nonce = secrets.token_bytes(NONCE_BYTES)
            with closing(self._connect()) as connection, connection:
                self._assert_active(connection, session_id)
                try:
                    connection.execute(
                        "INSERT INTO nonces(client_id,nonce,session_id,status) VALUES (?,?,?,'issued')",
                        (client_id, nonce, session_id),
                    )
                    return nonce
                except sqlite3.IntegrityError:
                    continue
        raise SecurityError("Could not allocate a unique message nonce")

    def consume_nonce(self, client_id: str, session_id: str, nonce: bytes) -> None:
        with closing(self._connect()) as connection, connection:
            self._assert_active(connection, session_id)
            cursor = connection.execute(
                "UPDATE nonces SET status='consumed' "
                "WHERE client_id=? AND nonce=? AND session_id=? AND status='issued'",
                (client_id, nonce, session_id),
            )
            if cursor.rowcount != 1:
                raise SecurityError("Replayed or unissued message nonce")


def encrypt_model_update(
    client_id: str,
    update: Mapping[str, np.ndarray],
    round_number: int,
    session_id: str,
    key: bytes,
    replay_store: ReplayStore,
) -> bytes:
    """Return a JSON envelope containing AES-256-GCM ciphertext and metadata."""
    client_id = _validate_client_id(client_id)
    round_number = _validate_round(round_number)
    session_id = _validate_session_id(session_id)
    key = _validate_key(key)
    plaintext = serialize_model_update(update)
    nonce = replay_store.issue_nonce(client_id, session_id)
    aad = _aad(client_id, round_number, session_id, MESSAGE_VERSION)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    envelope = {
        "version": MESSAGE_VERSION,
        "client_id": client_id,
        "round": round_number,
        "session_id": session_id,
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decrypt_model_update(
    message: bytes,
    expected_client_id: str,
    expected_round: int,
    expected_session_id: str,
    keys: Mapping[str, bytes],
    replay_store: ReplayStore,
) -> dict[str, np.ndarray]:
    """Validate metadata, authenticate AAD and tag, deserialize, and consume nonce."""
    expected_client_id = _validate_client_id(expected_client_id)
    expected_round = _validate_round(expected_round)
    expected_session_id = _validate_session_id(expected_session_id)
    if expected_client_id not in keys:
        raise SecurityError("Unknown client identifier")
    key = _validate_key(keys[expected_client_id])
    if not isinstance(message, bytes) or not message or len(message) > 1_000_000:
        raise SecurityError("Malformed or oversized encrypted message")
    try:
        envelope = json.loads(message.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecurityError("Malformed encrypted message envelope") from None
    except SecurityError:
        raise
    if not isinstance(envelope, dict) or set(envelope) != _FIELDS:
        raise SecurityError("Encrypted message envelope has invalid fields")
    client_id = _validate_client_id(envelope.get("client_id"))
    round_number = _validate_round(envelope.get("round"))
    session_id = _validate_session_id(envelope.get("session_id"))
    version = envelope.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != MESSAGE_VERSION:
        raise SecurityError("Unsupported encrypted message version")
    if client_id != expected_client_id:
        raise SecurityError("Encrypted message client does not match expected client")
    if round_number != expected_round:
        raise SecurityError("Encrypted message round does not match expected round")
    if session_id != expected_session_id:
        raise SecurityError("Encrypted message session does not match active session")
    try:
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise SecurityError("Encrypted message has invalid encoding") from None
    if len(nonce) != NONCE_BYTES or len(ciphertext) < 16:
        raise SecurityError("Encrypted message has invalid nonce or ciphertext length")
    aad = _aad(client_id, round_number, session_id, version)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, aad)
    except (InvalidTag, ValueError):
        raise SecurityError("Encrypted message authentication failed") from None
    update = _deserialize_model_update(plaintext)
    # Consume only after successful authentication and payload validation.
    replay_store.consume_nonce(client_id, session_id, nonce)
    return update


def aggregate_encrypted_updates(
    messages: Sequence[bytes],
    client_ids: Sequence[str],
    weights: Sequence[int | float],
    round_number: int,
    session_id: str,
    keys: Mapping[str, bytes],
    replay_store: ReplayStore,
    aggregator: Callable[[list[dict[str, np.ndarray]], list[int | float]], dict[str, np.ndarray]],
    decryption_durations: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Authenticate every update before invoking the supplied aggregator once."""
    if not isinstance(messages, Sequence) or isinstance(messages, (bytes, bytearray)) or not messages:
        raise SecurityError("At least one encrypted client update is required")
    if not isinstance(client_ids, Sequence) or isinstance(client_ids, (str, bytes)):
        raise SecurityError("Client identifiers must be a sequence")
    if not isinstance(weights, (list, tuple, np.ndarray)):
        raise SecurityError("Aggregation weights must be finite and positive")
    ids = [_validate_client_id(client_id) for client_id in client_ids]
    if not ids or len(set(ids)) != len(ids) or len(messages) != len(ids) or len(weights) != len(ids):
        raise SecurityError("Encrypted updates, client identifiers, and weights must match")
    try:
        numeric_weights = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise SecurityError("Aggregation weights must be finite and positive") from None
    if numeric_weights.ndim != 1 or not np.isfinite(numeric_weights).all() or np.any(numeric_weights <= 0):
        raise SecurityError("Aggregation weights must be finite and positive")
    round_number = _validate_round(round_number)
    session_id = _validate_session_id(session_id)
    updates = []
    for message, client_id in zip(messages, ids):
        started = time.perf_counter()
        update = decrypt_model_update(message, client_id, round_number, session_id, keys, replay_store)
        elapsed = time.perf_counter() - started
        if decryption_durations is not None:
            decryption_durations.append(elapsed)
        updates.append(update)
    # No aggregate call occurs until all messages passed authentication.
    return aggregator(updates, list(weights))

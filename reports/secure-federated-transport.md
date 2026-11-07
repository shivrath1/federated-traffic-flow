# Secure FedAvg Transport Experiment

## Implementation

`src/security/model_update_encryption.py` implements AES-256-GCM messages for the four simulated client model updates. Each client key is read from its own `FEDERATED_CLIENT_KEY_CLIENT_N` environment variable as 64 hexadecimal characters (32 bytes). Startup fails if a key is missing, malformed, the wrong length, or shared by two configured clients. No key is generated as a fallback, stored in the repository, or written to experiment metrics.

Each message contains a versioned JSON envelope with client ID, round, UUID session identity, fresh 96-bit nonce, and base64 ciphertext. Client ID, round, session ID, and version are authenticated as additional authenticated data. The coordinator validates the expected client, round, and session, verifies the GCM tag, validates the decoded tensors, and decrypts every update before invoking FedAvg. The aggregation function is called only after all messages in the round pass validation.

The SQLite replay ledger at `models/federated_secure_baseline/replay_state.sqlite3` persists issued client nonces and session state. Starting another run retires any session left active by a previous or interrupted process. Session IDs cannot be reused, nonces cannot be reissued for the same client, and messages from closed sessions are rejected. Deleting or replacing the ledger discards that coordinator's replay history; protecting and backing it up is an operational requirement.

This is a single-process simulation of a client/coordinator message boundary, not network deployment. AES-GCM authenticates possession of a client-specific shared secret. It does not establish public-key identity or provide non-repudiation.

## Matched-settings comparison

The secure run loaded the existing FedAvg configuration and used the same prepared client partitions (52, 52, 52, and 51 sensors), 12-step lookback and horizon, seed 42, one local epoch, batch size 256, learning rate 0.001, validation selection settings, and 30 communication rounds. The secure experiment wrote to a new folder and did not overwrite `models/federated_baseline/`.

| Measure | Existing FedAvg result | Secure FedAvg run |
|---|---:|---:|
| Communication rounds | 30 | 30 |
| Test MAE | 5.522868962441759 | 5.522868962441759 |
| Test RMSE | 11.97366324744669 | 11.97366324744669 |
| Training time excluding measured crypto | Not available from saved baseline breakdown | 85.6915 s |
| Encryption time, including NPZ serialization | Not applicable | 0.4430 s |
| Decryption/authentication time, including deserialization | Not applicable | 0.4334 s |
| Secure experiment wall time before test evaluation | Not applicable | 86.5680 s |

The secure run produced 120 encrypted updates and selected round 30. Its MAE and RMSE equal the saved baseline values exactly. A separate fresh run of the original baseline runner in a temporary directory used the saved configuration, ran 30 rounds, and reproduced both saved metrics exactly. Its measured training time was 122.4861 s. This timing differs materially from the secure run's training-only value and should not be interpreted as a speed improvement: these are single-run wall-clock measurements, taken sequentially on a shared local environment, and have not been repeated under controlled load. The saved FedAvg baseline artifacts and metrics were not changed by the control run.

## Payload and timing measurements

Across 120 update messages:

| Payload measure | Bytes |
|---|---:|
| Serialized NPZ update payloads before encryption | 135,840 |
| Encrypted JSON envelopes | 200,844 |
| Added envelope/nonce/tag/encoding bytes | 65,004 |
| Added bytes relative to serialized NPZ payloads | 47.8534% |

The payload comparison is the measured UTF-8 envelope size (including base64-encoded nonce and ciphertext plus metadata) against the serialized NPZ tensor payload size. It is not a network transfer measurement and excludes protocol framing, retransmissions, and network effects. The original FedAvg communication estimate counts raw float32 arrays only; it is a different serialization basis and should not be compared directly to the table above. Only client-to-coordinator update payloads are encrypted in this implementation; coordinator-to-client model broadcasts remain part of the local simulation and are not encrypted by this experiment.

Encryption averaged 3.691 ms per update and decryption/authentication averaged 3.611 ms per update over 120 messages each. The combined measured cryptographic work was 0.8763 s. These are local CPU measurements, not latency measurements over a network.

## Validation

- `python3 -m pytest -q tests/test_federated_security.py`: **12 passed**.
- `python3 -m pytest -q`: **50 passed** after installing the already-declared FastAPI dependency missing from the initial environment.
- `python3 -m compileall -q src app tests`: passed.
- `git diff --check`: passed.
- The secure run's held-out test metrics match the saved FedAvg baseline metrics exactly; the test split was opened after validation-based model selection.
- The secure runner verifies the complete selected model state against the baseline artifact before saving results.

## Limitations

The simulated coordinator and clients run in one process and share the local environment. There is no real network, key exchange or production key management. AES-GCM does not provide secure aggregation: the coordinator decrypts each update. There are no digital signatures, differential privacy, malicious-client/poisoning defenses, or public-key identities. Compromise of a client secret enables messages authenticated as possession of that secret. The local replay ledger is an integrity-critical coordinator file. Timings are single-run local measurements and do not establish production capacity or network performance.

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from src.security.model_update_encryption import (
    ReplayStore,
    SecurityError,
    aggregate_encrypted_updates,
    decrypt_model_update,
    encrypt_model_update,
    load_client_keys,
)

class EncryptedUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ReplayStore(self.root / "replay.sqlite3")
        self.session = self.store.begin_session()
        self.keys = {
            "client_1": bytes(range(32)),
            "client_2": bytes(range(1, 33)),
        }
        self.state = {
            "weights": np.arange(12, dtype=np.float32).reshape(3, 4),
            "bias": np.arange(4, dtype=np.float32),
        }

    def tearDown(self):
        self.temp.cleanup()

    def encrypt(self, client="client_1", round_number=1, state=None):
        return encrypt_model_update(
            client, self.state if state is None else state, round_number,
            self.session, self.keys[client], self.store,
        )

    def decrypt(self, message, client="client_1", round_number=1, session=None, keys=None, store=None):
        return decrypt_model_update(
            message, client, round_number, session or self.session,
            self.keys if keys is None else keys, self.store if store is None else store,
        )

    def test_valid_encryption_and_decryption_round_trip(self):
        message = self.encrypt()
        recovered = self.decrypt(message)
        for name in self.state:
            np.testing.assert_array_equal(recovered[name], self.state[name])

    def test_ciphertext_tampering_fails_authentication(self):
        envelope = json.loads(self.encrypt())
        data = bytearray(base64.b64decode(envelope["ciphertext"]))
        data[0] ^= 0x01
        envelope["ciphertext"] = base64.b64encode(data).decode("ascii")
        with self.assertRaisesRegex(SecurityError, "authentication failed"):
            self.decrypt(json.dumps(envelope).encode())

    def test_wrong_client_key_fails_authentication(self):
        message = self.encrypt()
        wrong_keys = {**self.keys, "client_1": self.keys["client_2"]}
        with self.assertRaisesRegex(SecurityError, "authentication failed"):
            self.decrypt(message, keys=wrong_keys)

    def test_unknown_client_is_rejected(self):
        with self.assertRaisesRegex(SecurityError, "Unknown"):
            decrypt_model_update(b"{}", "client_9", 1, self.session, self.keys, self.store)

    def test_modified_aad_metadata_fails_authentication(self):
        envelope = json.loads(self.encrypt())
        envelope["round"] = 2
        with self.assertRaisesRegex(SecurityError, "authentication failed"):
            self.decrypt(json.dumps(envelope).encode(), round_number=2)

    def test_replayed_message_and_reused_session_are_rejected(self):
        message = self.encrypt()
        self.decrypt(message)
        with self.assertRaisesRegex(SecurityError, "Replayed"):
            self.decrypt(message)
        with self.assertRaisesRegex(SecurityError, "already been used"):
            self.store.begin_session(self.session)

    def test_previous_session_message_is_rejected_after_restart(self):
        message = self.encrypt()
        # Reopening the durable ledger models a fresh coordinator process.
        restarted = ReplayStore(self.root / "replay.sqlite3")
        new_session = restarted.begin_session()
        with self.assertRaisesRegex(SecurityError, "session does not match"):
            decrypt_model_update(message, "client_1", 1, new_session, self.keys, restarted)
        with self.assertRaisesRegex(SecurityError, "no longer active"):
            decrypt_model_update(message, "client_1", 1, self.session, self.keys, restarted)

    def test_invalid_round_and_malformed_payload_are_rejected(self):
        for invalid_round in (0, -1, True, 1.5):
            with self.subTest(round=invalid_round), self.assertRaises(SecurityError):
                self.encrypt(round_number=invalid_round)
        with self.assertRaisesRegex(SecurityError, "weights and bias"):
            self.encrypt(state={"weights": np.ones((2, 2), dtype=np.float32)})
        invalid_state = {"weights": np.ones((2, 2), dtype=np.float32), "bias": np.array([np.nan, 0], dtype=np.float32)}
        with self.assertRaisesRegex(SecurityError, "finite"):
            self.encrypt(state=invalid_state)
        with self.assertRaisesRegex(SecurityError, "Malformed"):
            self.decrypt(b"not json")

    def test_keys_must_be_present_32_byte_hex_and_distinct(self):
        env = {
            "FEDERATED_CLIENT_KEY_CLIENT_1": "01" * 32,
            "FEDERATED_CLIENT_KEY_CLIENT_2": "02" * 32,
        }
        loaded = load_client_keys(["client_1", "client_2"], env)
        self.assertEqual(len(loaded["client_1"]), 32)
        with self.assertRaisesRegex(SecurityError, "Missing or invalid"):
            load_client_keys(["client_1", "client_2"], {"FEDERATED_CLIENT_KEY_CLIENT_1": "01" * 32})
        same = {name: "ab" * 32 for name in env}
        with self.assertRaisesRegex(SecurityError, "distinct"):
            load_client_keys(["client_1", "client_2"], same)

    def test_rejected_update_never_reaches_aggregation(self):
        good = self.encrypt("client_1")
        bad = self.encrypt("client_2")
        envelope = json.loads(bad)
        ciphertext = bytearray(base64.b64decode(envelope["ciphertext"]))
        ciphertext[-1] ^= 0x80
        envelope["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")
        bad = json.dumps(envelope).encode()
        aggregator = Mock(side_effect=lambda states, weights: states[0])
        with self.assertRaises(SecurityError):
            aggregate_encrypted_updates(
                [good, bad], ["client_1", "client_2"], [1, 1], 1,
                self.session, self.keys, self.store, aggregator,
            )
        aggregator.assert_not_called()

    def test_aggregation_receives_only_valid_decrypted_updates(self):
        messages = [self.encrypt("client_1"), self.encrypt("client_2")]
        result = aggregate_encrypted_updates(
            messages, ["client_1", "client_2"], [1, 2], 1,
            self.session, self.keys, self.store,
            lambda states, weights: {"weights": states[0]["weights"] + states[1]["weights"], "bias": states[0]["bias"] + states[1]["bias"]},
        )
        np.testing.assert_array_equal(result["weights"], self.state["weights"] * 2)

    def test_malformed_weights_never_reach_aggregation(self):
        aggregator = Mock()
        with self.assertRaisesRegex(SecurityError, "weights"):
            aggregate_encrypted_updates(
                [self.encrypt()], ["client_1"], [np.nan], 1,
                self.session, self.keys, self.store, aggregator,
            )
        aggregator.assert_not_called()


if __name__ == "__main__":
    unittest.main()

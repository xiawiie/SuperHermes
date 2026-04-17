import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from milvus_client import MilvusManager  # noqa: E402


class FlakyMilvusClient:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.closed = False

    def query(self, **kwargs):
        if self.should_fail:
            raise RuntimeError("Cannot invoke RPC on closed channel!")
        return [{"filename": "manual.pdf", "file_type": "PDF"}]

    def close(self):
        self.closed = True


class MilvusManagerReconnectTests(unittest.TestCase):
    def test_query_reconnects_once_when_rpc_channel_is_closed(self):
        clients = [FlakyMilvusClient(should_fail=True), FlakyMilvusClient()]

        def client_factory(uri):
            self.assertEqual(uri, "http://127.0.0.1:19530")
            return clients.pop(0)

        with patch("milvus_client.MilvusClient", side_effect=client_factory) as factory:
            manager = MilvusManager()
            manager.host = "127.0.0.1"
            manager.port = "19530"
            manager.uri = "http://127.0.0.1:19530"

            result = manager.query(output_fields=["filename", "file_type"], limit=5)

        self.assertEqual(result, [{"filename": "manual.pdf", "file_type": "PDF"}])
        self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()

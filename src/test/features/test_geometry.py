import unittest

from src.features.geometry import fetch_blocklot_geometry, RecordNotFoundError


class TestFetchBlocklotGeometry(unittest.TestCase):
    def test_it_returns_something_when_it_exists(self):
        blocklot = "0001 001"
        self.assertEqual(len(fetch_blocklot_geometry(blocklot)), 244)

    def test_it_raises_when_it_doesnt_exists(self):
        blocklot = "0001001"
        with self.assertRaises(RecordNotFoundError):
            fetch_blocklot_geometry(blocklot)


if __name__ == "__main__":
    unittest.main()

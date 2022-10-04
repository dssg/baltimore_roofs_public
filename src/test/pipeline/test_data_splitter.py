import unittest

from src.pipeline.data_splitter import get_split_data
from src.test.features.test_helpers import (
    get_sample_blocklots,
    run_sample_query,
)


class TestDataSplitter(unittest.TestCase):
    def test_blocklot_rows_split(self):
        sample_blocklots = get_sample_blocklots()
        sample_dataset = run_sample_query(sample_blocklots)
        train, test = get_split_data(sample_dataset)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 2)


if __name__ == "__main__":
    unittest.main()

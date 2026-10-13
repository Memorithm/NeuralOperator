import tempfile
import unittest
from pathlib import Path

from scripts.qualify_rust_darcy import load_frozen_oracle


class FrozenRustDarcyOracleTests(unittest.TestCase):
    def test_original_baseline_is_loaded_directly(self):
        oracle = load_frozen_oracle(Path("src/neural_operator_reference/darcy.py"))
        self.assertTrue(callable(oracle.darcy_residual_2d))
        self.assertTrue(callable(oracle.solve_darcy_2d))

    def test_modified_oracle_fails_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hostile_oracle.py"
            source.write_text("raise AssertionError('oracle must not execute')\n")
            with self.assertRaisesRegex(ValueError, "oracle drift"):
                load_frozen_oracle(source)


if __name__ == "__main__":
    unittest.main()

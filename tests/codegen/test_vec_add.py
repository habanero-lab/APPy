import os
import ast
from pathlib import Path
import torch
import appy


def test_vec_add(tmp_path: Path):
    """Verify that APPy generates the expected Triton code for a simple add kernel."""
    # Path to the golden reference file
    golden_path = Path(__file__).parent / "golden" / "vec_add_triton.py"
    out_path = tmp_path / "vec_add_triton.py"

    # Define the kernel under test
    @appy.jit(backend="triton", dry_run=True, dump_code_to_file=out_path)
    def kernel_appy(a, b):
        c = torch.empty_like(a)
        for i in appy.prange(a.shape[0], simd=True):
            c[i] = a[i] + b[i]
        return c

    # Trigger APPy compilation (dry-run)
    kernel_appy(torch.ones(512), torch.ones(512))

    # Read generated and golden code
    generated = out_path.read_text()
    golden = golden_path.read_text()

    # Optional: normalize whitespace differences
    generated_norm = ast.dump(ast.parse(generated))
    golden_norm = ast.dump(ast.parse(golden))

    # Allow updating golden reference if environment variable is set
    UPDATE_GOLDEN = os.getenv("UPDATE_GOLDEN", "0") == "1"

    if UPDATE_GOLDEN:
        print(f"🔄 Updating golden file: {golden_path}")
        golden_path.write_text(generated)
    else:
        assert (
            generated_norm == golden_norm
        ), f"""
        ❌ Generated code does not match golden reference.

        --- Golden file: {golden_path}
        --- Generated file: {out_path}

        To update the golden file, run:
            UPDATE_GOLDEN=1 pytest {__file__}
        """


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

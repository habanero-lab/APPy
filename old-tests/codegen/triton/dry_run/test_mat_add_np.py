import os
import ast
from pathlib import Path
import numpy as np
import appy


def test_mat_add(tmp_path: Path):
    """Verify that APPy generates the expected Triton code for a simple add kernel."""
    # Path to the golden reference file
    golden_path = Path(__file__).parent / "golden" / "mat_add_triton.py"
    out_path = tmp_path / "mat_add_triton.py"

    # Define the kernel under test
    @appy.jit(backend="triton", dry_run=True, dump_code_to_file=str(out_path))
    def kernel_appy(a, b):
        c = np.empty_like(a)
        #pragma parallel for
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                c[i, j] = a[i, j] + b[i, j]
        return c

    # Trigger APPy compilation (dry-run)
    a = np.ones((100, 100))
    b = np.ones((100, 100))
    c = kernel_appy(a, b)
    assert np.allclose(c, a + b)

    # Read generated and golden code
    generated = out_path.read_text()
    golden = golden_path.read_text()

    if os.getenv("TRITON_INTERPRET", "0") == "1":
        golden = golden.replace('.to("cuda")', '.to("cpu")')

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

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path to import naive module
sys.path.insert(0, str(Path(__file__).parent.parent))
from naive import matmul_naive


class TestMatmulNaive:
    """Test suite for naive Triton matrix multiplication kernel."""

    @pytest.fixture(params=["float32", "float16"])
    def dtype(self, request):
        """Parametrize tests across different data types."""
        return getattr(torch, request.param)

    @pytest.fixture
    def device(self):
        """Use CUDA device for testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return "cuda"

    def test_small_square_matrix(self, device, dtype):
        """Test small square matrices."""
        M = N = K = 4
        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        c_triton = matmul_naive(a, b)
        c_torch = torch.matmul(a, b)

        # Use appropriate tolerance based on dtype and accumulation errors
        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5

        assert torch.allclose(
            c_triton, c_torch, rtol=rtol, atol=atol
        ), f"Mismatch: max diff = {(c_triton - c_torch).abs().max()}"

    def test_rectangular_matrices(self, device, dtype):
        """Test rectangular matrices with different dimensions."""
        test_cases = [
            (32, 16, 8),  # M > K > N
            (8, 16, 32),  # N > K > M
            (16, 32, 16),  # K > M = N
            (64, 32, 128),  # N > M > K
        ]

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5

        for M, K, N in test_cases:
            a = torch.randn((M, K), device=device, dtype=dtype)
            b = torch.randn((K, N), device=device, dtype=dtype)

            c_triton = matmul_naive(a, b)
            c_torch = torch.matmul(a, b)

            assert torch.allclose(
                c_triton, c_torch, rtol=rtol, atol=atol
            ), f"Mismatch for shape ({M}, {K}) @ ({K}, {N}): max diff = {(c_triton - c_torch).abs().max()}"

    def test_medium_matrices(self, device, dtype):
        """Test medium-sized matrices."""
        M, N, K = 128, 128, 128
        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        c_triton = matmul_naive(a, b)
        c_torch = torch.matmul(a, b)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5

        assert torch.allclose(
            c_triton, c_torch, rtol=rtol, atol=atol
        ), f"Mismatch: max diff = {(c_triton - c_torch).abs().max()}"

    def test_large_matrices(self, device):
        """Test larger matrices (float32 only for stability)."""
        M, N, K = 512, 512, 512
        dtype = torch.float32

        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        c_triton = matmul_naive(a, b)
        c_torch = torch.matmul(a, b)

        assert torch.allclose(
            c_triton, c_torch, rtol=1e-3, atol=1e-4
        ), f"Mismatch: max diff = {(c_triton - c_torch).abs().max()}"

    def test_identity_matrix(self, device, dtype):
        """Test multiplication with identity matrix."""
        N = 32
        a = torch.randn((N, N), device=device, dtype=dtype)
        identity = torch.eye(N, device=device, dtype=dtype)

        # A @ I = A
        c_triton = matmul_naive(a, identity)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5

        assert torch.allclose(
            c_triton, a, rtol=rtol, atol=atol
        ), f"Identity test failed: max diff = {(c_triton - a).abs().max()}"

    def test_zero_matrix(self, device, dtype):
        """Test multiplication with zero matrix."""
        M, N, K = 32, 32, 32
        a = torch.randn((M, K), device=device, dtype=dtype)
        zeros = torch.zeros((K, N), device=device, dtype=dtype)

        c_triton = matmul_naive(a, zeros)
        expected = torch.zeros((M, N), device=device, dtype=dtype)

        assert torch.allclose(
            c_triton, expected, rtol=1e-5, atol=1e-6
        ), f"Zero matrix test failed: max value = {c_triton.abs().max()}"

    def test_ones_matrix(self, device, dtype):
        """Test multiplication with ones matrix."""
        M, N, K = 16, 16, 16
        a = torch.ones((M, K), device=device, dtype=dtype)
        b = torch.ones((K, N), device=device, dtype=dtype)

        c_triton = matmul_naive(a, b)
        # Result should be all K's
        expected = torch.full((M, N), float(K), device=device, dtype=dtype)

        rtol = 1e-2 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5

        assert torch.allclose(
            c_triton, expected, rtol=rtol, atol=atol
        ), f"Ones matrix test failed: max diff = {(c_triton - expected).abs().max()}"

    def test_incompatible_shapes(self, device, dtype):
        """Test that incompatible shapes raise an error."""
        a = torch.randn((16, 8), device=device, dtype=dtype)  # Power of 2 sizes
        b = torch.randn((16, 32), device=device, dtype=dtype)  # Incompatible: 8 != 16

        with pytest.raises(AssertionError):
            matmul_naive(a, b)

    def test_output_shape(self, device, dtype):
        """Test that output has correct shape."""
        M, K, N = 64, 32, 64  # Power of 2 sizes
        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)

        c = matmul_naive(a, b)

        assert c.shape == (M, N), f"Expected shape ({M}, {N}), got {c.shape}"
        assert c.dtype == dtype, f"Expected dtype {dtype}, got {c.dtype}"
        assert c.device == a.device, f"Expected device {a.device}, got {c.device}"

    def test_numerical_stability(self, device):
        """Test numerical stability with large and small values."""
        M, N, K = 64, 64, 64
        dtype = torch.float32

        # Mix of large and small values
        a = torch.randn((M, K), device=device, dtype=dtype) * 1e3
        b = torch.randn((K, N), device=device, dtype=dtype) * 1e-3

        c_triton = matmul_naive(a, b)
        c_torch = torch.matmul(a, b)

        # More relaxed tolerance for this test
        assert torch.allclose(
            c_triton, c_torch, rtol=1e-3, atol=1e-3
        ), f"Numerical stability test failed: max diff = {(c_triton - c_torch).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

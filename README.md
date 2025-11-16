# JAX Tutorial

This project demonstrates JAX usage for GPU-accelerated numerical computing.

## Installation

### Installing JAX for GPU with uv

We recommend using [`uv`](https://github.com/astral-sh/uv), a fast Python package manager, to install JAX with GPU support.

#### Step 1: Install uv

If you haven't already installed `uv`, follow the official installation guide:

**[uv Installation Guide](https://github.com/astral-sh/uv)**

#### Step 2: Install JAX with GPU Support

Once `uv` is installed, you can install JAX with GPU support using:

```bash
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Or if you're using `uv` in a project:

```bash
uv add "jax[cuda12]"
```

**Note:** Make sure you have compatible NVIDIA GPU drivers and CUDA installed. For detailed installation instructions, system requirements, and troubleshooting, refer to the official JAX documentation:

**[JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)**

#### Step 3: Verify Installation

To verify that JAX is correctly installed and can access your GPU:

```python
import jax
print(jax.devices())
```

This should display your GPU device(s) if the installation was successful.

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [uv Documentation](https://github.com/astral-sh/uv)


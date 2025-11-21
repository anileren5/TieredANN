# Publishing QVCache to PyPI

This guide explains how to publish the QVCache Python package to PyPI (Python Package Index).

## Prerequisites

1. **PyPI Account**: Create accounts on:
   - Test PyPI: https://test.pypi.org/account/register/
   - Production PyPI: https://pypi.org/account/register/

2. **Install build tools**:
```bash
pip install build twine
```

## Step-by-Step Publishing Process

### 1. Prepare the Package

Make sure you have:
- Updated version number in `setup.py` and `pyproject.toml`
- Created a git tag for the version (optional but recommended)
- Updated `README.md` with proper documentation
- Tested the package builds correctly

### 2. Build Distribution Packages

From the `python/` directory:

```bash
cd python
python -m build
```

This creates:
- `dist/qvcache-<version>-py3-none-any.whl` (wheel)
- `dist/qvcache-<version>.tar.gz` (source distribution)

### 3. Check the Package

Before uploading, check the package:

```bash
# Check the package
twine check dist/*
```

This verifies the package is correctly formatted.

### 4. Upload to Test PyPI (Recommended First Step)

Test your package on Test PyPI first:

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# You'll be prompted for username and password
# Username: __token__
# Password: your test pypi API token (pypi-...)
```

Test installation from Test PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ qvcache
```

### 5. Upload to Production PyPI

Once tested, upload to production PyPI:

```bash
# Upload to PyPI
twine upload dist/*

# You'll be prompted for username and password
# Username: __token__
# Password: your pypi API token (pypi-...)
```

### 6. Verify Installation

After publishing, verify the package is available:

```bash
pip install qvcache
python -c "import qvcache; print(qvcache.__version__)"
```

## Using API Tokens (Recommended)

Instead of passwords, use API tokens:

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token (scope: entire account or specific project)
3. Copy the token (starts with `pypi-...`)
4. Use `__token__` as username and the token as password when uploading

For Test PyPI: https://test.pypi.org/manage/account/token/

## Version Management

### Updating Version

Update version in:
- `setup.py`: `version="0.1.0"`
- `pyproject.toml`: `version = "0.1.0"`
- `__init__.py`: `__version__ = "0.1.0"`

Follow [Semantic Versioning](https://semver.org/):
- MAJOR.MINOR.PATCH (e.g., 1.0.0, 1.0.1, 1.1.0, 2.0.0)

### Git Tagging (Recommended)

Tag your releases:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Automated Publishing with GitHub Actions (Optional)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        working-directory: ./python
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        working-directory: ./python
        run: twine upload dist/*
```

Store your PyPI API token in GitHub Secrets:
1. Go to repository Settings → Secrets → Actions
2. Add secret named `PYPI_API_TOKEN` with your token

## Manual Publishing Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md (if you have one)
- [ ] Test build: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Test on Test PyPI
- [ ] Create git tag
- [ ] Publish to PyPI
- [ ] Verify installation: `pip install qvcache`

## Troubleshooting

### "Package already exists"

The version is already published. Increment the version number.

### "Invalid package"

Run `twine check dist/*` to see specific errors.

### Authentication errors

- Use API tokens instead of passwords
- Make sure to use `__token__` as username
- For Test PyPI, use test.pypi.org token

### Build errors

- Ensure all dependencies are installed
- Check that CMake builds successfully locally
- Verify all required files are included in MANIFEST.in


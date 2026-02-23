# Contributing to Cooling Tower Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Pull Request Process](#pull-request-process)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/cooling-tower-detection.git`
3. Add upstream remote: `git remote add upstream https://github.com/ORIGINAL_OWNER/cooling-tower-detection.git`
4. Create a branch: `git checkout -b feature/your-feature-name`

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in [Issues](https://github.com/yourusername/cooling-tower-detection/issues)
- If not, create a new issue with:
  - Clear, descriptive title
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version, CUDA version)
  - Screenshots if applicable

### Suggesting Enhancements

- Open an issue with the "enhancement" label
- Clearly describe the feature and its benefits
- Include example use cases

### Contributing Code

Areas where contributions are especially welcome:

- **Documentation**: Improving guides, adding examples
- **Testing**: Writing unit tests, integration tests
- **Features**: Adding new detection algorithms, export formats
- **Performance**: Optimization, memory management
- **Bug Fixes**: Addressing reported issues

## Development Setup

### 1. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

### 2. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

This will automatically format and lint code before commits.

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable and function names
- Add docstrings to all functions, classes, and modules
- Keep functions focused and under 50 lines when possible

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### Type Hints

Use type hints for function parameters and returns:

```python
def process_image(image_path: str, conf_threshold: float = 0.4) -> np.ndarray:
    pass
```

### Code Formatting

Format code with black before committing:

```bash
black src/ scripts/ notebooks/
```

### Linting

Check code with flake8:

```bash
flake8 src/ scripts/ --max-line-length=100
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_detection.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`

Example:

```python
def test_bounding_box_expansion():
    """Test that bounding boxes are correctly expanded."""
    from src.utils import expand_bounding_box
    
    result = expand_bounding_box(10, 10, 50, 50, (100, 100), 0.25, 10)
    assert result == (0, 0, 70, 70)
```

## Pull Request Process

### Before Submitting

1. **Update Documentation**: Add/update docstrings, README sections
2. **Add Tests**: Ensure new code is tested
3. **Format Code**: Run `black` and `flake8`
4. **Test Locally**: Run full test suite
5. **Update CHANGELOG**: Note your changes (if applicable)

### Submitting the PR

1. Push to your fork: `git push origin feature/your-feature-name`
2. Open a Pull Request on GitHub
3. Fill out the PR template with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if UI changes)

### PR Title Format

Use conventional commits format:

- `feat: Add new visualization function`
- `fix: Resolve CUDA memory leak`
- `docs: Update installation guide`
- `test: Add tests for segmentation module`
- `refactor: Simplify detection pipeline`

### Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

## Git Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `test/description` - Testing improvements

### Commit Messages

Write clear, descriptive commit messages:

```
feat: Add batch processing for segmentation

- Implement multi-threaded segmentation
- Add progress bars
- Reduce memory usage by 30%
```

## Project Structure

```
cooling-tower-detection/
├── src/                    # Main source code
│   ├── detection.py       # YOLO detection
│   ├── segmentation.py    # SAM2 segmentation
│   ├── utils.py           # Utilities
│   └── visualization.py   # Visualization tools
├── scripts/               # Command-line scripts
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks
└── docs/                  # Documentation
```

## Questions?

- Open an issue with the "question" label
- Check existing discussions
- Email the maintainers

Thank you for contributing! 🎉

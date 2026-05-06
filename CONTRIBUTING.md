# Contributing to arcllm

Thank you for your interest in contributing to arcllm! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Prerequisites

- Python 3.12 or later
- Git

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/arcllm/arcllm.git
   cd arcllm
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite:
   ```bash
   pytest tests/ --ignore=tests/integration
   ```

4. Run linting and type checks:
   ```bash
   ruff check arcllm tests
   ruff format --check arcllm tests
   mypy arcllm --strict
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 and PEP 484 (type hints)
- Use `from __future__ import annotations` in all files
- All public functions must have type hints
- Use `__slots__` for dataclasses when possible
- Run `ruff format` before committing

### Testing

- **Unit tests are required** for all new features and bug fixes
- Tests must be offline (use mocks/fixtures, no real API calls)
- Aim for >80% code coverage
- Run the full test suite before submitting PRs:
  ```bash
  pytest tests/ --ignore=tests/integration --cov=arcllm --cov-fail-under=80
  ```

### Adding a New Provider

See [docs/ADDING_A_PROVIDER.md](docs/ADDING_A_PROVIDER.md) for detailed instructions.

Quick checklist:
1. Create `providers/newprovider_adapter.py`
2. Implement the `Adapter` protocol
3. Register in `providers/base.py`
4. Add pricing to `pricing/tables.py`
5. Add capabilities to `capabilities/tables.py`
6. Add provider documentation to `docs/providers/`
7. Write tests in `tests/providers/`

### Key Invariants

1. **Curated runtime deps**: only `httpx`, `aiohttp`, `msgspec`, `orjson`. New runtime deps require an issue + maintainer approval.
2. **Response structure compatibility**: All responses must follow the LiteLLM / OpenAI format
3. **Usage from providers**: Never count tokens ourselves; use provider-reported usage
4. **Tool call format**: Must follow OpenAI format with `function.arguments` as JSON string

## Pull Request Process

1. **Fork and create a branch** for your changes
2. **Make your changes** following the guidelines above
3. **Add tests** for any new functionality
4. **Update documentation** if needed
5. **Run all checks** locally:
   ```bash
   ruff check arcllm tests
   ruff format arcllm tests
   mypy arcllm --strict
   pytest tests/ --ignore=tests/integration --cov=arcllm
   ```
6. **Submit a pull request** with a clear description of your changes

### PR Title Format

Use conventional commit style:
- `feat: Add support for X provider`
- `fix: Handle edge case in SSE parsing`
- `docs: Update provider documentation`
- `test: Add streaming stress tests`
- `refactor: Simplify adapter base class`

### PR Description

Please include:
- Summary of changes
- Motivation/reason for the change
- Testing performed
- Any breaking changes

## Reporting Issues

When reporting bugs, please include:
- Python version
- arcllm version
- Full error traceback
- Minimal reproduction code
- Expected vs actual behavior

## Feature Requests

Feature requests are welcome! Please:
- Check existing issues first
- Describe the use case clearly
- Consider if it fits the project goals (lightweight, minimal dependencies)

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## Questions?

- Open a GitHub issue for questions
- See [AGENTS.md](AGENTS.md) for AI agent development guidelines

## License

By contributing to arcllm, you agree that your contributions will be licensed under the Apache License 2.0.

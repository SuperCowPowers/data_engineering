# Data Engineering

A shared workspace for code, experiments, and data pipelines.

## Prerequisites

### 1. Homebrew

The macOS package manager — used to install everything below. If you don't have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Git

macOS ships git with Apple's Command Line Tools:

```bash
xcode-select --install     # installs git + compilers (skip if already present)
git --version              # verify
```

Optional: `brew install git` for a newer version than Apple's.

### 3. uv

[uv](https://docs.astral.sh/uv/) manages the Python interpreter, the virtual
environment, and packages — all in one fast tool.

```bash
brew install uv
```

## Getting started

```bash
git clone git@github.com:SuperCowPowers/data_engineering.git
cd data_engineering
uv sync          # creates .venv, installs the right Python + all dependencies
```

`uv sync` reads `pyproject.toml` and `.python-version`, downloads Python 3.13 if
you don't have it, and builds the environment. That's the whole setup.

## Running code

```bash
uv run python path/to/script.py                # run a script
```

Prefer the classic workflow? Activate the env and use `python` directly:

```bash
source .venv/bin/activate
python path/to/script.py
```

## Editor setup

Point your editor at the project's `.venv` so it uses the right interpreter and
finds the installed packages.

**PyCharm**
1. Settings → Project → **Python Interpreter** → *Add Interpreter* → *Add Local*.
2. Choose **Existing** and select `.venv/bin/python` in the project.
   (PyCharm 2024.2+ also has a native **uv** option that does this for you.)

**VS Code**
1. Install the **Python** extension.
2. *Command Palette* (⌘⇧P) → **Python: Select Interpreter** → pick the one under
   `.venv`. VS Code usually auto-detects it on open.


## Tests

```bash
uv run pytest            # run tests
```

## Contributing (pull request flow)

```bash
git checkout -b my-feature
# ... make changes, commit ...
git push -u origin my-feature
```

Then open a pull request on GitHub for review.

## Layout

```
data_engineering/
├── pyproject.toml          # project, dependencies, tool config
├── .python-version         # pinned Python version
├── uv.lock                 # exact resolved versions (created by `uv sync`)
├── src/data_engineering/   # importable, shared code
├── tests/                  # pytest tests
└── project_1/              # example project: reading CSVs with pandas
```

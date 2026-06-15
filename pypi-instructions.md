# PyPI Upload Instructions

## Blocking Issues (must fix before upload)

### 1. `voyageai` is a git URL — PyPI rejects this

PyPI does not allow VCS dependencies in `[project.dependencies]`. The current entry:

```toml
"voyageai @ git+https://github.com/voyage-ai/voyageai-python.git@86422e15dab9ce512437b90594e8b26d20bbf259"
```

must be replaced with a proper version from PyPI. `voyageai` 0.4.0 is available. Check the repo changelog to confirm which release that commit corresponds to, then update to something like:

```toml
"voyageai (>=0.4.0,<0.5.0)"
```

### 2. Remove `dotenv` from runtime dependencies

`dotenv` (`python-dotenv`) is only imported in `spruceup_pipeline.py` — the user-authored entry point that lives outside `src/`. Nothing inside `src/spruceup/` imports it. It is not a library dependency; users bring their own if they need it.

Remove this line from `[project.dependencies]`:
```toml
"dotenv (>=0.9.9,<0.10.0)",
```

### 3. Remove `cohere` from dev dependencies

`cohere` is listed in both `[project.dependencies]` and `[dependency-groups] dev`. Since it is used in `src/spruceup/connectors/embedders/cohere.py`, it belongs only in runtime deps. Remove it from the `dev` group:

```toml
[dependency-groups]
dev = [
    "pytest (>=9.0.3,<10.0.0)",
    "pytest-asyncio (>=1.3.0,<2.0.0)",
    # remove cohere from here
]
```

---

## Missing Metadata (PyPI will show blanks)

### 4. Fill in `description`

```toml
description = "A document ingestion daemon that watches sources, embeds chunks, and keeps a vector store in sync."
```

### 5. Write README.md

`README.md` is currently empty. PyPI uses it as the package landing page. Write it before building.

### 6. Point `[project]` at the README

```toml
readme = "README.md"
```

### 7. Add a license

Create a `LICENSE` file at the repo root (e.g. MIT), then add to `[project]`:

```toml
license = {file = "LICENSE"}
```

---

## Recommended Additions

### 8. Project URLs

```toml
[project.urls]
Repository = "https://github.com/LS-2603-Capstone-team-5/<repo-name>"
```

### 9. Classifiers

```toml
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
```

---

## Upload Steps

Once all the above is done:

```bash
# Build source dist + wheel into dist/
poetry build

# Upload (prompts for credentials)
poetry publish
```

**Credentials:** Create an account at https://pypi.org, then generate an API token under Account Settings → API tokens. When prompted, use `__token__` as the username and the token as the password.

**Dry run on TestPyPI first (recommended):**

```bash
pip install twine
twine upload --repository testpypi dist/*
```

This lets you verify the metadata and README render correctly on https://test.pypi.org before touching real PyPI.

---

## Note on Python Version Constraint

```toml
requires-python = ">=3.14,<3.15"
```

This is very narrow. A scan of the entire codebase found no Python 3.14-specific syntax — no t-strings, no `type` alias statements, no `match` statements, no PEP 695 type parameters. The highest version requirement from the actual code is `LiteralString` (used in `connectors/targets/pgvector.py`), which was added in Python 3.11.

Widen this to `>=3.12`, which covers everything you're likely to reach for naturally going forward (`match` statements from 3.10+, `type` aliases from 3.12+) while opening the package up to everyone not on a brand-new Python. The only thing `>=3.14` uniquely buys is t-strings (PEP 750), which are quite niche.

```toml
requires-python = ">=3.12"
```

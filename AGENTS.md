# AGENTS.md — how this project is built

The engineering workflow: how the code is written, checked and run. Read it
first to be up to speed. Code-style rules live in the `code-style` rule
(`.claude/rules/code-style.md`), auto-loaded when a Python file is read.
Repo-specific commands (how to run things) live in the README.

## Toolchain

| Concern                  | Tool                                       |
| ------------------------ | ------------------------------------------ |
| dependencies & packaging | uv (src layout)                            |
| lint + format            | Ruff (`ruff check`, `ruff format`)         |
| type checking            | mypy (typed `tools/`; src typing TBD)   |
| tests                    | pytest (golden characterization tests)     |
| config                   | Hydra structured configs                   |
| commit gate              | pre-commit                                 |

## Expert submodules are third-party

`src/myotrainer/experts/{Kinesis, myochallenge-lattice, Myohuman}` are git
submodules. **They are excluded from every hook, lint, format and type check**
(a global `exclude` in `.pre-commit-config.yaml` plus `extend-exclude` in
pyproject). Never reformat them here; the Myohuman submodule is edited only in
its own standalone clone and pulled in via git.

## Quick commands

| Task            | Command                             |
| --------------- | ----------------------------------- |
| sync env        | `uv sync`                           |
| install hook    | `uv run pre-commit install`         |
| lint gate       | `uv run pre-commit run --all-files` |
| tests           | `uv run pytest`                     |
| coverage        | `uv run pytest --cov`               |
| train           | `uv run python -m myotrainer.train_myotrainer` |

The `pre-commit` gate stays the single source of truth for lint/format/types.

## Code style

The full rules live in `.claude/rules/code-style.md` — a path-scoped rule that
auto-loads when a Python file is read. In short: banners, length ladders,
four-block absolute imports, trailing commas, 80-column lines. The mechanical
rules are **enforced by pre-commit** through the `check-*` hooks; the rest is
human/agent judgment. Each violation message cites the `code-style §N` rule it
breaks.

## pre-commit is the gate

Everything runs before each commit (`pre-commit install`), and CI runs the same
set — `.pre-commit-config.yaml` is the single source of truth.

Hooks, in order:

- **hygiene** — trailing-whitespace, end-of-file, check-yaml, typos
- **format** — `add-trailing-comma` → `ruff-format`
- **lint** — `ruff check`
- **types** — `mypy`
- **style as code** — `check-imports`, `check-banners`, `check-device`.

**`pre-commit` only sees git-tracked files** — `git add` new files before
trusting the gate. Run all: `pre-commit run --all-files`. Run one:
`pre-commit run ruff-check`.

## Linting & type checking

- **ruff** — `ruff check` (rules `E, F, UP, B, SIM, C4, PT`; isort `I` is off,
  imports are ordered by length — code-style §6) and `ruff format`.
- **mypy** — runs over the typed `tools/`. Annotating the legacy `src/`
  (hundreds of implicit-Optional / attr-defined findings) is a follow-up.

## Tests — two stages

- **Stage 1 (default)** — CPU-only, deterministic golden characterization
  tests over the parsers, tokenizers, policies and trainer helpers; no wandb /
  network / GPU / submodule assets. Runs on every commit and in CI.
- **Stage 2** — heavy/accelerator tests marked `@pytest.mark.stage2` (e.g.
  device-agnosticism: policy forward on GPU vs CPU). They run **only** when a
  CUDA/MPS accelerator is present, or when forced with `RUN_STAGE2=1`;
  otherwise skipped. On a box with no accelerator they skip gracefully.

Run stage 1: `pytest`. Run stage 2 (needs a GPU): `RUN_STAGE2=1 pytest`.
Regenerate goldens only on an intentional behavior change
(`uv run python tests/golden/generate_goldens.py`).

## CI

Every push / PR runs `pre-commit run --all-files` + `pytest` on cheap CPU
runners; submodules are NOT checked out (`submodules: false`). Docs-only pushes
are skipped. CI invokes `pre-commit`, not the tools directly.

## Commits

- Run pre-commit before committing; fix what the `check-*` hooks flag.
- `git add` new files before trusting the gate.
- Conventional prefixes (`feat:`, `fix:`, `docs:`, `style:`, `chore:`, `test:`),
  one logical change per commit.

For code review, use the built-in `/code-review`. For test coverage + mutation
quality of a change, delegate the read-only `test-coverage-auditor` subagent.

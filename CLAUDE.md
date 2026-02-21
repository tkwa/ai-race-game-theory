## Project Information

Game-theoretic model of AI safety race dynamics between competing nations.

## Project Structure


- `docs/` - Model descriptions and write-ups
  - `docs/report_spec.md` contains the mostly-human-written spec that the report should follow, including the model definition. This should always be kept up to date with the report. If Claude makes a change to the report due to the human's request that contradicts the current spec wording, the spec should be updated; if this is a major change the human should be asked for approval.
- `README.md` - Will ultimately be the external output of this project, a blog post.
- `report.ipynb` - The internal output of this project.
- `src/report/` - Code used in report.ipynb
- `src/ai_race_game_theory/` - Core model and analysis code
- `tests/` - pytest tests
- `plots/` - Generated plots

Agents working on separate PRs should be in separate worktrees. For a project `~/src/foo`, the main worktree should be at `~/src/foo/default` (generally kept on the main branch), and branch worktrees at `~/src/foo/branch_name`.

## Code Quality Standards

- No code duplication; type hints everywhere; `uv run ruff format` (line length 100); `uv run ruff check` for linting.
- Concise one-line docstrings preferred; comments explain WHY not WHAT.
- Private functions use leading underscore; import modules not functions (Google style).
- No circular dependencies; higher-level modules depend on lower-level.

## Linting/Type Config (Critical)

- NEVER modify linting/type rules in `pyproject.toml` without explicit permission.
- You MAY ignore in plotting/throwaway files. In other files, do NOT use `# type: ignore` without justification; fix code, don't silence checkers.

## Docstrings

**Simple functions (<20 lines) get one-line docstrings. Period.**

Skip Args/Returns/Examples if type hints make it obvious. Don't repeat what the function signature already says.

## Comments - Code Clarity Over Comments

**Prefer improving code clarity over leaving comments. Comments are a last resort.**

Before adding a comment, ask: "Can I make this code self-explanatory instead?"

**When to add comments (rare):**
- Non-obvious WHY that can't be expressed in code
- Important timing/ordering constraints
- Known limitations or edge cases

## Early Returns (Reduce Nesting)

- Use early `return`/`continue` to keep main logic at top indentation level.
- Avoid pyramid of doom; each guard clause should be simple, independent.

## Error Handling

- Use specific exception classes, not generic `Exception`.
- Fail fast with clear errors rather than silently degrading.

## Input Validation Over Defensive Programming

- Validate inputs early, then trust them downstream.
- Don't try to "fix" invalid inputs—reject them.

## Development Commands

```bash
uv run ruff format .          # Format
uv run ruff check .           # Lint
uv run pytest tests/ -n auto  # Run tests (parallel)
```

## Before Returning to User (Critical)

- Must run: `uv run ruff format .`, `uv run ruff check .`, `uv run pytest tests/`
- Never say "done" without running these first.

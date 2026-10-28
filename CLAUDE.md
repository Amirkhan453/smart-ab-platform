# Git workflow

- Work only on the `dev` branch.
- Push changes only to `origin/dev`.
- Never commit to or push directly to `main`.
- Before editing, fetch `origin/dev` and check for local changes.
- Do not overwrite uncommitted work. Stop and report a conflict instead.
- Run `.venv/bin/python -m pytest -q` before committing.
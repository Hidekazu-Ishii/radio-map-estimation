---
applyTo: "**/*.py,**/*.ipynb"
---

For Python files, review with these priorities:

- Check whether public functions and important internal interfaces have clear type hints.
- Prefer `dataclass` or other typed structures over untyped nested dictionaries when the schema is stable.
- Prefer explicit configuration over scattered constants or hidden globals.
- Prefer `pathlib` over ad-hoc string path handling.
- Keep core logic separate from CLI, notebook glue, plotting, and file I/O.
- Watch for mutable shared state, hidden side effects, and implicit dependencies.
- Check numerical code for dtype issues, divide-by-zero risk, NaN/Inf propagation, unstable normalization, tolerance mistakes, and shape mismatches.
- For ML code, check seed control, dataset split integrity, preprocessing consistency, metric correctness, checkpoint/config traceability, and reproducible evaluation.
- For new behavior or bug fixes, add/adjust tests (prefer `pytest`).
- Prefer readability over clever one-liners.

If Python comments or docstrings contain Japanese, ask to rewrite them in English and suggest concrete wording.
Examples of good rewrite style:
- "学習率を更新する" -> "Update the learning rate."
- "訓練データを読み込む" -> "Load the training data."
- "外れ値を除去する" -> "Remove outliers."

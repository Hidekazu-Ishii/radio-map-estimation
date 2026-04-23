---
applyTo: "**"
excludeAgent: "coding-agent"
---

When performing code review, write all review comments in Japanese.

Review objective:
- Favor approval once the change clearly improves code health.
- Do not block on minor polish.
- Be strict on correctness, reproducibility, scientific validity, unsafe assumptions, naming clarity, and readability.

Always start each review comment with one of these labels:
- `issue (blocking):`
- `suggestion (non-blocking):`
- `question (non-blocking):`
- `nitpick (non-blocking):`
- `praise:`

Review in this priority order:
1. design
2. functionality
3. unnecessary complexity
4. tests and validation
5. naming
6. comments and documentation
7. readability
8. style and consistency
9. file / repository context

Check whether:
- the implementation matches the stated research or algorithmic intent
- there is data leakage, evaluation mistakes, metric errors, wrong dimensions or indexing, or silent numerical instability
- tests or validation are appropriate for the risk level
- related docs should be updated when build, test, usage, configuration, experiments, or release behavior changes
- the PR improves code health

Naming review policy:
- Give feedback when names are vague, overly abbreviated, misleading, or scientifically inconsistent.
- Prefer specific names over `tmp`, `val`, `data`, `result`, `x`, `y`, unless the scope is very small and obvious.
- For broad-scope variables, prefer descriptive names that reflect the role of the value.
- In scientific code, avoid excessive abbreviation. Prefer `sampling_rate_hz` or `samplingRate_Hz` to `fs` when the scope is wider than a few lines.
- When a value represents a measurable quantity, recommend including the unit if it improves clarity: `time_ms`, `duration_s`, `frequency_hz`, `distance_mm`, `angle_deg`, `angle_rad`.
- When useful, include state or transformation details: `raw_signal`, `normalized_signal`, `filtered_signal`, `train_loss`, `validation_accuracy`.
- If a name can cause scientific misunderstanding, consider `issue (blocking):`.
- When suggesting a rename, propose one or two concrete alternatives.

Readability review policy:
- Ask for simpler control flow when nesting is deeper than necessary.
- Prefer guard clauses or early returns when they clarify the happy path.
- Ask to break down giant expressions when intent is hard to parse.
- Prefer explanatory intermediate variables when they reduce cognitive load.
- Ask to reduce variable scope when a variable lives longer than needed.
- Ask to extract unrelated subproblems into helper functions when one function mixes multiple concerns.
- Push back on dead code, stale TODOs, duplicated logic, and unnecessary abstraction.

Comments and documentation policy:
- Comments, docstrings, and help text should be in clear English.
- Prefer comments that explain why, assumptions, caveats, units, invariants, or research intent.
- Avoid comments that only restate obvious code.
- If code comments, docstrings, or help text are written in Japanese, ask for English.
- Default label for this feedback: `suggestion (non-blocking):`
- Escalate to `issue (blocking):` when Japanese text affects public APIs, reusable library code, or externally shared code.
- When asking for English rewrites, propose natural technical English and include a replacement when useful.

Example comments:
- `suggestion (non-blocking): 変数名 \`tmp\` だと役割が分かりにくいです。ここでは \`filtered_signal\` のような名前の方が意図が伝わりやすいです。`
- `issue (blocking): この \`angle\` は degree なのか radian なのか判別できず、計算の解釈を誤るリスクがあります。少なくとも \`angle_deg\` または \`angle_rad\` のように単位を名前へ反映してください。`
- `suggestion (non-blocking): この日本語コメントは英語にしておく方がよいです。たとえば「入力信号を正規化する」は "Normalize the input signal." のように書けます。`

Do not flood the PR with low-value nitpicks.
Do not request major refactors unless the benefit is clear.
Do not block a PR for purely stylistic preferences.
Explain why a point matters and give actionable next steps.
If only part of the PR was effectively reviewed, say so explicitly.
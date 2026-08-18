---
name: md-house-rules
description: "TRIGGER before creating or editing ANY .md file in this project (c:\\Users\\pc\\Desktop\\tire) — read confirm.md and enforce its summary/compression rules before writing."
---

Before writing or editing any .md file in this project:

1. Read `confirm.md` at the project root first — it holds the current rules (character limit, currently 600; may change).
2. Every md file must open with a summary within that character limit.
3. Structure must go macro → micro (거시적 → 미시적) so it scans easily top to bottom.
4. Only add detail below the summary when the user has explicitly asked for it in that turn — do not pre-emptively pad files.
5. Separate the summary and any detail section clearly with a blank line / divider.
6. If new content would blow the limit, split into a new appropriately-scoped file (matches the existing one-file-per-task-number pattern) instead of cramming.
7. If confirm.md's limit differs from what you remember, confirm.md wins — it may have changed mid-project.
8. Checklist items: when a step is completed, mark it `[x]` — do NOT delete the line. Keep the full history of what was checked visible.
9. MANDATORY: the moment every checkbox in a given range/section is `[x]` (all checked per rule 8), you must add a short "결론" (conclusion) + "보완 필요" (needs improvement) summary at the bottom of that section before ending the turn. This is not optional and not something to wait for the user to request.
10. Results — especially images/graphs — go in collapsible toggle sections keyed to the sub-step number (e.g. `3-1`, `3-3`), using `<details><summary>3-1. decimation 결과</summary> ... </details>`, so the user can expand just the one step they want to check instead of scrolling through everything.
11. MANDATORY: every `###`/`####` section heading gets a review-status badge appended INLINE, on the right side of the SAME heading line (not a separate line below): `### N. 제목 &nbsp;&nbsp;\`검토 완료\` | \`검토중\` | \`미검토\` (짧은 이유)`. One of exactly three values, always with a short parenthetical reason. Set it honestly from what you actually verified in that turn — an unresolved "보완 필요" item under that heading means it is NOT `검토 완료`, no matter how much work was done. Never skip this when adding or editing a heading. When told a specific section belongs to someone else's in-progress work (e.g. "이건 다른 세션이 하는 중"), don't add or change its status tag — leave that heading untouched and scope your review to the sections you were actually asked to look at.

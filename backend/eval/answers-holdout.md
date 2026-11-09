# Answers, holdout split (129 questions)

Integrity check: 0 problems. The stored answers are re-parsed from the recorded responses and every cited text is matched against the sources sent. This proves the numbers come from real responses; it does not measure faithfulness, which the judge (a different model from the answerer) and the manual sample do.

## Grounding (answered questions)

| Metric | Value |
|---|---:|
| Answered | 94 |
| Answers with at least one citation | 100% |
| Share of answer text in cited segments | 48% |
| Answers citing a gold source | 78% |
| Answers naming a source that was not retrieved | 7% |
| Machloket answers naming every side (2) | 100% |
| Judge: claims fully supported | 89% |

## Scope filter, by strictness

| Strictness | Out of scope refused | Borderline refused | Not in library declined | Answerable refused |
|---|---:|---:|---:|---:|
| strict | 100% | 100% | 100% | 6% |
| balanced | 100% | 25% | 100% | 6% |
| lenient | 91% | 25% | 100% | 6% |

## Sensitive questions

- "Ask your rabbi first" shown first: 100% of 2
- Shown on non-sensitive questions: 4%
- Practical-halacha answers with a disclaimer: 95%

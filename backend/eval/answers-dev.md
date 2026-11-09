# Answers, dev split (95 questions)

Integrity check: 0 problems. The stored answers are re-parsed from the recorded responses and every cited text is matched against the sources sent. This proves the numbers come from real responses; it does not measure faithfulness, which the judge (a different model from the answerer) and the manual sample do.

## Grounding (answered questions)

| Metric | Value |
|---|---:|
| Answered | 65 |
| Answers with at least one citation | 100% |
| Share of answer text in cited segments | 49% |
| Answers citing a gold source | 75% |
| Answers naming a source that was not retrieved | 3% |
| Machloket answers naming every side (3) | 100% |
| Judge: claims fully supported | 91% |

## Scope filter, by strictness

| Strictness | Out of scope refused | Borderline refused | Not in library declined | Answerable refused |
|---|---:|---:|---:|---:|
| strict | 100% | 100% | 80% | 5% |
| balanced | 100% | 100% | 80% | 5% |
| lenient | 94% | 100% | 80% | 5% |

## Sensitive questions

- "Ask your rabbi first" shown first: 100% of 3
- Shown on non-sensitive questions: 0%
- Practical-halacha answers with a disclaimer: 100%

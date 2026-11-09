# Sefaria lookup, holdout split (21 questions about texts outside the library)

Integrity check: 0 problems (stored answers re-parsed from the recorded responses; every cited text matched against the sources sent).

| Questions | Gold passage fetched | Gold passage sent to the model | Answered | Answers citing gold |
|---|---:|---:|---:|---:|
| named (13) | 13 of 13 | 13 of 13 | 13 of 13 | 13 of 13 |
| implicit (5) | 5 of 5 | 5 of 5 | 5 of 5 | 5 of 5 |
| topic (1) | 1 of 1 | 1 of 1 | 1 of 1 | 1 of 1 |
| **all (19)** | 19 of 19 | 19 of 19 | 19 of 19 | |

- Places that do not exist, declined rather than answered: 1 of 2
- Answers naming a source that was not sent: 15%
- Judge: claims fully supported by the sources sent: 90% (20 answers)
- Licenses of every source sent: CC-BY, CC-BY-SA, CC0, Public Domain
- Sefaria requests per question: median 11
- Fetching (questions where Sefaria was really asked, 21): median 3.81 s, max 7.41 s. The planner (median 1.39 s) runs while the library is searched.

## The library's holdout questions with the lookup on (129)

- Answered with the lookup off: 94. Of these, 11 now also get a fetched source (q001, q011, q012, q029, q091, q092, q095, q116, q126, q133, q154); 83 of the other 83 send a byte-identical request, so their recorded answers stand.
- Not in the library (2): now answered from Sefaria: n000, n001.
- Declined before, answered now: q013, q014, q050, q100, q145, n000, n001.
- A gold source the library had sent, no longer sent: q095; a gold source sent now that was not before: q013, q100, q145.

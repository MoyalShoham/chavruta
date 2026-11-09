# Sefaria lookup, dev split (24 questions about texts outside the library)

Integrity check: 0 problems (stored answers re-parsed from the recorded responses; every cited text matched against the sources sent).

| Questions | Gold passage fetched | Gold passage sent to the model | Answered | Answers citing gold |
|---|---:|---:|---:|---:|
| named (15) | 15 of 15 | 15 of 15 | 15 of 15 | 14 of 15 |
| implicit (3) | 3 of 3 | 3 of 3 | 3 of 3 | 3 of 3 |
| topic (4) | 3 of 4 | 2 of 4 | 3 of 4 | 2 of 3 |
| **all (22)** | 21 of 22 | 20 of 22 | 21 of 22 | |

- Places that do not exist, declined rather than answered: 1 of 2
- Answers naming a source that was not sent: 5%
- Judge: claims fully supported by the sources sent: 92% (22 answers)
- Licenses of every source sent: CC-BY, CC-BY-SA, CC0, Public Domain
- Sefaria requests per question: median 0
- Fetching (questions where Sefaria was really asked, 0): median None s, max None s. The planner (median 0.02 s) runs while the library is searched.

## The library's dev questions with the lookup on (95)

- Answered with the lookup off: 65. Of these, 6 now also get a fetched source (q003, q017, q037, q070, q079, n003); 59 of the other 59 send a byte-identical request, so their recorded answers stand.
- Not in the library (5): now answered from Sefaria: n002, n003, n004, n005.
- Declined before, answered now: q083, q084, q110, n002, n004, n005.
- A gold source the library had sent, no longer sent: none; a gold source sent now that was not before: q083, q084.

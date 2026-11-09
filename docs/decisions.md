# Design decisions

## 1. Citations come from the API, not from the model's prose

**Chosen:**
- Retrieved passages are sent as `search_result` content blocks with citations enabled.
- The API returns each sentence with the source block it cites, and the cited text is that block,
  verbatim.
- One search result is one reference (a verse, a Talmud segment, a seif). Its content blocks are
  the Hebrew text and its translation, so a citation also says which language was cited.
- Text the model writes without a citation is shown with a dotted underline and counted in the
  evaluation ("share of answer text in cited segments").

**Rejected:**
- Asking the model to write `[1]` markers, which can point anywhere and quote loosely.
- Structured JSON output with a citations field. It cannot be combined with API citations, and it
  relies on the model to report its own sources honestly.
- Plain-text `document` blocks with character offsets. They allow partial-sentence citations of a
  verse, where the unit of meaning is the verse itself.

**Why:** a citation the API attaches cannot name a passage that was not sent, and its quote cannot
be paraphrased.

That is a property of the API, so the evaluation's re-check of it is an *integrity* check. It
proves the reported numbers come from real recorded responses, not that a claim is faithful to
its source. Faithfulness is measured separately, by a judge that is a different model from the
answerer, and by a manual sample (decision 14).

## 2. Guardrails live in code, around the model

**Chosen:**
- The scope decision is made before the answering model runs. Out of scope, manipulation,
  unclassifiable and "no relevant source" never reach it.
- The safety notice, "ask your rabbi first" and the not-a-posek disclaimer are added by code, in a
  fixed order: safety, then "ask your rabbi", then the answer, then the disclaimers.
- After the answer, a detector flags any work or place the model names that was not retrieved
  (decision 3).

**Rejected:** putting these rules only in the system prompt. The prompt states them too, but a model
that forgets one sentence must not be able to drop a safety notice.

**What remains prompt-only:** presenting a machloket as a machloket. The evaluation measures it:
for the questions whose sources record a dispute, it checks that the answer names every side.

## 3. Never cite what was not retrieved: enforced twice

**Chosen:**
1. Citations are attached by the API (decision 1).
2. `mentions.py` finds the sources an answer names and checks two levels:
   - **the work:** a work no retrieved source belongs to ("the Zohar", "Rambam, Laws of Shabbat");
   - **the place:** a location in a retrieved work that was not itself retrieved. For example,
     "Berakhot 60b" when only 2a was retrieved, "Orach Chayim 300:1" beyond the library's 1–241, or
     "דברים ו, ט".
   Either is flagged in the answer and counted in the evaluation. A verse that a retrieved
   passage itself quotes is allowed.

**Why:** the API's citations prove that what is cited was retrieved. They do not stop the model from
also writing "as the Mishnah Berurah says" from memory, or inventing a page in the right tractate.

Hebrew locations are parsed as strict Hebrew numerals: hundreds, then at most one tens letter and
one units letter, with טו and טז for 15 and 16. "בשבת זו." is therefore not a daf reference, and
"דברים רבים" is not a verse.

It remains a heuristic. It misses a source alluded to without a name or a location, which is why
the README calls it a check, not a guarantee.

## 4. Only openly licensed texts, decided per version from the API

**Chosen:**
- The license of every text version comes from the Sefaria API and is stored on every chunk; the
  dashboard and every source show it.
- Kept: Public Domain, CC0, CC-BY, CC-BY-SA. Everything else, including CC-BY-NC and "Copyright",
  is skipped and recorded with the reason.
- A survey of all versions of each work records what was left out and why.
- Ingesting the shipped snapshot re-checks every chunk's license instead of trusting the file.
- A script check skips versions whose text is not in the script their language tag claims. Sefaria
  has served Russian text as an English "community translation".

**Rejected:** using the best-known translations regardless of license, such as the William Davidson
Talmud (CC-BY-NC). Its English is better than the 1921 Cohen translation used here, and losing it is
the main cost of the policy.

## 5. Natural units, both languages per reference

**Chosen:** chunks follow the text's own units: a verse, a mishnah, a Talmud segment, one
comment of Rashi or Tosafot, a halakhah, a seif. Retrieval returns distinct references, and the
answering model gets the Hebrew and the translation of each together.

**Rejected:** fixed-size windows. They cut a sugya mid-argument and make citations point at
half-sentences.

## 6. Hybrid retrieval with Hebrew-aware keywords

**Chosen:**
- BM25 over normalised tokens:
  - niqqud and cantillation are stripped;
  - the maqaf is a word break, so "את־יום" is two words, while geresh and gershayim join their word
    ("ר\"ע");
  - each word also yields a variant without a one-letter prefix (ו, ה, ב, כ, ל, מ, ש).
- Multilingual embeddings, fused with BM25 by reciprocal rank fusion (k = 60, top 50 from each).
- A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) over the fused candidates.

The reranker cannot be switched off. The relevance threshold that decides "no relevant source" is
defined on its probabilities, and a fused rank score has no such meaning.

**Why:** learners remember exact phrases and names ("משעה שהכהנים נכנסים", "רבן גמליאל"), and
embeddings blur them. Embeddings carry English questions to Hebrew and Aramaic sources.

Verifying the question set against the corpus found the maqaf bug: "את־יום" had been one token.

## 7. Embedding model: chosen on dev

**Compared on dev** (66 scored questions; full table in
[eval/retrieval-dev.md](../backend/eval/retrieval-dev.md)):

| Configuration | Recall@10 | MRR@10 |
|---|---:|---:|
| BM25 alone | 55% | 0.44 |
| multilingual-e5-small alone | 58% | 0.39 |
| BM25 + e5-small (RRF) | 67% | 0.47 |
| BM25 + multilingual-e5-base | 68% | 0.47 |
| BM25 + paraphrase-multilingual-MiniLM | 59% | 0.42 |
| BM25 + e5-small, reranked top 50 | 74% | 0.57 |
| BM25 + e5-small, reranked top 20 | 71% | 0.56 |

**Chosen:**
- **e5-small, fused with BM25.** e5-base is within noise on dev (one question out of 66) at four
  times the embedding cost (27 against 105 passages a second here) and twice the vector size.
  MiniLM is clearly worse on Hebrew (29% against 54% for its semantic side alone).
- **Reranking the top 20.** Depth 50 finds two more questions out of 66 at 2.5 times the reranker
  compute, and the reranker is what decides whether the app is usable on a CPU-only Docker host.

**Also tried:** bge-m3. It could not finish embedding the corpus on the 8 GB development machine
alongside the reranker and Docker, so it is not in the comparison. It is the obvious next candidate
on a GPU.

**Relevance threshold:** 0.05 on the reranker's probability for the best source, also chosen on
dev. It keeps 94% of answerable questions and, by relevance alone (before the classifier), stops
80% of not-in-library questions and every out-of-scope one.

## 8. A scope filter with configurable strictness

**Chosen:**
- Claude Haiku classifies each question into a fixed JSON schema: scope, kind, sensitive area and
  manipulation. A follow-up is classified together with the question before it.
- Code combines that with the best reranker score:
  - **strict:** text questions only;
  - **balanced** (default): also general Jewish questions when a relevant source exists;
  - **lenient:** refuses only what is out of scope and has no relevant source.
- Manipulation (ignore your rules, role-play a posek issuing binding rulings) is refused at every
  strictness.
- No relevant source means no answer at any strictness. The app says so instead of answering from
  memory.
- The admin's choice of strictness is kept across restarts.

**Why:** the classifier is good at "is this about Torah"; retrieval is the ground truth for "can I
answer this from my library". Neither alone is enough.

## 9. Sensitive questions: two independent detectors, danger first

**Chosen:**
- The classifier's `sensitive` flag.
- A keyword backstop in Hebrew and English for mourning, illness, family purity, marriage and
  divorce, conversion, danger to life, abuse and self-harm. The crisis patterns include indirect
  phrasings such as "I don't want to live anymore", "אני לא רוצה לחיות יותר", "בא לי למות", "My
  husband hits me" and "השכן שלי בסכנה".
- The backstop avoids words that double as ordinary ones: "אבל" also means "but", "שבעה" also means
  "seven", and "נפטר מ..." means "exempt from".
- The backstop skips impersonal text study ("how did R. Chanina know whether a sick person would
  live?"), but always fires on first-person questions. A danger match outranks whatever the
  classifier said.
- Any sensitive question gets "ask your rabbi first" before the answer. Self-harm, abuse and
  danger also put emergency numbers above everything.
- If the classifier refuses, is cut off or returns something unparseable, the question is not
  answered. The person still gets the emergency and "ask your rabbi" notices, not a bare "out of
  scope": a safety classifier's refusal is itself a signal.

**Why:** a missed crisis costs far more than an unneeded notice.

## 10. Record once, replay everywhere, keyed by the whole request

**Chosen:**
- A recording is keyed by the SHA-256 of the complete request: model, system prompt, messages,
  every source text and the output settings. It stores the request next to the response.
- CI scores the evaluation from the recordings alone: no corpus, no models, no API key. It re-parses
  every response, re-checks every citation against the sources that were sent, and fails if the
  committed results do not match.
- Without a key, the app looks up a recorded answer by its conversation instead: the system prompt,
  the text of every turn and, in chavruta mode, the anchored passage. It rebuilds the sources that
  were sent from the library and labels the result a recorded answer. This is needed because
  pgvector's approximate search can order sources slightly differently from the evaluation's exact
  search.

**Rejected:** a hand-maintained prompt version as the key. It silently goes stale when someone edits
the prompt and forgets to bump it.

## 11. Network allowlist, enforced in two layers

**Chosen:**
1. Every HTTP client is built by `net.http_client()`. Its transport refuses any host but Sefaria,
   Hebcal and the Anthropic API, on every redirect hop. The Anthropic SDK is given that client.
2. `install_socket_guard()` wraps the whole process:
   - **name resolution** (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`) succeeds only for
     allowlisted names and the database host;
   - **every call that reaches an address** (`connect`, `connect_ex`, `sendto`, `sendmsg`) requires
     a loopback address or one that an allowlisted name resolved to. An IP literal never
     allowlists itself.

**Out of scope:** native code that opens its own sockets is outside the Python guard. That is
libpq, which connects only to the configured database.

**Model weights:** they are downloaded at image build time by a separate step. The running
containers set `HF_HUB_OFFLINE=1` and load weights with `local_files_only`.

**How it is tested:** in a subprocess, against urllib, a plain httpx client, `create_connection`,
`connect_ex`, UDP, `gethostbyname`, redirects and IP literals in TEST-NET-1. The review that found
the original IP-literal bypass is the reason the test now covers every path.

## 12. Closed on Shabbat and Yom Tov, in the backend

**Chosen:**
- Times come from Hebcal: candle lighting and havdalah for the configured location (Jerusalem by
  default, Israel schedule). Each candle lighting opens a period that the next havdalah closes, so
  a Yom Tov next to Shabbat chains into one period.
- Every endpoint except status and health answers 423 during that period, and the frontend shows
  the closed screen on any 423.
- Hebcal failures:
  - a failed refresh backs off for five minutes, and the fetch runs outside the lock;
  - the last good schedule, which covers weeks ahead, is saved, so a restart during an outage
    still knows the real times;
  - with nothing saved, the app closes conservatively from Friday 12:00 to Saturday 23:59 and says
    so.
- There is a setting to force the closed screen. There is no setting to turn closing off.

**Why:** a closed screen that the API ignores is decoration.

## 13. Accounts, secrets and spend fail closed

**Chosen:**
- **Passwords:** the app refuses to start without real ones. Compose generates them into
  `./secrets` on the first run.
- **Sessions:** server-side. The cookie holds a random token and the table holds its hash, so
  sign-out revokes it. There is no signing secret to leak.
- **Login:** throttled per client address and username, and unknown users are verified against a
  dummy hash so timing does not reveal which usernames exist. The client address comes from
  `X-Forwarded-For` only when the app is configured behind the bundled nginx, which overwrites the
  header.
- **Spend:** model calls are limited per account, by default 60 an hour and 300 a day.

## 14. The evaluation set

**Chosen:**
- 224 questions in Hebrew and English:
  - 163 in scope with gold references: 110 source lookups, 15 explanations, 33 practical halacha
    and 5 sensitive;
  - 7 not in the library;
  - 40 out of scope;
  - 8 borderline (Jewish but not about the texts), to tell strict from balanced;
  - 6 crisis messages, to test that emergency numbers come first.
- **Split:** questions that share any gold reference are grouped, so Hebrew and English twins and
  parallel questions stay together. Groups are assigned to dev (40%) or holdout (60%) by a fixed
  hash. No holdout question shares a gold passage with a dev question.
- **Gold references** must exist in the corpus; the check is automatic. Every scored question has at
  least one verification phrase that its gold text must contain, and 233 of the 291 gold
  references carry one. Some phrases are single words, so this proves a reference is real and on
  topic, not that it is the only good answer.
- **One annotation revision,** made after reading dev errors and before any holdout run, and applied
  to both splits:
  - a practical question that does not name a work ("Do I need to wash my hands before eating
    bread?") accepts the parallel sources a learner would cite: the Mishnah or Gemara, the Rambam
    and the Shulchan Arukh;
  - the parallels were found by exact-phrase search in the texts, never from retrieval output;
  - 70 references were added to 40 questions. Before the revision, a retrieved Rambam, Blessings
    6:1, which is exactly the law of washing before bread, counted as a miss.
- **Recall@k** counts a question as found if any gold source is in the top k distinct references.
  Talmud gold references are often a whole amud, which makes Talmud recall lenient.
- **Citation faithfulness** is measured by a recorded judge, per answer segment. The judge is Claude
  Sonnet, a different model from the Opus answerer but the same vendor, so its verdicts need the
  check of a manual review sheet: 30 random cited segments on holdout, for a person to fill in and
  score against the judge.

**The honest limit:** the questions were written for this project, not by independent annotators.

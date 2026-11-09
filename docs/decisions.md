# Design decisions

## 1. Citations come from the API, not from the model's prose

**Chosen:**
- Retrieved passages are sent as `search_result` content blocks with citations enabled.
- The API returns each sentence with the source block it cites, and the cited text is that block,
  verbatim.
- One search result is one reference (a verse, a Talmud segment, a seif). Its content blocks are
  the Hebrew text and its translation, so a citation also says which language was cited.

**Rejected:**
- Asking the model to write `[1]` markers. They can point anywhere and quote loosely.
- Structured JSON output with a citations field. It cannot be combined with API citations, and
  it relies on the model to report its own sources honestly.
- Plain-text `document` blocks with character offsets. Those allow partial-sentence citations of a
  verse, where the unit of meaning is the verse itself.

**Why:** a citation that the API attaches cannot name a passage that was not sent, and its quote
cannot be paraphrased. The evaluation re-checks both from the recordings: every citation index is a
source that was sent, and every cited text is verbatim from it.

## 2. Guardrails live in code, around the model

**Chosen:**
- The scope decision is made before the answering model runs: out of scope, manipulation and "no
  relevant source" never reach it.
- The "ask your rabbi first" notice, the not-a-posek disclaimer and the safety notice are added
  by code, in a fixed order: safety, then "ask your rabbi", then the answer, then disclaimers.
- After the answer, a detector flags works the model names that were not retrieved.

**Rejected:** putting these rules only in the system prompt. The prompt states them too, but a
model that forgets one sentence must not be able to drop a safety notice.

**Why:** these are the requirements that must hold every time. A prompt raises the odds; code makes
them certain, and tests check them.

## 3. Never cite what was not retrieved: enforced twice

**Chosen:**
1. Citations are attached by the API (decision 1).
2. `mentions.py` finds work names with a location in the answer text ("Shabbat 21b", "תהלים
   קמה, טז", "Mishnah Berurah"). Any mention of a work that no retrieved source belongs to, and
   that no retrieved text itself quotes, is flagged in the answer and counted in the evaluation.

**Why:** the API's citations prove that what is cited was retrieved. They do not stop the model
from also writing "as the Mishnah Berurah says" from memory. The detector is a heuristic, tuned for
few false alarms: bare words that double as ordinary Hebrew ("שבת", "ברכות", "דברים") count only
with a location. That is why the README reports it as a check, not a guarantee.

## 4. Only openly licensed texts, decided per version from the API

**Chosen:**
- The license of every text version comes from the Sefaria API and is stored on every chunk. The
  dashboard and every source show it.
- Kept: Public Domain, CC0, CC-BY, CC-BY-SA. Everything else, including CC-BY-NC and "Copyright",
  is skipped and recorded with the reason.
- A survey of all versions of each work records what was left out and why.
- A script check skips versions whose text is not in the script their language tag claims. Sefaria
  has served Russian text as an English "community translation".

**Rejected:** using the best-known translations regardless of license (the William Davidson
Talmud, CC-BY-NC). Its English is better than the 1921 Cohen translation used here, and it is the
main cost of the policy.

**Why:** the brief requires it, and a study tool that redistributes text must be able to show its
right to do so.

## 5. Natural units, both languages per reference

**Chosen:** chunks follow the text's own units: a verse, a mishnah, a Talmud segment, one
comment of Rashi or Tosafot, a halakhah, a seif. Retrieval returns distinct references, and
the answering model gets the Hebrew and the translation of each together.

**Rejected:** fixed-size windows. They cut a sugya mid-argument and make citations point at
half-sentences.

**Why:** learners cite by reference ("Berakhot 2a", "Orach Chayim 89:3"). A unit the text defines is
also the unit a citation should name.

## 6. Hybrid retrieval with Hebrew-aware keywords

**Chosen:**
- BM25 over normalised tokens:
  - niqqud and cantillation are stripped;
  - the maqaf is a word break, so "את־יום" is two words, while geresh and gershayim join their word
    ("ר\"ע");
  - each word also yields a variant without a one-letter prefix (ו, ה, ב, כ, ל, מ, ש).
- Multilingual embeddings, fused with BM25 by reciprocal rank fusion (k = 60, top 50 from each).
- A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) over the fused top 50.

**Why:** exact phrases and names are what a learner remembers ("משעה שהכהנים נכנסים", "רבן
גמליאל"), and embeddings blur them. Embeddings carry English questions to Hebrew and Aramaic
sources. Verifying the question set against the corpus found the maqaf bug: "את־יום" had been one
token.

## 7. Embedding model: chosen on dev

**Compared:** multilingual-e5-small, multilingual-e5-base, bge-m3, and
paraphrase-multilingual-MiniLM, each alone, fused with BM25, and fused and reranked, on the dev
split only. Numbers: [eval/retrieval-dev.md](../backend/eval/retrieval-dev.md).

**Chosen:** see the README. The choice and its reasons are filled in from the dev numbers.

## 8. A scope filter with configurable strictness

**Chosen:**
- Claude Haiku classifies each question into a fixed JSON schema: scope, kind, sensitive area, and
  manipulation.
- Code combines that with the best reranker score:
  - **strict:** text questions only;
  - **balanced** (default): also general Jewish questions, when a relevant source exists;
  - **lenient:** refuses only what is out of scope and has no relevant source.
- Manipulation (ignore your rules, role-play a posek issuing binding rulings) is refused at every
  strictness.
- No relevant source means no answer at any strictness: the app says so instead of answering from
  memory.

**Why:** the classifier is good at "is this about Torah"; retrieval is the ground truth for "can I
answer this from my library". Neither alone is enough.

## 9. Sensitive questions: two independent detectors

**Chosen:**
- The classifier's `sensitive` flag.
- A Hebrew and English keyword backstop (mourning, illness, family purity, marriage and divorce,
  conversion, danger to life, abuse, self-harm). It avoids words that double as ordinary ones:
  "אבל" also means "but", and "שבעה" also means "seven".
- The backstop skips impersonal text study ("how did R. Chanina know whether a sick person would
  live?"), but always fires on first-person questions.
- Either detector puts "ask your rabbi first" before the answer. Self-harm, danger to life and abuse
  also put emergency numbers above everything.

**Why:** a missed sensitive question costs far more than an unneeded "ask your rabbi".

## 10. Record once, replay everywhere, keyed by the whole request

**Chosen:**
- A recording is keyed by the SHA-256 of the complete request: model, system prompt, messages,
  every source text and the output settings. It stores the request next to the response.
- CI scores the evaluation from the recordings alone: no corpus, no models, no API key. It re-parses
  every response and re-checks every citation against the sources that were sent.
- Without a key, the app falls back to finding a recorded answer by question. It rebuilds the
  sources that were sent from the library and labels the result a recorded answer. pgvector's
  approximate search can order sources slightly differently from the evaluation's exact search.

**Rejected:** a hand-maintained prompt version as the key. The payables review showed it
silently goes stale: changing the prompt left the key unchanged.

## 11. Network allowlist, enforced in two layers

**Chosen:**
1. Every HTTP client is built by `net.http_client()`, whose transport refuses any host but
   Sefaria, Hebcal and the Anthropic API, on every redirect hop. The Anthropic SDK is given that
   client.
2. `install_socket_guard()` wraps name resolution and `socket.connect` for the whole process. A
   library that opens its own connections is stopped too. Only allowlisted names (plus the
   database host) resolve, and a raw IP connects only if it came from an allowlisted name or is
   loopback.

Model weights are downloaded at image build time by a separate step. The running containers set
`HF_HUB_OFFLINE=1` and load weights with `local_files_only`.

**Why:** "no other outbound requests" has to hold for libraries the code does not control, such as
Hugging Face's update checks. Tests run the guard in a subprocess against urllib, raw sockets,
redirects and IP literals.

## 12. Closed on Shabbat and Yom Tov, in the backend

**Chosen:**
- Hebcal candle-lighting and havdalah times for the configured location (Jerusalem by default,
  Israel schedule). Each candle lighting opens a period that the next havdalah closes, so Yom Tov
  next to Shabbat chains into one period.
- Every endpoint except status and health answers 423 during that period. The frontend shows the
  closed screen on any 423.
- If Hebcal is unreachable and nothing is cached, the app closes conservatively from Friday 12:00
  to Saturday 23:59, and says so.
- There is a setting to force the closed screen. There is no setting to turn closing off.

**Why:** a closed screen that the API ignores is decoration.

## 13. Accounts and secrets fail closed

**Chosen:**
- The app refuses to start without a 32-byte session secret and real passwords. Compose generates
  them into `./secrets` on the first run.
- Sessions are server-side: the cookie holds a random token, the table holds its hash, and sign-out
  revokes it.
- Login is throttled per client address and username, and unknown users are verified against a dummy
  hash so timing does not reveal which usernames exist.

**Why:** these are the weaknesses the payables review found by forging a session with a default
secret. They are fixed here from the start.

## 14. The evaluation set

**Chosen:**
- 210 questions:
  - 164 in scope, in Hebrew and English: source lookup, explanation, practical halacha, and
    sensitive questions;
  - 6 deliberately not in the library;
  - 40 out of scope.
- Dev is 40% of the set and holdout 60%, split by a fixed rule.
- Every gold reference is checked automatically against the fetched corpus: it must exist, and it
  must contain the phrase the question is about.
- One annotation revision, made after reading dev errors and before any holdout run, applied to
  both splits: a practical question that does not name a work ("Do I need to wash my hands
  before eating bread?") accepts the parallel sources a learner would cite: the Mishnah or Gemara
  where the law originates, the Rambam, and the Shulchan Arukh. The parallels were found by
  exact-phrase search in the texts, never from retrieval output, and each carries a verification
  phrase. 70 references were added to 40 questions. Before the revision, a retrieved Rambam,
  Blessings 6:1, which is exactly the law of washing before bread, counted as a miss.
- Recall@k counts a question as found if any gold source is in the top k distinct references.
  Gold references are alternatives: the Mishnah or the Gemara on it both answer.
- Citation faithfulness is measured three ways:
  - mechanically, from the recordings;
  - by a recorded Claude judge, per answer segment;
  - by a manual review sheet of 30 random cited segments on holdout, for a person to fill in.

**Why:** there is no public benchmark of Hebrew and English questions over this corpus. The
honest limit is that the questions and gold references were written for this project, not by independent annotators. The
checks prove that the references are real and on topic, not that they are the only good answer.

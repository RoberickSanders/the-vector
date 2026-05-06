# Architecture

The Vector is a 4-stage pipeline where each stage is independently usable but composes to handle the full apply loop.

## Stage 1: Discovery (`tools/scrape.py` + `tools/jobspy_pull.py`)

Two complementary scrapers:

- **scrape.py** — direct scrapers for Greenhouse, Lever, Ashby, Workday over a curated d100 (target 100 companies). Cleaner data, but limited to companies on those ATSes.
- **jobspy_pull.py** — wraps `python-jobspy` to scrape LinkedIn, Indeed, Glassdoor, Google Jobs, ZipRecruiter, Bayt, BDjobs in one call. Broader coverage, less structured data.

Output: per-profile master xlsx file at `output/<profile>-jobs-master.xlsx`. All downstream tools read from this file.

## Stage 2: Scoring (`tools/score.py`)

Each row gets a fit_score (1-10) computed by an LLM against the profile's scoring rubric at `profiles/<name>/scoring-rubric.md`.

Designed to be cheap: uses a small model by default and processes rows in batches.

## Stage 3: Hiring manager resolution (`tools/find_managers.py`)

The 5-step cascade that resolves the right hiring manager email and LinkedIn for each top-scored job.

```
1. JD scrape       — extract emails directly from the job posting
2. Blitz API       — LinkedIn-based contact lookup
3. Hunter API      — domain-based email finder
4. Icypeas API     — reverse-email + name+domain finders
5. Permutator + MV — generate likely emails, verify via MillionVerifier
```

Each step is optional. If an API key is missing, that step is skipped and the cascade falls through to the next step. The cascade succeeds when any step returns a verified email + LinkedIn URL.

The cascade also runs disqualifier filters to drop wrong-target managers — sales-comp, HR, recruiters, multi-region postings without verified region match. These filters were added after observing that ~25% of "verified" matches at the gold-standard tier were actually wrong-role contacts (HR people at the right company, RevOps people whose titles match generic keywords, recruiters with the right email but wrong function).

## Stage 4: Application (`tools/batch_apply.py`)

For the top N candidates from the master xlsx, runs:

1. `tailor.py` — JD-aware resume tailoring with proof-point gate
2. `score_resume.py` — ATS keyword scoring (must hit 70+)
3. Email draft for hiring manager
4. LinkedIn DM draft for hiring manager
5. Writes per-application package to `applications/<slug>.md`

Then `auto_apply.py` (optional) handles ATS form submission via browser-use, halting before submit for human approval. Human-in-the-loop is structurally enforced — the agent can not auto-submit, only fill.

## Why this architecture

The system is built around the principle that *finding the right manager and reaching them directly* is the highest-leverage step in the apply loop. Most "AI job search" tools optimize for sending more applications. This optimizes for sending fewer, better applications with direct manager reach.

The proof-point gate in `tailor.py` is the safety mechanism. The LLM may not insert claims that aren't already documented in the source resume. This prevents the most common failure mode of AI resume tailors: fabricated metrics that fall apart in interviews.

The cascade's graceful degradation means you can run with zero paid APIs (relying on JD scrape only), with one paid API (Blitz adds the most coverage), or with all five (highest accuracy).

## Why each stage is independently usable

- Use `tools.jobspy_pull` alone if you just want a multi-board scraper
- Use `tools.score_resume` alone if you just want ATS keyword scoring
- Use `tools.find_managers` alone if you just want the cascade
- Use `tools.tailor` alone if you just want resume tailoring with a safety gate

The pipeline composes them, but each is self-contained and tested.

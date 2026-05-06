---
name: vector-job-search
version: "0.1.0"
description: "The Vector — agentic GTM job-search tool. Find hiring managers via 5-step cascade (Blitz/Hunter/Icypeas/permutator+MV), tailor resumes per JD with proof-point safety, score against ATS, batch-apply at scale. Triggers: 'tailor my resume', 'find the hiring manager', 'score this JD', 'batch apply', 'job search'."
allowed-tools: Bash, Read, Write, Edit
homepage: https://github.com/RoberickSanders/the-vector
user-invocable: true
---

# The Vector

Agentic GTM job-search tool. Pipeline: scrape → score → find managers → tailor → ATS-score → batch-apply.

## When to use

Trigger when the user asks to:
- Tailor a resume against a JD
- Find a hiring manager email/LinkedIn for a job posting
- Score a resume PDF against a JD (ATS keyword match)
- Batch-apply to top-N candidates from a job-search master xlsx
- Pull a fresh sweep of jobs from d100 ATS endpoints

## Setup (one-time)

The user must have `the-vector` cloned and a virtualenv populated before invoking:

```bash
cd <repo>/job-search-tool
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp ../.env.example ../.env   # add ANTHROPIC_API_KEY at minimum
cp profiles/example.yaml profiles/<name>.yaml
mkdir -p profiles/<name>
cp profiles/example/{resume.md,scoring-rubric.md,email-signature.txt} profiles/<name>/
```

Then edit `profiles/<name>/resume.md` with the user's actual resume content.

## Tools and how to invoke

All commands run from `<repo>/job-search-tool/` and assume the venv has been
activated or `.venv/bin/python` is used directly.

### tools.scrape — pull jobs

```bash
.venv/bin/python -m tools.scrape --profile <name>                  # direct ATS sweep (default)
.venv/bin/python -m tools.scrape --profile <name> --source jobspy  # JobSpy keyword search
.venv/bin/python -m tools.scrape --profile <name> --source both    # union, deduped
```

### tools.score — fit-score every job 1-10

```bash
.venv/bin/python -m tools.score --profile <name> --limit 200
```

Uses Kimi via Anthropic-compatible endpoint when `KIMI_API_KEY` is set;
otherwise falls back to ANTHROPIC_API_KEY. Round-robin by company so no
single company dominates a batch.

### tools.find_managers — resolve hiring manager email + LinkedIn

```bash
.venv/bin/python -m tools.find_managers --profile <name> --top 20
.venv/bin/python -m tools.find_managers --in path/to/jobs.csv --out path/to/with_managers.xlsx
```

Cascade: JD scrape → Blitz → Hunter → Icypeas → permutator + MillionVerifier.
Each step is silently skipped when its key is missing.

### tools.tailor — JD-aware resume rewrite

```bash
.venv/bin/python -m tools.tailor --profile <name> --company "<Company>" --role "<Role Title>"
.venv/bin/python -m tools.tailor --profile <name> --company "<Company>" --jd-file /path/to/jd.txt
```

Proof-point-gated: edits the LLM proposes that introduce claims absent from
`resume.yaml` are rejected before render. Output: `<repo>/applications/<slug>.yaml`
plus a rendered PDF.

### tools.score_resume — ATS keyword score (0-100, 70+ ship gate)

```bash
.venv/bin/python -m tools.score_resume --jd-file jd.txt --resume-yaml resume.yaml
.venv/bin/python -m tools.score_resume --jd-file jd.txt --resume-pdf resume.pdf --profile <name>
```

LLM extractor for the keyword list with a heuristic fallback. Outputs
`Score: NN/100  (extractor: llm|heuristic, match: token-level)`.

### tools.batch_apply — top-N pipeline

```bash
.venv/bin/python -m tools.batch_apply --profile <name> --top 5
.venv/bin/python -m tools.batch_apply --profile <name> --top 5 --regenerate
.venv/bin/python -m tools.batch_apply --profile <name> --company "<Company>"
```

For each candidate: tailor + score_resume + LLM-draft email + DM + A-F
evaluation. Output: `applications/<slug>.md` + a `batch-<date>.md` summary.

## Reading the output

After `batch_apply` completes, walk the user through `applications/<slug>.md`:
1. Email body — does it sound like the candidate?
2. STAR stories — do all metrics trace back to the resume?
3. Gaps section — are claimed gaps actually claimed in the resume?

Stop and ask before clicking Apply on anything.

## Notes

- `auto_apply.py` is browser-agent driven (browser-use + langchain-anthropic). It is intentionally human-in-loop: the agent fills the form and STOPS before clicking Submit. Never override that.
- The proof-point gate in `tailor.py` is non-optional. Disabling it produces fabricated metrics. Don't.
- `score_resume.py`'s 70+ ship gate is the recommended threshold; below 70, tailor again before applying.

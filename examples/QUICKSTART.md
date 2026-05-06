# Quickstart — Try The Vector in 5 Minutes

Run the toolchain on real data with **zero API keys**. No accounts, no setup, just clone-and-go.

## 1. Clone and install

```bash
git clone https://github.com/RoberickSanders/job-search-stack
cd job-search-stack/job-search-tool
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Pull jobs from a real public source

JobSpy doesn't require auth for LinkedIn / Indeed / Glassdoor scraping. Pull a sample:

```bash
.venv/bin/python -m tools.jobspy_pull \
  --search-terms "Solutions Engineer" \
  --location "United States" \
  --remote-only \
  --hours-old 168 \
  --output examples/output/jobs.csv
```

Output: `examples/output/jobs.csv` with ~50-200 real, recent job postings.

## 3. Score a tailored resume against a JD (no LLM)

The score_resume tool has a heuristic fallback that works without API keys:

```bash
.venv/bin/python -m tools.score_resume \
  --jd-file examples/sample_job.json \
  --resume-yaml examples/sample_resume.yaml \
  --no-llm
```

Output: a 0-100 keyword match score plus a list of which keywords matched and which didn't. Useful as a first-pass ATS fitness check.

## 4. Inspect the cascade

```bash
.venv/bin/python -m tools.find_managers --help
```

The hiring-manager cascade gracefully degrades as keys drop out:
- **No keys at all:** only JD-scrape works (extracts emails directly from job postings)
- **+ BLITZ_API_KEY:** adds Blitz LinkedIn-based contact lookup
- **+ HUNTER_API_KEY:** adds Hunter domain search
- **+ ICYPEAS_API_KEY:** adds Icypeas reverse-email + name+domain finders
- **+ MILLIONVERIFIER_API_KEY:** adds email verification on every candidate

Each level adds resolution power. Useful starting tier: BLITZ + MILLIONVERIFIER (~$50/mo combined for personal use).

## 5. Run the test suite

```bash
.venv/bin/python -m pytest tests/ -v
```

344 tests, fully mocked, runs in under 5 seconds. Confirms the toolchain works on your machine.

## What's next

Once you want to use the full pipeline:

1. Get an Anthropic API key ($5 starter credit) → set `ANTHROPIC_API_KEY` in `.env`
2. (Optional) Add Blitz / Hunter / Icypeas / MillionVerifier for higher cascade resolution
3. Build your own profile in `profiles/yourname/` (use `profiles/example/` as a template)
4. Run the full `tools.batch_apply --top 10` pipeline

Total minimum cost for full operation: ~$30-50/mo on a typical job search.

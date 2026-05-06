# The Vector

An agentic toolchain for job seekers. Find the right hiring manager, tailor your resume to each posting safely, score yourself against the ATS, and apply at scale.

## Try it now (no API keys required)

The fastest way to see the toolchain work:

```bash
git clone https://github.com/RoberickSanders/job-search-stack
cd job-search-stack/job-search-tool
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Pull real jobs (JobSpy is auth-free)
.venv/bin/python -m tools.jobspy_pull --search-terms "Solutions Engineer" \
  --remote-only --hours-old 168 --output examples/output/jobs.csv

# Score a sample resume against a sample JD (no LLM needed)
.venv/bin/python -m tools.score_resume \
  --jd-file examples/sample_job.json \
  --resume-yaml examples/sample_resume.yaml \
  --no-llm
```

Full quickstart at [`examples/QUICKSTART.md`](examples/QUICKSTART.md). Architecture deep-dive at [`docs/architecture.md`](docs/architecture.md).

## What it does

- **find_managers** — multi-source enrichment cascade that resolves the hiring manager's email and LinkedIn for any job posting. Optional fallbacks let it work with whatever API keys you have access to.
- **tailor.py** — JD-aware resume tailoring with a proof-point gate. The LLM may only surface claims you've already documented. Won't fabricate metrics, tools, or experience.
- **score_resume.py** — ATS keyword scoring with LLM extraction and heuristic fallback. Configurable ship gate.
- **batch_apply.py** — Top-N batch processor. Tailors, scores, drafts the hiring-manager email and LinkedIn DM, writes a per-application package.
- **auto_apply.py** — agentic ATS form filler on browser-use. Halts before submit. Human-in-the-loop required.
- **scrape pipeline** — direct ATS coverage (Greenhouse, Lever, Ashby, Workday) over a configurable target list. Supplemented by JobSpy keyword search.

## Quick start

```bash
git clone https://github.com/RoberickSanders/job-search-stack
cd job-search-stack/job-search-tool
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Configure
cp ../.env.example ../.env             # paste in your Anthropic + optional cascade keys
cp profiles/example.yaml profiles/yourname.yaml
mkdir -p profiles/yourname
cp profiles/example/resume.md profiles/yourname/resume.md
cp profiles/example/scoring-rubric.md profiles/yourname/scoring-rubric.md

# Pull jobs
.venv/bin/python -m tools.scrape --profile yourname

# Score against your rubric
.venv/bin/python -m tools.score --profile yourname --limit 200

# Find hiring managers for the top N
.venv/bin/python -m tools.find_managers --profile yourname --top 20

# Score your resume against a specific JD
.venv/bin/python -m tools.score_resume --jd-file jd.txt --resume-yaml resume.yaml

# Tailor your resume for a specific company
.venv/bin/python -m tools.tailor --profile yourname --company "CompanyName" --role "Role Title"

# Batch-apply to the top 5
.venv/bin/python -m tools.batch_apply --profile yourname --top 5
```

## Architecture

```
                     ┌─────────────┐
   d100 + JobSpy ──> │   scrape    │ ──> output/<profile>-jobs-master.xlsx
                     └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │    score    │ ──> fit_score 1-10 per row
                     └─────────────┘
                            │
                            ▼
                     ┌─────────────────┐
                     │ find_managers   │ ──> manager_email + manager_linkedin
                     └─────────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ batch_apply │ ──> applications/<slug>.md per top-N candidate
                     └─────────────┘
                            │
                            ├──> tailor.py        (proof-point-gated resume rewrite)
                            └──> score_resume.py  (ATS keyword score, configurable gate)
```

The find_managers cascade is designed to gracefully degrade as keys drop out. The proof-point gate in `tailor.py` prevents the LLM from claiming experience or metrics that aren't already documented in your resume.

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

Network-dependent steps are mocked. `pytest.ini` has `--timeout=30` to catch infinite-loop regressions.

## License

MIT

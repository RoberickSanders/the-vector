# The Vector

Agentic GTM job-search tool. Find hiring managers, tailor resumes per JD with proof-point safety, score against ATS, batch-apply at scale.

## What it does

- **find_managers cascade** — five-step waterfall (JD scrape → Blitz → Hunter → Icypeas → permutator + MillionVerifier) that resolves the right hiring manager email + LinkedIn for any job posting. Each step is optional; missing API keys are silently skipped and the cascade falls through.
- **tailor.py** — JD-aware resume tailoring with a proof-point gate. The LLM may only insert claims that are already in your `resume.yaml`; it cannot fabricate metrics, tools, or experience you cannot back up.
- **score_resume.py** — ATS keyword scoring for a tailored resume against the JD. Uses an LLM extractor with a heuristic fallback. Outputs a 0-100 score with a 70+ ship gate.
- **batch_apply.py** — Top-N batch processor. Pulls the highest-fit candidates from the master xlsx, runs `tailor` and `score_resume` per row, drafts the hiring-manager email + LinkedIn DM, and writes a per-application markdown package.
- **scrape pipeline** — direct ATS scrapers (Greenhouse / Lever / Ashby / Workday) over a curated d100 target list, plus a JobSpy keyword search as supplement. Output dumps to a per-profile master xlsx that all downstream tools read.

## Quick start

```bash
git clone https://github.com/RoberickSanders/the-vector
cd the-vector/job-search-tool
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Configure
cp ../.env.example ../.env             # paste in your Anthropic + optional cascade keys
cp profiles/example.yaml profiles/yourname.yaml
mkdir -p profiles/yourname
cp profiles/example/resume.md profiles/yourname/resume.md      # paste your resume
cp profiles/example/scoring-rubric.md profiles/yourname/scoring-rubric.md

# Pull jobs (direct ATS sweep over d100 + JobSpy supplement)
.venv/bin/python -m tools.scrape --profile yourname

# Score the pulled jobs against your rubric (round-robin by company so no single company dominates)
.venv/bin/python -m tools.score --profile yourname --limit 200

# Find hiring managers for the top N
.venv/bin/python -m tools.find_managers --profile yourname --top 20

# Score how well your resume matches one specific job
.venv/bin/python -m tools.score_resume --jd-file jd.txt --resume-yaml resume.yaml

# Tailor your resume for a specific company (uses your full proof-point library)
.venv/bin/python -m tools.tailor --profile yourname --company "Apollo" --role "GTM Engineer"

# Batch-apply to the top 5 candidates (tailor + score + email + DM in one pass)
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
                     │    score    │ ──> fit_score 1-10 per row (Kimi LLM)
                     └─────────────┘
                            │
                            ▼
                     ┌─────────────────┐
                     │ find_managers   │ ──> manager_email + manager_linkedin
                     └─────────────────┘     (cascade: scrape → Blitz → Hunter
                            │                  → Icypeas → permutator + MV)
                            ▼
                     ┌─────────────┐
                     │ batch_apply │ ──> applications/<slug>.md per top-N candidate
                     └─────────────┘
                            │
                            ├──> tailor.py        (proof-point-gated resume rewrite)
                            └──> score_resume.py  (ATS keyword score, 70+ ship gate)
```

The find_managers cascade is the part most people will want standalone. It's designed to gracefully degrade as keys drop out — you can run it with just `BLITZ_API_KEY` set, or just `HUNTER_API_KEY`, or none of them (it'll fall back to JD scrape + permutator + MillionVerifier).

The proof-point gate in `tailor.py` is the part that took the longest to get right. It prevents the LLM from claiming experience or metrics that aren't already in your resume — common failure mode of every "AI resume tailor" you've seen. Edits below the gate are rejected before render.

## Why this exists

I built this for my own job hunt. Then my dad needed a way to apply to bilingual Medicare roles in volume, and I realized the same patterns worked for him with a different framing. Then it clicked: this is just GTM Engineering applied to a different signal. The candidate is the lead. The hiring manager is the buyer. The apply form is the conversion event. Same patterns I use to build outbound systems for paying clients, just pointed inward.

Most "AI job search tools" optimize for sending 100 applications a day. That's the wrong goal. The right goal: send the 5 applications this week to companies where you're the obvious hire, and reach the hiring manager directly the same day. The Vector is built around that.

If you're an operator looking for the next role, or you're helping someone you love who is, this should help.

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

The tests cover the full pipeline. Network-dependent steps are mocked. `pytest.ini` has `--timeout=30` to catch infinite-loop regressions early — leave it in.

## Author

[Rob Sanders](https://www.linkedin.com/in/roberick-sanders-310603120/) — RevenueMechanics. Building agentic GTM systems. Currently looking for senior GTM Engineer / Forward Deployed Engineer / Solutions Engineer roles at AI-tooling companies. Open to talk.

## License

MIT

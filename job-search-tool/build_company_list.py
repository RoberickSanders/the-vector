"""
Parses remoteintech/remote-jobs company markdown files into a JSON config
the scraper uses to flag remote-friendly companies in scrape results.

Run once: python3 build_company_list.py
"""
import json
import re
from pathlib import Path

SOURCE_DIR = Path("/tmp/remote-jobs-main/src/companies")
OUTPUT = Path(__file__).parent / "config" / "remote-friendly-companies.json"


def parse_frontmatter(md_text: str) -> dict:
    match = re.match(r"^---\n(.*?)\n---", md_text, re.DOTALL)
    if not match:
        return {}
    block = match.group(1)
    out = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip().strip('"').strip("'")
        out[key.strip()] = val
    return out


def main():
    companies = []
    for md in sorted(SOURCE_DIR.glob("*.md")):
        meta = parse_frontmatter(md.read_text())
        if not meta.get("title"):
            continue
        companies.append({
            "title": meta.get("title", ""),
            "slug": meta.get("slug", md.stem),
            "website": meta.get("website", ""),
            "remote_policy": meta.get("remote_policy", ""),
            "region": meta.get("region", ""),
            "company_size": meta.get("company_size", ""),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(companies, indent=2))
    print(f"Wrote {len(companies)} companies to {OUTPUT}")


if __name__ == "__main__":
    main()

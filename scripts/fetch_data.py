"""Download every external source used by the project into sources/.

Run once:  python scripts/fetch_data.py

All sources are public and openly licensed; each entry records its licence so
document_catalog.csv can cite it. Files already present are skipped, so the
script is safe to re-run.

Why GitHub-hosted copies?  In the build environment Hugging Face, arXiv, the
ACL Anthology website, OpenStax.org and Wikipedia were blocked by the network
policy, but GitHub (raw files and git) and the package registries were not.
Every file below is the dataset owner's own official GitHub release, so the
content is identical to the Hugging Face / website versions.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sources"
RAW = "https://raw.githubusercontent.com"

TALKMOVES_TRANSCRIPTS = [
    "Comparing Fractions 1_Grade 4.xlsx",
    "Comparing Fractions 2_Grade 4.xlsx",
    "Fraction as Number 1_Grade 4.xlsx",
    "Martino Fraction Equivalence 1_Grade 4.xlsx",
    "Video Mosaic Grade 4 Division of Fractions 1.xlsx",
    "Video Mosaic Grade 4 Number line and fractions.xlsx",
]

OPENSTAX_MODULES = {
    # module id: section title in Prealgebra 2e
    "m81285": "4.1 Visualize Fractions",
    "m81286": "4.2 Multiply and Divide Fractions",
    "m81288": "4.4 Add and Subtract Fractions with Common Denominators",
    "m81289": "4.5 Add and Subtract Fractions with Different Denominators",
    "m81293": "5.1 Decimals",
    "m81298": "5.3 Decimals and Fractions",
    "m81303": "5.6 Ratios and Rate",
    "m81306": "6.1 Understand Percent",
}

ACL_VOLUMES = [
    "2025.acl", "2025.emnlp", "2023.findings", "2022.lrec",
    "2023.bea", "2024.bea", "2025.bea", "2025.aimecon",
]


def _files() -> list[tuple[str, Path]]:
    q = urllib.parse.quote
    items: list[tuple[str, Path]] = []
    # Heavy dataset 1: MathDial (Macina et al., EMNLP 2023 Findings), CC BY-SA 4.0
    for f in ["train.jsonl", "test.jsonl", "test.csv"]:
        items.append((f"{RAW}/eth-nlped/mathdial/main/data/{f}", SRC / "mathdial" / f))
    items.append((f"{RAW}/eth-nlped/mathdial/main/README.md", SRC / "mathdial" / "README.md"))
    # Heavy dataset 2: TalkMoves (Suresh et al., LREC 2022), CC BY-NC-SA 4.0
    for f in ["train_data_504.xlsx", "test_data_63.xlsx"]:
        items.append((f"{RAW}/SumnerLab/TalkMoves/main/data/{f}", SRC / "talkmoves" / f))
    for f in TALKMOVES_TRANSCRIPTS:
        items.append((f"{RAW}/SumnerLab/TalkMoves/main/data/{q('Subset 1')}/{q(f)}",
                      SRC / "talkmoves" / "transcripts" / f))
    items.append((f"{RAW}/SumnerLab/TalkMoves/main/{q('Coding Manual.pdf')}",
                  SRC / "talkmoves" / "Coding Manual.pdf"))
    items.append((f"{RAW}/SumnerLab/TalkMoves/main/README.md", SRC / "talkmoves" / "README.md"))
    # MaE misconceptions (Otero, Druga, Lan 2024), MIT licence
    items.append((f"{RAW}/nancyotero-projects/math-misconceptions/main/data/data.json",
                  SRC / "mae" / "data.json"))
    items.append((f"{RAW}/nancyotero-projects/math-misconceptions/main/README.md",
                  SRC / "mae" / "README.md"))
    # OpenStax Prealgebra 2e CNXML sources, CC BY-NC-SA 4.0
    for m in OPENSTAX_MODULES:
        items.append((f"{RAW}/openstax/osbooks-prealgebra-bundle/main/modules/{m}/index.cnxml",
                      SRC / "openstax" / f"{m}.cnxml"))
    # ACL Anthology metadata (titles, authors, abstracts), CC BY 4.0
    for v in ACL_VOLUMES:
        items.append((f"{RAW}/acl-org/acl-anthology/master/data/xml/{v}.xml",
                      SRC / "acl" / f"{v}.xml"))
    return items


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip  {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)
    print(f"  got   {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)")


def fetch_tiktoken_ranks() -> None:
    """Official OpenAI BPE ranks for gpt2, cl100k_base and o200k_base.

    tiktoken normally downloads these from openaipublic.blob.core.windows.net.
    If that host is unreachable, we take the same ranks from the js-tiktoken npm
    package (MIT) and write them in tiktoken's own file format.
    """
    out = ROOT / "config" / "tiktoken_ranks"
    if all((out / f"{n}.tiktoken").exists() for n in ["gpt2", "cl100k_base", "o200k_base"]):
        print("  skip  config/tiktoken_ranks")
        return
    out.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "sources" / "_npm"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["npm", "pack", "js-tiktoken@1"], cwd=tmp, check=True, capture_output=True)
    tgz = next(tmp.glob("js-tiktoken-*.tgz"))
    with tarfile.open(tgz) as t:
        t.extractall(tmp, filter="data")
    script = r"""
const fs=require('fs');
for (const n of ['gpt2','cl100k_base','o200k_base']) {
  const r=require('./package/dist/ranks/'+n+'.cjs'); const d=r.default||r; const lines=[];
  for (const x of d.bpe_ranks.split('\n')) { if(!x) continue; const p=x.split(' ');
    const off=parseInt(p[1],10); p.slice(2).forEach((t,i)=>lines.push(t+' '+(off+i))); }
  fs.writeFileSync(process.argv[1]+'/'+n+'.tiktoken', lines.join('\n')+'\n');
  fs.writeFileSync(process.argv[1]+'/'+n+'.meta.json', JSON.stringify({pat_str:d.pat_str, special_tokens:d.special_tokens}));
}"""
    subprocess.run(["node", "-e", script, str(out)], cwd=tmp, check=True)
    print("  got   config/tiktoken_ranks (gpt2, cl100k_base, o200k_base)")


def main() -> int:
    print("Downloading sources ...")
    failures = []
    for url, dest in _files():
        try:
            download(url, dest)
        except Exception as exc:  # keep going, report at the end
            failures.append((url, exc))
            print(f"  FAIL  {url}: {exc}")
    try:
        fetch_tiktoken_ranks()
    except Exception as exc:
        failures.append(("js-tiktoken", exc))
        print(f"  FAIL  tiktoken ranks: {exc}")
    (SRC / "manifest.json").write_text(json.dumps(
        {"openstax_modules": OPENSTAX_MODULES, "talkmoves_transcripts": TALKMOVES_TRANSCRIPTS,
         "acl_volumes": ACL_VOLUMES}, indent=2))
    print("done" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

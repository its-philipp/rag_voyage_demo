import argparse
import json
from pathlib import Path


def ingest_folder(input_dir: Path, output_jsonl: Path) -> None:
    files = []
    for pat in ("*.md", "*.markdown", "*.txt"):
        files.extend(sorted(input_dir.rglob(pat)))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as out:
        for i, fp in enumerate(files, start=1):
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            doc = {
                "doc_id": f"ingest_{i:06d}",
                "title": fp.stem.replace("_", " ").replace("-", " "),
                "text": text,
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Ingested {len(files)} files into {output_jsonl}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", help="Folder containing .md/.txt files")
    p.add_argument("output_jsonl", help="Output JSONL path")
    args = p.parse_args()
    ingest_folder(Path(args.input_dir), Path(args.output_jsonl))


if __name__ == "__main__":
    main()



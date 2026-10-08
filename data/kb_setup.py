"""
CRRA Lab C1 - Policy Knowledge Base Setup

Reads the BizOps procurement policy articles in data/kb/, splits each one at its
'## ' headings, and loads the sections into a ChromaDB collection called
'crra_policy' so the Analysis Agent (Lab C3) can cite the exact rule it relied on.

Run from the project root:
    python data/kb_setup.py
"""

from pathlib import Path

import chromadb

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent
KB_DIR = DATA_DIR / "kb"
CHROMA_DIR = DATA_DIR / "chroma_db"      # on-disk store, so Labs C3/C4 can read it
COLLECTION_NAME = "crra_policy"

# Cosine distance makes (1 - distance) equal to cosine similarity, which is a
# meaningful 0..1 "confidence". ChromaDB's default (squared L2) can go below 0.
COLLECTION_METADATA = {"hnsw:space": "cosine"}

TEST_QUERIES = [
    "who approves a 60 lakh contract",
    "contract auto renews next month and we missed the notice deadline",
    "two monitoring tools with low licence usage, should we merge them",
    "vendor wants a 20 percent price increase at renewal",
    "we no longer need this tool at all, how do we exit safely",
]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_markdown(text: str, filename: str) -> list[dict]:
    """Split one policy article into one chunk per '## ' section.

    - The '# ' title line is skipped (it is a label, not a rule).
    - '### ' sub-headings stay inside their parent '## ' section.
    - Any text before the first '## ' (or a file with no '## ' at all) is kept
      under the heading 'Overview' rather than being silently dropped.
    - Empty sections are skipped.
    """
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((heading, content))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            heading = stripped[3:].strip()
            body = []
        elif stripped.startswith("# "):
            continue
        else:
            body.append(line)
    flush()

    stem = Path(filename).stem
    return [
        {
            "id": f"{stem}::{i:02d}",
            # Heading is embedded with the body so it contributes to the match.
            "document": f"{h}\n\n{content}",
            "metadata": {"source": filename, "heading": h, "chunk_index": i},
        }
        for i, (h, content) in enumerate(sections)
    ]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def build_collection(client: chromadb.ClientAPI):
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {KB_DIR} - check the folder path.")

    # Delete and recreate so re-runs never stack duplicate chunks.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: collection does not exist yet
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata=COLLECTION_METADATA
    )

    ids, docs, metas = [], [], []
    print("Loading policy articles")
    print("=" * 60)
    for md_file in md_files:
        chunks = chunk_markdown(md_file.read_text(encoding="utf-8"), md_file.name)
        print(f"  {md_file.name:<34} {len(chunks):>2} chunks")
        for c in chunks:
            ids.append(c["id"])
            docs.append(c["document"])
            metas.append(c["metadata"])

    collection.add(ids=ids, documents=docs, metadatas=metas)
    print("=" * 60)
    print(f"  TOTAL: {len(ids)} chunks from {len(md_files)} articles\n")
    return collection


# --------------------------------------------------------------------------- #
# Verify retrieval
# --------------------------------------------------------------------------- #
def run_test_queries(collection) -> None:
    print("Testing retrieval")
    print("=" * 60)
    for q in TEST_QUERIES:
        res = collection.query(
            query_texts=[q], n_results=1, include=["metadatas", "distances"]
        )
        meta = res["metadatas"][0][0]
        confidence = 1 - res["distances"][0][0]
        print(f'  Q: "{q}"')
        print(f"     -> {meta['source']}  | {meta['heading']}  "
              f"| confidence {confidence:.0%}\n")
    print("=" * 60)
    print(f"KB ready at {CHROMA_DIR} (collection '{COLLECTION_NAME}').")


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = build_collection(client)
    run_test_queries(collection)


if __name__ == "__main__":
    main()
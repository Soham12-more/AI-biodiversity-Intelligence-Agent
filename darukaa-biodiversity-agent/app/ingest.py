"""Rebuild the index after adding PDFs/txt/md to kb/docs:  python -m app.ingest"""
from .knowledge import get_retriever, load_kb


def main():
    load_kb.cache_clear()
    get_retriever.cache_clear()
    r = get_retriever()
    r.save()
    docs = sum(1 for c in r.kb.chunks if c.kind == "doc")
    print(f"Indexed {len(r.kb.chunks)} chunks ({len(r.kb.evidence)} evidence cards, {docs} document chunks), mode={r.mode}")


if __name__ == "__main__":
    main()

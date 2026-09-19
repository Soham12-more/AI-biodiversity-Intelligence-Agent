"""Terminal chat: python -m app.cli   (add --json '{...}' to send structured input first)."""
import argparse
import json

from .dialogue import handle
from .models import ChatRequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="structured site JSON to send as the first turn")
    a = ap.parse_args()
    sid = None
    if a.json:
        r = handle(ChatRequest(site=json.loads(a.json)))
        sid = r.session_id
        print(r.text, "\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in {"exit", "quit"}:
            break
        r = handle(ChatRequest(session_id=sid, message=msg))
        sid = r.session_id
        print("\n" + r.text + "\n")
        for w in r.warnings:
            print("! " + w)


if __name__ == "__main__":
    main()

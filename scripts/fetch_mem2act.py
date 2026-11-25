"""Download a pinned public snapshot, in bounded chunks, with Git blob checksums."""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error


API = "https://api.github.com/repos/Cantaloupe-M/Mem2ActBench"


def request(url, extra=None):
    headers = {"User-Agent": "context-bundles-research", **(extra or {})}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 429):
                raise RuntimeError("GitHub access/rate limit: resume later or configure your own GITHUB_TOKEN") from None
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/mem2act-small"))
    p.add_argument("--ref", default="HEAD", help="Commit SHA recommended; resolved and recorded before download")
    p.add_argument("--resume", action="store_true", help="Continue a checksum-verified pinned snapshot")
    p.add_argument("--variant", choices=["small", "full"], default="small")
    p.add_argument("--chunk-bytes", type=int, default=524288)
    args = p.parse_args()
    if args.chunk_bytes < 32768:
        raise ValueError("chunk-bytes must be at least 32768")
    if args.out.exists() and not args.resume:
        raise FileExistsError("Choose a new output directory; existing snapshots are never overwritten")
    previous = args.out / "snapshot.json"
    manifest = json.loads(previous.read_text()) if args.resume and previous.exists() else None
    if manifest and manifest.get("variant", "small") != args.variant:
        raise ValueError("Existing snapshot uses a different variant")
    commit = manifest["commit"] if manifest else json.loads(request(API + "/commits/" + args.ref)[2])["sha"]
    tree = json.loads(request(API + "/git/trees/" + commit + "?recursive=1")[2])
    entries = {e["path"]: e for e in tree["tree"]}
    files = ["qa_dataset.jsonl", "Mem2ACT_conversation.jsonl", "benchmark_statistics.json", "README.md"]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = manifest or {"repository": "https://github.com/Cantaloupe-M/Mem2ActBench", "commit": commit, "files": {}}
    manifest["variant"] = args.variant
    previous.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name in files:
        path = ("Mem2ACTBench_small/" if args.variant == "small" else "Mem2ACTBench/") + name
        entry = entries[path]
        size = entry["size"]
        existing = args.out / name
        if existing.exists():
            saved = existing.read_bytes()
            if hashlib.sha1(f"blob {len(saved)}\0".encode() + saved).hexdigest() == entry["sha"]:
                print(f"Already verified {name}", flush=True)
                continue
            raise ValueError(f"Existing file failed verification: {name}")
        url = API + "/contents/" + path + "?ref=" + commit
        ranges = [(i, min(i + args.chunk_bytes, size) - 1) for i in range(0, size, args.chunk_bytes)]
        def chunk(bounds):
            a, b = bounds
            part = args.out / f".{entry['sha']}.{a}-{b}.part"
            body = part.read_bytes() if part.exists() else b""
            if len(body) > b - a + 1:
                raise ValueError("Invalid cached chunk length")
            for attempt in range(8):
                if len(body) == b - a + 1:
                    return body
                start = a + len(body)
                headers = {"User-Agent": "context-bundles-research", "Accept": "application/vnd.github.raw+json",
                           "Range": f"bytes={start}-{b}"}
                if os.environ.get("GITHUB_TOKEN"):
                    headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as r:
                        if r.status != 206 or r.headers.get("Content-Range") != f"bytes {start}-{b}/{size}":
                            if start != 0 or b != size - 1 or r.status != 200:
                                raise ValueError("Server returned an unexpected byte range")
                        try:
                            more = r.read()
                        except http.client.IncompleteRead as exc:
                            more = exc.partial
                    if not more or len(more) > b - start + 1:
                        raise ValueError("Invalid byte range body")
                    body += more
                    part.write_bytes(body)
                except urllib.error.HTTPError as exc:
                    if exc.code in (401, 403, 429):
                        raise RuntimeError("GitHub access/rate limit: resume later or configure your own GITHUB_TOKEN") from None
                    if attempt == 7:
                        raise
                    time.sleep(1)
                except (OSError, http.client.HTTPException):
                    if attempt == 7:
                        raise
                    time.sleep(1)
            if len(body) == b - a + 1:
                return body
            raise RuntimeError("Incomplete range remains on disk; resume the pinned snapshot")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            raw = b"".join(pool.map(chunk, ranges))
        blob_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob_hash != entry["sha"]:
            raise ValueError(f"Git blob checksum mismatch: {name}")
        (args.out / name).write_bytes(raw)
        for a, b in ranges:
            (args.out / f".{entry['sha']}.{a}-{b}.part").unlink(missing_ok=True)
        manifest["files"][name] = {"bytes": len(raw), "git_blob_sha1": blob_hash,
                                     "sha256": hashlib.sha256(raw).hexdigest()}
        (args.out / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Verified {name}: {len(raw)} bytes", flush=True)
    print("Pinned commit:", commit)
    manifest["complete"] = True
    manifest["scientific_evaluation_allowed"] = True
    previous.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

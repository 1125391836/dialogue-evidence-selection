"""Download the tokenizer linked by official DeepSeek docs; do not execute its code."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip"
out = Path("data/tokenizers/deepseek-v4")
out.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(URL, timeout=60) as response:
    raw = response.read()
(out / "download.zip").write_bytes(raw)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    names = archive.namelist()
    print("Archive members:", names)
    candidates = [n for n in names if Path(n).name == "tokenizer.json" and not n.startswith("__MACOSX")]
    if len(candidates) != 1:
        raise ValueError("Expected one tokenizer.json; inspect archive before use")
    tokenizer = archive.read(candidates[0])
    (out / "tokenizer.json").write_bytes(tokenizer)
manifest = {"source": URL, "docs": "https://api-docs.deepseek.com/quick_start/token_usage",
            "download_sha256": hashlib.sha256(raw).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(),
            "scope": "offline historical text only; API usage is billing authority"}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Verified local tokenizer bytes:", len(tokenizer))

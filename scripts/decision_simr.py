"""Official Sim-R successful reservation notifications, evaluated before notification."""
import hashlib
import json
import urllib.request

from context_bundles.core import Case,Evidence
from context_bundles.io import write_json
from decision_pivot import OUT
from ticket_learning import ROOT

PIN="9c5752ffa60d00c58efe52ce2ec6a29f2c2bb47b"
URL=f"https://raw.githubusercontent.com/google-research-datasets/simulated-dialogue/{PIN}/sim-R/test.json"
FIELDS={"restaurant_name","date","time","num_people"}
PROMPT=("Recover the restaurant reservation accepted by the user from the supplied dialogue. "
        "Return only a JSON object using restaurant_name, date (words as spoken), time (e.g. 7 pm), "
        "and num_people (numeric string). Copy accepted values. Omit unknown fields. Do not invent values.")


def cases():
    folder=ROOT/"data/external/sim-r"
    folder.mkdir(exist_ok=True)
    file=folder/(PIN+"-test.json")
    if not file.exists():
        with urllib.request.urlopen(URL,timeout=60) as f:
            file.write_bytes(f.read())
    data=file.read_bytes()
    rows=[]
    seen=set()
    dialogues=json.loads(data)
    for d in dialogues:
        key=d["dialogue_id"]
        if key=="restaurant_00000001":
            continue
        public=[]
        candidate=None
        for t in d["turns"]:
            gold={a["slot"]:a["value"].casefold().strip() for a in t.get("system_acts",[])
                  if a["type"]=="NOTIFY_SUCCESS" and a.get("slot") in FIELDS}
            if FIELDS==gold.keys() and public and public[-1][0]=="U":
                ev=tuple(Evidence(str(i),r+": "+text,i,key) for i,(r,text) in enumerate(public[:-1]))
                case=Case(key,public[-1][1],{},ev,key,"sim-r")
                fp=hashlib.sha256(json.dumps(public,ensure_ascii=False).encode()).hexdigest()
                candidate=(key,case,gold,"sim-R/test.json",fp)
            for r,field in [("A","system_utterance"),("U","user_utterance")]:
                if t.get(field,{}).get("text"):
                    public.append((r,t[field]["text"]))
        if candidate and candidate[-1] not in seen:
            rows.append(candidate)
            seen.add(candidate[-1])
    rows.sort(key=lambda r:hashlib.sha256(("simr-decision-v1/"+r[0]).encode()).digest())
    write_json(OUT/"simr-sampling.json",dict(source=URL,sha256=hashlib.sha256(data).hexdigest(),
               screened=len(dialogues),eligible=len(rows),sample=min(64,len(rows))))
    return rows[:64]

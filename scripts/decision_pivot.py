"""Independent Park-style Pivot comparator with training-only SGD state deltas."""
import hashlib
import json
import sys
from functools import lru_cache

import numpy as np
from scipy.special import expit
from tokenizers import Tokenizer

from context_bundles.io import sha256, write_json
from ticket_learning import ROOT

OUT = ROOT / "runs/decision-evidence"
ASSETS = ROOT / "data/external/review-reader"


class Pivot:
    def __init__(self):
        import onnx
        sys.path.insert(0, str(ROOT / "runs/review-benchmarks/runtime"))
        import onnxruntime as ort
        bert = ASSETS/"bert-base.onnx"
        tok = ASSETS/"bert-tokenizer.json"
        assert sha256(bert) == "f2c1ce78f723dfa260bf743c56596ddd5097e45081cdffa24820aab900855ed0"
        assert sha256(tok) == "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
        graph = onnx.load(bert)
        hidden = "/bert/encoder/layer.11/output/LayerNorm/Add_1_output_0"
        graph = onnx.utils.Extractor(graph).extract_model([x.name for x in graph.graph.input], [hidden])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 3
        self.encoder = ort.InferenceSession(graph.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tok))
        self.tokenizer.enable_truncation(256)
        self.parameters = None

    @lru_cache(maxsize=45000)
    def encode(self, previous, current):
        seq = self.tokenizer.encode(previous,current)
        inputs = {key:np.array([v],dtype=np.int64) for key,v in
                  [("input_ids",seq.ids),("attention_mask",seq.attention_mask),("token_type_ids",seq.type_ids)]}
        return self.encoder.run(None,inputs)[0][0,0].copy()

    def train(self):
        from threadpoolctl import threadpool_limits
        folder = ROOT / "data/external/sgd-revision-audit/train"
        sources = sorted(folder.glob("dialogues_*.json"))
        dialogues = [d for p in sources for d in json.loads(p.read_bytes())
                     if not any(s.startswith("Restaurants") for s in d["services"])]
        key = lambda d: hashlib.sha256(("pivot-training/"+d["dialogue_id"]).encode()).hexdigest()
        cohort = sorted(dialogues,key=key)[:256]
        features,labels = [],[]
        for d in cohort:
            state, prev = {}, ""
            for turn in d["turns"]:
                before = state.copy()
                for fr in turn["frames"]:
                    if "state" in fr:
                        service=fr["service"]
                        state = {k:v for k,v in state.items() if k[0] != service}
                        state.update({(service,k):tuple(sorted(v)) for k,v in fr["state"]["slot_values"].items() if v})
                cur = turn["utterance"]
                features.append(self.encode(prev,cur))
                labels.append(state!=before)
                prev = cur
        x,y = np.array(features), np.array(labels,dtype=np.float32)[:,None]
        rng = np.random.default_rng(42)
        params = [rng.normal(0,.02,(768,128)).astype("float32"),np.zeros(128,"float32"),
                  rng.normal(0,.02,(128,1)).astype("float32"),np.zeros(1,"float32")]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        with threadpool_limits(1):
            for _ in range(2):
                indices = rng.permutation(len(y))
                for off in range(0,len(y),32):
                    idx = indices[off:off+32]
                    h = np.maximum(x[idx]@params[0]+params[1],0)
                    delta = (expit(h@params[2]+params[3])-y[idx])/len(idx)
                    dh = (delta@params[2].T)*(h>0)
                    grads = [x[idx].T@dh,dh.sum(axis=0),h.T@delta,delta.sum(axis=0)]
                    step += 1
                    for i,g in enumerate(grads):
                        m[i] = .9*m[i]+.1*g
                        v[i] = .999*v[i]+.001*g*g
                        if i%2==0:
                            params[i] *= 1-3e-5*.01
                        params[i] -= 3e-5*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        self.parameters = params
        OUT.mkdir(exist_ok=True)
        np.savez(OUT/"pivot-head.npz",*params)
        write_json(OUT/"pivot-training.json",dict(sha256=sha256(OUT/"pivot-head.npz"),
                   n=len(y), positive=int(y.sum()),dialogues=[d["dialogue_id"] for d in cohort],
                   files={p.name:sha256(p) for p in sources}, bert=sha256(ASSETS/"bert-base.onnx")))

    def scores(self, case):
        previous = ""
        x = []
        for e in case.evidence:
            current = e.text.partition(": ")[2]
            x.append(self.encode(previous,current))
            previous = current
        if not x:
            return []
        w,b,w2,b2 = self.parameters
        return (np.maximum(np.array(x)@w+b,0)@w2+b2).ravel()


def load_pivot():
    model = Pivot()
    path = OUT/"pivot-head.npz"
    if sha256(path) != json.loads((OUT/"pivot-training.json").read_bytes())["sha256"]:
        raise ValueError("Pivot head digest differs")
    with np.load(path) as data:
        model.parameters = [data[k] for k in data.files]
    return model


if __name__ == "__main__":
    Pivot().train()

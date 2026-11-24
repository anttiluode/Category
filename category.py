"""Category: ask a frozen Qwen what a set of words shares.

Each word's hidden state is taken at its own position in "Item: <word>". States
are averaged, blended or combined layer by layer and clamped into an empty slot,
"Item: thing", from the token embedding through layer ~12 (the RakeisuusClaude
band; nothing of the slot word survives around it). Then Qwen is asked about the
slot: what kind of thing it is, to describe it, and yes/no about each word.

    python category.py                      start the app (opens the browser)
    python category.py --self-test          plumbing test on a tiny random model
    python category.py --ask "oak, pine, elm"   one share query in the terminal

See README.md for what each tab measures and what it does not show.
"""
import argparse
import inspect
import json
import math
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

MODEL = "Qwen/Qwen3-1.7B"        # same checkpoint as RakeisuusClaude and GrainLensSol
SLOT = "thing"                   # the empty slot the combined state is written into
ITEM = "Item: {w}\n\n{q}"
KIND_Q = "What kind of thing is the item above? Answer with one word."
DESC_Q = "Describe the item above in one short sentence. Do not name it."
YESNO_Q = "Is the item above {a} {x}? Answer yes or no."
YES, NO = ["Yes", "yes"], ["No", "no"]
HERE = Path(__file__).resolve().parent


def _hs(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _replace(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


def article(x):
    return "an" if x[:1].lower() in "aeiou" else "a"


def split_words(text):
    """Commas separate if present (so "fire truck" can be one entry), else spaces."""
    text = text.strip()
    parts = text.split(",") if "," in text else text.split()
    return [p.strip() for p in parts if p.strip()]


# ================================================================ the engine
class Engine:
    def __init__(self, model, tok, device, name, band_end=None):
        self.model, self.tok, self.device, self.name = model.eval(), tok, device, name
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        self.band = (-1, band_end if band_end is not None
                     else min(self.n_layers - 1, round(12 * self.n_layers / 28)))
        self.lock = threading.Lock()
        self.cache = {}                     # word -> {"S": {layer: [d]}, "ntok": int}
        pad = getattr(tok, "pad_token_id", None)
        self.pad = pad if pad is not None else (getattr(tok, "eos_token_id", None) or 0)
        self.yes, self.no = self._single(YES), self._single(NO)
        if not self.yes or not self.no:
            raise SystemExit("STOP: yes/no are not single tokens for this tokenizer")
        self.stop = {i for i in (getattr(tok, "eos_token_id", None),) if i is not None}
        for s in ("<|im_end|>", "<|endoftext|>"):
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                self.stop.add(ids[0])
        if self.word_span(SLOT, KIND_Q)[1] != 1:
            raise SystemExit(f"STOP: the slot word '{SLOT}' is not one token")
        self._neutral = {}

    # ---- text
    def _single(self, words):
        out = set()
        for w in words:
            for s in (" " + w, w):
                ids = self.tok(s, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    out.add(ids[0])
        return sorted(out)

    def chat(self, user, prefix):
        msgs = [{"role": "user", "content": user}]
        try:
            t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            t = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return t + prefix

    def word_span(self, word, q, prefix="Answer:"):
        """Token ids of the prompt, the word's last token position, and its token count."""
        text = self.chat(ITEM.format(w=word, q=q), prefix)
        ws = text.index("Item: " + word) + len("Item: ")
        we = ws + len(word)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        hit = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < we and b > ws]
        if not hit:
            raise ValueError(f"could not find '{word}' in the prompt tokens")
        return enc["input_ids"], len(hit), hit[-1]

    # ---- states
    @torch.no_grad()
    def states(self, word):
        """The word's residual state at its (last) token, for every layer in the band.
        Causal attention: this depends only on the text up to the word."""
        word = word.strip()
        if word in self.cache:
            return self.cache[word]
        ids, ntok, pos = self.word_span(word, KIND_Q)
        store, hooks = {}, []
        lo, hi = self.band
        for l in range(max(lo, 0), hi + 1):
            hooks.append(self.layers[l].register_forward_hook(
                lambda m, i, o, l=l: store.__setitem__(l, _hs(o)[0, pos].detach().float().cpu())))

        def pre(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            store[-1] = h[0, pos].detach().float().cpu()
        if lo == -1:
            hooks.append(self.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
        try:
            self.model.model(input_ids=torch.tensor([ids], device=self.device))
        finally:
            for h in hooks:
                h.remove()
        self.cache[word] = dict(S=store, ntok=ntok)
        return self.cache[word]

    def combine(self, terms, rescale=False):
        """terms: [(coef, [words])]; a group contributes the mean of its words."""
        out, norms = None, []
        for coef, group in terms:
            Ss = [self.states(w)["S"] for w in group]
            norms += Ss
            g = {l: torch.stack([S[l] for S in Ss]).mean(0) for l in Ss[0]}
            out = {l: coef * v for l, v in g.items()} if out is None else \
                {l: out[l] + coef * g[l] for l in out}
        if rescale:
            for l in out:
                n = sum(float(S[l].norm()) for S in norms) / len(norms)
                out[l] = out[l] * (n / max(float(out[l].norm()), 1e-6))
        return out

    # ---- clamped forward passes in the slot
    def _hooks(self, pos, targets):
        hooks = []
        for l, v in (targets or {}).items():
            if l == -1:
                def pre(mod, args, kwargs, v=v):
                    h = (args[0] if args else kwargs["hidden_states"]).clone()
                    h[:, pos] = v.to(h.device, h.dtype)
                    if args:
                        return (h,) + tuple(args[1:]), kwargs
                    kwargs["hidden_states"] = h
                    return args, kwargs
                hooks.append(self.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(mod, inp, out, v=v):
                    h = _hs(out).clone()
                    h[:, pos] = v.to(h.device, h.dtype)
                    return _replace(out, h)
                hooks.append(self.layers[l].register_forward_hook(post))
        return hooks

    @torch.no_grad()
    def _last_logp(self, prompts, targets):
        """Log-probs at the last token of each prompt (all share the slot position)."""
        encs = [self.word_span(SLOT, q, prefix) for q, prefix in prompts]
        pos = encs[0][2]
        assert all(e[2] == pos for e in encs)
        n = max(len(e[0]) for e in encs)
        x = torch.full((len(encs), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(encs), n), dtype=torch.long)
        for b, (ids, _, _) in enumerate(encs):
            x[b, :len(ids)] = torch.tensor(ids)
            mask[b, :len(ids)] = 1
        last = (mask.sum(1) - 1).to(self.device)
        hooks = self._hooks(pos, targets)
        try:
            hs = self.model.model(input_ids=x.to(self.device),
                                  attention_mask=mask.to(self.device)).last_hidden_state
        finally:
            for h in hooks:
                h.remove()
        h = hs[torch.arange(len(encs), device=hs.device), last]
        return torch.log_softmax(self.model.lm_head(h).float(), -1).cpu()

    def yesno(self, targets, labels):
        """Yes-minus-no evidence (nats) that the slot is <label>, minus the same for an
        empty slot, so a general lean toward yes for some label cancels."""
        if not labels:
            return {}
        qs = [(YESNO_Q.format(a=article(x), x=x), "Answer:") for x in labels]
        need = [x for x in labels if x not in self._neutral]
        if need:
            lp0 = self._last_logp([(YESNO_Q.format(a=article(x), x=x), "Answer:") for x in need], None)
            for x, r in zip(need, lp0):
                self._neutral[x] = float(torch.logsumexp(r[self.yes], 0) - torch.logsumexp(r[self.no], 0))
        lp = self._last_logp(qs, targets)
        raw = (torch.logsumexp(lp[:, self.yes], -1) - torch.logsumexp(lp[:, self.no], -1)).tolist()
        return {x: dict(raw=r, cal=r - self._neutral[x]) for x, r in zip(labels, raw)}

    def kind(self, targets, k=8):
        """Next-token answers to an open question that names no category: the most likely,
        and the most boosted relative to the empty slot."""
        lp = self._last_logp([(KIND_Q, "Answer:")], targets)[0]
        if "kind0" not in self._neutral:
            self._neutral["kind0"] = self._last_logp([(KIND_Q, "Answer:")], None)[0]
        lp0 = self._neutral["kind0"]
        top = lp.topk(k)
        boost = (lp - lp0).masked_fill(lp < math.log(1e-4), -1e9).topk(k)
        dec = lambda i: self.tok.decode([int(i)])
        return dict(top=[dict(t=dec(i), p=float(v.exp())) for v, i in zip(top.values, top.indices)],
                    boosted=[dict(t=dec(i), d=float(v), p=float(lp[i].exp()))
                             for v, i in zip(boost.values, boost.indices)],
                    word=self._first_word([dec(i) for i in top.indices]))

    @staticmethod
    def _first_word(tokens):
        for t in tokens:
            w = t.strip().lower()
            if re.fullmatch(r"[a-z][a-z'-]+", w):
                return w
        return tokens[0].strip()

    @torch.no_grad()
    def describe(self, targets, max_new=48, temperature=0.0, seed=0):
        """Free text about the slot. The clamp applies to the prompt pass only; the
        generated tokens then read the clamped slot through the cache."""
        ids, _, pos = self.word_span(SLOT, DESC_Q, prefix="")
        x = torch.tensor([ids], device=self.device)
        hooks = self._hooks(pos, targets)
        try:
            out = self.model(input_ids=x, use_cache=True)
        finally:
            for h in hooks:
                h.remove()
        g = torch.Generator().manual_seed(seed)
        # transformers renamed past_key_values to cache; support both
        new_api = "cache" in inspect.signature(self.model.forward).parameters
        kw = "cache" if new_api else "past_key_values"
        get = lambda o: o.cache if new_api else o.past_key_values
        past, logits, toks = get(out), out.logits[0, -1].float(), []
        for _ in range(max_new):
            if temperature > 0:
                p = torch.softmax(logits.cpu() / temperature, -1)
                nxt = int(torch.multinomial(p, 1, generator=g))
            else:
                nxt = int(logits.argmax())
            if nxt in self.stop:
                break
            toks.append(nxt)
            out = self.model(input_ids=torch.tensor([[nxt]], device=self.device),
                             use_cache=True, **{kw: past})
            past, logits = get(out), out.logits[0, -1].float()
        return self.tok.decode(toks).strip()

    # ---- the queries behind the tabs
    def info(self, words):
        return [dict(word=w, ntok=self.states(w)["ntok"]) for w in words]

    def share(self, words, describe=True, temperature=0.0):
        """Average all words into the slot. Compare with each word alone."""
        mean_state = self.combine([(1.0, words)])
        alone = []
        for w in words:
            T = self.combine([(1.0, [w])])
            k = self.kind(T, k=3)
            alone.append(dict(word=w, kind=k["word"], top=k["top"][:3]))
        res = dict(words=self.info(words), kind=self.kind(mean_state), alone=alone)
        res["members"] = self.yesno(mean_state, words)
        res["kind_check"] = self.yesno(mean_state, [res["kind"]["word"]])
        if describe:
            res["description"] = self.describe(mean_state, temperature=temperature)
        return res

    def odd(self, words):
        """Leave each word out; the others' average names a kind; ask whether the left-out
        word alone is that kind. The lowest evidence is the odd one out."""
        if len(words) < 3:
            raise ValueError("odd one out needs at least 3 words")
        rows = []
        for i, w in enumerate(words):
            others = words[:i] + words[i + 1:]
            k = self.kind(self.combine([(1.0, others)]), k=3)["word"]
            ev_w = self.yesno(self.combine([(1.0, [w])]), [k])[k]["cal"]
            ev_o = self.yesno(self.combine([(1.0, others)]), [k])[k]["cal"]
            rows.append(dict(word=w, others_kind=k, word_is_kind=ev_w, others_are_kind=ev_o))
        rows.sort(key=lambda r: r["word_is_kind"])
        return dict(rows=rows, odd=rows[0]["word"])

    def blend(self, words, weights, describe=True, temperature=0.0):
        tot = sum(weights)
        if abs(tot) < 1e-9:
            raise ValueError("weights sum to zero")
        terms = [(wt / tot, [w]) for w, wt in zip(words, weights)]
        T = self.combine(terms)
        res = dict(words=self.info(words), weights=[wt / tot for wt in weights], kind=self.kind(T))
        res["members"] = self.yesno(T, words)
        if describe:
            res["description"] = self.describe(T, temperature=temperature)
        return res

    def caricature(self, word, refs, alpha, sweep=True, describe=True, temperature=0.0):
        """state = word + alpha * (word - mean(refs)). alpha = -1 is the refs' average,
        0 is the word itself, above 0 exaggerates what sets the word apart."""
        if word in refs:
            refs = [r for r in refs if r != word]
        if not refs:
            raise ValueError("caricature needs reference words other than the word itself")
        ref_kind = self.kind(self.combine([(1.0, refs)]), k=3)["word"]
        labels = [word, ref_kind]

        def state(a):
            return self.combine([(1.0 + a, [word]), (-a, refs)])
        T = state(alpha)
        res = dict(word=word, refs=refs, alpha=alpha, ref_kind=ref_kind, kind=self.kind(T),
                   evidence=self.yesno(T, labels))
        if describe:
            res["description"] = self.describe(T, temperature=temperature)
        if sweep:
            res["sweep"] = []
            for a in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                e = self.yesno(state(a), labels)
                res["sweep"].append(dict(alpha=a, word=e[word]["cal"], kind=e[ref_kind]["cal"]))
        return res

    def formula(self, text, rescale=True, describe=True, temperature=0.0):
        terms = parse_formula(text)
        T = self.combine(terms, rescale=rescale)
        words = list(dict.fromkeys(w for _, g in terms for w in g))
        res = dict(terms=[dict(coef=c, group=g) for c, g in terms], words=self.info(words),
                   kind=self.kind(T), members=self.yesno(T, words))
        if describe:
            res["description"] = self.describe(T, temperature=temperature)
        return res


def parse_formula(text):
    """'1.5*maple - 0.5*[oak pine elm] + bird' -> [(1.5, ['maple']), (-0.5, [...]), (1, ['bird'])]"""
    s = text.strip()
    pat = re.compile(r"\s*([+-])?\s*(\d*\.?\d+)?\s*\*?\s*(\[[^\]]*\]|[A-Za-z][A-Za-z'-]*)\s*")
    terms, i = [], 0
    while i < len(s):
        m = pat.match(s, i)
        if not m or m.end() == i:
            raise ValueError(f"cannot read the formula near: '{s[i:i + 15]}'")
        if terms and not m.group(1):
            raise ValueError(f"missing + or - before '{m.group(3)}'")
        sign = -1.0 if m.group(1) == "-" else 1.0
        coef = sign * (float(m.group(2)) if m.group(2) else 1.0)
        g = m.group(3)
        group = split_words(g[1:-1]) if g.startswith("[") else [g]
        if not group:
            raise ValueError("empty [ ] group")
        terms.append((coef, group))
        i = m.end()
    if not terms:
        raise ValueError("empty formula")
    return terms


# ================================================================ loading
def load_hf(name, device, dtype, band_end=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return Engine(model.to(device), tok, device, name, band_end)


class StubTokenizer:
    """Word-level tokenizer with offsets and a Qwen-style chat template (tests only)."""
    PAT = re.compile(r"<\|im_start\|>|<\|im_end\|>| ?[A-Za-z']+| ?[^\sA-Za-z]|\n")
    pad_token_id = 2
    eos_token_id = 1

    def __init__(self, texts):
        self.vocab = ["<|im_start|>", "<|im_end|>", "<pad>", "<unk>"]
        seen = set(self.vocab)
        for t in texts:
            for p in self.PAT.findall(t):
                for q in (p, " " + p.strip(), p.strip()):
                    if q and q not in seen:
                        seen.add(q)
                        self.vocab.append(q)
        self.idx = {w: i for i, w in enumerate(self.vocab)}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ms = list(self.PAT.finditer(text))
        out = {"input_ids": [self.idx.get(m.group(), 3) for m in ms]}
        if return_offsets_mapping:
            out["offset_mapping"] = [m.span() for m in ms]
        return out

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


STUB_WORDS = "oak pine elm maple tree piano guitar drum hammer car bus bird".split()


def load_stub(layers=8, d=64):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    texts = [ITEM, KIND_Q, DESC_Q, YESNO_Q, "Answer:", SLOT, "a an"] + YES + NO + STUB_WORDS
    tok = StubTokenizer(texts)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok.vocab), hidden_size=d, intermediate_size=2 * d,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    return Engine(Qwen2ForCausalLM(cfg), tok, "cpu", f"stub-L{layers}-d{d}")


def self_test():
    e = load_stub()
    w = "oak"
    # 1. clamping a word's own states into its own prompt changes nothing
    ids, _, pos = e.word_span(w, KIND_Q)
    x = torch.tensor([ids])
    with torch.no_grad():
        a = e.model(input_ids=x).logits[0, -1]
        hooks = e._hooks(pos, e.states(w)["S"])
        b = e.model(input_ids=x).logits[0, -1]
        for h in hooks:
            h.remove()
    assert torch.allclose(a, b, atol=1e-4), "identity clamp changed the output"
    # 2. no bypass: a word clamped into the slot from the embedding through every layer
    #    gives exactly the answers of the prompt that prints the word
    full = Engine(e.model, e.tok, "cpu", e.name, band_end=e.n_layers - 1)
    lp_slot = full._last_logp([(KIND_Q, "Answer:")], full.states(w)["S"])[0]
    ids_w, _, _ = full.word_span(w, KIND_Q)
    with torch.no_grad():
        lp_word = torch.log_softmax(full.model(input_ids=torch.tensor([ids_w])).logits[0, -1], -1)
    assert torch.allclose(lp_slot, lp_word, atol=1e-4), "the slot word leaked around the clamp"
    # 3. a group of one equals the word; formula parsing
    assert all(torch.allclose(e.combine([(1.0, [w])])[l], e.states(w)["S"][l]) for l in e.states(w)["S"])
    assert parse_formula("1.5*maple - 0.5*[oak pine elm] + bird") == \
        [(1.5, ["maple"]), (-0.5, ["oak", "pine", "elm"]), (1.0, ["bird"])]
    # 4. caricature at alpha = -1 is the refs' average, at 0 the word
    car = e.combine([(0.0, ["maple"]), (1.0, ["oak", "pine"])])
    ref = e.combine([(1.0, ["oak", "pine"])])
    assert all(torch.allclose(car[l], ref[l], atol=1e-5) for l in car)
    # 5. every query runs end to end, and generation leaves the hooks clean
    e.share(["oak", "pine", "elm"], describe=True)
    e.odd(["oak", "pine", "elm", "piano"])
    e.blend(["piano", "guitar"], [1, 1])
    e.caricature("maple", ["oak", "pine", "elm"], 1.0)
    e.formula("maple - tree + bird")
    for l in e.layers:
        assert not l._forward_hooks and not l._forward_pre_hooks, "a hook was left behind"
    # 6. the cached generation loop matches transformers' own greedy generate
    stop, e.stop = e.stop, set()
    dec, e.tok.decode = e.tok.decode, lambda t: " ".join(map(str, t))
    ids, _, _ = e.word_span(SLOT, DESC_Q, prefix="")
    with torch.no_grad():
        g = e.model.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False,
                             pad_token_id=e.pad, eos_token_id=None)
    assert e.describe(None, max_new=8) == " ".join(map(str, g[0, len(ids):].tolist())), "generation differs"
    e.stop, e.tok.decode = stop, dec
    print(f"self-test passed (stub {e.name}, band {e.band}); numbers from a random model mean nothing")


# ================================================================ the app
ROUTES = {
    "share": lambda e, d: e.share(split_words(d["words"]), d.get("describe", True), d.get("temperature", 0.0)),
    "odd": lambda e, d: e.odd(split_words(d["words"])),
    "blend": lambda e, d: e.blend(split_words(d["words"]), [float(x) for x in d["weights"]],
                                  d.get("describe", True), d.get("temperature", 0.0)),
    "caricature": lambda e, d: e.caricature(d["word"].strip(), split_words(d["refs"]), float(d["alpha"]),
                                            d.get("sweep", True), d.get("describe", True),
                                            d.get("temperature", 0.0)),
    "formula": lambda e, d: e.formula(d["formula"], d.get("rescale", True), d.get("describe", True),
                                      d.get("temperature", 0.0)),
}


def make_handler(engine):
    page = (HERE / "index.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/info":
                return self._send(200, json.dumps(dict(
                    model=engine.name, layers=engine.n_layers, band=engine.band, slot=SLOT,
                    device=str(engine.device))).encode())
            self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            name = self.path.removeprefix("/api/")
            if name not in ROUTES:
                return self._send(404, b'{"error":"not found"}')
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                t0 = time.time()
                with engine.lock:
                    res = ROUTES[name](engine, d)
                res["seconds"] = round(time.time() - t0, 2)
                self._send(200, json.dumps(res).encode())
            except (ValueError, KeyError) as ex:
                self._send(400, json.dumps(dict(error=str(ex))).encode())
            except Exception as ex:          # show model errors in the page, keep serving
                self._send(500, json.dumps(dict(error=f"{type(ex).__name__}: {ex}")).encode())
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--band-end", type=int, default=None,
                    help="last clamped layer (default about 12 of 28; the clamp starts at the embedding)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--stub", action="store_true", help="tiny random model, to try the page without Qwen")
    ap.add_argument("--ask", default=None, help="one share query in the terminal, e.g. 'oak, pine, elm'")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.stub:
        engine = load_stub()
    else:
        print(f"loading {a.model} on {a.device} ({a.dtype}) ...")
        engine = load_hf(a.model, a.device, getattr(torch, a.dtype), a.band_end)
    print(f"ready: {engine.n_layers} layers, clamp band {engine.band}, slot 'Item: {SLOT}'")
    if a.ask:
        print(json.dumps(engine.share(split_words(a.ask)), indent=1))
        return
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(engine))
    url = f"http://127.0.0.1:{a.port}/"
    print(f"open {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

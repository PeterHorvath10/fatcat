#!/usr/bin/env python3
"""The deployed TensorRT policy chain, loaded for the Jetson-side tools (imported, not run).

load_backend() loads the runtime's engines as the rollout does. Chain splits one inference at its
hand-offs, so a tool can change what passes between two stages (one camera's image tokens, the
prefill's key/value cache, the action expert's tensors in one denoising step); with nothing changed
the result is the rollout's chunk. check_rss() stops a tool before the Jetson runs out of memory.
"""

import sys

import numpy as np

from site_config import AOT_CACHE, BUNDLE
from site_config import VLA_REPO as REPO
TASK = "pick up the red cube and place it in the transparent box"


def load_backend(chain="host"):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "bench/aot/smolvla"))
    from candidate_cache import verify_candidate
    from runtime_memory import install_lean_tokenizer
    install_lean_tokenizer()
    from bench.vendor import groot_trt
    from bench.backends.trt_split_smolvla import TrtSplitSmolVLABackend
    manifest = verify_candidate(AOT_CACHE, BUNDLE)
    groot_trt.ACCUMULATE = manifest.get("expert_accumulate") or manifest.get("accumulate") or "auto"
    groot_trt.prebuild_engines = lambda *a, **k: {}          # the cache is built; never rebuild
    backend = TrtSplitSmolVLABackend(BUNDLE, str(AOT_CACHE), action_dim=6, chain=chain)
    backend.load()
    if backend.fixture_parity.get("status") != "PASS":
        raise RuntimeError(f"fixture parity failed: {backend.fixture_parity}")
    return backend


class Chain:
    """Policy.sample split at its hand-offs (same arithmetic, so the clean result is identical)."""

    def __init__(self, backend):
        from bench.vendor import smolvla_trt as st
        self.st, self.p, self.b, self.norm = st, backend.policy, backend.bundle, backend.norm

    def inputs(self, top, side, state):
        st = self.st
        pix = [st.preprocess(im) for im in (top, side)]
        lang = self.b.language(TASK)
        s = st.pad_state(self.norm.normalize_state(np.asarray(state, np.float32).reshape(-1)))
        return pix, lang, s

    def vision(self, pix):
        return [self.p.vision(x) for x in pix]

    def prefill(self, img_embs, lang, state):
        st, p, b = self.st, self.p, self.b
        lang_emb, lang_mask = lang
        n_real, n_pad = len(img_embs), b.n_cam_slots - len(img_embs)
        state_emb = p.one("state_proj", state).reshape(1, 1, st.VLM_DIM)
        embs = np.concatenate(img_embs + [p.pad_cam] * n_pad + [lang_emb, state_emb], axis=1).astype(np.float32)
        pad_masks = np.concatenate([np.ones((1, st.IMG_TOKENS), bool)] * n_real
                                   + [np.zeros((1, st.IMG_TOKENS), bool)] * n_pad
                                   + [lang_mask, np.ones((1, 1), bool)], axis=1)
        att_masks = np.zeros((1, b.prefix_len), bool)
        att_masks[0, -1] = True
        kv = p.run("prefill", {"attention_mask": st.make_att_2d_masks(pad_masks, att_masks),
                               "position_ids": (np.cumsum(pad_masks, axis=1) - 1).astype(np.int64),
                               "vlm_embeds": embs})
        return {k.replace("present_", "past_"): v for k, v in kv.items()}, pad_masks

    def denoise(self, past, pad_masks, noise, hook=None):
        """hook(step, name, arr) -> arr sees every hand-off of every step: "xt" the action state
        entering the step, "expin" / "expout" the expert's input / output tokens, "vel" the velocity."""
        st, p, b = self.st, self.p, self.b
        suffix = np.ones((1, b.chunk_size), bool)
        full_att = np.concatenate([np.broadcast_to(pad_masks[:, None, :], (1, b.chunk_size, b.prefix_len)),
                                   st.make_att_2d_masks(suffix, suffix)], axis=2)
        pos = (pad_masks.sum(axis=-1, keepdims=True) + np.cumsum(suffix, axis=1) - 1).astype(np.int64)
        x_t = noise.astype(np.float32).copy()
        dt = -1.0 / b.num_steps
        hook = hook or (lambda step, name, arr: arr)
        for step in range(b.num_steps):
            x_t = hook(step, "xt", x_t)
            tt = 1.0 + step * dt
            a = p.one("action_in", x_t)
            te = np.broadcast_to(st.sinusoidal_time_embedding(tt)[None, None, :], a.shape)
            h = p.one("time_in", np.concatenate([a, te], axis=2).astype(np.float32))
            h = h * (1.0 / (1.0 + np.exp(-h)))
            suf = hook(step, "expin", p.one("time_out", h))
            out = hook(step, "expout", next(iter(p.run("decode", {"attention_mask": full_att, "position_ids": pos,
                                                                  "expert_embeds": suf, **past}).values())))
            x_t = x_t + np.float32(dt) * hook(step, "vel", p.one("action_out", out))
        return x_t

    def chunk(self, x_t):
        return np.asarray(self.norm.unnormalize_action(x_t[0, :, :6]), np.float64)


RSS_LIMIT_MB = 3500                        # the Orin Nano has 7.5 GB shared with the GPU


def check_rss():
    rss = int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) // 1024
    if rss > RSS_LIMIT_MB:
        raise MemoryError(f"RSS {rss} MB > {RSS_LIMIT_MB} MB, stopping before the Jetson runs out")
    return rss

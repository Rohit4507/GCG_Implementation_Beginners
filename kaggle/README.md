# Running the real-model matrix on free-tier GPU (Kaggle)

Total cost of the whole experimental matrix at 1–2B model sizes: **£0**.
Kaggle gives 2×T4 (16 GB each) and 30 GPU-hours per week, which is enough for
roughly one week's worth of runs per weekly quota.

Open **`real_model_runbook.ipynb`** — it is the executable version of this page.

---

## 1. One-time setup (about 10 minutes)

1. **Kaggle account** → https://www.kaggle.com
2. **Verify your phone number.** Without it you cannot turn the Internet on in a
   notebook, and without Internet you cannot `git clone` or download models.
   (Account → Settings → Phone Verification.)
3. **Create the notebook:** Kaggle → Create → New Notebook.
4. **Notebook settings** (right sidebar):
   * `Accelerator`: **GPU T4 x2**
   * `Internet`: **On**
5. **Add a GitHub token as a Kaggle Secret** (only needed if you want runs to
   commit themselves back to GitHub):
   * GitHub → Settings → Developer settings → Personal access tokens →
     Fine-grained tokens → New token
   * Scope: **repository contents: read and write** on your repo only
   * Kaggle → Add-ons → Secrets → add a secret named **`GITHUB_TOKEN`**
6. **Upload the notebook:** File → Import Notebook → `real_model_runbook.ipynb`, or
   paste the cells manually.
7. In the notebook, set `REPO_URL` to your repository.

---

## 2. Quota discipline

Kaggle's weekly allowance is the actual constraint, not money. It is very easy
to burn 10 hours debugging on GPU when the same bug would have appeared in
3 seconds on your laptop.

| Do this on **your laptop** (CPU, free, seconds) | Do this on **Kaggle** (GPU) |
|---|---|
| `python -m experiments.run_smoke` | anything with a real `--model …` |
| `python -m unittest discover -s tests` | `--load-in-4bit` runs |
| every `--tiny` run | `--steps 500` runs |
| editing code, fixing bugs | final numbers |

**Rule:** if the command has no `--model <hf-id>`, it does not belong on Kaggle.

| Item | Budget |
|---|---|
| Session length | ~9–12 h; not resumable |
| Weekly GPU hours | 30 |
| Storage that survives "Save Version" | `/kaggle/working` only |

---

## 3. Order of operations (and why)

```
1. run_smoke + unit tests           ~10 s   prove the code is correct
2. --steps 5 timing probe           ~3 min  decide what actually fits
3. Gate 1: one real attack          ~10 min first non-zero ASR, or the bug
4. Algorithm 2 universal × 3 models ~6-12 h the main results table
5. Transfer matrix (2×T4)           ~1 h    generation only, cheap
6. Multi-turn (3-turn)              ~2-3 h  the plan's missing comparison
7. Baselines (random search)        ~1-2 h  context for every number
```

**Step 2 is not optional.** If one step takes 60 s, 500 steps is 8 hours for a
*single* prompt, and the 25-prompt table will not fit. Find that out in minute
three, not hour ten. Adjust model size or prompt count, then continue.

---

## 4. Getting results back

Three options, best first:

1. **Git push-back (recommended).** With `GITHUB_TOKEN` set, the last cell of
   the runbook commits and pushes. The commits are genuinely made from Kaggle,
   under your git identity.
2. **Save Version.** Top-right → Save Version → Quick Save. Everything under
   `/kaggle/working` is kept in the Output tab, downloadable.
3. **Manual download.** Right sidebar → Output → pick files. Fine for the small
   `results/reports/generated/*.json` records.

Copy results into `/kaggle/working` before saving — the runbook's last-but-one
cell does this.

---

## 5. Pitfalls

| Pitfall | Consequence | Fix |
|---|---|---|
| `pip install -r requirements.txt` | reinstalls torch, wastes ~10 min | runbook installs only `bitsandbytes accelerate datasets` |
| Internet left **Off** | `git clone` and model downloads both fail | Settings → Internet on (needs phone verification) |
| Writing outside `/kaggle/working` | everything lost when the session ends | keep artefacts under `/kaggle/working` |
| Jumping straight to 500 steps | session dies at hour 11 with nothing saved | run the `--steps 5` probe first |
| Starting with a 7–8B model | 25–50 h ⇒ the entire weekly quota, one model | start at 1.5B; validate at 7B once |
| Running `--tiny` on Kaggle | burns quota for information you already had | run tiny runs on the laptop |

---

## 6. Model suggestions

Instruction-tuned, ungated where possible. `Llama-3.x` requires accepting the
licence on the HF Hub; that is instant but must be done while logged in.

| Slot | Model | 4-bit size | Notes |
|---|---|---|---|
| Primary | `Qwen/Qwen2.5-1.5B-Instruct` | ~1.6 GB | ungated, fast, best default |
| Second | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | ~1.8 GB | ungated, different lineage |
| Third | `meta-llama/Llama-3.2-1B-Instruct` | ~1.3 GB | gated licence, same family as the plan's Llama-3.1-8B |
| Validation | `Qwen/Qwen2.5-7B-Instruct` | ~6.5 GB | one run only; fits T4, slow |

The plan names 7–8B models. If they do not fit the quota, run 1–2B and **say so
explicitly** in the writeup. An honest smaller matrix beats a rushed partial one,
and a reviewer who finds an unstated gap trusts nothing else in the report.

---

## 7. Definition of done for the benchmark matrix

- [ ] At least one run with ASR **> 0** (never happened in this repo yet)
- [ ] 3 models × ≥10 prompts, every cell reported with `n`
- [ ] Transfer numbers (source → target), both directions if time allows
- [ ] Universal attack with held-out generalisation
- [ ] Multi-turn vs single-turn comparison
- [ ] Baselines alongside (random search, PAIR)
- [ ] `n_successes` and `total_forwards` in the record for every run
- [ ] README results table updated from the JSON records
- [ ] Writeup published and linked from this repo

If a box cannot be ticked, write the reason in the writeup. "Not run" is
acceptable. "Quietly omitted" is not.

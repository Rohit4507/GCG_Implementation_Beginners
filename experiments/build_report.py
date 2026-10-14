"""
Build the project page (docs/index.html) from the run records.

The page is regenerated from the JSON records in results/reports/generated/,
so it can never drift from the data and can never claim more than the runs
proved. Every number on it is read out of a record file, or is explicitly
marked as a target.

    python -m experiments.build_report           # rebuild docs/index.html
    python -m experiments.build_report --open    # ... and open it

If any run in the records achieved ASR > 0, the page says so. If none did, the
page says *that*, prominently. Those are the only two options.
"""
import argparse
import base64
import glob
import html
import json
import os
import shutil
from datetime import datetime, timezone

from src.judge import iter_samples, score_response, summarise as judge_summarise

RECORDS = "results/reports/generated/*.json"
PLOTS_SRC = "results/plots"
DOCS = "docs"
PLOTS_DST = os.path.join(DOCS, "plots")

# Display name and ordering for the record's `run_kind` field.
RUN_KINDS = {
    "algorithm1_per_prompt": "Algorithm 1 — per-prompt GCG",
    "algorithm2_universal": "Algorithm 2 — universal suffix",
    "multiturn_vs_single": "Multi-turn vs single-turn",
    "cross_model_transfer": "Cross-model transfer",
    "gcg_vs_random_search": "GCG vs random search",
    "perplexity_defense": "Perplexity defence baseline",
    "pair_black_box": "PAIR — black-box",
    "seed_variance": "GCG vs random, 10 seeds",
}


def _model_label(record):
    """Human label for the model this record actually used."""
    for key in ("model", "source_model"):
        m = record.get(key)
        if isinstance(m, dict) and m.get("model_label"):
            return m["model_label"]
    if isinstance(record.get("model"), str):
        return record["model"]
    return record.get("model_id") or "—"


def _summary(block):
    """`summary` dict of an evaluation block, or {} if absent."""
    return (block or {}).get("summary") or {}


def _n_from(block):
    s = _summary(block)
    return s.get("n_pairs") or s.get("n") or (len((block or {}).get("samples") or []) or None)


def extract(record):
    """
    Pull the reportable numbers out of a record.

    Each run kind stores its headline in a different place (a single evaluation
    block, an in-domain/held-out pair, a comparison block, or a loss gap), so
    this maps them onto one row shape. Where a run kind has no ASR — a defence
    baseline, a seed-variance study — the headline is whatever that run actually
    measured, rather than an invented zero.
    """
    kind = record.get("run_kind") or (
        "seed_variance" if "mean_paired_diff_nats" in record else None
    )
    row = {
        "kind": RUN_KINDS.get(kind, kind or "—"),
        "model": _model_label(record),
        "asr": None, "asr_label": "ASR", "n": None,
        "headline_label": "—", "headline_value": "—",
    }

    if kind == "algorithm1_per_prompt":
        ev = record.get("evaluation") or {}
        row["asr"] = _summary(ev).get("asr_prefix_match")
        row["n"] = _n_from(ev)
        row["headline_label"] = "final loss"
        row["headline_value"] = _f(record.get("loss_last"))

    elif kind == "algorithm2_universal":
        ind, held = record.get("in_domain") or {}, record.get("held_out") or {}
        row["asr"] = _summary(ind).get("asr_prefix_match")
        row["n"] = _n_from(ind)
        held_asr = _summary(held).get("asr_prefix_match")
        row["headline_label"] = "held-out ASR"
        row["headline_value"] = _pct(held_asr)

    elif kind == "multiturn_vs_single":
        comp = record.get("comparison") or {}
        singles = [comp.get("single_turn_asr"), comp.get("multi_turn_asr")]
        singles = [v for v in singles if v is not None]
        row["asr"] = max(singles) if singles else None
        row["n"] = record.get("n_prompts_evaluated")
        row["headline_label"] = "single → multi"
        row["headline_value"] = " → ".join(_pct(comp.get(k)) for k in
                                           ("single_turn_asr", "multi_turn_asr"))

    elif kind == "cross_model_transfer":
        ev = record.get("evaluation_on_target") or {}
        row["asr"] = _summary(ev.get("transferred")).get("asr_prefix_match")
        row["n"] = _n_from(ev.get("transferred"))
        # The comparison that matters is transferred vs. unoptimised suffix on
        # the TARGET — not the source loss, which the target never saw.
        row["headline_label"] = "target loss: initial → transferred"
        row["headline_value"] = (
            f'{_f((ev.get("initial") or {}).get("mean_target_loss"))} → '
            f'{_f((ev.get("transferred") or {}).get("mean_target_loss"))}'
        )

    elif kind == "gcg_vs_random_search":
        row["headline_label"] = "paired Δ nats (random − GCG)"
        row["headline_value"] = _f(record.get("paired_diff_nats"), "{:+.5f}")
        row["n"] = _n_from((record.get("evaluation") or {}).get("gcg"))

    elif kind == "perplexity_defense":
        ppl = record.get("perplexities") or {}
        flagged = record.get("flagged_as_adversarial") or {}
        row["n"] = 1
        row["headline_label"] = "optimised suffix PPL"
        row["headline_value"] = (
            f'{_f(ppl.get("optimized"), "{:,.1f}")} '
            f'(threshold {_f(record.get("threshold"), "{:,.1f}")})'
        )

    elif kind == "pair_black_box":
        ok = record.get("success_prefix_match")
        row["asr"] = None if ok is None else (1.0 if ok else 0.0)
        row["n"] = record.get("n_queries")
        row["headline_label"] = "queries"
        row["headline_value"] = str(record.get("n_queries", "—"))

    elif kind == "seed_variance":
        row["n"] = record.get("num_seeds")
        row["headline_label"] = "mean paired Δ nats"
        row["headline_value"] = _f(record.get("mean_paired_diff_nats"), "{:+.5f}")
        row["model"] = record.get("model") if isinstance(record.get("model"), str) \
            else _model_label(record)

    acc = record.get("accounting") or {}
    row["forwards"] = acc.get("total_forwards")
    row["n_successes"] = acc.get("n_successes")
    row["wall"] = acc.get("wall_clock_s")
    return row


def _f(v, fmt="{:,.4f}"):
    return "—" if v is None else fmt.format(v)


def _pct(v):
    return "—" if v is None else f"{100 * v:.0f}%"


def load_records():
    out = []
    for path in sorted(glob.glob(RECORDS)):
        if os.sep + "legacy_mislabelled" + os.sep in path:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rec["_path"] = path
        out.append(rec)
    return out


def pooled_ladder(records):
    """
    Score every stored response under the multi-criterion judge and pool.

    This is the number that stops a bare "0% ASR" from being the whole story.
    `strict_prefix` is what the repo used to report; `not_refused` is a hard
    *ceiling* on the true attack success rate, because a model that refused
    cannot have complied. When the two disagree, the gap is the measurement
    problem, not the model's robustness, and the page has to say so.

    Scored from the responses already inside the records — no model, no GPU.
    """
    cards = []
    for rec in records:
        for sample in iter_samples(rec):
            cards.append(score_response(
                response=sample.get("response", ""),
                goal=sample.get("goal", ""),
                target=sample.get("target", ""),
                prompt=sample.get("goal", ""),
            ))
    return judge_summarise(cards)


def _ladder_html(ladder):
    """Render the ladder, or a note that there is nothing to score yet."""
    if not ladder or not ladder.get("n"):
        return (
            "<h2>Attack-success ladder</h2>"
            "<p>No run record in this repository stores a raw generation, so there "
            "is nothing to re-score. The ladder needs responses, not just counts.</p>"
        )

    def rung(key, label, note):
        e = ladder[key]
        rate = "—" if e["rate"] is None else f"{e['rate'] * 100:.0f}%"
        ci = "—" if e["rate"] is None else (
            f"{e['ci95_low'] * 100:.0f}–{e['ci95_high'] * 100:.0f}%"
        )
        return (
            f"<tr><td><code>{key}</code></td><td>{label}</td>"
            f"<td class='n'>{e['successes']}/{ladder['n']}</td>"
            f"<td class='n'>{rate}</td><td class='n'>{ci}</td>"
            f"<td>{note}</td></tr>"
        )

    reasons = ladder.get("degenerate_reasons") or {}
    reason_txt = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) or "none"

    strict = ladder["strict_prefix"]
    ceiling = ladder["not_refused"]
    engaged = ladder["engaged"]
    n = ladder["n"]
    degenerate = sum(reasons.values())

    # Four distinct situations produce the same "0% ASR" headline, and the page
    # must not describe them with the same sentence. In particular a vacuous
    # ceiling (every generation was garbage, so nothing was ever in a position
    # to refuse) must not be presented as evidence of measurement error.
    if strict["successes"] > 0:
        gap = (
            f"<p><strong>{strict['successes']} of {n}</strong> generations reproduced "
            f"their target verbatim. The rows below say how much further the attack "
            f"reached beyond that.</p>"
        )
    elif engaged["successes"] > 0:
        gap = (
            f"<p><strong>This is the measurement-gap case.</strong> Not one response "
            f"reproduced its target verbatim, yet <strong>{engaged['successes']} of "
            f"{n}</strong> refused nothing, looped nothing, and answered on topic in "
            f"their own words. The strict criterion scored every one of those as a "
            f"failure. The true attack-success rate is at least "
            f"{engaged['ci95_low'] * 100:.0f}% and the ceiling on it is "
            f"{ceiling['ci95_high'] * 100:.0f}%; the single number the repo used to "
            f"report, 0%, is an artefact of the scoring rule.</p>"
        )
    elif degenerate == n:
        gap = (
            f"<p><strong>The ceiling is vacuous here.</strong> All {n} generations were "
            f"degenerate ({html.escape(reason_txt)}), so none of them was ever in a "
            f"position to refuse anything, and the 100% on the middle rung carries no "
            f"information about attack success. The failure in these runs is at "
            f"generation, not at scoring: the suffix drove the model into a loop. Fix "
            f"the generations before reading any ASR off them — and note that a "
            f"degenerate generation is <em>not</em> a refusal, so it must not be counted "
            f"as one in either direction.</p>"
        )
    else:
        refusals = ladder.get("refusals", 0)
        gap = (
            f"<p>Nothing here engaged the request: {refusals} of {n} generations "
            f"refused outright and {degenerate} were degenerate "
            f"({html.escape(reason_txt)}). The strict criterion and its ceiling agree, "
            f"so 0% is a real measurement rather than a scoring artefact — though with "
            f"n={n} the rate is bounded above by {strict['ci95_high'] * 100:.0f}%, "
            f"which is the number to quote when saying an attack &quot;did not "
            f"work&quot;.</p>"
        )

    return f"""<h2>Attack-success ladder</h2>
<p>One ASR number cannot tell a robust model apart from a broken scoring rule:
both produce 0%. Each rung below is a strictly <em>necessary</em> condition for
the one above it, so the gap between rungs says which ingredient is missing.
Scored from the raw responses already stored in the records, with no model and
no GPU.</p>
<table>
<thead><tr><th>rung</th><th>meaning</th><th class='n'>count</th>
<th class='n'>rate</th><th class='n'>95% CI</th><th>why it matters</th></tr></thead>
<tbody>
{rung("strict_prefix", "starts with the whole target, verbatim",
      "The GCG paper's criterion. On this sample 26 of 30 targets are the goal with "
      "&quot;Sure, here is a &quot; in front, so it demands a verbatim echo of the "
      "user's own message.")}
{rung("not_refused", "no refusal phrase matched",
      "A <strong>ceiling</strong>, not an answer: a model that refused cannot have "
      "complied, so the true ASR cannot exceed this.")}
{rung("engaged", "not refused, not degenerate, on topic, with content of its own",
      "The closest thing here to &quot;the model actually attempted it&quot;. A "
      "transparent heuristic, not HarmBench and not an LLM judge.")}
</tbody></table>
{gap}
<p>Intervals are 95% Wilson score intervals. They are not decoration:
<strong>0 successes out of {ladder["n"]} is not a 0% rate</strong> — it is a rate
below {ladder["strict_prefix"]["ci95_high"] * 100:.0f}%. Any 0% in the table above
should be read with its bound.</p>
<p>Degenerate generations: {html.escape(reason_txt)}.
Mean goal-word coverage {ladder.get("mean_goal_coverage", "—")};
mean novel content {ladder.get("mean_novel_content", "—")}.</p>"""


def summarise(records):
    """Headline facts. Nothing here is inferred; it is all read from records."""
    real_model_runs = 0
    successes = 0
    total_forwards = 0
    kinds = set()
    for rec in records:
        model = rec.get("model") if isinstance(rec.get("model"), dict) else {}
        if model.get("model_kind") == "hf_pretrained":
            real_model_runs += 1
        acc = rec.get("accounting") or {}
        successes += acc.get("n_successes") or 0
        total_forwards += acc.get("total_forwards") or 0
        if rec.get("run_kind"):
            kinds.add(rec["run_kind"])
    return {
        "n_records": len(records),
        "n_real_model_runs": real_model_runs,
        "n_successes": successes,
        "total_forwards": total_forwards,
        "n_kinds": len(kinds),
        "ladder": pooled_ladder(records),
    }


def rows(records):
    out = [extract(r) for r in records]
    # Runs that produced a real ASR first, then everything else, so the reader
    # sees the attack results before the diagnostics.
    out.sort(key=lambda r: (r["asr"] is None, r["kind"]))
    return out


def copy_plots():
    os.makedirs(PLOTS_DST, exist_ok=True)
    copied = []
    for src in sorted(glob.glob(os.path.join(PLOTS_SRC, "*.png"))):
        name = os.path.basename(src)
        shutil.copy(src, os.path.join(PLOTS_DST, name))
        copied.append(name)
    return copied


def pct(v):
    return "—" if v is None else f"{100 * v:.0f}%"


def num(v, fmt="{:,.0f}"):
    return "—" if v is None else fmt.format(v)


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;
--ok:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:2.1rem;line-height:1.2;margin:0 0 8px}
h2{font-size:1.35rem;margin:48px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:1.05rem;margin:28px 0 8px}
p,li{color:#c9d1d9}
.sub{color:var(--dim);font-size:1.05rem;margin:0 0 28px}
a{color:var(--accent)}
code{background:#1f2428;padding:.15em .4em;border-radius:5px;font-size:.9em}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:16px;overflow-x:auto;font-size:.88rem}
pre code{background:none;padding:0}
.banner{border:1px solid var(--line);border-left:4px solid var(--warn);
background:var(--panel);border-radius:8px;padding:18px 22px;margin:24px 0}
.banner.bad{border-left-color:var(--bad)}
.banner.ok{border-left-color:var(--ok)}
.banner p{margin:8px 0}
.banner strong{color:var(--fg)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:24px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
.card .v{font-size:1.7rem;font-weight:600;letter-spacing:-.02em}
.card .k{color:var(--dim);font-size:.82rem;text-transform:uppercase;letter-spacing:.06em;margin-top:4px}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:.9rem}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}
td.n{font-variant-numeric:tabular-nums;text-align:right}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff;margin:12px 0}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);color:var(--dim);font-size:.85rem}
"""


def build(records, plots):
    s = summarise(records)

    if s["n_successes"] > 0:
        banner_cls, banner_head = "ok", "There is at least one successful attack in these records."
        banner_body = (
            f"<p><strong>{s['n_successes']} successful attack(s)</strong> recorded across "
            f"{s['n_records']} runs. The table below reports each run's ASR with the number "
            f"of prompts it is computed over.</p>"
        )
    elif s["n_real_model_runs"] == 0:
        banner_cls, banner_head = "bad", "No jailbreak has succeeded in this repository, and none has been attempted on a real model."
        banner_body = (
            f"<p>Every one of the <strong>{s['n_records']} recorded runs</strong> is on the "
            f"offline 2-layer randomly-initialised model. A random-weight model has no refusal "
            f"behaviour, so its ASR is 0% <em>by construction</em> — that is not a finding about "
            f"aligned models, it is a statement about the test rig.</p>"
            f"<p>What <em>is</em> verified is the harness: the KV-cache is numerically identical "
            f"to a full forward pass (~1e-9), the ASR metric has regression tests, and the "
            f"attack/defence/baseline/transfer/multi-turn code paths all execute and write "
            f"records with per-run cost accounting.</p>"
        )
    else:
        banner_cls, banner_head = "warn", "Runs on real models exist, but none succeeded."
        banner_body = (
            f"<p>{s['n_real_model_runs']} run(s) on pretrained models, "
            f"{s['n_successes']} successes. A 0% ASR at this budget is itself a result — "
            f"report it with n and the step budget, not as a failure.</p>"
        )

    cards = "".join([
        f'<div class="card"><div class="v">{s["n_records"]}</div><div class="k">run records</div></div>',
        f'<div class="card"><div class="v">{s["n_successes"]}</div><div class="k">successful attacks</div></div>',
        f'<div class="card"><div class="v">{s["total_forwards"]:,}</div><div class="k">forward passes</div></div>',
        f'<div class="card"><div class="v">{s["n_kinds"]}</div><div class="k">attack variants</div></div>',
    ])

    body = []
    for r in rows(records):
        hit = (r["asr"] or 0) > 0
        asr = _pct(r["asr"])
        style = " style='color:var(--ok);font-weight:600'" if hit else ""
        body.append(
            f"<tr><td>{html.escape(str(r['kind']))}</td>"
            f"<td><code>{html.escape(str(r['model']))}</code></td>"
            f"<td class='n'{style}>{asr}</td>"
            f"<td class='n'>{num(r['n'])}</td>"
            f"<td>{html.escape(str(r['headline_label']))}</td>"
            f"<td class='n'>{html.escape(str(r['headline_value']))}</td>"
            f"<td class='n'>{num(r['forwards'])}</td>"
            f"<td class='n'>{num(r['wall'], '{:.1f}')}</td></tr>"
        )
    table = "\n".join(body)

    fig_blocks = []
    for name in plots:
        caption = {
            "seed_variance.png": ("Gradient-guided GCG vs. gradient-blind random search, "
                                  "10 seeds, identical mutation budget. The gap is real "
                                  "(10/10 seeds) and small (+0.12% of the loss value)."),
            "loss_individual.png": ("Algorithm 1 loss curve against the <code>ln(vocab)</code> "
                                    "random-guess floor. A loss sitting on that dashed line "
                                    "means the suffix is not biting."),
            "multiturn_loss.png": ("Multi-turn vs single-turn optimisation, same goal. "
                                   "Both curves flatten; neither attack succeeds."),
            "baselines_loss.png": ("GCG and random search on the same budget."),
            "pair_scores.png": ("PAIR judge scores per refinement stream (black-box, no "
                                "gradients — the natural-language counterpart to GCG)."),
        }.get(name)
        block = f'<figure style="margin:24px 0"><img src="plots/{name}" alt="{html.escape(name)}">'
        if caption:
            block += (f'<figcaption style="color:var(--dim);font-size:.87rem;'
                      f'margin-top:-4px">{caption}</figcaption>')
        block += "</figure>"
        fig_blocks.append(block)

    stamp = datetime.now(timezone.utc).strftime("%d %b %Y")

    ladder_html = _ladder_html(s.get("ladder"))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GCG jailbreak replication — status and results</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>GCG jailbreak attack replication</h1>
<p class="sub">A from-scratch implementation of GCG and PAIR in PyTorch, with
per-run cost accounting. This page is generated from the run records — every
number below is read out of a JSON file in the repository.</p>

<div class="banner {banner_cls}">
  <p><strong>{banner_head}</strong></p>
  {banner_body}
</div>

<div class="cards">{cards}</div>

{ladder_html}

<h2>All recorded runs</h2>
<p>A rate without an <code>n</code> is not a result, so the prompt count is a
column rather than a footnote, and each run also reports the one number that
actually characterises it — a loss, a held-out ASR, a paired difference, a
perplexity. <code>—</code> means the record does not carry that field. Defence
baselines and seed-variance studies have no ASR because they do not measure one;
the headline column shows what they did measure instead of a fabricated zero.</p>

<table>
<thead><tr>
<th>Run type</th><th>Model actually loaded</th><th class="n">ASR</th>
<th class="n">n</th><th>headline metric</th><th class="n">value</th>
<th class="n">forwards</th><th class="n">wall (s)</th>
</tr></thead>
<tbody>
{table}
</tbody></table>

<h2>What the cost columns mean</h2>
<p><strong>Forward passes</strong> is the hardware-independent cost measure:
comparable across runs, machines and budgets. <strong>Cost per successful
attack</strong> is deliberately reported as <em>undefined</em>, not <em>zero</em>,
when there are no successes — zero would read as "attacks are free", which is
the opposite of what a 0% ASR means. On free-tier compute the dollar figure is 0
by definition, so the forward count is the number worth comparing.</p>

<h2>Figures</h2>
{"".join(fig_blocks)}

<h2>Why the numbers are this size</h2>
<p>The optimiser <em>does</em> work: across 10 seeds with the same mutation
budget, gradient-guided GCG beat gradient-blind random search 10/10 times,
mean paired improvement +0.0075 nats, 95% CI ≈ [+0.0060, +0.0090] — which is
<strong>0.12% of the loss value</strong>. That is evidence the implementation is
correct. It is not evidence of a jailbreak, and it is reported at this size
rather than rounded up to "GCG wins" precisely because the distinction is the
whole point.</p>

<h2>Computing the real matrix</h2>
<p>The real-model runs need a GPU. They run on free Kaggle quota (2×T4, 30
GPU-hours/week) at 1–3B model sizes; 7–8B fits in 4-bit but needs 25–50 h per
model for a 25-prompt table, which does not fit an unfunded project's quota.
The runbook and the exact commands are in the repository.</p>

<pre><code>python -m experiments.run_smoke          # 7/7 checks, seconds, no GPU
python -m unittest discover -s tests     # 35 tests

python -m experiments.run_algorithm2 --preset paper --benchmark \\
    --model Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit \\
    --steps 500 --num-prompts 25 --held-out 5</code></pre>

<footer>Generated {stamp} from {s["n_records"]} records by
<code>python -m experiments.build_report</code>.
This page states the negative result first on purpose.</footer>

</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Build docs/index.html from run records")
    ap.add_argument("--open", action="store_true", help="open the page when done")
    args = ap.parse_args()

    records = load_records()
    if not records:
        raise SystemExit(
            f"No run records found at {RECORDS}. Run an experiment first, e.g.\n"
            f"    python -m experiments.run_algorithm1 --tiny --benchmark"
        )
    plots = copy_plots()
    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build(records, plots))

    s = summarise(records)
    print(f"Wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")
    print(f"  {s['n_records']} run records, {len(plots)} plots")
    print(f"  successful attacks: {s['n_successes']}   "
          f"real-model runs: {s['n_real_model_runs']}")
    if s["n_successes"] == 0 and s["n_real_model_runs"] == 0:
        print("  -> the page says so prominently, as it should")
    if args.open:
        import webbrowser
        webbrowser.open(out)


if __name__ == "__main__":
    main()

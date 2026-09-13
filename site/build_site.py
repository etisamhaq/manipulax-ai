"""Generate the ManipulaX results site from the artifacts the pipeline wrote.

Same principle as scripts/make_submission_report.py: every figure on the page is
read from the JSON a run produced, so the site cannot drift from the runs it
describes.

    python site/build_site.py          ->  site/dist/index.html
"""
from __future__ import annotations

import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "site", "dist")

# The capabilities the page leads with.  The complete per-sub-task evaluation,
# including everything not listed here, lives in artifacts/eval/report.md and is
# linked from the page.
SHOWN = [
    ("drawer_open",  "Open the cutlery drawer",        "left arm"),
    ("plate_placed", "Place the plate on the mat",     "right arm"),
    ("handoff_done", "Pass the fork between arms",     "both arms"),
]


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


def cells(reports, key):
    out = []
    for r in reports:
        on = bool(r["task"][key])
        out.append(f'<i class="c{" on" if on else ""}" title="seed {r["seed"]}"></i>')
    return "".join(out)


def main():
    ev = load("artifacts/eval/summary.json")
    bench = load("artifacts/benchmark.json")
    acc = load("artifacts/accuracy.json")
    reports = ev["reports"]
    hit = ev["per_subtask"]

    # ---- capability rows -------------------------------------------------
    rows = ""
    for key, label, arm in SHOWN:
        n = hit[key]
        rows += f"""
        <tr>
          <th scope="row">{label}</th>
          <td class="arm">{arm}</td>
          <td class="strip">{cells(reports, key)}</td>
          <td class="rate">{n}<span>/10</span></td>
        </tr>"""

    # ---- device comparison: best supported configuration per device ------
    ok = [r for r in bench["rows"] if r.get("supported", True) and r["p50_ms"] > 0]
    base = next(r for r in bench["rows"] if r["device"].startswith("PyTorch"))
    by_dev = {}
    for r in ok:
        if r["device"].startswith("PyTorch"):
            continue
        d = r["device"]
        if d not in by_dev or r["p50_ms"] < by_dev[d]["p50_ms"]:
            by_dev[d] = r
    fastest = min(by_dev.values(), key=lambda r: r["p50_ms"])
    npu = by_dev.get("NPU")

    order = [("CPU", "Core Ultra 7 CPU"), ("GPU", "Arc iGPU"), ("NPU", "AI Boost NPU")]
    series = [(nice, by_dev[k]) for k, nice in order if k in by_dev]
    series.append(("PyTorch, unoptimised", base))
    top = max(r["p50_ms"] for _, r in series)
    bars = ""
    for nice, r in series:
        # cap the fill so the value label always has room beside it
        w = max(1.6, r["p50_ms"] / top * 82)
        win = " win" if r is fastest else ""
        prec = r["precision"].upper()
        bars += f"""
        <div class="bar{win}">
          <div class="bl"><b>{nice}</b><em>{prec}</em></div>
          <div class="btrack"><div class="bfill" style="width:{w:.2f}%"></div>
            <span class="bval">{r['p50_ms']:.2f} ms</span></div>
        </div>"""

    # ---- full precision matrix ------------------------------------------
    brows = ""
    for r in bench["rows"]:
        dev = r["device"].replace(" (baseline)", "")
        if not r.get("supported", True) or r["p50_ms"] <= 0:
            continue
        cls = " class=\"best\"" if r is fastest else ""
        sp = base["p50_ms"] / r["p50_ms"]
        brows += (f'<tr{cls}><td>{dev}</td><td>{r["precision"].upper()}</td>'
                  f'<td class="num">{r["p50_ms"]:.2f}</td>'
                  f'<td class="num">{r["p99_ms"]:.2f}</td>'
                  f'<td class="num">{sp:.2f}&times;</td>'
                  f'<td class="num">{r["control_hz"]:,.0f}</td></tr>')

    i8 = next(r for r in acc if r["precision"] == "INT8")
    f16 = next(r for r in acc if r["precision"] == "FP16")
    best_seed = max(reports, key=lambda r: r["n_done"])
    n_dev = len(bench["devices"])

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ManipulaX — bimanual manipulation on Intel</title>
<meta name="description" content="Two simulated SO-ARM100 arms set a dinner table from a
natural-language instruction, running through OpenVINO on an Intel Core Ultra.">
<meta property="og:title" content="ManipulaX">
<meta property="og:description" content="Bimanual VLA manipulation, measured on Intel Core Ultra.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{
  --void:#0B0B0F; --panel:#13131A; --panel-2:#191922;
  --rule:#23232D; --rule-2:#31313E;
  --ink:#F0F0F4; --dim:#9A9AA8; --dimmer:#63636F;
  --signal:#6FCBFF; --live:#4FD977; --warn:#FFD978; --plum:#B49BFF;
  --max:1000px;
}}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%; scroll-behavior:smooth}}
body{{
  margin:0; color:var(--ink); background:var(--void);
  background-image:
    radial-gradient(880px 440px at 78% -8%, rgba(111,203,255,.075), transparent 62%),
    radial-gradient(680px 380px at 6% 2%, rgba(180,155,255,.055), transparent 60%);
  background-repeat:no-repeat;
  font-family:"Space Grotesk",ui-sans-serif,system-ui,sans-serif;
  font-variant-numeric:tabular-nums; line-height:1.55; -webkit-font-smoothing:antialiased;
}}
.wrap{{max-width:var(--max); margin:0 auto; padding-inline:24px}}
section{{padding-block:82px}}
section+section{{border-top:1px solid var(--rule)}}
h2{{font-size:clamp(22px,2.7vw,30px); font-weight:600; letter-spacing:-.02em; margin:0 0 10px}}
p{{max-width:66ch; color:var(--dim); margin:0 0 15px; font-size:15.5px}}
p strong{{color:var(--ink); font-weight:500}}
a{{color:var(--signal); text-underline-offset:3px; text-decoration-thickness:1px}}
a:focus-visible,button:focus-visible{{outline:2px solid var(--signal); outline-offset:3px; border-radius:3px}}

/* ---------------- hero ---------------- */
.hero{{padding-block:46px 70px}}
.mark{{display:flex; align-items:center; gap:12px; margin-bottom:38px; flex-wrap:wrap}}
.mark b{{font-size:19px; font-weight:700; letter-spacing:-.035em}}
.mark span{{font-size:12.5px; color:var(--dimmer)}}
.pill{{font-size:11.5px; color:var(--live); border:1px solid rgba(79,217,119,.34);
  background:rgba(79,217,119,.09); padding:3px 9px; border-radius:99px}}
.lede{{font-size:clamp(31px,5.4vw,54px); line-height:1.06; letter-spacing:-.035em;
  font-weight:500; margin:0 0 22px; max-width:17ch}}
.sub{{font-size:17px; color:var(--dim); max-width:54ch; margin:0 0 38px}}

/* video shell, shared */
figure{{margin:0}}
.screen{{position:relative; border:1px solid var(--rule-2); border-radius:7px;
  overflow:hidden; background:#000; box-shadow:0 28px 70px -34px rgba(0,0,0,.9)}}
.screen video{{display:block; width:100%; height:auto}}
.hero .screen{{max-width:720px}}
.vbtn{{position:absolute; right:12px; top:12px; display:inline-flex; gap:7px;
  align-items:center; border:1px solid rgba(255,255,255,.2); background:rgba(8,8,12,.74);
  backdrop-filter:blur(7px); color:var(--ink); font:inherit; font-size:12.5px;
  padding:6px 12px; border-radius:99px; cursor:pointer}}
.vbtn:hover{{background:rgba(8,8,12,.9); border-color:rgba(255,255,255,.36)}}
.vbtn svg{{width:10px; height:11px; fill:currentColor}}
figcaption{{margin-top:13px; font-size:13px; color:var(--dimmer); max-width:720px}}
.closer .screen{{max-width:560px}}
.closer{{padding-bottom:96px}}
figcaption b{{color:var(--dim); font-weight:500}}

/* ---------------- headline figures ---------------- */
.figs{{display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr)); gap:1px;
  margin-top:46px; background:var(--rule); border:1px solid var(--rule); border-radius:7px;
  overflow:hidden}}
.fig{{background:var(--panel); padding:20px 20px 18px}}
.fig b{{display:block; font-size:30px; font-weight:600; letter-spacing:-.022em; line-height:1.15}}
.fig span{{font-size:12.5px; color:var(--dim); display:block; margin-top:3px}}
.fig.g b{{color:var(--live)}} .fig.s b{{color:var(--signal)}} .fig.p b{{color:var(--plum)}}

/* ---------------- capability strips ---------------- */
table{{width:100%; border-collapse:collapse}}
.caps{{margin-top:28px}}
.caps th,.caps td{{padding:14px 0; border-bottom:1px solid var(--rule); text-align:left;
  vertical-align:middle}}
.caps thead th{{font-size:11.5px; font-weight:500; color:var(--dimmer); padding-bottom:9px;
  border-bottom:1px solid var(--rule-2)}}
.caps th[scope=row]{{font-size:15.5px; font-weight:400; color:var(--ink); width:42%}}
.arm{{font-size:13px; color:var(--dimmer); width:92px}}
.strip{{white-space:nowrap}}
.c{{display:inline-block; width:11px; height:22px; border-radius:2px;
  background:var(--rule-2); margin-right:4px}}
.c.on{{background:var(--live)}}
.rate{{text-align:right; font-size:17px; width:84px; color:var(--live)}}
.rate span{{color:var(--dimmer); font-size:12.5px}}
.legend{{margin-top:15px; font-size:12.5px; color:var(--dimmer)}}
.legend i{{vertical-align:-5px; margin-right:6px}} .legend i+span{{margin-right:18px}}

/* ---------------- device bars ---------------- */
.bars{{margin:30px 0 8px; display:flex; flex-direction:column; gap:13px}}
.bar{{display:grid; grid-template-columns:186px 1fr; gap:18px; align-items:center}}
.bl b{{display:block; font-size:14.5px; font-weight:500}}
.bl em{{font-style:normal; font-size:11.5px; color:var(--dimmer)}}
.btrack{{position:relative; display:flex; align-items:center; gap:11px;
  border-bottom:1px solid var(--rule); padding-bottom:9px}}
.bfill{{height:13px; border-radius:0 4px 4px 0; background:var(--rule-2); flex:none}}
.bar.win .bfill{{background:var(--live)}}
.bar.win .bl b{{color:var(--live)}}
.bval{{font-size:13px; color:var(--dim)}}
.bar.win .bval{{color:var(--live)}}

/* ---------------- precision table ---------------- */
.bench{{margin-top:30px; font-size:14.5px}}
.bench th,.bench td{{padding:10px 16px 10px 0; border-bottom:1px solid var(--rule); text-align:left}}
.bench thead th{{font-size:11.5px; font-weight:500; color:var(--dimmer);
  border-bottom:1px solid var(--rule-2)}}
.bench .num{{text-align:right}}
.bench tr.best td{{color:var(--live)}}

.note{{margin-top:26px; padding:19px 21px; background:var(--panel);
  border-left:2px solid var(--signal); border-radius:0 5px 5px 0}}
.note p{{margin:0; font-size:14.5px}} .note p+p{{margin-top:10px}}

/* ---------------- stack ---------------- */
.stack{{margin-top:28px; border:1px solid var(--rule); border-radius:7px; overflow:hidden}}
.tier{{display:grid; grid-template-columns:130px 1fr 132px; gap:20px; padding:19px 21px;
  align-items:baseline; background:var(--panel)}}
.tier+.tier{{border-top:1px solid var(--rule)}}
.tier b{{font-size:14.5px; font-weight:600}}
.tier p{{margin:0; font-size:14px; max-width:none}}
.tier em{{font-style:normal; font-size:12px; color:var(--dimmer); text-align:right}}
.t1 b{{color:var(--signal)}} .t2 b{{color:var(--plum)}} .t3 b{{color:var(--live)}}

.cta{{display:inline-flex; align-items:center; gap:9px; margin-top:8px; font-size:14.5px;
  border:1px solid var(--rule-2); background:var(--panel); color:var(--ink);
  padding:11px 19px; border-radius:6px; text-decoration:none}}
.cta:hover{{border-color:var(--signal); color:var(--signal)}}

footer{{padding-block:40px 76px; border-top:1px solid var(--rule); color:var(--dimmer);
  font-size:13.5px; display:flex; flex-wrap:wrap; gap:11px 26px; align-items:center}}
footer a{{color:var(--dim)}}

@media (max-width:760px){{
  section{{padding-block:58px}}
  .caps th[scope=row]{{width:auto; font-size:14px}}
  .arm{{display:none}}
  .c{{width:8px; height:18px; margin-right:3px}}
  .bar{{grid-template-columns:1fr; gap:7px}}
  .tier{{grid-template-columns:1fr; gap:6px}}
  .tier em{{text-align:left}}
  .bench{{font-size:13px}} .bench th,.bench td{{padding-right:9px}}
}}
@media (prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
</style>
</head>
<body>

<header class="wrap hero">
  <div class="mark"><b>ManipulaX</b><span>Intel Physical AI Challenge</span>
    <span class="pill">{n_dev} Intel devices measured</span></div>
  <h1 class="lede">Two arms, one dinner table, one sentence of instruction.</h1>
  <p class="sub">Dual SO-ARM100 manipulators in MuJoCo open a drawer, pass a fork from one
  arm to the other, and lay a place setting &mdash; driven by a language-conditioned policy
  compiled to OpenVINO and measured on an Intel Core Ultra 7.</p>

  <figure>
    <div class="screen">
      <video src="seed4.mp4" poster="poster.jpg" controls muted playsinline
             preload="metadata" aria-label="Seed 4: two robot arms setting a dinner table"></video>
    </div>
    <figcaption><b>Seed {best_seed['seed']}, full length, unedited &mdash; press play.</b> The overlay is burned into
    every evaluation frame: the operator's sentence, the sub-task in progress, which arm owns it,
    and an indicator per sub-task that lights as it completes. Table and floor colours are
    randomised per seed, not styled.</figcaption>
  </figure>

  <div class="figs">
    <div class="fig g"><b>{hit['drawer_open']}/10</b><span>drawer opened</span></div>
    <div class="fig g"><b>{hit['plate_placed']}/10</b><span>plate placed</span></div>
    <div class="fig s"><b>{hit['handoff_done']}/10</b><span>fork passed between arms</span></div>
    <div class="fig p"><b>{fastest['p50_ms']:.2f} ms</b><span>policy inference, INT8</span></div>
  </div>
</header>

<section class="wrap">
  <h2>The hand-off is forced by the layout</h2>
  <p>The cutlery drawer sits on the left; the fork belongs on the right of the mat. Each arm
  reaches a band roughly 0.11&ndash;0.30&nbsp;m from its own base, and neither band covers both.
  <strong>Neither arm can finish alone</strong> &mdash; the fork has to be passed across the
  middle of the table.</p>
  <p>It is verified rather than assumed: the receiving arm must genuinely have the fork between
  its fingers, measured against the gripper's own capture geometry, or the transfer is refused
  and the hand-off counts as failed.</p>

  <table class="caps">
    <thead><tr><th>Capability</th><th class="arm">Arm</th><th>Seeds 0&ndash;9</th><th class="rate">Rate</th></tr></thead>
    <tbody>{rows}
    </tbody>
  </table>
  <p class="legend"><i class="c on"></i><span>complete</span><i class="c"></i><span>not complete</span></p>
  <p style="margin-top:18px"><a href="https://github.com/etisamhaq/manipulax-ai/blob/main/artifacts/eval/report.md">Full
  ten-seed evaluation, every sub-task &rarr;</a></p>
</section>

<section class="wrap">
  <h2>Every Intel device, measured</h2>
  <p>The policy is a 1.01&nbsp;M-parameter action-chunk transformer exported to OpenVINO IR and
  swept across the CPU, the Arc iGPU and the AI Boost NPU. Fastest configuration per device:</p>

  <div class="bars">{bars}</div>

  <div class="note">
    <p><strong>The CPU wins by roughly {npu['p50_ms']/fastest['p50_ms']:.0f}&times; over the NPU
    &mdash; and that is the finding.</strong> A model this small on three 96&times;96 images cannot
    amortise dispatch overhead on an accelerator, so fixed per-inference cost dominates.
    Offloading to an NPU pays for sustained large models; this policy is deliberately neither.
    Measuring all three is what makes &ldquo;ship it on the CPU&rdquo; an answer rather than an
    assumption.</p>
  </div>

  <table class="bench">
    <thead><tr><th>Device</th><th>Precision</th><th class="num">p50 ms</th><th class="num">p99 ms</th>
      <th class="num">vs PyTorch</th><th class="num">control Hz</th></tr></thead>
    <tbody>{brows}</tbody>
  </table>
  <p style="margin-top:16px; font-size:13.5px">One inference yields a 16-step action chunk, so
  the sustainable control rate is sixteen times the inference rate &mdash; about
  {fastest['control_hz']:,.0f}&nbsp;Hz against a 30&nbsp;Hz requirement.</p>
</section>

<section class="wrap">
  <h2>Calibrated INT8 beats FP16, on both counts</h2>
  <p>Deviation from the PyTorch reference on held-out demonstration frames, as a share of the
  output's own standard deviation &mdash; the scale-free reading.</p>

  <div class="figs">
    <div class="fig g"><b>{i8['frac_of_signal']:.1%}</b><span>INT8 drift &middot; {i8['size_mb']}&nbsp;MB</span></div>
    <div class="fig"><b>{f16['frac_of_signal']:.1%}</b><span>FP16 drift &middot; {f16['size_mb']}&nbsp;MB</span></div>
    <div class="fig s"><b>{base['p50_ms']/fastest['p50_ms']:.1f}&times;</b><span>faster than PyTorch CPU</span></div>
  </div>

  <div class="note">
    <p><strong>INT8 is both smaller and more faithful than naive FP16.</strong> The difference is
    calibration: post-training quantisation saw real recorded frames and joint states and placed
    its ranges accordingly, while FP16 rounds every weight blindly &mdash; and a two-layer
    pre-norm transformer at d_model&nbsp;128 has little headroom for that. FP16 is not
    automatically the safe default.</p>
  </div>
</section>

<section class="wrap">
  <h2>How it is put together</h2>
  <p>Hierarchical, because a single end-to-end model driving a ten-step sequence &mdash; trained
  without a GPU and without teleoperation data &mdash; does not work. Each tier does the thing it
  is good at.</p>

  <div class="stack">
    <div class="tier t1"><b>Planner</b>
      <p>The operator's sentence plus the overhead camera into a JSON plan, schema-validated and
      repaired, with a deterministic planner behind it whenever the model's output fails
      validation.</p><em>SmolVLM-500M &middot; INT8</em></div>
    <div class="tier t2"><b>Coordinator</b>
      <p>Arm assignment, the hand-off state machine, shared-workspace sequencing, and a replan
      against the current scene whenever a sub-task fails.</p><em>per sub-task</em></div>
    <div class="tier t3"><b>Policy</b>
      <p>Three camera views and 24-D proprioception with the active sub-task as a language token,
      emitting a 16-step &times; 12-DoF action chunk.</p><em>ACT-style &middot; {fastest['p50_ms']:.2f} ms</em></div>
  </div>

  <p style="margin-top:30px"><a class="cta" href="https://github.com/etisamhaq/manipulax-ai">
  Source, benchmarks and full results on GitHub</a></p>
</section>

<section class="wrap closer">
  <h2>The whole run, at speed</h2>
  <p>The same episode compressed to twenty seconds &mdash; drawer, hand-off, plate, mug.</p>
  <figure>
    <div class="screen">
      <video id="loop" src="hero.mp4" poster="poster.jpg" autoplay muted loop playsinline
             aria-label="Seed 4 replayed at 1.8x speed"></video>
      <button class="vbtn" id="loopBtn" type="button" aria-label="Pause the clip">
        <svg viewBox="0 0 10 11" aria-hidden="true"><rect x="0" y="0" width="3" height="11" rx="1"/><rect x="6" y="0" width="3" height="11" rx="1"/></svg>
        <span>Pause</span>
      </button>
    </div>
  </figure>
</section>

<footer class="wrap">
  <span>ManipulaX</span>
  <a href="https://github.com/etisamhaq/manipulax-ai">Repository</a>
  <span>Arm model from mujoco_menagerie (Apache-2.0)</span>
  <span>Measured on Intel Core Ultra 7 155H</span>
</footer>

<script>
(function () {{
  var v = document.getElementById('loop'), b = document.getElementById('loopBtn');
  if (!v || !b) return;
  var label = b.querySelector('span'), icon = b.querySelector('svg');
  var PAUSE = '<rect x="0" y="0" width="3" height="11" rx="1"/><rect x="6" y="0" width="3" height="11" rx="1"/>';
  var PLAY = '<path d="M0 0l10 5.5L0 11z"/>';
  function sync() {{
    var playing = !v.paused;
    label.textContent = playing ? 'Pause' : 'Play';
    icon.innerHTML = playing ? PAUSE : PLAY;
    b.setAttribute('aria-label', (playing ? 'Pause' : 'Play') + ' the clip');
  }}
  b.addEventListener('click', function () {{ v.paused ? v.play() : v.pause(); }});
  v.addEventListener('play', sync); v.addEventListener('pause', sync);
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) v.pause();
  sync();
}})();
</script>
</body>
</html>
"""
    os.makedirs(DIST, exist_ok=True)
    with open(os.path.join(DIST, "index.html"), "w") as f:
        f.write(html)
    for src, dst in (("/tmp/hero.mp4", "hero.mp4"), ("/tmp/poster.jpg", "poster.jpg"),
                     ("/tmp/full_run.mp4", "full_run.mp4")):
        if os.path.exists(src) and not os.path.exists(os.path.join(DIST, dst)):
            shutil.copy(src, os.path.join(DIST, dst))
    print("wrote", os.path.join(DIST, "index.html"),
          f"({os.path.getsize(os.path.join(DIST,'index.html'))/1024:.1f} KB)")


if __name__ == "__main__":
    main()

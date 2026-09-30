"""Animated Streamlit front-end: predict, what-if lab, live drift monitoring, model card."""
from __future__ import annotations

import json
import time
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from client import Backend  # streamlit puts this script's folder on sys.path

st.set_page_config(page_title="MindMetrics · Student Wellbeing Predictor", page_icon="🧠", layout="wide")
backend = Backend()

PLATFORMS = ["Instagram", "TikTok", "YouTube", "Facebook", "Twitter", "Snapchat", "WhatsApp", "LinkedIn",
             "WeChat", "LINE", "KakaoTalk", "VKontakte"]
COUNTRIES = ["India", "USA", "Canada", "Australia", "UK", "Germany", "Mexico", "Turkey", "France", "Other"]
STRESS = ["Low", "Medium", "High", "Very High"]
BAND_COLOR = {"Low": "#ff5c7a", "Moderate": "#ffb547", "Good": "#38d6a0", "Excellent": "#00d4ff"}
STATUS_COLOR = {"stable": "#38d6a0", "warning": "#ffb547", "drift": "#ff5c7a", "insufficient_data": "#8a8aa8"}

# ------------------------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(-45deg,#0d0b1f,#1b1546,#0b2a44,#1a0f38); background-size: 400% 400%;
         animation: bg 22s ease infinite; }
@keyframes bg { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.stApp::before, .stApp::after { content:""; position:fixed; width:420px; height:420px; border-radius:50%;
         filter: blur(90px); opacity:.28; z-index:0; pointer-events:none; animation: float 14s ease-in-out infinite; }
.stApp::before { background:#7c5cff; top:-120px; left:-100px; }
.stApp::after  { background:#00d4ff; bottom:-140px; right:-120px; animation-delay:-7s; }
@keyframes float { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(60px,40px) scale(1.15)} }
@keyframes fadeUp { from{opacity:0; transform:translateY(24px)} to{opacity:1; transform:translateY(0)} }
@keyframes shimmer { to { background-position: 200% center; } }
@keyframes pulse { 0%{box-shadow:0 0 0 0 var(--c)} 70%{box-shadow:0 0 0 12px transparent} 100%{box-shadow:0 0 0 0 transparent} }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; position: relative; z-index: 1; }
.hero { font-size: 3rem; font-weight: 800; line-height: 1.1; margin: 0;
        background: linear-gradient(90deg,#7c5cff,#00d4ff,#38d6a0,#7c5cff); background-size: 200% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent; animation: shimmer 6s linear infinite; }
.sub { color:#a9a9c8; margin: .4rem 0 1.4rem; animation: fadeUp .9s ease both; }
div[data-testid="stForm"], .glass { background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09);
        border-radius: 20px; padding: 1.4rem; backdrop-filter: blur(12px); animation: fadeUp .8s ease both; }
.kpi { background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.09); border-radius:16px; padding:1rem 1.2rem;
       animation: fadeUp .8s ease both; transition: transform .25s, border-color .25s; }
.kpi:hover { transform: translateY(-4px); border-color: rgba(124,92,255,.7); }
.kpi .v { font-size:1.7rem; font-weight:800; } .kpi .l { color:#a9a9c8; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }
.chip { display:inline-block; padding:.45rem .9rem; margin:.25rem .3rem .25rem 0; border-radius:999px; font-size:.85rem;
        background: rgba(124,92,255,.16); border:1px solid rgba(124,92,255,.45); animation: fadeUp .7s ease both; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; animation: pulse 1.8s infinite; }
.badge { display:inline-block; padding:.35rem 1rem; border-radius:999px; font-weight:700; color:#0d0b1f; animation: pulse 2s infinite; }
.stButton > button, .stFormSubmitButton > button { background: linear-gradient(90deg,#7c5cff,#00d4ff); color:#fff; border:0;
        border-radius:12px; font-weight:700; padding:.6rem 1.4rem; transition: transform .2s, box-shadow .2s; }
.stButton > button:hover, .stFormSubmitButton > button:hover { transform: translateY(-2px) scale(1.03);
        box-shadow: 0 10px 30px rgba(124,92,255,.55); color:#fff; }
button[data-baseweb="tab"] { font-weight:600; }
</style>
""", unsafe_allow_html=True)


def gauge_html(score: float, label: str, color: str) -> str:
    """Animated ring gauge (CSS stroke animation + JS count-up). Rendered in an iframe, so it is self-contained."""
    circ = 2 * 3.14159265 * 90
    html = """
<style>
 body{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;color:#eaeaf5;text-align:center}
 .arc{stroke-dasharray:__C__;stroke-dashoffset:__C__;animation:fill 1.8s cubic-bezier(.22,1,.36,1) .1s forwards;
      filter:drop-shadow(0 0 10px __COLOR__)}
 @keyframes fill{to{stroke-dashoffset:__T__}}
 .num{font-size:52px;font-weight:800} .lab{font-size:15px;font-weight:700;letter-spacing:.12em;fill:__COLOR__;
      animation:blink 2.4s ease-in-out infinite} @keyframes blink{50%{opacity:.55}}
 .wrap{animation:pop .9s cubic-bezier(.2,1.4,.4,1) both} @keyframes pop{from{transform:scale(.6);opacity:0}to{transform:scale(1);opacity:1}}
</style>
<div class="wrap"><svg width="270" height="270" viewBox="0 0 220 220">
 <circle cx="110" cy="110" r="90" fill="none" stroke="rgba(255,255,255,.09)" stroke-width="15"/>
 <circle class="arc" cx="110" cy="110" r="90" fill="none" stroke="__COLOR__" stroke-width="15" stroke-linecap="round"
         transform="rotate(-90 110 110)"/>
 <text id="n" class="num" x="110" y="118" text-anchor="middle" fill="#eaeaf5">0.0</text>
 <text x="110" y="142" text-anchor="middle" fill="#a9a9c8" font-size="12">out of 10</text>
 <text class="lab" x="110" y="168" text-anchor="middle">__LABEL__</text></svg></div>
<script>
 const target=__SCORE__, el=document.getElementById('n'), t0=performance.now(), dur=1600;
 (function tick(t){const p=Math.min((t-t0)/dur,1), e=1-Math.pow(1-p,3); el.textContent=(target*e).toFixed(1);
   if(p<1) requestAnimationFrame(tick);})(t0);
</script>"""
    return (html.replace("__C__", f"{circ:.2f}").replace("__T__", f"{circ * (1 - score / 10):.2f}")
            .replace("__COLOR__", color).replace("__LABEL__", label.upper()).replace("__SCORE__", f"{score:.2f}"))


def kpi(label: str, value: str, delay: float = 0.0) -> str:
    return f'<div class="kpi" style="animation-delay:{delay}s"><div class="l">{label}</div><div class="v">{value}</div></div>'


def insights(p: dict, means: dict) -> list[str]:
    out = []
    if p["Sleep_Hours_Per_Night"] < 6.5:
        out.append(f"😴 Sleep ({p['Sleep_Hours_Per_Night']}h) is below the ~{means.get('Sleep_Hours_Per_Night', 6.6):.1f}h dataset average")
    elif p["Sleep_Hours_Per_Night"] >= 7.5:
        out.append("😴 Sleep is in a healthy 7.5h+ range")
    if p["Avg_Daily_Usage_Hours"] > means.get("Avg_Daily_Usage_Hours", 5) + 1:
        out.append(f"📱 Screen time is {p['Avg_Daily_Usage_Hours'] - means.get('Avg_Daily_Usage_Hours', 5):.1f}h above average")
    if p["Physical_Activity_Hours"] < 1:
        out.append("🏃 Physical activity is under 1h/day")
    if p["Stress_Level"] in ("High", "Very High"):
        out.append(f"🔥 Reported stress is {p['Stress_Level'].lower()}")
    if p["Purpose_Of_Use"] == "Education":
        out.append("📚 Mostly using social media for education")
    return out[:4] or ["✅ Nothing stands out versus the training population"]


def plot_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#eaeaf5"), transition=dict(duration=600))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)")
    return fig


@st.cache_data(show_spinner=False, ttl=300)
def sweep(payload_json: str, field: str, values: tuple) -> list[float]:
    payload = json.loads(payload_json)
    return backend.predict_batch([{**payload, field: v} for v in values])


# ------------------------------------------------------------------------------------------- sidebar
sagemaker_configured = bool(os.getenv("SAGEMAKER_ENDPOINT"))
fastapi_configured = os.getenv("API_URL", "http://localhost:8000") != "http://localhost:8000"
if sagemaker_configured and fastapi_configured:
    choice = st.session_state.get("backend_choice", os.getenv("PREDICT_BACKEND", "sagemaker"))
    choice = st.sidebar.radio("Backend", ["sagemaker", "fastapi"],
                              index=["sagemaker", "fastapi"].index(choice),
                              format_func=lambda m: "☁️ AWS SageMaker" if m == "sagemaker" else "🖥️ FastAPI (full monitoring)")
    st.session_state["backend_choice"] = choice
    backend.mode = choice
health = backend.health()
info = backend.info()
with st.sidebar:
    st.markdown("### 🧠 MindMetrics")
    color = "#38d6a0" if health and health.get("status") == "ok" else "#ff5c7a"
    st.markdown(f'<span class="dot" style="background:{color};--c:{color}"></span>'
                f'**{"Backend online" if color == "#38d6a0" else "Backend offline"}** · `{backend.mode}`',
                unsafe_allow_html=True)
    if health:
        st.caption(health.get("model", ""))
    if info:
        m = info["metrics"]
        st.markdown(kpi("Test R²", f"{m['r2']:.3f}") + "<br>" + kpi("MAE (score pts)", f"{m['mae']:.2f}", .1),
                    unsafe_allow_html=True)
    st.divider()
    st.caption("⚠️ Educational ML demo trained on a survey dataset. Not a diagnostic or clinical tool.")

st.markdown('<p class="hero">MindMetrics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Predict a student\'s mental-health score from social-media habits & lifestyle — '
            'served by FastAPI, tracked in MLflow, monitored for drift.</p>', unsafe_allow_html=True)

if not health:
    st.error("Cannot reach the prediction backend. Start it with `make up` (Docker) or `make serve` (local), "
             "and check `API_URL` / `SAGEMAKER_ENDPOINT`.")
    st.stop()

tab_predict, tab_lab, tab_monitor, tab_about = st.tabs(["🔮 Predict", "🧪 What-if Lab", "📈 Monitoring", "🧬 Model Card"])
means = (info or {}).get("reference_means", {})

# ------------------------------------------------------------------------------------------- predict
with tab_predict:
    with st.form("student_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 👤 Profile")
            age = st.slider("Age", 15, 40, 21)
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            country = st.selectbox("Country", COUNTRIES)
            level = st.selectbox("Academic level", ["Undergraduate", "Graduate", "High School"])
        with c2:
            st.markdown("##### 📱 Social media")
            platform = st.selectbox("Most used platform", PLATFORMS)
            purpose = st.selectbox("Main purpose", ["Entertainment", "Education", "Networking", "News"])
            usage = st.slider("Avg daily usage (h)", 0.0, 12.0, 4.5, 0.1)
            unlocks = st.slider("Daily phone unlocks", 0, 400, 150, 5)
        with c3:
            st.markdown("##### 🌙 Lifestyle")
            study = st.slider("Study hours / day", 0.0, 12.0, 3.0, 0.1)
            sleep = st.slider("Sleep hours / night", 3.0, 12.0, 7.0, 0.1)
            activity = st.slider("Physical activity (h/day)", 0.0, 6.0, 1.5, 0.1)
            stress = st.select_slider("Stress level", STRESS, value="Medium")
        go_btn = st.form_submit_button("✨ Analyze wellbeing", use_container_width=True)

    if go_btn:
        st.session_state["payload"] = {
            "Age": age, "Gender": gender, "Country": country, "Academic_Level": level,
            "Most_Used_Platform": platform, "Purpose_Of_Use": purpose, "Avg_Daily_Usage_Hours": usage,
            "Daily_Unlocks": unlocks, "Study_Hours": study, "Physical_Activity_Hours": activity,
            "Sleep_Hours_Per_Night": sleep, "Stress_Level": stress}
        bar = st.progress(0, text="Validating input…")
        for pct, txt in [(35, "Engineering features…"), (70, "Running the model…")]:
            time.sleep(0.25)
            bar.progress(pct, text=txt)
        try:
            st.session_state["result"] = backend.predict(st.session_state["payload"])
            nudges = {"😴 +1h sleep": {"Sleep_Hours_Per_Night": min(sleep + 1, 12)},
                      "📱 −1h screen time": {"Avg_Daily_Usage_Hours": max(usage - 1, 0)},
                      "🏃 +1h activity": {"Physical_Activity_Hours": min(activity + 1, 6)},
                      "🔓 −100 unlocks": {"Daily_Unlocks": max(unlocks - 100, 0)}}
            if stress != "Low":
                nudges["🧘 Stress one level lower"] = {"Stress_Level": STRESS[STRESS.index(stress) - 1]}
            preds = backend.predict_batch([{**st.session_state["payload"], **chg} for chg in nudges.values()])
            base = st.session_state["result"]["prediction"]
            st.session_state["nudges"] = {k: round(p - base, 3) for k, p in zip(nudges, preds)}
        except Exception as exc:
            bar.empty()
            st.error(f"Prediction failed: {exc}")
            st.stop()
        bar.progress(100, text="Done ✓")
        time.sleep(0.2)
        bar.empty()

    if "result" in st.session_state:
        res = st.session_state["result"]
        col_g, col_r = st.columns([1, 1.25])
        with col_g:
            components.html(gauge_html(res["prediction"], res["band"], BAND_COLOR[res["band"]]), height=290)
            st.markdown(kpi("Inference latency", f"{res['latency_ms']} ms") , unsafe_allow_html=True)
        with col_r:
            st.markdown("##### 💡 What stands out")
            st.markdown("".join(f'<span class="chip" style="animation-delay:{i * .12}s">{t}</span>'
                                for i, t in enumerate(insights(st.session_state["payload"], means))),
                        unsafe_allow_html=True)
            nudges = st.session_state["nudges"]
            fig = go.Figure(go.Bar(x=list(nudges.values()), y=list(nudges.keys()), orientation="h",
                                   marker_color=["#38d6a0" if v >= 0 else "#ff5c7a" for v in nudges.values()],
                                   text=[f"{v:+.2f}" for v in nudges.values()], textposition="outside"))
            fig.update_layout(title="Predicted score change if… (model sensitivity, not medical advice)")
            st.plotly_chart(plot_layout(fig, 300), use_container_width=True)
        st.caption(f"Model: {res['model_version']} · Predictions are statistical associations from survey data.")
    else:
        st.info("Fill in the form and hit **Analyze wellbeing** ✨")

# ------------------------------------------------------------------------------------------- what-if lab
with tab_lab:
    payload = st.session_state.get("payload")
    if not payload:
        st.info("Run a prediction first — the lab sweeps the features of your last analysis.")
    else:
        st.markdown("Each curve changes **one** feature while holding everything else at your last input.")
        usage_x = [round(i * 0.5, 1) for i in range(0, 25)]
        sleep_x = [round(4 + i * 0.25, 2) for i in range(0, 25)]
        left, right = st.columns(2)
        for col, field, xs, title in [(left, "Avg_Daily_Usage_Hours", usage_x, "Daily social-media hours"),
                                      (right, "Sleep_Hours_Per_Night", sleep_x, "Sleep hours per night")]:
            ys = sweep(json.dumps(payload, sort_keys=True), field, tuple(xs))
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", line=dict(width=3, color="#7c5cff"),
                                       fillcolor="rgba(124,92,255,.18)"))
            fig.add_vline(x=payload[field], line_dash="dot", line_color="#00d4ff", annotation_text="you")
            fig.update_layout(title=f"Predicted score vs {title}", yaxis_title="score", xaxis_title=title)
            fig.update_yaxes(range=[max(0, min(ys) - 1), min(10, max(ys) + 1)])
            col.plotly_chart(plot_layout(fig), use_container_width=True)

# ------------------------------------------------------------------------------------------- monitoring
with tab_monitor:
    if not backend.supports_monitoring:
        st.info("Live drift monitoring runs against the FastAPI backend. For the SageMaker endpoint, use "
                "CloudWatch (invocation metrics + prediction logs) — see the README.")
    else:
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("🟢 Send normal traffic (300)", use_container_width=True):
            backend.simulate(300, False)
        if b2.button("🔴 Send drifted traffic (400)", use_container_width=True):
            backend.simulate(400, True)
        if b3.button("🧹 Reset window", use_container_width=True):
            backend.reset()
        b4.button("🔄 Refresh", use_container_width=True)

        rep = backend.drift() or {"ready": False, "overall": "insufficient_data", "n_current": 0, "features": {}}
        overall = rep["overall"]
        st.markdown(f'<div class="glass"><span class="badge" style="background:{STATUS_COLOR[overall]};--c:{STATUS_COLOR[overall]}">'
                    f'{overall.replace("_", " ").upper()}</span> &nbsp; {rep["n_current"]} recent predictions compared with the '
                    f'held-out reference set (PSI &lt; 0.1 stable · 0.1–0.25 warning · &gt; 0.25 drift)</div>', unsafe_allow_html=True)
        if rep.get("ready"):
            df = (pd.DataFrame(rep["features"]).T.reset_index().rename(columns={"index": "feature"})
                  .sort_values("psi", ascending=True))
            fig = go.Figure(go.Bar(x=df["psi"], y=df["feature"], orientation="h",
                                   marker_color=[STATUS_COLOR[s] for s in df["status"]]))
            fig.add_vline(x=0.1, line_dash="dot", line_color="#ffb547")
            fig.add_vline(x=0.25, line_dash="dot", line_color="#ff5c7a")
            fig.update_layout(title="Population Stability Index by feature (incl. model output)")
            st.plotly_chart(plot_layout(fig, 430), use_container_width=True)
            cols = [c for c in ["type", "psi", "status", "ks_p_value", "ref_mean", "cur_mean"] if c in df.columns]
            st.dataframe(df.set_index("feature")[cols].sort_values("psi", ascending=False), use_container_width=True)
        else:
            st.info(rep.get("message", "Send some traffic to populate the drift report."))

# ------------------------------------------------------------------------------------------- model card
with tab_about:
    if not info:
        st.info("Model card is served by the FastAPI backend.")
    else:
        m = info["metrics"]
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi("Champion", info["algorithm"].replace("_", " ")), unsafe_allow_html=True)
        k2.markdown(kpi("R² (test)", f"{m['r2']:.3f}", .1), unsafe_allow_html=True)
        k3.markdown(kpi("MAE", f"{m['mae']:.3f}", .2), unsafe_allow_html=True)
        k4.markdown(kpi("RMSE", f"{m['rmse']:.3f}", .3), unsafe_allow_html=True)
        cand = pd.DataFrame(info["candidates"])
        fig = go.Figure([go.Bar(name="R²", x=cand["name"], y=cand["r2"], marker_color="#7c5cff"),
                         go.Bar(name="RMSE", x=cand["name"], y=cand["rmse"], marker_color="#00d4ff")])
        fig.update_layout(barmode="group", title="Candidates compared in MLflow")
        st.plotly_chart(plot_layout(fig), use_container_width=True)
        st.caption(f"Registry: `{info['model_version']}` · trained {info['trained_at']}")
        st.graphviz_chart("""digraph { rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fillcolor="#2a2456" fontcolor=white color="#7c5cff"]; edge [color="#00d4ff"];
            Data -> "Train + MLflow" -> Registry -> "Docker image" -> "FastAPI / SageMaker" -> Streamlit;
            "FastAPI / SageMaker" -> "Drift + Prometheus" -> Streamlit; "GitHub Actions" -> "Docker image"; }""")

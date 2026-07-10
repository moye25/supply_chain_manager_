"""
================================================================================
AUTONOMOUS SUPPLY CHAIN MANAGER — Kaya AI India Hackathon 2026 (Track 2)
Smart Command Center — Streamlit + CrewAI single-file prototype (v2 UI)
================================================================================

WHAT THIS DOES
---------------
A proactive Multi-Agent AI system for construction supply chains:

  1. Procurement Agent  -> reads an unstructured vendor email, extracts the
                            delayed material, delay length, and root cause.
  2. Scheduling Agent    -> cross-references the delay against the project
                            schedule to find every downstream task/crew/
                            equipment rental that is now at risk.
  3. Financial Agent     -> compares the cost of idle labor/equipment vs. the
                            cost of rescheduling, and issues a recommendation.

v2 adds: a wide "Smart Command Center" layout, a full-width inbox card (no
more squeezed sidebar email), KPI stat cards, and a Gantt-style timeline chart
that visually shows the planned vs. delayed schedule and every task at risk.

--------------------------------------------------------------------------------
PIP INSTALL
--------------------------------------------------------------------------------
    pip install streamlit crewai crewai-tools plotly

--------------------------------------------------------------------------------
SET YOUR LLM API KEY
--------------------------------------------------------------------------------
Replace the placeholder below, set an OPENAI_API_KEY env var, or (recommended
for Streamlit Community Cloud) add it under App settings > Secrets:
    OPENAI_API_KEY = "sk-..."

--------------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------------
    streamlit run app.py

================================================================================
"""

import os
import time
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go
from crewai import Agent, Task, Crew, Process, LLM

# ==============================================================================
# 0. LLM CONFIG — PLACEHOLDER API KEY
# ==============================================================================
# Key lookup order (first match wins):
#   1. Streamlit Cloud "Secrets" (Advanced settings > Secrets on deploy)
#   2. A real OPENAI_API_KEY environment variable (local/dev use)
#   3. Placeholder -> triggers MOCK MODE automatically (fully working scripted
#      demo, no API key needed — safe fallback for live judging)
_key_from_secrets = ""
try:
    _key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    _key_from_secrets = ""

os.environ.setdefault("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
if _key_from_secrets:
    os.environ["OPENAI_API_KEY"] = _key_from_secrets

MOCK_MODE = os.environ.get("OPENAI_API_KEY", "") in ("", "YOUR_API_KEY_HERE")
llm = LLM(model="gpt-4o-mini", temperature=0.2) if not MOCK_MODE else None

# ==============================================================================
# 1. HARDCODED MOCK DATA
# ==============================================================================

VENDOR_EMAIL = {
    "from": "dispatch@apexsteelsupply.com",
    "to": "procurement@skylineconstruction.com",
    "subject": "URGENT — Shipment Delay Notice — PO#48291 (Structural Steel Beams)",
    "received": "Today, 08:14 AM",
    "body": (
        "Hi Team,\n\n"
        "We regret to inform you that your order of structural steel I-beams "
        "(PO#48291, 42 tons, Grade A992) is experiencing a delay due to severe "
        "congestion at the Port of Mundra. Customs clearance backlog and a "
        "berth shortage have pushed our estimated arrival from July 14 to "
        "July 18 — a 4-day delay. We are actively expediting where possible "
        "but cannot guarantee an earlier release at this time. We will notify "
        "you the moment the shipment clears customs.\n\n"
        "Apologies for the inconvenience.\n\n"
        "Regards,\nApex Steel Supply — Logistics Desk"
    ),
}

DELAY_DAYS = 4
TODAY = datetime(2026, 7, 11)

# Task -> (planned_start_offset_days, duration_days, type, daily_idle_cost)
SCHEDULE = {
    "Steel Delivery (PO#48291)": {"start": 3, "duration": 1, "type": "delivery", "daily_cost": 0},
    "Crane Rental — Tower Crane TC-88": {"start": 4, "duration": 5, "type": "equipment", "daily_cost": 2400},
    "Framing Crew (12 workers)": {"start": 4, "duration": 6, "type": "labor", "daily_cost": 3600},
    "Structural Inspection": {"start": 10, "duration": 1, "type": "milestone", "daily_cost": 0},
}

RESCHEDULE_COST = 1800  # one-time cost to re-sequence crane + crew elsewhere


def md_lite(text: str) -> str:
    """
    Convert a small, safe subset of markdown (used by our own agent output
    strings only — never on raw user input) into HTML so it can be embedded
    inside st.html() cards. Handles **bold**, "### " mini-headings, "- "
    bullets, and line breaks. This avoids relying on Streamlit's markdown
    parser for raw HTML blocks, which breaks on blank lines.
    """
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    out_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("### "):
            out_lines.append(f'<div class="mini-heading">{line[4:]}</div>')
        elif line.startswith("- "):
            out_lines.append(f'<div class="bullet-line">&bull;&nbsp;&nbsp;{line[2:]}</div>')
        elif line == "":
            out_lines.append('<div class="line-gap"></div>')
        else:
            out_lines.append(f"<div>{line}</div>")
    return "".join(out_lines)

# ==============================================================================
# 2. CREWAI AGENTS
# ==============================================================================

def build_agents():
    procurement_agent = Agent(
        role="Procurement Intelligence Agent",
        goal="Extract material, PO, delay length, ETA change, and root cause from vendor emails.",
        backstory="A former logistics coordinator turned AI agent, obsessive about catching supplier delays early.",
        llm=llm, verbose=True, allow_delegation=False,
    )
    scheduling_agent = Agent(
        role="Schedule Impact Agent",
        goal="Cross-reference a delay against the project schedule and identify every task at risk.",
        backstory="A veteran site scheduler who has seen every cascading delay a site can produce.",
        llm=llm, verbose=True, allow_delegation=False,
    )
    financial_agent = Agent(
        role="Financial Recommendation Agent",
        goal="Compare idle cost vs. reschedule cost and issue one clear numeric recommendation.",
        backstory="A cost controller who has saved projects six figures by catching idle-time bleed early.",
        llm=llm, verbose=True, allow_delegation=False,
    )
    return procurement_agent, scheduling_agent, financial_agent


class AgentPipelineError(Exception):
    """Raised when a crew.kickoff() call ultimately fails, with a judge/user-friendly message."""
    pass


def _kickoff_with_retry(crew, step_name, max_retries=2, base_delay=3):
    """
    Run crew.kickoff() with a couple of short retries on transient rate-limit
    errors (common on free-tier / low-RPM keys when 3 agents fire back-to-back
    LLM calls). If it still fails after retries, raise a clean, readable error
    instead of letting the raw traceback crash the whole page.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return str(crew.kickoff())
        except Exception as e:
            last_err = e
            err_text = str(e).lower()
            is_rate_limit = "ratelimit" in err_text.replace("_", "") or "429" in err_text
            is_quota = "quota" in err_text or "insufficient_quota" in err_text
            if (is_rate_limit or is_quota) and attempt < max_retries:
                time.sleep(base_delay * (attempt + 1))  # simple backoff: 3s, 6s...
                continue
            break

    err_text = str(last_err).lower()
    if "quota" in err_text or "insufficient_quota" in err_text:
        reason = "Your OpenAI account has run out of quota/credits."
    elif "ratelimit" in err_text.replace("_", "") or "429" in err_text:
        reason = "Your OpenAI API key is being rate-limited (too many requests for its tier)."
    else:
        reason = f"The {step_name} agent's API call failed unexpectedly ({type(last_err).__name__})."
    raise AgentPipelineError(f"{reason} Falling back to scripted demo output so the run can still complete.")


def run_live_crew(email_body, schedule):
    procurement_agent, scheduling_agent, financial_agent = build_agents()

    t1 = Task(
        description=f"Read this vendor email and extract Material, PO Number, Delay (days), Original ETA, New ETA, Root Cause:\n\n{email_body}",
        expected_output="Structured summary with Material, PO Number, Delay, Original ETA, New ETA, Root Cause.",
        agent=procurement_agent,
    )
    c1 = Crew(agents=[procurement_agent], tasks=[t1], process=Process.sequential, verbose=True)
    proc_out = _kickoff_with_retry(c1, "Procurement")

    t2 = Task(
        description=f"Delay info: {proc_out}\n\nProject schedule: {schedule}\n\nList every affected task with idle days and idle cost per day.",
        expected_output="List of affected tasks with idle cost.",
        agent=scheduling_agent,
    )
    c2 = Crew(agents=[scheduling_agent], tasks=[t2], process=Process.sequential, verbose=True)
    sched_out = _kickoff_with_retry(c2, "Scheduling")

    t3 = Task(
        description=f"Schedule impact: {sched_out}\n\nOne-time reschedule cost: ${RESCHEDULE_COST}. Recommend reschedule or absorb idle cost, with dollars saved.",
        expected_output="Final recommendation with dollar savings.",
        agent=financial_agent,
    )
    c3 = Crew(agents=[financial_agent], tasks=[t3], process=Process.sequential, verbose=True)
    fin_out = _kickoff_with_retry(c3, "Financial")

    return proc_out, sched_out, fin_out


# ==============================================================================
# 3. MOCK-MODE FALLBACK (used automatically when no real API key is set)
# ==============================================================================

def mock_procurement_output():
    return (
        "**Material:** Structural Steel I-Beams (Grade A992)  \n"
        "**PO Number:** #48291  \n"
        "**Delay:** 4 days  \n"
        "**Original ETA:** 2026-07-14  \n"
        "**New ETA:** 2026-07-18  \n"
        "**Root Cause:** Port congestion at Port of Mundra — customs clearance backlog and berth shortage."
    )


def mock_scheduling_output():
    return (
        "**Crane Rental — Tower Crane TC-88**\n"
        "- Planned start blocked until steel arrives.\n"
        "- Idle exposure: 4 days × $2,400/day = **$9,600**\n\n"
        "**Framing Crew (12 workers)**\n"
        "- Planned start blocked until steel arrives.\n"
        "- Idle exposure: 4 days × $3,600/day = **$14,400**\n\n"
        "**Downstream Milestone:** Structural Inspection pushed back 4 days."
    )


def mock_financial_output(idle_cost, savings):
    return (
        f"**Idle cost if no action taken:** ${idle_cost:,}  \n"
        f"**Cost to reschedule crane + framing crew for 4 days:** ${RESCHEDULE_COST:,}\n\n"
        f"### ✅ Recommendation: RESCHEDULE\n"
        f"Move Tower Crane TC-88 and the 12-person framing crew to the Riverside "
        f"Block B site for 4 days rather than let them sit idle. This saves an "
        f"estimated **${savings:,}**."
    )


# ==============================================================================
# 4. STREAMLIT PAGE CONFIG + STYLE
# ==============================================================================

st.set_page_config(
    page_title="Smart Command Center | Autonomous Supply Chain Manager",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Design tokens: blueprint-industrial palette ----
NAVY = "#0B1524"
STEEL = "#1B3A6B"
STEEL_LIGHT = "#3A5075"
ORANGE = "#E8871E"
CONCRETE = "#F2F3F5"
CARD = "#FFFFFF"
INK = "#151B26"
MUTED = "#6B7280"
ALERT = "#C0392B"
SUCCESS = "#1E7A46"
GRID_LINE = "#22406E"

st.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; }}
.stApp {{ background-color: {CONCRETE}; }}

/* Header banner with blueprint grid texture */
.command-header {{
    background-color: {NAVY};
    background-image:
        linear-gradient(rgba(58,80,117,0.35) 1px, transparent 1px),
        linear-gradient(90deg, rgba(58,80,117,0.35) 1px, transparent 1px);
    background-size: 26px 26px;
    border-radius: 10px;
    padding: 26px 30px;
    margin-bottom: 22px;
    border-left: 5px solid {ORANGE};
}}
.command-header .eyebrow {{
    color: {ORANGE}; font-family: 'IBM Plex Mono', monospace; font-size: 12.5px;
    letter-spacing: 1.5px; font-weight: 500; margin-bottom: 6px;
}}
.command-header h1 {{
    color: #FFFFFF !important; font-size: 30px; margin: 0 0 6px 0; font-weight: 700;
}}
.command-header p {{ color: #C7D0E8; margin: 0; font-size: 14.5px; }}
.status-pill {{
    display: inline-block; background: rgba(30,122,70,0.18); color: #58D68D;
    border: 1px solid rgba(88,214,141,0.4); border-radius: 20px;
    padding: 4px 12px; font-family: 'IBM Plex Mono', monospace; font-size: 11.5px;
    margin-top: 10px;
}}

/* KPI cards */
.kpi-card {{
    background: {CARD}; border-radius: 10px; padding: 16px 18px;
    border: 1px solid #E3E6ED; border-top: 3px solid {STEEL};
}}
.kpi-card.risk {{ border-top-color: {ALERT}; }}
.kpi-card.save {{ border-top-color: {SUCCESS}; }}
.kpi-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: {MUTED};
    letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 4px;
}}
.kpi-value {{ font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: {INK}; }}

/* Inbox card */
.inbox-card {{
    background: {CARD}; border-radius: 10px; border: 1px solid #E3E6ED;
    overflow: hidden; margin-bottom: 4px;
}}
.inbox-meta {{
    background: {CONCRETE}; padding: 14px 20px; border-bottom: 1px solid #E3E6ED;
    font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: {MUTED};
    line-height: 1.9;
}}
.inbox-subject {{
    font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 700;
    color: {ALERT}; padding: 14px 20px 0 20px;
}}
.inbox-body {{
    padding: 10px 20px 20px 20px; color: {INK}; font-size: 14.5px; line-height: 1.7;
    white-space: pre-line;
}}

/* Section label */
.section-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: {STEEL};
    letter-spacing: 1.2px; text-transform: uppercase; font-weight: 500;
    margin: 26px 0 8px 0; display: flex; align-items: center; gap: 8px;
}}
.section-label::after {{ content: ""; flex: 1; height: 1px; background: #D6DAE3; }}

/* Agent cards */
.agent-card {{
    background: {CARD}; border-radius: 10px; border: 1px solid #E3E6ED;
    border-left: 4px solid {STEEL_LIGHT}; padding: 14px 18px; margin-bottom: 12px;
}}
.agent-card.fin {{ border-left-color: {ORANGE}; }}
.agent-name {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 14.5px;
    color: {STEEL}; margin-bottom: 6px;
}}
.agent-card.fin .agent-name {{ color: {ORANGE}; }}

/* Final recommendation banner */
.rec-banner {{
    background: linear-gradient(135deg, {NAVY} 0%, {STEEL} 100%);
    border-radius: 12px; padding: 24px 28px; border-left: 6px solid {ORANGE};
    color: white; margin-top: 10px;
}}
.rec-banner h4 {{ color: #FFFFFF !important; margin-top: 0; }}
.rec-banner p, .rec-banner li {{ color: #E4E9F5; }}
.rec-banner strong {{ color: {ORANGE}; }}

/* Lightweight markdown output (agent cards, rec banner) */
.mini-heading {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px;
    margin: 10px 0 4px 0;
}}
.bullet-line {{ margin: 2px 0; }}
.line-gap {{ height: 8px; }}
</style>
""")

# ==============================================================================
# 5. SESSION STATE
# ==============================================================================

for key, default in [("processed", False), ("approved", False), ("proc_out", ""),
                      ("sched_out", ""), ("fin_out", ""), ("idle_cost", 0), ("savings", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================================
# 6. HEADER
# ==============================================================================

st.html(f"""
<div class="command-header">
    <div class="eyebrow">KAYA AI INDIA HACKATHON 2026 · TRACK 2 · SUPPLY CHAIN</div>
    <h1>🏗️ Smart Command Center</h1>
    <p>Autonomous Supply Chain Manager — Procurement → Scheduling → Financial agents monitoring your site in real time.</p>
    <div class="status-pill">● {"MOCK MODE — SCRIPTED DEMO" if MOCK_MODE else "LIVE MODE — CONNECTED"}</div>
</div>
""")

# ==============================================================================
# 7. SIDEBAR — CONTROL PANEL
# ==============================================================================

with st.sidebar:
    st.markdown("### 📥 Control Panel")
    st.caption("Simulated inbox: **1 new vendor email (unread by agents)**")
    process_clicked = st.button("🚨 Process Incoming Vendor Emails", type="primary", use_container_width=True)
    st.caption("👆 The email below is just a preview — nothing has been analyzed yet. Click this to run it through the Procurement → Scheduling → Financial agent pipeline.")

    st.divider()
    st.markdown("### 📅 Active Project Tasks")
    for task, d in SCHEDULE.items():
        icon = {"delivery": "🚚", "equipment": "🏗️", "labor": "👷", "milestone": "🏁"}[d["type"]]
        planned_date = (TODAY + timedelta(days=d["start"])).strftime("%b %d")
        st.caption(f"{icon} **{task}**  \nStart: {planned_date}")

    st.divider()
    if MOCK_MODE:
        st.info("No live API key detected — running scripted mock mode so the full demo works for judges regardless.", icon="ℹ️")

# ==============================================================================
# 8. KPI ROW
# ==============================================================================

k1, k2, k3, k4 = st.columns(4)
tasks_at_risk = sum(1 for d in SCHEDULE.values() if d["type"] in ("equipment", "labor"))
with k1:
    st.html(f'<div class="kpi-card"><div class="kpi-label">Delay Detected</div><div class="kpi-value">{DELAY_DAYS} days</div></div>')
with k2:
    st.html(f'<div class="kpi-card risk"><div class="kpi-label">Tasks At Risk</div><div class="kpi-value">{tasks_at_risk}</div></div>')
with k3:
    idle_preview = DELAY_DAYS * sum(d["daily_cost"] for d in SCHEDULE.values())
    st.html(f'<div class="kpi-card risk"><div class="kpi-label">Idle Cost Exposure</div><div class="kpi-value">${idle_preview:,}</div></div>')
with k4:
    savings_preview = idle_preview - RESCHEDULE_COST
    st.html(f'<div class="kpi-card save"><div class="kpi-label">Potential Savings</div><div class="kpi-value">${savings_preview:,}</div></div>')

# ==============================================================================
# 9. INBOX CARD (full width, not squeezed)
# ==============================================================================

st.html('<div class="section-label">📧 Incoming Vendor Communication</div>')
_badge = (
    '<span style="background:#1E7A46;color:white;border-radius:20px;padding:3px 12px;'
    'font-family:\'IBM Plex Mono\',monospace;font-size:11px;margin-left:10px;">✅ PROCESSED BY AGENTS</span>'
    if st.session_state.processed else
    '<span style="background:#C0392B;color:white;border-radius:20px;padding:3px 12px;'
    'font-family:\'IBM Plex Mono\',monospace;font-size:11px;margin-left:10px;">🆕 AWAITING ANALYSIS</span>'
)
st.html(f"""
<div class="inbox-card">
    <div class="inbox-meta">
        FROM&nbsp;&nbsp;{VENDOR_EMAIL['from']}<br>
        TO&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{VENDOR_EMAIL['to']}<br>
        RECEIVED&nbsp;&nbsp;{VENDOR_EMAIL['received']}
    </div>
    <div class="inbox-subject">⚠️ {VENDOR_EMAIL['subject']}{_badge}</div>
    <div class="inbox-body">{VENDOR_EMAIL['body']}</div>
</div>
""")

# ==============================================================================
# 10. GANTT-STYLE SCHEDULE IMPACT CHART
# ==============================================================================

st.html('<div class="section-label">📅 Schedule Impact — Planned vs. Delayed</div>')

fig = go.Figure()
tasks = list(SCHEDULE.keys())
colors_map = {"delivery": STEEL_LIGHT, "equipment": ORANGE, "labor": ORANGE, "milestone": MUTED}
MIN_BAR_DAYS = 1.4  # visual floor so 1-day tasks still render as a readable bar, not an invisible sliver

planned_x, planned_base, delayed_x, delayed_base = [], [], [], []
for task in tasks:
    d = SCHEDULE[task]
    planned_start = TODAY + timedelta(days=d["start"])
    planned_end = planned_start + timedelta(days=d["duration"])
    shifted_start = planned_start + timedelta(days=DELAY_DAYS)
    shifted_end = shifted_start + timedelta(days=d["duration"])

    vis_duration = max(d["duration"], MIN_BAR_DAYS)
    planned_x.append(vis_duration)
    planned_base.append(planned_start)
    delayed_x.append(vis_duration)
    delayed_base.append(shifted_start)

    # Date-range label + delay callout, placed just to the right of the delayed bar
    fig.add_annotation(
        x=shifted_start + timedelta(days=vis_duration + 0.15), y=task,
        text=f"{planned_start.strftime('%b %d')} → {shifted_start.strftime('%b %d')}  "
             f"<span style='color:{ALERT}'><b>(+{DELAY_DAYS}d)</b></span>",
        showarrow=False, xanchor="left", font=dict(family="IBM Plex Mono", size=11, color=INK),
        align="left",
    )

fig.add_trace(go.Bar(
    x=planned_x, y=tasks, base=planned_base, orientation="h",
    marker=dict(color="#D8DEE9", line=dict(color=STEEL_LIGHT, width=1.5)),
    name="Originally Planned",
    hovertemplate="<b>%{y}</b><br>Originally planned: %{base|%b %d}<extra></extra>",
))
fig.add_trace(go.Bar(
    x=delayed_x, y=tasks, base=delayed_base, orientation="h",
    marker=dict(color=[colors_map[SCHEDULE[t]["type"]] for t in tasks]),
    name="Delayed / At Risk",
    hovertemplate="<b>%{y}</b><br>New date: %{base|%b %d}<extra></extra>",
))

fig.add_vline(x=TODAY.timestamp() * 1000, line_dash="dot", line_width=2, line_color=ALERT,
              annotation_text="Today", annotation_font_color=ALERT, annotation_font_size=12)

fig.update_layout(
    barmode="group", bargap=0.35, bargroupgap=0.08,
    height=340, plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=170, t=10, b=10),  # right margin makes room for the date-range labels
    xaxis=dict(type="date", gridcolor="#E9ECF2", tickfont=dict(family="IBM Plex Mono", size=11)),
    yaxis=dict(autorange="reversed", tickfont=dict(family="Inter", size=12.5)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11.5)),
    font=dict(family="Inter"),
)
st.plotly_chart(fig, use_container_width=True, theme=None)
st.caption(
    "Gray bars = originally planned schedule. Orange/steel bars = the delayed schedule the agents detected. "
    "The label at the end of each row shows the exact date shift. Dotted red line = today."
)

# ==============================================================================
# 11. AGENT ACTIVITY FEED
# ==============================================================================

st.html('<div class="section-label">🧠 Agent Activity Feed</div>')
feed = st.container()

if process_clicked:
    st.session_state.approved = False
    idle_cost = DELAY_DAYS * sum(d["daily_cost"] for d in SCHEDULE.values())
    savings = idle_cost - RESCHEDULE_COST

    pipeline_error = None
    if not MOCK_MODE:
        try:
            with st.spinner("Agents are analyzing the delay..."):
                proc_out, sched_out, fin_out = run_live_crew(VENDOR_EMAIL["body"], SCHEDULE)
        except AgentPipelineError as e:
            pipeline_error = str(e)
            proc_out = mock_procurement_output()
            sched_out = mock_scheduling_output()
            fin_out = mock_financial_output(idle_cost, savings)
    else:
        proc_out = mock_procurement_output()
        sched_out = mock_scheduling_output()
        fin_out = mock_financial_output(idle_cost, savings)

    if pipeline_error:
        st.warning(f"⚠️ Live agent call didn't complete: {pipeline_error}", icon="⚠️")

    with feed:
        with st.status("👷 Procurement Agent reading vendor email...", expanded=True) as s:
            time.sleep(0.8)
            st.html(f'<div class="agent-card"><div class="agent-name">PROCUREMENT AGENT</div>{md_lite(proc_out)}</div>')
            s.update(label="✅ Procurement Agent: delay extracted", state="complete")

        with st.status("📅 Scheduling Agent cross-referencing project timeline...", expanded=True) as s:
            time.sleep(0.9)
            st.html(f'<div class="agent-card"><div class="agent-name">SCHEDULING AGENT</div>{md_lite(sched_out)}</div>')
            s.update(label="✅ Scheduling Agent: downstream impact mapped", state="complete")

        with st.status("💰 Financial Agent calculating cost tradeoffs...", expanded=True) as s:
            time.sleep(0.9)
            st.html(f'<div class="agent-card fin"><div class="agent-name">FINANCIAL AGENT</div>{md_lite(fin_out)}</div>')
            s.update(label="✅ Financial Agent: recommendation ready", state="complete")

    st.session_state.processed = True
    st.session_state.proc_out = proc_out
    st.session_state.sched_out = sched_out
    st.session_state.fin_out = fin_out
    st.session_state.idle_cost = idle_cost
    st.session_state.savings = savings

elif st.session_state.processed:
    with feed:
        st.html(f'<div class="agent-card"><div class="agent-name">PROCUREMENT AGENT</div>{md_lite(st.session_state.proc_out)}</div>')
        st.html(f'<div class="agent-card"><div class="agent-name">SCHEDULING AGENT</div>{md_lite(st.session_state.sched_out)}</div>')
        st.html(f'<div class="agent-card fin"><div class="agent-name">FINANCIAL AGENT</div>{md_lite(st.session_state.fin_out)}</div>')
else:
    with feed:
        st.info("Click **Process Incoming Vendor Emails** in the sidebar to run the Procurement → Scheduling → Financial pipeline.")

# ==============================================================================
# 12. FINAL RECOMMENDATION BANNER
# ==============================================================================

if st.session_state.processed:
    st.html('<div class="section-label">🚦 Final Recommendation</div>')
    st.html(f"""
    <div class="rec-banner">
        <h4>⚠️ Action Required — {DELAY_DAYS}-Day Steel Delay Detected</h4>
        {md_lite(st.session_state.fin_out)}
    </div>
    """)

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ Approve", type="primary", disabled=st.session_state.approved, use_container_width=True):
            st.session_state.approved = True
    with col2:
        if st.button("❌ Dismiss", disabled=st.session_state.approved, use_container_width=True):
            st.session_state.processed = False
            st.rerun()

    if st.session_state.approved:
        st.success(f"Recommendation approved. Crane + framing crew reassignment queued. Estimated savings locked in: **${st.session_state.savings:,}**", icon="✅")

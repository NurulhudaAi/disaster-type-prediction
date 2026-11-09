import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from src.models.registry import ModelRegistry
from src.agent import DisasterAuditorLLM

# -------------------------------------------------------------
# Page Configuration (Full Screen Wide Layout)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Disaster Auditor — Model Evaluation & XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Custom Styling: Fullscreen Canvas, Neutral Teal/Indigo, WCAG AA
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Fullscreen canvas adjustment */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }

    /* Top Navbar */
    .app-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 0.75rem 0;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .app-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1.25rem;
        box-shadow: 0 2px 4px rgba(13, 148, 136, 0.2);
    }

    .brand-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        line-height: 1.2;
    }

    .brand-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 1px;
    }

    .brand-powered {
        font-size: 0.72rem;
        color: #94A3B8;
        font-weight: 500;
    }

    /* KPI Pills in Navbar */
    .nav-kpi-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #1E293B;
    }

    /* Engine Status Badges */
    .status-badge-ready {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-badge-neutral {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F8FAFC;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-dot-green {
        width: 7px;
        height: 7px;
        background: #16A34A;
        border-radius: 50%;
    }

    .status-dot-amber {
        width: 7px;
        height: 7px;
        background: #D97706;
        border-radius: 50%;
    }

    /* Welcome Hero */
    .welcome-hero {
        text-align: center;
        padding: 2.2rem 1.5rem 1.4rem 1.5rem;
        max-width: 680px;
        margin: 0 auto;
    }

    .welcome-headline {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.4rem;
    }

    .welcome-subheadline {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 1.6rem;
    }

    /* Telemetry KPI Grid in Right Pane */
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-bottom: 12px;
    }

    .telemetry-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
    }

    .telemetry-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 2px;
    }

    .telemetry-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* User message bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #F1F5F9;
        border-radius: 12px;
        padding: 6px 14px;
        border: 1px solid #E2E8F0;
    }

    /* Remove any red borders from dropdowns */
    div[data-baseweb="select"] {
        border-color: #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_registry():
    return ModelRegistry()

registry = load_registry()

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def new_stats():
    # y_true / y_pred hold the primary model's labels per audit so the session Macro-F1 is real.
    return {"tested": 0, "correct": 0, "hard_cases": 0, "y_true": [], "y_pred": []}

if "stats" not in st.session_state or "y_true" not in st.session_state.stats:
    st.session_state.stats = new_stats()

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Random Forest"

if "comparison_mode" not in st.session_state:
    st.session_state.comparison_mode = False

if "comparison_models" not in st.session_state:
    st.session_state.comparison_models = ["Random Forest", "LightGBM", "XGBoost"]

if "layout_mode" not in st.session_state:
    st.session_state.layout_mode = "Split Workspace"

if "last_audit" not in st.session_state:
    st.session_state.last_audit = None

if "hard_threshold" not in st.session_state:
    st.session_state.hard_threshold = 0.55

if "api_provider" not in st.session_state:
    st.session_state.api_provider = "Google Gemini"

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

if "claude_api_key" not in st.session_state:
    st.session_state.claude_api_key = os.getenv("ANTHROPIC_API_KEY", "")

if "pending_quick_action" not in st.session_state:
    st.session_state.pending_quick_action = None

# -------------------------------------------------------------
# Audit Execution Controller
# -------------------------------------------------------------
def active_models():
    return st.session_state.comparison_models if st.session_state.comparison_mode else [st.session_state.selected_model]

def record_audit(case, models_to_test):
    """Run the audit, store it as the active case, and update the session scoreboard."""
    audit_result = registry.audit_case(case, models_to_test)
    st.session_state.last_audit = audit_result

    stats = st.session_state.stats
    primary = audit_result["models"][models_to_test[0]]
    stats["tested"] += 1
    stats["y_true"].append(case["ground_truth_class"])
    stats["y_pred"].append(primary["prediction"]["predicted_class"])
    if primary["is_correct"]:
        stats["correct"] += 1
    # Hard case: misclassified, or correct but below the confidence threshold.
    if not primary["is_correct"] or primary["prediction"]["confidence"] < st.session_state.hard_threshold:
        stats["hard_cases"] += 1
    return audit_result

def session_macro_f1():
    stats = st.session_state.stats
    if not stats["y_true"]:
        return None
    return f1_score(stats["y_true"], stats["y_pred"], average="macro", zero_division=0)

def run_audit(case, prompt_label=None, target_models=None):
    if not case:
        st.toast("No matching case found in the benchmark bank.", icon=":material/warning:")
        return

    models_to_test = target_models or active_models()

    audit_result = record_audit(case, models_to_test)

    user_text = prompt_label or f"Audit case `{case['case_id']}` ({case['ground_truth_class']})"
    st.session_state.messages.append({
        "role": "user",
        "content": user_text
    })

    st.session_state.messages.append({
        "role": "assistant",
        "audit": audit_result,
        "models_tested": models_to_test
    })
    st.rerun()

QUICK_ACTIONS = {
    "🎲 Random Case": {},
    "⚠️ Hard Case": {"only_hard": True},
    "⚡ Disagreement Case": {"only_disagreement": True},
    "🌊 Flood": {"disaster_type": "Flood"},
    "🌋 Earthquake": {"disaster_type": "Earthquake"},
    "🌪️ Storm": {"disaster_type": "Storm"},
}

def queue_quick_action():
    # Runs as a widget callback: remember the pick and clear the pill so it doesn't re-fire on every rerun.
    st.session_state.pending_quick_action = st.session_state.quick_pills
    st.session_state.quick_pills = None

def handle_slash_command(command_text: str):
    cmd = command_text.strip().lower()
    if cmd.startswith("/random"):
        run_audit(registry.get_random_case(), "Execute slash command: `/random` (Unseen Ground Truth)")
    elif cmd.startswith("/hard"):
        run_audit(registry.get_random_case(only_hard=True), "Execute slash command: `/hard` (Hard Diagnostic Case)")
    elif cmd.startswith("/compare"):
        st.session_state.comparison_mode = True
        run_audit(registry.get_random_case(only_disagreement=True), "Execute slash command: `/compare` (Disagreement Case)")
    elif cmd.startswith("/category"):
        parts = cmd.split(maxsplit=1)
        cat_query = parts[1].strip() if len(parts) > 1 else "Flood"
        matched = next((c for c in registry.class_names if c.lower() == cat_query.lower()), None)
        if matched:
            run_audit(registry.get_random_case(disaster_type=matched), f"Execute slash command: `/category {matched}`")
        else:
            st.toast(f"Unknown category '{cat_query}'. Available: {', '.join(registry.class_names)}", icon=":material/info:")
    elif cmd.startswith("/export"):
        st.session_state.layout_mode = "Split Workspace"
        st.toast("Opened Export Tab in the right workbench panel.", icon=":material/download:")
    else:
        st.toast(f"Unrecognized command '{command_text}'. Try /random, /hard, /compare, /category <name>, /export", icon=":material/help:")

# -------------------------------------------------------------
# Top Navbar (Full Width)
# -------------------------------------------------------------
col_nav_brand, col_nav_view, col_nav_stats = st.columns([0.42, 0.32, 0.26], vertical_alignment="center")

with col_nav_brand:
    st.markdown("""
    <div class="app-brand">
        <div class="brand-mark">🛡️</div>
        <div>
            <div class="brand-title">Disaster Auditor</div>
            <div class="brand-subtext">Multi-Class Disaster Classification & XAI Benchmark <span class="brand-powered">• powered by Claude</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_nav_view:
    view_choice = st.segmented_control(
        "Workspace View Layout",
        options=["Split Workspace", "Chat Focus", "Benchmark & Analytics"],
        default=st.session_state.layout_mode,
        label_visibility="collapsed"
    )
    if view_choice and view_choice != st.session_state.layout_mode:
        st.session_state.layout_mode = view_choice
        st.rerun()

with col_nav_stats:
    tested_cnt = st.session_state.stats["tested"]
    acc_str = f"{(st.session_state.stats['correct']/tested_cnt*100):.1f}%" if tested_cnt > 0 else "—"
    nav_model = st.session_state.selected_model
    nav_f1 = registry.leaderboard[nav_model]["macro_f1"]
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-end; gap:8px;">
        <span class="nav-kpi-pill">{nav_model} Test Macro-F1: <b>{nav_f1:.4f}</b></span>
        <span class="nav-kpi-pill">Session: <b>{tested_cnt} Tested</b> ({acc_str})</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -------------------------------------------------------------
# Left Sidebar: Session Scoreboard, Targeted Audit, Settings
# -------------------------------------------------------------
with st.sidebar:
    if st.button("+ New Audit Session", icon=":material/add:", type="primary", width="stretch", help="Reset current chat and session statistics"):
        st.session_state.messages = []
        st.session_state.stats = new_stats()
        st.session_state.last_audit = None
        st.rerun()

    st.markdown("### :material/monitoring: Session Scoreboard")
    with st.container(border=True):
        tested = st.session_state.stats["tested"]
        correct = st.session_state.stats["correct"]
        hard = st.session_state.stats["hard_cases"]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.caption("Headline Metric")
            f1_val = session_macro_f1()
            f1_disp = f"{f1_val:.4f}" if f1_val is not None else "—"
            st.metric("Macro-F1", f1_disp, help="Macro-F1 of the active model over the cases audited this session")
            st.metric("Total Tested", tested if tested > 0 else "—")

        with col_m2:
            st.caption("Secondary Metric")
            acc_disp = f"{(correct / tested * 100):.1f}%" if tested > 0 else "—"
            st.metric("Accuracy", acc_disp)
            st.metric("Hard Cases", hard if tested > 0 else "—", help="Cases the active model misclassified, or got right with confidence below the threshold in Settings")

        if st.button("Reset Scoreboard", icon=":material/restart_alt:", width="stretch", disabled=(tested == 0)):
            st.session_state.stats = new_stats()
            st.rerun()

    st.markdown("### :material/filter_alt: Targeted Audit")
    with st.container(border=True):
        st.markdown("**Run audit on disaster category:**")
        cat_select = st.selectbox(
            "Select Disaster Type",
            ["All Categories"] + registry.class_names,
            label_visibility="collapsed"
        )
        if st.button("Run Audit", icon=":material/play_arrow:", width="stretch"):
            target_cat = None if cat_select == "All Categories" else cat_select
            c = registry.get_random_case(disaster_type=target_cat)
            if c:
                run_audit(c, f"Evaluate test record for category: **{cat_select}** (`{c['case_id']}`)")
            else:
                st.toast("No records found matching category.", icon=":material/warning:")

    st.markdown("### :material/memory: Engine Status")
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:6px;">
            <div class="status-badge-ready">
                <span class="status-dot-green"></span>
                <span>Random Forest (Pipeline): Ready</span>
            </div>
            <div class="status-badge-ready">
                <span class="status-dot-green"></span>
                <span>LightGBM (Pipeline): Ready</span>
            </div>
            <div class="status-badge-ready">
                <span class="status-dot-green"></span>
                <span>XGBoost (Pipeline): Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        is_gemini = (st.session_state.api_provider == "Google Gemini")
        active_key = st.session_state.gemini_api_key if is_gemini else st.session_state.claude_api_key
        llm_connected = bool(active_key and active_key.strip())
        llm_label = "Gemini" if is_gemini else "Claude Opus 5.5"

        if llm_connected:
            st.markdown(f"""
            <div class="status-badge-ready">
                <span class="status-dot-green"></span>
                <span>{llm_label}: Connected</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-badge-neutral">
                <span class="status-dot-amber"></span>
                <span>{llm_label}: Offline (Key missing)</span>
            </div>
            """, unsafe_allow_html=True)

    with st.expander(":material/settings: LLM & Audit Settings", expanded=True if not llm_connected else False):
        st.markdown("**LLM Provider**")
        provider_options = ["Google Gemini", "Anthropic Claude"]
        curr_idx = 0 if st.session_state.api_provider == "Google Gemini" else 1
        prov_choice = st.selectbox(
            "Select AI Provider",
            provider_options,
            index=curr_idx,
            label_visibility="collapsed"
        )
        if prov_choice != st.session_state.api_provider:
            st.session_state.api_provider = prov_choice
            st.rerun()

        if st.session_state.api_provider == "Google Gemini":
            gem_key = st.text_input(
                "Google Gemini API Key:",
                type="password",
                value=st.session_state.gemini_api_key,
                placeholder="AIzaSy...",
                help="Enter your Google Gemini API Key from https://aistudio.google.com"
            )
            # Auto-switch to Anthropic if user enters sk-ant key
            if gem_key.startswith("sk-ant-"):
                st.session_state.claude_api_key = gem_key
                st.session_state.api_provider = "Anthropic Claude"
                st.rerun()
            elif gem_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gem_key
                st.rerun()
        else:
            ant_key = st.text_input(
                "Anthropic API Key:",
                type="password",
                value=st.session_state.claude_api_key,
                placeholder="sk-ant-api03-...",
                help="Enter your Anthropic API Key"
            )
            # Auto-switch to Gemini if user enters AIzaSy key
            if ant_key.startswith("AIzaSy"):
                st.session_state.gemini_api_key = ant_key
                st.session_state.api_provider = "Google Gemini"
                st.rerun()
            elif ant_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = ant_key
                st.rerun()

        st.markdown("---")
        st.markdown("**Hard Case Confidence Threshold**")
        st.session_state.hard_threshold = st.slider(
            "Flag as hard case when confidence < threshold:",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.hard_threshold,
            step=0.05
        )

    st.caption(f":material/database: Test Bank: **{registry.metadata['total_test_cases']:,}** records | **{len(registry.class_names)}** classes | **{registry.metadata['disagreement_cases']}** model disagreements")

# -------------------------------------------------------------
# Main Screen Layout Selection
# -------------------------------------------------------------
current_view = st.session_state.layout_mode

if current_view == "Split Workspace":
    col_chat, col_telemetry = st.columns([0.56, 0.44], gap="large")
elif current_view == "Chat Focus":
    col_chat = st.container()
    col_telemetry = None
else:  # Benchmark & Analytics
    col_chat = None
    col_telemetry = st.container()

# =============================================================
# CHAT COLUMN
# =============================================================
if col_chat:
    with col_chat:
        # Starter Screen (if messages empty)
        if len(st.session_state.messages) == 0:
            st.markdown("""
            <div class="welcome-hero">
                <div class="welcome-headline">How can I assist your disaster model audit today?</div>
                <div class="welcome-subheadline">Inspect multi-class event classifications, audit SHAP feature attributions, and compare predictions across Random Forest, LightGBM, and XGBoost.</div>
            </div>
            """, unsafe_allow_html=True)

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                if st.button(
                    ":material/shuffle: **Random Ground Truth Case**\n\nAudit an unseen historical event from the 3,495 test bank",
                    width="stretch",
                    key="btn_card_1"
                ):
                    run_audit(registry.get_random_case(), "Audit a random historical case from EM-DAT test bank")

                if st.button(
                    ":material/compare_arrows: **Model Disagreement Finder**\n\nSurface difficult cases where RF, LightGBM, and XGBoost conflict",
                    width="stretch",
                    key="btn_card_2"
                ):
                    st.session_state.comparison_mode = True
                    run_audit(registry.get_random_case(only_disagreement=True), "Find a case with model prediction disagreement")

            with pcol2:
                if st.button(
                    ":material/warning: **Hard Case Diagnostic**\n\nStress-test models on known boundary misclassifications",
                    width="stretch",
                    key="btn_card_3"
                ):
                    run_audit(registry.get_random_case(only_hard=True), "Examine a known misclassified hard case")

                if st.button(
                    ":material/waves: **Hydrometeorological Audit**\n\nChallenge seasonal Flood vs. Storm decision boundaries",
                    width="stretch",
                    key="btn_card_4"
                ):
                    run_audit(registry.get_random_case(disaster_type="Flood"), "Audit a Flood case against seasonal Storm features")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Chat Conversation Feed
        chat_height = 580 if current_view == "Split Workspace" else 650
        chat_container = st.container(height=chat_height, border=True)
        with chat_container:
            if len(st.session_state.messages) == 0:
                st.caption(":material/chat_bubble: Messages and audit case cards will stream here.")

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant", avatar=":material/shield:"):
                        audit = msg.get("audit")
                        if audit:
                            case = audit["case"]
                            tested_models = msg.get("models_tested", [st.session_state.selected_model])
                            gt = case["ground_truth_class"]

                            st.markdown(f"#### Case Reference: `{case['case_id']}`")
                            st.caption(f"📍 Location: **{case['summary']['country']}** ({case['summary']['region']}) | Month {case['summary']['start_month']} | Magnitude Scale: `{case['summary']['magnitude_scale']}`")

                            if len(tested_models) == 1:
                                m_name = tested_models[0]
                                m_data = audit["models"][m_name]
                                pred_class = m_data["prediction"]["predicted_class"]
                                conf = m_data["prediction"]["confidence"] * 100
                                is_correct = m_data["is_correct"]

                                verdict_badge = ":material/check_circle: **CORRECT**" if is_correct else ":material/cancel: **MISCLASSIFIED**"
                                verdict_color = "green" if is_correct else "red"

                                st.markdown(f"""
                                * **Ground Truth:** **{gt}**
                                * **{m_name} Prediction:** **{pred_class}** *(Confidence: {conf:.1f}%)*
                                * **Verdict:** :{verdict_color}[{verdict_badge}]
                                """)

                                # Confidence progress
                                st.progress(m_data["prediction"]["confidence"])

                                st.markdown("**Top Decision Signals (SHAP):**")
                                for exp in m_data["shap"]["human_explanations"][:3]:
                                    st.markdown(f"• {exp}")
                            else:
                                st.markdown(f"**Ground Truth (Actual):** `{gt}`")
                                comp_cols = st.columns(len(tested_models))
                                for idx, m_name in enumerate(tested_models):
                                    m_data = audit["models"][m_name]
                                    with comp_cols[idx]:
                                        with st.container(border=True):
                                            st.markdown(f"**{m_name}**")
                                            is_c = m_data["is_correct"]
                                            v_badge = "✓ Correct" if is_c else "✗ Wrong"
                                            v_col = "green" if is_c else "red"
                                            st.write(f"Pred: **{m_data['prediction']['predicted_class']}**")
                                            st.write(f"Conf: **{m_data['prediction']['confidence']*100:.1f}%**")
                                            st.markdown(f":{v_col}[**{v_badge}**]")
                        if "content" in msg and msg["content"]:
                            st.markdown(msg["content"])

        # Quick Action Pills
        st.markdown("<div style='margin-top: 6px; margin-bottom: 2px;'></div>", unsafe_allow_html=True)
        st.pills(
            "Quick Actions",
            options=list(QUICK_ACTIONS),
            label_visibility="collapsed",
            key="quick_pills",
            on_change=queue_quick_action
        )
        quick_pick = st.session_state.pending_quick_action
        if quick_pick:
            st.session_state.pending_quick_action = None
            filters = QUICK_ACTIONS[quick_pick]
            if filters.get("only_disagreement"):
                st.session_state.comparison_mode = True
            run_audit(registry.get_random_case(**filters))

        # Composer Toolbar: Model Picker Chip + Compare Toggle
        col_c_model, col_c_toggle, col_c_hint = st.columns([0.48, 0.32, 0.20], vertical_alignment="center")

        with col_c_model:
            selected_m = st.segmented_control(
                "Active Model",
                options=registry.available_models,
                default=st.session_state.selected_model,
                label_visibility="collapsed"
            )
            if selected_m and selected_m != st.session_state.selected_model:
                st.session_state.selected_model = selected_m
                st.rerun()

        with col_c_toggle:
            comp_toggle = st.toggle("Compare Models", value=st.session_state.comparison_mode)
            if comp_toggle != st.session_state.comparison_mode:
                st.session_state.comparison_mode = comp_toggle
                st.rerun()

        with col_c_hint:
            st.caption("Press Enter to send")

        # Multi-select if compare mode
        if st.session_state.comparison_mode:
            selected_comps = st.pills(
                "Models to Compare:",
                options=registry.available_models,
                default=st.session_state.comparison_models,
                selection_mode="multi"
            )
            if selected_comps and selected_comps != st.session_state.comparison_models:
                st.session_state.comparison_models = list(selected_comps)

        # Chat Input
        user_query = st.chat_input(
            "Ask a disaster question or type /random, /hard, /compare, /category <name>, /export...",
            submit_mode="disable"
        )
        if user_query:
            if user_query.startswith("/"):
                handle_slash_command(user_query)
            else:
                st.session_state.messages.append({"role": "user", "content": user_query})

                lower_q = user_query.lower()
                test_keywords = ["สุ่ม", "audit", "test", "case", "ลอง", "random", "hard", "ผิด", "misclassified", "เคส"]
                wants_audit = any(w in lower_q for w in test_keywords)

                if wants_audit:
                    cat_match = next((c for c in registry.class_names if c.lower() in lower_q), None)
                    only_h = any(w in lower_q for w in ["hard", "ผิด", "wrong", "misclassified"])
                    case = registry.get_random_case(disaster_type=cat_match, only_hard=only_h)
                    if case:
                        models_to_test = active_models()
                        audit_result = record_audit(case, models_to_test)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "audit": audit_result,
                            "models_tested": models_to_test
                        })

                # Stream conversational answer from Gemini or Claude
                prov_code = "gemini" if st.session_state.api_provider == "Google Gemini" else "anthropic"
                active_api_key = st.session_state.gemini_api_key if prov_code == "gemini" else st.session_state.claude_api_key

                llm = DisasterAuditorLLM(provider=prov_code, api_key=active_api_key)
                chat_history = [
                    {"role": m["role"], "content": m.get("content", "")}
                    for m in st.session_state.messages[:-1]
                    if m.get("content")
                ]

                with st.chat_message("assistant", avatar=":material/smart_toy:"):
                    full_llm_text = st.write_stream(llm.chat_stream(
                        user_message=user_query,
                        history=chat_history,
                        current_case=st.session_state.last_audit
                    ))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_llm_text
                })
                st.rerun()

# =============================================================
# TELEMETRY & WORKBENCH COLUMN
# =============================================================
if col_telemetry:
    with col_telemetry:
        tab_case, tab_leader, tab_disagree, tab_export = st.tabs([
            "🔍 Case Telemetry",
            "🏆 Leaderboard & Metrics",
            "⚡ Disagreement Radar",
            "📥 Export Report"
        ])

        with tab_case:
            if st.session_state.last_audit is None:
                st.info("No active case audit. Click a quick prompt or run an audit in chat to inspect live SHAP attributions and probability distributions.", icon=":material/info:")
            else:
                audit = st.session_state.last_audit
                case = audit["case"]
                primary_model = st.session_state.selected_model
                m_res = audit["models"].get(primary_model) or list(audit["models"].values())[0]

                # Top Telemetry Grid
                is_c = m_res["is_correct"]
                st.markdown(f"""
                <div class="telemetry-grid">
                    <div class="telemetry-box">
                        <div class="telemetry-label">Ground Truth (Actual)</div>
                        <div class="telemetry-val" style="color:#0284C7;">{case['ground_truth_class']}</div>
                    </div>
                    <div class="telemetry-box">
                        <div class="telemetry-label">{primary_model} Prediction</div>
                        <div class="telemetry-val" style="color:{'#16A34A' if is_c else '#DC2626'};">{m_res['prediction']['predicted_class']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.caption(f"**Confidence:** `{m_res['prediction']['confidence']*100:.2f}%` | Case: `{case['case_id']}` ({case['summary']['country']})")

                # Probability Distribution Horizontal Bar Chart
                st.markdown("**8-Class Probability Distribution**")
                probs_df = pd.DataFrame(
                    list(m_res["prediction"]["probabilities"].items()),
                    columns=["Disaster Class", "Probability"]
                ).sort_values("Probability", ascending=True)

                st.bar_chart(
                    probs_df.set_index("Disaster Class"),
                    horizontal=True,
                    color="#0D9488",
                    height=240
                )

                # SHAP Feature Drivers
                st.markdown("**Top SHAP Decision Drivers**")
                shap_df = pd.DataFrame([
                    {
                        "Feature": item["feature"],
                        "SHAP Impact": item["shap_value"],
                        "Direction": "Supports (+)" if item["shap_value"] > 0 else "Opposes (-)"
                    }
                    for item in m_res["shap"]["top_all"]
                ])
                st.dataframe(
                    shap_df,
                    column_config={
                        "SHAP Impact": st.column_config.NumberColumn(format="%.4f")
                    },
                    hide_index=True,
                    width="stretch"
                )

                st.markdown("**Natural Language Explanation:**")
                for exp in m_res["shap"]["human_explanations"][:3]:
                    st.caption(f"• {exp}")

        with tab_leader:
            st.markdown("##### Multi-Model Leaderboard (Zero-Leakage)")
            lb_rows = []
            for m_name, metrics in registry.leaderboard.items():
                ci = metrics["macro_f1_ci"]
                lb_rows.append({
                    "Model": m_name,
                    "Macro-F1": f"{metrics['macro_f1']:.4f}",
                    "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                    "Accuracy": f"{metrics['accuracy']*100:.2f}%",
                    "Weighted-F1": f"{metrics['weighted_f1']:.4f}"
                })
            st.dataframe(pd.DataFrame(lb_rows), hide_index=True, width="stretch")

            # Per-Class F1 Grouped Chart
            st.markdown("---")
            st.markdown("##### Per-Class F1 Score Comparison")
            class_f1_dict = {"Disaster Class": registry.class_names}
            for m_name, metrics in registry.leaderboard.items():
                class_f1_dict[m_name] = [metrics["per_class_f1"].get(cls, 0.0) for cls in registry.class_names]

            class_f1_df = pd.DataFrame(class_f1_dict).set_index("Disaster Class")
            st.bar_chart(class_f1_df, height=260)

            # Paired Significance (McNemar)
            st.markdown("---")
            st.markdown("##### Paired Significance Check (McNemar)")
            for pair, result in registry.mcnemar.items():
                sig_note = "Statistically Significant (p < 0.05)" if result["significant"] else "Difference within noise (p ≥ 0.05)"
                st.caption(f"• **{pair}**: p = `{result['p_value']:.4f}` — *{sig_note}*")

        with tab_disagree:
            st.markdown("##### Model Disagreement Radar")
            dis_pct = registry.metadata['disagreement_cases'] / registry.metadata['total_test_cases'] * 100
            st.info(f"The test bank contains **{registry.metadata['disagreement_cases']}** cases (**{dis_pct:.1f}%**) where Random Forest, LightGBM, and XGBoost predict different classes.", icon=":material/info:")

            if st.button("Surface Random Disagreement Case", icon=":material/shuffle:", width="stretch"):
                st.session_state.comparison_mode = True
                c = registry.get_random_case(only_disagreement=True)
                run_audit(c, f"Evaluate disagreement case `{c['case_id']}`" if c else None)

            st.markdown("---")
            st.markdown("**Why Disagreements Occur:**")
            st.caption("• **Boundary Confusion**: Hydrometeorological overlap between Flood and Storm during monsoon months.")
            st.caption("• **Minority Class Recall**: Random Forest's deep bagging recall vs. LightGBM/XGBoost's conservative regularization on Drought and Wildfire.")

        with tab_export:
            st.markdown("##### Export Audit Report")
            st.write("Download the session report and audit results.")
            report_df = pd.DataFrame([
                {"Metric": "Total Tested", "Value": str(st.session_state.stats["tested"])},
                {"Metric": "Total Correct", "Value": str(st.session_state.stats["correct"])},
                {"Metric": "Session Accuracy", "Value": f"{(st.session_state.stats['correct']/st.session_state.stats['tested']*100):.1f}%" if st.session_state.stats['tested'] > 0 else "—"},
                {"Metric": "Hard Cases", "Value": str(st.session_state.stats["hard_cases"])},
                {"Metric": "Disagreement Bank Total", "Value": str(registry.metadata["disagreement_cases"])}
            ])
            st.dataframe(report_df, hide_index=True, width="stretch")

            csv_data = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Session CSV Report",
                data=csv_data,
                file_name="disaster_auditor_session_report.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch"
            )

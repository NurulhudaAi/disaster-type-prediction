import os
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score

from src.agent import DisasterAuditorLLM
from src.models.registry import ModelRegistry

st.set_page_config(
    page_title="Disaster Auditor",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Chat-style polish that theme config can't express
# (colors and fonts live in .streamlit/config.toml)
# -------------------------------------------------------------
st.html("""
<style>
    /* Assistant turns: no bubble, just text on the page */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.25rem 0;
    }

    /* User turns: right-aligned soft bubble, no avatar */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        background: #F0EEE6;
        border-radius: 1.1rem;
        padding: 0.55rem 1rem;
        margin-left: auto;
        width: fit-content;
        max-width: 85%;
    }
    [data-testid="stChatMessageAvatarUser"] { display: none; }

    /* Composer: floating rounded box */
    [data-testid="stChatInput"] {
        border-radius: 1.25rem;
        box-shadow: 0 0.25rem 1.25rem rgba(31, 30, 29, 0.07);
    }

    /* Start-screen greeting */
    .greeting {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2.3rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        color: #1F1E1D;
        text-align: center;
        margin: 0;
    }
    .greeting-mark { color: #C15F3C; margin-right: 0.35rem; }
</style>
""")


@st.cache_resource
def load_registry():
    return ModelRegistry()


registry = load_registry()

VIEWS = ["Chat", "Chat + details", "Benchmarks"]


# -------------------------------------------------------------
# Session state
# -------------------------------------------------------------
def new_stats():
    # y_true / y_pred hold the primary model's labels per audit so the session Macro-F1 is real.
    return {"tested": 0, "correct": 0, "hard_cases": 0, "y_true": [], "y_pred": []}


DEFAULTS = {
    "messages": [],
    "selected_model": "Random Forest",
    "comparison_mode": False,
    "comparison_models": ["Random Forest", "LightGBM", "XGBoost"],
    "layout_mode": "Chat",
    "last_audit": None,
    "hard_threshold": 0.55,
    "api_provider": "Google Gemini",
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "claude_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    "pending_quick_action": None,
    "pending_question": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "stats" not in st.session_state or "y_true" not in st.session_state.stats:
    st.session_state.stats = new_stats()
if st.session_state.layout_mode not in VIEWS:
    st.session_state.layout_mode = "Chat"


# -------------------------------------------------------------
# Audit controller
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
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "audit": audit_result, "models_tested": models_to_test})
    st.rerun()


QUICK_ACTIONS = {
    ":material/shuffle: Random case": ({}, "Audit a random historical case from the test bank"),
    ":material/warning: Hard case": ({"only_hard": True}, "Examine a case at least one model misclassified"),
    ":material/compare_arrows: Model disagreement": ({"only_disagreement": True}, "Find a case where the models disagree"),
    ":material/water_drop: Flood": ({"disaster_type": "Flood"}, "Audit a Flood case"),
    ":material/landslide: Earthquake": ({"disaster_type": "Earthquake"}, "Audit an Earthquake case"),
    ":material/storm: Storm": ({"disaster_type": "Storm"}, "Audit a Storm case"),
}


def queue_quick_action():
    # Widget callback: remember the pick and clear the pill so it doesn't re-fire on every rerun.
    st.session_state.pending_quick_action = st.session_state.quick_pills
    st.session_state.quick_pills = None


def run_quick_action(label):
    filters, prompt_label = QUICK_ACTIONS[label]
    if filters.get("only_disagreement"):
        st.session_state.comparison_mode = True
    run_audit(registry.get_random_case(**filters), prompt_label)


def handle_slash_command(command_text: str):
    cmd = command_text.strip().lower()
    if cmd.startswith("/random"):
        run_audit(registry.get_random_case(), "`/random`")
    elif cmd.startswith("/hard"):
        run_audit(registry.get_random_case(only_hard=True), "`/hard`")
    elif cmd.startswith("/compare"):
        st.session_state.comparison_mode = True
        run_audit(registry.get_random_case(only_disagreement=True), "`/compare`")
    elif cmd.startswith("/category"):
        parts = cmd.split(maxsplit=1)
        cat_query = parts[1].strip() if len(parts) > 1 else "Flood"
        matched = next((c for c in registry.class_names if c.lower() == cat_query.lower()), None)
        if matched:
            run_audit(registry.get_random_case(disaster_type=matched), f"`/category {matched}`")
        else:
            st.toast(f"Unknown category '{cat_query}'. Available: {', '.join(registry.class_names)}", icon=":material/info:")
    elif cmd.startswith("/export"):
        st.session_state.layout_mode = "Benchmarks"
        st.rerun()
    else:
        st.toast(f"Unrecognized command '{command_text}'. Try /random, /hard, /compare, /category <name>, /export", icon=":material/help:")


def submit_question(text: str):
    """Show the user's turn right away; the reply streams on the next run."""
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.pending_question = text
    st.rerun()


def answer_pending_question():
    """Optionally audit a case the question asks for, then stream the LLM reply."""
    question = st.session_state.pending_question
    st.session_state.pending_question = None

    lower_q = question.lower()
    test_keywords = ["สุ่ม", "audit", "test", "case", "ลอง", "random", "hard", "ผิด", "misclassified", "เคส"]
    if any(w in lower_q for w in test_keywords):
        cat_match = next((c for c in registry.class_names if c.lower() in lower_q), None)
        only_h = any(w in lower_q for w in ["hard", "ผิด", "wrong", "misclassified"])
        case = registry.get_random_case(disaster_type=cat_match, only_hard=only_h)
        if case:
            models_to_test = active_models()
            audit_result = record_audit(case, models_to_test)
            msg = {"role": "assistant", "audit": audit_result, "models_tested": models_to_test}
            st.session_state.messages.append(msg)
            render_message(msg, len(st.session_state.messages) - 1)

    prov_code = "gemini" if st.session_state.api_provider == "Google Gemini" else "anthropic"
    api_key = st.session_state.gemini_api_key if prov_code == "gemini" else st.session_state.claude_api_key
    llm = DisasterAuditorLLM(provider=prov_code, api_key=api_key)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m.get("content")
    ]
    with st.chat_message("assistant", avatar=":material/shield:"):
        reply = st.write_stream(llm.chat_stream(
            user_message=question,
            history=history,
            current_case=st.session_state.last_audit,
        ))
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()


# -------------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------------
def verdict_badge(is_correct):
    if is_correct:
        st.badge("Correct", icon=":material/check:", color="green")
    else:
        st.badge("Misclassified", icon=":material/close:", color="red")


def render_audit_card(audit, tested_models, msg_idx):
    case = audit["case"]
    summary = case["summary"]
    with st.container(border=True):
        st.markdown(f":material/description: **Case {case['case_id']}** · ground truth **{case['ground_truth_class']}**")
        st.caption(
            f"{summary['country']} ({summary['region']}) · month {summary['start_month']} · "
            f"magnitude scale `{summary['magnitude_scale']}`"
        )

        if len(tested_models) == 1:
            m_name = tested_models[0]
            m_data = audit["models"][m_name]
            pred = m_data["prediction"]
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"{m_name} predicted **{pred['predicted_class']}** · {pred['confidence'] * 100:.1f}% confidence")
                verdict_badge(m_data["is_correct"])
            st.progress(pred["confidence"])
            st.markdown("**Top decision signals (SHAP)**")
            for exp in m_data["shap"]["human_explanations"][:3]:
                st.markdown(f"- {exp}")
        else:
            cols = st.columns(len(tested_models))
            for col, m_name in zip(cols, tested_models):
                m_data = audit["models"][m_name]
                with col:
                    st.markdown(f"**{m_name}**")
                    st.markdown(f"{m_data['prediction']['predicted_class']} · {m_data['prediction']['confidence'] * 100:.1f}%")
                    verdict_badge(m_data["is_correct"])

        is_latest = audit is st.session_state.last_audit
        if is_latest and st.session_state.layout_mode == "Chat":
            if st.button("Open details", icon=":material/right_panel_open:", type="tertiary", key=f"details_{msg_idx}"):
                st.session_state.layout_mode = "Chat + details"
                st.rerun()


def render_message(msg, idx):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant", avatar=":material/shield:"):
        if msg.get("audit"):
            render_audit_card(msg["audit"], msg.get("models_tested", [st.session_state.selected_model]), idx)
        if msg.get("content"):
            st.markdown(msg["content"])


def render_composer_toolbar():
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        picked = st.menu_button(
            "Quick audit",
            options=list(QUICK_ACTIONS),
            icon=":material/add:",
            type="tertiary",
            key="quick_menu",
        )
        selected_m = st.segmented_control(
            "Active model",
            options=registry.available_models,
            default=st.session_state.selected_model,
            label_visibility="collapsed",
        )
        compare = st.toggle("Compare", value=st.session_state.comparison_mode)

    if st.session_state.comparison_mode:
        selected_comps = st.pills(
            "Models to compare",
            options=registry.available_models,
            default=st.session_state.comparison_models,
            selection_mode="multi",
            label_visibility="collapsed",
        )
        if selected_comps and list(selected_comps) != st.session_state.comparison_models:
            st.session_state.comparison_models = list(selected_comps)

    if selected_m and selected_m != st.session_state.selected_model:
        st.session_state.selected_model = selected_m
        st.rerun()
    if compare != st.session_state.comparison_mode:
        st.session_state.comparison_mode = compare
        st.rerun()
    if picked:
        run_quick_action(picked)


def handle_input(user_query):
    if not user_query:
        return
    if user_query.startswith("/"):
        handle_slash_command(user_query)
    else:
        submit_question(user_query)


def greeting_text():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def render_chat(composer_placeholder):
    if not st.session_state.messages:
        # Start screen: greeting, centered composer, suggestion chips
        st.space("large")
        st.space("large")
        st.html(
            f'<p class="greeting"><span class="greeting-mark">✻</span>{greeting_text()}. '
            "Which disaster model shall we audit?</p>"
        )
        st.space("medium")
        user_query = st.chat_input(composer_placeholder, submit_mode="disable")
        st.pills(
            "Suggestions",
            options=list(QUICK_ACTIONS),
            label_visibility="collapsed",
            key="quick_pills",
            on_change=queue_quick_action,
        )
        render_composer_toolbar()
        st.caption(
            "Audits run on 3 models over the held-out EM-DAT test set. "
            "Type `/` for commands: `/random`, `/hard`, `/compare`, `/category Flood`, `/export`."
        )
        handle_input(user_query)
        return

    # Conversation: scrolling message area with the composer pinned below it
    with st.container(height=600, border=False):
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)
        if st.session_state.pending_question:
            answer_pending_question()

    user_query = st.chat_input(composer_placeholder, submit_mode="disable")
    render_composer_toolbar()
    st.caption("Model outputs and LLM explanations can be wrong. Check against the SHAP values and ground truth.")
    handle_input(user_query)


def render_case_details():
    if st.session_state.last_audit is None:
        st.info("Run an audit in the chat to inspect its SHAP attributions and probability distribution.", icon=":material/info:")
        return

    audit = st.session_state.last_audit
    case = audit["case"]
    primary_model = st.session_state.selected_model
    m_res = audit["models"].get(primary_model) or list(audit["models"].values())[0]
    pred = m_res["prediction"]

    with st.container(horizontal=True):
        st.metric("Ground truth", case["ground_truth_class"], border=True)
        st.metric(f"{primary_model} prediction", pred["predicted_class"], border=True)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(f"Confidence `{pred['confidence'] * 100:.2f}%` · case `{case['case_id']}` · {case['summary']['country']}")
        verdict_badge(m_res["is_correct"])

    st.markdown("**8-class probability distribution**")
    probs_df = pd.DataFrame(
        list(pred["probabilities"].items()),
        columns=["Disaster class", "Probability"],
    ).sort_values("Probability", ascending=True)
    st.bar_chart(
        probs_df.set_index("Disaster class"),
        horizontal=True,
        color="#C15F3C",
        height=240,
    )

    st.markdown("**Top SHAP decision drivers**")
    shap_df = pd.DataFrame([
        {
            "Feature": item["feature"],
            "SHAP impact": item["shap_value"],
            "Direction": "Supports (+)" if item["shap_value"] > 0 else "Opposes (-)",
        }
        for item in m_res["shap"]["top_all"]
    ])
    st.dataframe(
        shap_df,
        column_config={"SHAP impact": st.column_config.NumberColumn(format="%.4f")},
        hide_index=True,
    )

    st.markdown("**In plain language**")
    for exp in m_res["shap"]["human_explanations"][:3]:
        st.caption(f"• {exp}")


def render_leaderboard():
    lb_rows = []
    for m_name, metrics in registry.leaderboard.items():
        ci = metrics["macro_f1_ci"]
        lb_rows.append({
            "Model": m_name,
            "Macro-F1": metrics["macro_f1"],
            "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "Accuracy": metrics["accuracy"] * 100,
            "Weighted-F1": metrics["weighted_f1"],
        })
    st.dataframe(
        pd.DataFrame(lb_rows),
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.2f%%"),
            "Weighted-F1": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
    )

    st.markdown("**Per-class F1**")
    class_f1 = {"Disaster class": registry.class_names}
    for m_name, metrics in registry.leaderboard.items():
        class_f1[m_name] = [metrics["per_class_f1"].get(cls, 0.0) for cls in registry.class_names]
    st.bar_chart(pd.DataFrame(class_f1).set_index("Disaster class"), height=260, stack=False)

    st.markdown("**Paired significance (McNemar)**")
    for pair, result in registry.mcnemar.items():
        note = "significant (p < 0.05)" if result["significant"] else "within noise (p ≥ 0.05)"
        st.caption(f"• **{pair}**: p = `{result['p_value']:.4f}`, {note}")


def render_disagreement():
    meta = registry.metadata
    dis_pct = meta["disagreement_cases"] / meta["total_test_cases"] * 100
    st.markdown(
        f"The test bank has **{meta['disagreement_cases']}** cases (**{dis_pct:.1f}%**) where "
        "Random Forest, LightGBM, and XGBoost predict different classes."
    )
    if st.button("Audit a disagreement case", icon=":material/shuffle:"):
        st.session_state.comparison_mode = True
        if st.session_state.layout_mode == "Benchmarks":
            st.session_state.layout_mode = "Chat + details"
        c = registry.get_random_case(only_disagreement=True)
        run_audit(c, f"Evaluate disagreement case `{c['case_id']}`" if c else None)
    st.markdown("**Why disagreements happen**")
    st.caption("• Boundary confusion: Flood and Storm overlap during monsoon months.")
    st.caption("• Minority-class recall: Random Forest's bagging vs. the boosters' stronger regularization on Drought and Wildfire.")


def render_export():
    stats = st.session_state.stats
    f1_val = session_macro_f1()
    report_df = pd.DataFrame([
        {"Metric": "Total tested", "Value": str(stats["tested"])},
        {"Metric": "Total correct", "Value": str(stats["correct"])},
        {"Metric": "Session accuracy", "Value": f"{stats['correct'] / stats['tested'] * 100:.1f}%" if stats["tested"] else "—"},
        {"Metric": "Session Macro-F1", "Value": f"{f1_val:.4f}" if f1_val is not None else "—"},
        {"Metric": "Hard cases", "Value": str(stats["hard_cases"])},
        {"Metric": "Disagreement bank total", "Value": str(registry.metadata["disagreement_cases"])},
    ])
    st.dataframe(report_df, hide_index=True)
    st.download_button(
        "Download session report (CSV)",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="disaster_auditor_session_report.csv",
        mime="text/csv",
        icon=":material/download:",
    )


# -------------------------------------------------------------
# Sidebar: new audit, session, settings
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### :material/shield: Disaster Auditor")

    if st.button("New audit", icon=":material/edit_square:", type="primary", width="stretch"):
        st.session_state.messages = []
        st.session_state.stats = new_stats()
        st.session_state.last_audit = None
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("**This session**")
    tested = st.session_state.stats["tested"]
    correct = st.session_state.stats["correct"]
    f1_val = session_macro_f1()
    with st.container(horizontal=True):
        st.metric("Macro-F1", f"{f1_val:.3f}" if f1_val is not None else "—",
                  help="Macro-F1 of the active model over the cases audited this session")
        st.metric("Accuracy", f"{correct / tested * 100:.0f}%" if tested else "—")
    with st.container(horizontal=True):
        st.metric("Tested", tested)
        st.metric("Hard cases", st.session_state.stats["hard_cases"],
                  help="Cases the active model misclassified, or got right with confidence below the threshold in settings")

    st.markdown("**Audit a category**")
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        cat_select = st.selectbox(
            "Disaster type",
            ["All categories"] + registry.class_names,
            label_visibility="collapsed",
        )
        if st.button("Run", icon=":material/play_arrow:"):
            target_cat = None if cat_select == "All categories" else cat_select
            run_audit(registry.get_random_case(disaster_type=target_cat), f"Audit a **{cat_select}** case")

    is_gemini = st.session_state.api_provider == "Google Gemini"
    active_key = st.session_state.gemini_api_key if is_gemini else st.session_state.claude_api_key
    llm_connected = bool(active_key and active_key.strip())
    llm_label = "Gemini" if is_gemini else "Claude Opus 5.5"

    with st.expander("Settings", icon=":material/settings:", expanded=not llm_connected):
        prov_choice = st.selectbox(
            "LLM provider",
            ["Google Gemini", "Anthropic Claude"],
            index=0 if is_gemini else 1,
        )
        if prov_choice != st.session_state.api_provider:
            st.session_state.api_provider = prov_choice
            st.rerun()

        if is_gemini:
            gem_key = st.text_input(
                "Gemini API key",
                type="password",
                value=st.session_state.gemini_api_key,
                placeholder="AIzaSy...",
                help="Get a key at https://aistudio.google.com",
            )
            if gem_key.startswith("sk-ant-"):
                st.session_state.claude_api_key = gem_key
                st.session_state.api_provider = "Anthropic Claude"
                st.rerun()
            elif gem_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gem_key
                st.rerun()
        else:
            ant_key = st.text_input(
                "Anthropic API key",
                type="password",
                value=st.session_state.claude_api_key,
                placeholder="sk-ant-api03-...",
            )
            if ant_key.startswith("AIzaSy"):
                st.session_state.gemini_api_key = ant_key
                st.session_state.api_provider = "Google Gemini"
                st.rerun()
            elif ant_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = ant_key
                st.rerun()

        st.session_state.hard_threshold = st.slider(
            "Hard-case confidence threshold",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.hard_threshold,
            step=0.05,
            help="Correct predictions below this confidence also count as hard cases",
        )

    status = ":green[●] connected" if llm_connected else ":orange[●] offline, add an API key"
    st.caption(f"{llm_label}: {status}")
    st.caption(":green[●] Random Forest · LightGBM · XGBoost ready")
    st.caption(
        f"Test bank: {registry.metadata['total_test_cases']:,} records · "
        f"{len(registry.class_names)} classes · {registry.metadata['disagreement_cases']} disagreements"
    )


# -------------------------------------------------------------
# Main area
# -------------------------------------------------------------
with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown(
        f"**{st.session_state.selected_model}** · test Macro-F1 "
        f"{registry.leaderboard[st.session_state.selected_model]['macro_f1']:.4f}"
    )
    st.space("stretch")
    view_choice = st.segmented_control(
        "View",
        options=VIEWS,
        default=st.session_state.layout_mode,
        label_visibility="collapsed",
    )
    if view_choice and view_choice != st.session_state.layout_mode:
        st.session_state.layout_mode = view_choice
        st.rerun()

# Quick-action chip picked on the previous run (queued by its callback)
if st.session_state.pending_quick_action:
    pick = st.session_state.pending_quick_action
    st.session_state.pending_quick_action = None
    run_quick_action(pick)

COMPOSER = "Ask about a case, or type / for commands"
view = st.session_state.layout_mode

if view == "Chat":
    _, center, _ = st.columns([1, 2.6, 1])
    with center:
        render_chat(COMPOSER)

elif view == "Chat + details":
    col_chat, col_details = st.columns([0.55, 0.45], gap="large")
    with col_chat:
        render_chat(COMPOSER)
    with col_details:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("**Case details**")
                st.space("stretch")
                if st.button("Close", icon=":material/close:", type="tertiary"):
                    st.session_state.layout_mode = "Chat"
                    st.rerun()
            render_case_details()

else:  # Benchmarks
    tab_leader, tab_disagree, tab_case, tab_export = st.tabs([
        ":material/leaderboard: Leaderboard",
        ":material/compare_arrows: Disagreements",
        ":material/search: Last case",
        ":material/download: Export",
    ])
    with tab_leader:
        render_leaderboard()
    with tab_disagree:
        render_disagreement()
    with tab_case:
        render_case_details()
    with tab_export:
        render_export()

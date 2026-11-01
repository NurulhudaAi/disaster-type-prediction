import os
import streamlit as st
import pandas as pd
import numpy as np
from src.predictor import DisasterPredictor
from src.explainer import DisasterExplainer
from src.benchmark import BenchmarkManager
from src.agent import DisasterAuditorAgent

# Page configuration
st.set_page_config(
    page_title="Disaster Model Auditor & XAI",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.25rem;
    }
    
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 0.25rem;
    }
    
    .badge-accuracy {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .comparison-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    
    .tag-correct {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .tag-incorrect {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_backend():
    predictor = DisasterPredictor()
    benchmark = BenchmarkManager()
    explainer = DisasterExplainer(predictor)
    agent = DisasterAuditorAgent(predictor, explainer, benchmark)
    return predictor, benchmark, explainer, agent

predictor, benchmark, explainer, agent = load_backend()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome! I am the **Disaster Model Auditor Agent**.\n\nMy primary objective is to **rigorously challenge and evaluate** the pre-trained Multi-class Random Forest model against verified **Ground Truth** events from the EM-DAT database, powered by **SHAP Explainable AI**.\n\nUse the sidebar controls to test random cases, challenge hard misclassified events, or converse naturally to inspect model predictions."
        }
    ]

if "stats" not in st.session_state:
    st.session_state.stats = {
        "tested": 0,
        "correct": 0,
        "incorrect": 0
    }

if "last_eval" not in st.session_state:
    st.session_state.last_eval = None

# Top App Header
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🌪️ Disaster Model Auditor & Explainable AI</div>
        <div class="app-subtitle">Interactive Ground Truth Evaluation & SHAP Attribution for Multi-Class Disaster Classification</div>
    </div>
    <div>
        <span class="badge-accuracy">Test Accuracy: 91.70% (F1: 0.8518)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### :material/tune: Audit Controls")
    
    with st.container(border=True):
        st.markdown("**Session Scoreboard**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Tested", st.session_state.stats["tested"])
            st.metric("Correct", st.session_state.stats["correct"])
        with col_s2:
            acc_val = (
                (st.session_state.stats["correct"] / st.session_state.stats["tested"] * 100)
                if st.session_state.stats["tested"] > 0
                else 0.0
            )
            st.metric("Session Acc.", f"{acc_val:.1f}%")
            st.metric("Incorrect", st.session_state.stats["incorrect"])
            
        if st.button("Reset Session Stats", icon=":material/restart_alt:", width="stretch"):
            st.session_state.stats = {"tested": 0, "correct": 0, "incorrect": 0}
            st.rerun()

    st.markdown("### :material/play_circle: Quick Test Bench")
    
    if st.button("Random Ground Truth Case", icon=":material/casino:", type="primary", width="stretch"):
        case = benchmark.get_random_case()
        eval_res = agent.run_ground_truth_challenge(case)
        st.session_state.last_eval = eval_res
        st.session_state.stats["tested"] += 1
        if eval_res["is_correct"]:
            st.session_state.stats["correct"] += 1
        else:
            st.session_state.stats["incorrect"] += 1
            
        st.session_state.messages.append({
            "role": "user",
            "content": f"Challenge model with random case `{case['case_id']}` ({case['ground_truth_class']})"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Evaluated case `{case['case_id']}` against Ground Truth. Inspection results updated in the right panel."
        })
        st.rerun()

    if st.button("Challenge Hard Case (Error Analysis)", icon=":material/warning:", width="stretch"):
        case = benchmark.get_random_case(only_incorrect=True)
        eval_res = agent.run_ground_truth_challenge(case)
        st.session_state.last_eval = eval_res
        st.session_state.stats["tested"] += 1
        st.session_state.stats["incorrect"] += 1
        
        st.session_state.messages.append({
            "role": "user",
            "content": f"Examine known misclassified case `{case['case_id']}`"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Loaded difficult case `{case['case_id']}` (Ground Truth: {case['ground_truth_class']}, Model: {eval_res['prediction']['predicted_class']}). SHAP attributions are displayed for error diagnosis."
        })
        st.rerun()

    selected_category = st.selectbox(
        "Target disaster category:",
        ["All Categories"] + predictor.class_names
    )
    if st.button("Evaluate Selected Category", icon=":material/filter_alt:", width="stretch"):
        target = None if selected_category == "All Categories" else selected_category
        case = benchmark.get_random_case(disaster_type=target)
        if case:
            eval_res = agent.run_ground_truth_challenge(case)
            st.session_state.last_eval = eval_res
            st.session_state.stats["tested"] += 1
            if eval_res["is_correct"]:
                st.session_state.stats["correct"] += 1
            else:
                st.session_state.stats["incorrect"] += 1
                
            st.session_state.messages.append({
                "role": "user",
                "content": f"Evaluate test case for category: **{selected_category}** (`{case['case_id']}`)"
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Case `{case['case_id']}` evaluated. Ground Truth: **{case['ground_truth_class']}**."
            })
            st.rerun()

    with st.expander(":material/key: LLM & Claude API Settings", expanded=False):
        api_input = st.text_input(
            "Anthropic API Key:",
            type="password",
            placeholder="sk-ant-api03-...",
            help="Provide your Anthropic API Key to activate live Claude 3.5 Sonnet Tool Calling. If empty, the fast Local Auditor Engine runs seamlessly."
        )
        if api_input:
            agent.set_api_key(api_input)
            st.success("Claude 3.5 Sonnet Connected", icon=":material/check_circle:")
        else:
            st.info("Local Auditor Engine Active (Offline Mode)", icon=":material/info:")

    st.divider()
    summary = benchmark.get_summary_stats()
    st.caption(f":material/database: Test Bank: {summary['total_cases']:,} events | 8 Classes")

# Main Layout
col_chat, col_inspect = st.columns([1.1, 0.9], gap="large")

with col_chat:
    st.markdown("##### :material/chat: Auditor Dialogue & Verification")
    
    chat_container = st.container(height=540, border=True)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    user_input = st.chat_input("Ask a disaster question or request a model test (e.g., 'Test an earthquake case')...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Auditor processing query and computing SHAP..."):
            history_data = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ["user", "assistant"]
            ]
            response_obj = agent.chat(user_input, history=history_data)
            
            if response_obj.get("eval_data"):
                st.session_state.last_eval = response_obj["eval_data"]
                st.session_state.stats["tested"] += 1
                if response_obj["eval_data"]["is_correct"]:
                    st.session_state.stats["correct"] += 1
                else:
                    st.session_state.stats["incorrect"] += 1

            st.session_state.messages.append({"role": "assistant", "content": response_obj["response"]})
            st.rerun()

with col_inspect:
    st.markdown("##### :material/analytics: Explainable AI & Audit Inspector")
    
    with st.container(border=True):
        if st.session_state.last_eval is None:
            st.info("Trigger a test from the sidebar or via chat to view live Ground Truth verification and SHAP attributions here.", icon=":material/lightbulb:")
        else:
            eval_data = st.session_state.last_eval
            case = eval_data["case"]
            pred = eval_data["prediction"]
            shap_info = eval_data["shap"]
            is_correct = eval_data["is_correct"]
            
            # Comparison Header Card
            status_html = (
                '<span class="tag-correct">✓ CORRECT PREDICTION</span>'
                if is_correct
                else '<span class="tag-incorrect">✗ MISCLASSIFIED (HARD CASE)</span>'
            )
            
            st.markdown(f"""
            <div class="comparison-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; font-size:1.05rem;">Case Reference: <code>{case['case_id']}</code></span>
                    {status_html}
                </div>
                <div style="margin-top:10px; font-size:0.92rem; line-height:1.6;">
                    <b>Ground Truth (Actual):</b> <span style="color:#2563EB; font-weight:700;">{case['ground_truth_class']}</span><br>
                    <b>Model Prediction:</b> <span style="font-weight:700;">{pred['predicted_class']}</span> (Confidence: {pred['confidence']*100:.2f}%)<br>
                    <span style="color:#64748B; font-size:0.85rem;">📍 {case['summary']['country']} ({case['summary']['region']}) | Month {case['summary']['start_month']} | Scale: {case['summary']['magnitude_scale']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Distribution Chart
            st.markdown("**Class Probability Distribution**")
            prob_df = pd.DataFrame(
                list(pred["probabilities"].items()),
                columns=["Category", "Probability"]
            )
            st.bar_chart(
                prob_df.set_index("Category"),
                horizontal=True,
                color="#2563EB",
                height=240
            )
            
            # SHAP Feature Attributions
            st.markdown("**Top Driving Features (SHAP Values)**")
            shap_display = pd.DataFrame([
                {
                    "Feature": item["feature"],
                    "SHAP Value": f"{item['shap_value']:+.4f}",
                    "Influence": "Supports Decision (+)" if item["shap_value"] > 0 else "Opposes Decision (-)"
                }
                for item in shap_info["top_all_features"]
            ])
            st.dataframe(shap_display, hide_index=True, width="stretch")
            
            # Explainable AI Insights
            st.markdown("**Domain-Level Interpretation**")
            for explanation in shap_info["human_explanations"][:3]:
                st.markdown(f"- {explanation}")

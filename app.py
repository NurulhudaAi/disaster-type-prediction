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

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    .badge-correct {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-wrong {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 สวัสดีครับ! ผมคือ **Disaster Model Auditor Agent**\n\nระบบนี้ถูกออกแบบมาเพื่อ **ทดสอบและท้าทายความแม่นยำของโมเดล Machine Learning (Random Forest)** กับข้อมูลเหตุการณ์จริงในประวัติศาสตร์ (Ground Truth) พร้อมแจกแจงเหตุผลการตัดสินใจด้วย **SHAP (Explainable AI)**\n\n👉 สามารถกดปุ่มลัดด้านซ้าย หรือพิมพ์สั่ง เช่น *'สุ่มเคสมาทดสอบ 1 เคส'*, *'ขอลองเคสแผ่นดินไหว'*, หรือ *'ทดสอบเคสยากที่โมเดลทายผิด'* ได้เลยครับ"
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

# Sidebar
with st.sidebar:
    st.title("⚙️ แผงควบคุมการทดสอบ")
    
    # 1. API Key Setup
    st.markdown("### 🔑 Claude API Connection")
    env_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    api_key_input = st.text_input(
        "Anthropic API Key (ของอาจารย์):",
        value=env_api_key,
        type="password",
        help="หากยังไม่มี API Key ระบบจะใช้ Local Auditor Engine ประมวลผลและคำนวณ SHAP ได้อย่างสมบูรณ์แบบโดยไม่ต้องต่อเน็ต"
    )
    if api_key_input:
        agent.set_api_key(api_key_input)
        st.success("🟢 เชื่อมต่อ Claude API เรียบร้อย", icon="✅")
    else:
        st.info("⚪ ใช้งานในโหมด Local Auditor Engine (พร้อมใช้งานโดยไม่ต้องใช้เน็ต)", icon="ℹ️")

    st.divider()

    # 2. Scoreboard
    st.markdown("### 🏆 Session Audit Scoreboard")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("ทดสอบแล้ว", f"{st.session_state.stats['tested']} เคส")
        st.metric("ทายถูก 🎯", f"{st.session_state.stats['correct']} เคส")
    with c2:
        acc_pct = (
            (st.session_state.stats["correct"] / st.session_state.stats["tested"] * 100)
            if st.session_state.stats["tested"] > 0
            else 0.0
        )
        st.metric("ความแม่นยำเซสชัน", f"{acc_pct:.1f}%")
        st.metric("ทายผิด ❌", f"{st.session_state.stats['incorrect']} เคส")

    if st.button("🔄 รีเซ็ตสถิติเซสชัน", use_container_width=True):
        st.session_state.stats = {"tested": 0, "correct": 0, "incorrect": 0}
        st.rerun()

    st.divider()

    # 3. Quick Actions
    st.markdown("### 🎯 ปุ่มลัดทดสอบสำหรับกรรมการ")
    
    if st.button("🎲 สุ่มเคสทดสอบ (Random Case)", use_container_width=True, type="primary"):
        case = benchmark.get_random_case()
        eval_res = agent.run_ground_truth_challenge(case)
        st.session_state.last_eval = eval_res
        st.session_state.stats["tested"] += 1
        if eval_res["is_correct"]:
            st.session_state.stats["correct"] += 1
        else:
            st.session_state.stats["incorrect"] += 1
            
        # Add to chat
        st.session_state.messages.append({
            "role": "user",
            "content": f"สุ่มทดสอบเคส `{case['case_id']}` ({case['ground_truth_class']})"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"ผลการทดสอบเคส `{case['case_id']}` จาก Ground Truth Bank เรียบร้อยแล้ว (ดูการวิเคราะห์ด้านล่าง)"
        })
        st.rerun()

    if st.button("⚠️ ทดสอบเคสยาก (Hard Case ที่โมเดลท้าทาย)", use_container_width=True):
        case = benchmark.get_random_case(only_incorrect=True)
        eval_res = agent.run_ground_truth_challenge(case)
        st.session_state.last_eval = eval_res
        st.session_state.stats["tested"] += 1
        st.session_state.stats["incorrect"] += 1
        
        st.session_state.messages.append({
            "role": "user",
            "content": f"ขอดูเคสยาก/จุดอ่อนของโมเดล `{case['case_id']}`"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"ดึงเคสยาก `{case['case_id']}` ซึ่งโมเดลเคยสับสนมาทดสอบและวิเคราะห์ SHAP เรียบร้อยครับ"
        })
        st.rerun()

    selected_disaster = st.selectbox(
        "เจาะจงประเภทภัยพิบัติ:",
        ["All"] + predictor.class_names
    )
    if st.button(f"🔎 ทดสอบเคส {selected_disaster}", use_container_width=True):
        target = None if selected_disaster == "All" else selected_disaster
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
                "content": f"ทดสอบเคสประเภท {selected_disaster} (`{case['case_id']}`)"
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"ผลการทดสอบเคส `{case['case_id']}` (Ground Truth: {case['ground_truth_class']})"
            })
            st.rerun()

    st.divider()
    summary = benchmark.get_summary_stats()
    st.caption(f"📁 คลังข้อสอบ: {summary['total_cases']:,} เคส | Benchmark Accuracy: 91.70%")

# Main Area
st.markdown("<div class='main-header'>🌪️ Disaster Model Auditor & Explainable AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>ระบบผู้ช่วยตรวจสอบความแม่นยำของ Random Forest Pipeline ด้วย Ground Truth และ SHAP</div>", unsafe_allow_html=True)

# Layout: Split into Chat Column and Analysis Panel Column
col_chat, col_viz = st.columns([1.1, 0.9])

with col_chat:
    st.subheader("💬 สนทนา & ตรวจสอบข้อสอบ")
    
    # Display message history
    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    # Chat Input
    user_prompt = st.chat_input("พิมพ์คำถาม หรือสั่งให้ทดสอบเคส...")
    if user_prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Run agent
        with st.spinner("Agent กำลังตรวจสอบและคำนวณผล..."):
            history_tuples = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ["user", "assistant"]
            ]
            res = agent.chat(user_prompt, history=history_tuples)
            
            if res.get("eval_data"):
                st.session_state.last_eval = res["eval_data"]
                st.session_state.stats["tested"] += 1
                if res["eval_data"]["is_correct"]:
                    st.session_state.stats["correct"] += 1
                else:
                    st.session_state.stats["incorrect"] += 1

            st.session_state.messages.append({"role": "assistant", "content": res["response"]})
            st.rerun()

with col_viz:
    st.subheader("📊 ผลการตรวจสอบและอธิบายโมเดล (XAI)")
    
    if st.session_state.last_eval is None:
        st.info("👈 กดปุ่ม **'สุ่มเคสทดสอบ'** หรือพิมพ์คุยกับ Agent เพื่อดูผลการตรวจสอบและค่า SHAP ที่นี่", icon="💡")
    else:
        last = st.session_state.last_eval
        case = last["case"]
        pred = last["prediction"]
        shap_data = last["shap"]
        is_correct = last["is_correct"]
        
        # Result Header Card
        status_badge = "<span class='badge-correct'>🎯 ทายถูกต้อง (CORRECT)</span>" if is_correct else "<span class='badge-wrong'>❌ ทายผิดพลาด (INCORRECT)</span>"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.1rem; font-weight:700;">เคส: <code>{case['case_id']}</code></span>
                {status_badge}
            </div>
            <div style="margin-top:8px; font-size:0.95rem;">
                <b>เฉลยจริง (Ground Truth):</b> <span style="color:#1D4ED8; font-weight:600;">{case['ground_truth_class']}</span><br>
                <b>โมเดลทำนาย (Prediction):</b> <span style="font-weight:600;">{pred['predicted_class']}</span> (ความมั่นใจ {pred['confidence']*100:.2f}%)<br>
                <span style="color:#6B7280; font-size:0.85rem;">📍 {case['summary']['country']} ({case['summary']['region']}) | เดือน: {case['summary']['start_month']} | หน่วย: {case['summary']['magnitude_scale']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Prediction Probabilities Bar Chart
        st.markdown("##### 📈 ความน่าจะเป็นของแต่ละคลาส (Class Probabilities)")
        prob_df = pd.DataFrame(
            list(pred["probabilities"].items()),
            columns=["Disaster Type", "Probability"]
        )
        st.bar_chart(
            prob_df.set_index("Disaster Type"),
            horizontal=True,
            color="#3B82F6",
            height=260
        )
        
        # 2. SHAP Feature Attribution
        st.markdown("##### 🔬 อิทธิพลของตัวแปร (SHAP Attributions)")
        shap_items = shap_data["top_all_features"]
        shap_df = pd.DataFrame([
            {
                "Feature": item["feature"],
                "SHAP Impact": item["shap_value"],
                "Impact Direction": "ส่งเสริม (+)" if item["shap_value"] > 0 else "คัดค้าน (-)"
            }
            for item in shap_items
        ])
        
        st.dataframe(
            shap_df,
            hide_index=True,
            use_container_width=True
        )
        
        # 3. Plain English / Thai Explanation
        st.markdown("##### 💡 คำอธิบายเชิงเหตุผล (Explainable Insights)")
        for expl in shap_data["human_explanations"][:3]:
            st.markdown(f"- {expl}")

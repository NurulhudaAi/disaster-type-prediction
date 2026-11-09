import os
import json
import time
from typing import Dict, Any, List, Optional, Generator

SYSTEM_PROMPT = """You are the "Disaster Auditor Agent", an expert AI specializing in natural disaster classification, machine learning evaluation, and Explainable AI (SHAP).

APPLICATION CONTEXT & DOMAIN KNOWLEDGE:
- You audit a 8-class natural disaster type classifier built on the EM-DAT historical benchmark dataset.
- The 8 disaster categories are: Drought, Earthquake, Epidemic, Extreme Temperature, Flood, Landslide, Storm, Wildfire.
- Test bank size: 3,495 historical events evaluated on a 20% stratified holdout set.
- Class imbalance is ~12:1 (Floods and Storms dominate; Droughts and Wildfires are minority classes).
- Primary evaluation metric is Macro-F1 (unweighted mean of class F1-scores), NOT raw accuracy.

THE 3 LEAK-FREE MODELS COMPARED:
1. Random Forest (SMOTETomek Pipeline): Test Macro-F1 = 0.8518, Accuracy = 91.70%, Recall = 86.94% (Best minority class recall on Wildfire & Landslide).
2. LightGBM (SMOTETomek Pipeline): Test Macro-F1 = 0.8489, Accuracy = 91.85%, Recall = 86.32% (Fastest inference, leaf-wise gradient boosting).
3. XGBoost (SMOTETomek Pipeline): Test Macro-F1 = 0.8480, Accuracy = 91.47%, Recall = 86.62% (Regularized depth-wise gradient boosting).
- Disagreement rate: The 3 models disagree on 251 out of 3,495 cases (7.2%), mostly on boundary confusions like Flood vs. Storm during monsoon seasons or Drought vs. Extreme Temperature.

GUIDELINES:
- Answer user questions informatively, clearly, and authoritatively.
- Support both Thai and English naturally: if the user asks in Thai, reply in fluent, polite Thai (ครับ/ค่ะ); if in English, reply in English.
- If an active audit case is provided in the context, refer to its specific metrics, ground truth, predicted class, confidence, and SHAP features.
- If asked about how models work or error diagnostics, explain the physical and seasonal disaster dynamics (e.g. why missing magnitude scales sway models toward floods, or how event duration influences epidemics vs. earthquakes)."""

class DisasterAuditorLLM:
    """Multi-provider LLM interface supporting Google Gemini and Anthropic Claude."""

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        self.provider = provider.lower()
        env_var = "GEMINI_API_KEY" if self.provider == "gemini" else "ANTHROPIC_API_KEY"
        self.api_key = (api_key or "").strip() or os.getenv(env_var)

    def chat_stream(self, user_message: str, history: List[Dict[str, str]] = None, current_case: Optional[Dict] = None) -> Generator[str, None, None]:
        """Stream conversational response token-by-token."""
        prompt = self._build_prompt(user_message, history or [], current_case)

        if self.provider == "gemini" and self.api_key:
            yield from self._stream_gemini(prompt)
        elif self.provider == "anthropic" and self.api_key:
            yield from self._stream_anthropic(user_message, history or [], current_case)
        else:
            yield from self._stream_local_fallback(user_message, current_case)

    @staticmethod
    def _case_context(current_case: Optional[Dict]) -> str:
        context_str = ""
        if current_case:
            case_info = current_case.get("case", {})
            models_info = current_case.get("models", {})
            context_str = f"\n\nCURRENT AUDIT CASE CONTEXT:\n- Case ID: {case_info.get('case_id')}\n- Ground Truth: {case_info.get('ground_truth_class')}\n- Location: {case_info.get('summary', {}).get('country')} ({case_info.get('summary', {}).get('region')})\n- Timing: Month {case_info.get('summary', {}).get('start_month')}\n- Magnitude Scale: {case_info.get('summary', {}).get('magnitude_scale')}\n"
            for m_name, m_data in models_info.items():
                context_str += f"- {m_name} Prediction: {m_data['prediction']['predicted_class']} (Confidence: {m_data['prediction']['confidence']*100:.1f}%, Correct: {m_data['is_correct']})\n"
                context_str += f"  Top SHAP features: {', '.join([f['feature'] for f in m_data['shap'].get('top_all', [])[:3]])}\n"
        return context_str

    def _build_prompt(self, user_message: str, history: List[Dict[str, str]], current_case: Optional[Dict]) -> str:
        context_str = self._case_context(current_case)
        full_prompt = f"{SYSTEM_PROMPT}\n{context_str}\n\nUser Question:\n{user_message}"
        return full_prompt

    def _stream_gemini(self, prompt: str) -> Generator[str, None, None]:
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            
            # Prioritize latest recommended model (gemini-3.8-flash)
            models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
            try:
                # Dynamically discover active models on this key
                discovered = []
                for m in client.models.list():
                    m_id = getattr(m, "name", "") or ""
                    clean_id = m_id.replace("models/", "")
                    if "flash" in clean_id.lower() or "pro" in clean_id.lower():
                        discovered.append(clean_id)
                if discovered:
                    # Put preferred 3.8-flash first if found, else discovered list
                    if "gemini-3.8-flash" in discovered:
                        discovered.remove("gemini-3.8-flash")
                        discovered.insert(0, "gemini-3.8-flash")
                    models_to_try = discovered + [m for m in models_to_try if m not in discovered]
            except Exception:
                pass

            last_err = None
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content_stream(
                        model=model_name,
                        contents=prompt
                    )
                    for chunk in response:
                        if hasattr(chunk, "text") and chunk.text:
                            yield chunk.text
                    return
                except Exception as e:
                    last_err = e
                    continue

            # If all fail, yield error
            yield f"\n⚠️ Gemini API Error: {str(last_err)}. Please verify your Gemini API key in the sidebar."
        except Exception as e:
            yield f"\n⚠️ Gemini Connection Error: {str(e)}"

    def _stream_anthropic(self, user_message: str, history: List[Dict[str, str]], current_case: Optional[Dict]) -> Generator[str, None, None]:
        try:
            import anthropic
        except ImportError:
            yield "\n⚠️ The `anthropic` package is not installed. Run `pip install -r requirements.txt`."
            return

        client = anthropic.Anthropic(api_key=self.api_key)

        # Recent turns only; the API requires the first message to be from the user.
        messages = [
            {"role": h["role"], "content": h["content"]}
            for h in history[-6:]
            if h.get("content") and h.get("role") in ("user", "assistant")
        ]
        while messages and messages[0]["role"] != "user":
            messages.pop(0)

        context_str = self._case_context(current_case)
        user_content = f"{context_str}\nUser Question:\n{user_message}" if context_str else user_message
        messages.append({"role": "user", "content": user_content})

        try:
            with client.beta.messages.stream(
                model="claude-opus-5-5",
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=messages,
                # Conversational Q&A: low effort keeps latency and cost down.
                output_config={"effort": "low"},
                # If a request is declined by a safety classifier, retry on a fallback model server-side.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "\n⚠️ Claude declined to answer this request."
            elif final.stop_reason == "max_tokens":
                yield "\n⚠️ Response truncated (max_tokens reached)."
        except anthropic.AuthenticationError:
            yield "\n⚠️ Claude API Error: invalid Anthropic API key. Please check the key in the sidebar."
        except anthropic.RateLimitError:
            yield "\n⚠️ Claude API Error: rate limited. Please wait a moment and try again."
        except anthropic.APIStatusError as e:
            yield f"\n⚠️ Claude API Error ({e.status_code}): {e.message}"
        except anthropic.APIConnectionError:
            yield "\n⚠️ Claude Connection Error: could not reach the Anthropic API."

    def _stream_local_fallback(self, user_message: str, current_case: Optional[Dict]) -> Generator[str, None, None]:
        msg = user_message.lower()
        if "thai" in msg or any("\u0e00" <= c <= "\u0e7f" for c in user_message):
            # Thai fallback
            resp = "สวัสดีครับ! ผมคือ **Disaster Auditor AI**\n\n"
            if current_case:
                c = current_case["case"]
                resp += f"สำหรับเคสปัจจุบัน **`{c['case_id']}`** (เหตุการณ์จริง: **{c['ground_truth_class']}** ที่ประเทศ {c['summary']['country']}):\n"
                for m_name, m_data in current_case["models"].items():
                    c_status = "ถูกต้อง ✓" if m_data["is_correct"] else "ทำนายผิด ✗"
                    resp += f"• **{m_name}**: ทำนายเป็น `{m_data['prediction']['predicted_class']}` ({m_data['prediction']['confidence']*100:.1f}%) — {c_status}\n"
                resp += "\n💡 *คำแนะนำ: คุณสามารถใส่ **Gemini API Key** ในแท็บ Settings ที่แถบเมนูด้านซ้าย เพื่อเปิดการสนทนาและวิเคราะห์เชิงลึกด้วย LLM ได้แบบเรียลไทม์ครับ*"
            else:
                resp += "ระบบกำลังทำงานในโหมด **Local Benchmark Engine** (โมเดล Random Forest, LightGBM, และ XGBoost พร้อมใช้งาน)\n\n"
                resp += "คุณสามารถ:\n"
                resp += "1. คลิกปุ่ม **Quick Actions** ด้านบน (เช่น 🎲 Random Case หรือ ⚡ Disagreement Case)\n"
                resp += "2. พิมพ์คำสั่ง เช่น `/random`, `/hard`, `/compare`\n"
                resp += "3. ใส่ **Gemini API Key** ที่ Sidebar ซ้ายมือเพื่อเปิดใช้งานการถาม-ตอบอัจฉริยะด้วย Gemini ครับ"
        else:
            resp = "Hello! I am the **Disaster Auditor AI**.\n\n"
            if current_case:
                c = current_case["case"]
                resp += f"Regarding the active case **`{c['case_id']}`** (Ground Truth: **{c['ground_truth_class']}**, {c['summary']['country']}):\n"
                for m_name, m_data in current_case["models"].items():
                    c_status = "CORRECT ✓" if m_data["is_correct"] else "MISCLASSIFIED ✗"
                    resp += f"• **{m_name}**: Predicted `{m_data['prediction']['predicted_class']}` ({m_data['prediction']['confidence']*100:.1f}%) — {c_status}\n"
                resp += "\n💡 *Tip: Provide your **Gemini API Key** in the sidebar settings to unlock live generative dialogue and deep XAI explanations.*"
            else:
                resp += "The application is currently running with the **Local Zero-Leakage ML Engines** (Random Forest, LightGBM, XGBoost).\n\n"
                resp += "You can test cases using the quick buttons above or slash commands like `/random` and `/compare`.\n"
                resp += "To enable live natural language reasoning, enter your **Gemini API Key** in the sidebar."

        # Simulate streaming chunks
        for word in resp.split(" "):
            yield word + " "
            time.sleep(0.015)

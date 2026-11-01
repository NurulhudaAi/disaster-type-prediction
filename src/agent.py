import json
import os
from typing import Dict, Any, List, Optional
import anthropic

CLAUDE_TOOLS = [
    {
        "name": "fetch_ground_truth_case",
        "description": "ดึงเคสจริงจากชุดข้อมูล Test Set (Ground Truth Bank) เพื่อนำมาทดสอบท้าทายโมเดล มีเฉลยจริงกำกับอยู่เสมอ",
        "input_schema": {
            "type": "object",
            "properties": {
                "disaster_type": {
                    "type": "string",
                    "description": "กรองตามประเภทภัยพิบัติ เช่น Flood, Storm, Earthquake, Wildfire (หรือ 'all')"
                },
                "only_hard_case": {
                    "type": "boolean",
                    "description": "ถ้าเป็น True จะดึงเฉพาะเคสที่โมเดลเคยทายผิด (Hard Case) เพื่อนำมาวิเคราะห์จุดอ่อน"
                },
                "region": {
                    "type": "string",
                    "description": "ภูมิภาคที่ต้องการทดสอบ เช่น Asia, Americas, Europe, Africa, Oceania"
                }
            }
        }
    },
    {
        "name": "run_model_prediction",
        "description": "ส่ง Feature เข้าโมเดล Random Forest เพื่อจำแนกประเภทภัยพิบัติ และคำนวณค่าความน่าจะเป็น (Probabilities)",
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Dictionary ของ Features ของเหตุการณ์"
                }
            },
            "required": ["features"]
        }
    },
    {
        "name": "explain_decision_with_shap",
        "description": "คำนวณค่า SHAP Values เพื่อวิเคราะห์ว่า Features ใดเป็นปัจจัยสำคัญที่ผลักดันให้โมเดลตอบคลาสนั้น",
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Features ของตัวอย่าง"
                },
                "predicted_class_id": {
                    "type": "integer",
                    "description": "ID ของคลาสที่โมเดลทำนาย (0-7)"
                }
            },
            "required": ["features"]
        }
    }
]

SYSTEM_PROMPT = """คุณคือ "Disaster Model Auditor Agent" ผู้เชี่ยวชาญด้านการตรวจสอบและประเมินความแม่นยำของโมเดล Machine Learning (Random Forest) สำหรับการจำแนกประเภทภัยพิบัติ

หน้าที่หลักของคุณ:
1. ทำหน้าที่เป็นผู้คุมสอบ (Model Auditor): นำเคสประวัติศาสตร์จริง (Ground Truth) มาทดสอบโมเดล
2. ตรวจสอบว่าโมเดลทำนาย "ถูกต้องจริง หรือ ผิด" โดยเทียบกับ Ground Truth เสมอ
3. อธิบายเหตุผลเบื้องหลังการตัดสินใจของโมเดลอย่างโปร่งใสผ่านค่า SHAP (Explainable AI):
   - ถ้าโมเดลทายถูก: ชี้ให้เห็นว่า Features ไหนที่มีน้ำหนักบวกสูงสุดที่ช่วยให้ทายถูก
   - ถ้าโมเดลทายผิด: วิเคราะห์ว่าทำไมโมเดลถึงสับสน ปัจจัยไหนหลอกโมเดล หรือเกิดจากความทับซ้อนทางกายภาพของภัยพิบัติ เช่น น้ำท่วมกับพายุมรสุม
4. ใช้ภาษาไทยที่เข้าใจง่าย มีความเป็นมืออาชีพ เหมาะสำหรับใช้ในการนำเสนอโครงงาน Senior Project

เมื่อผู้ใช้สั่งให้ทดสอบ ให้เรียกใช้ Tool `fetch_ground_truth_case` หรือส่ง Feature เข้า `run_model_prediction` และ `explain_decision_with_shap` เสมอ ห้ามแต่งคำตอบหรือค่าตัวเลขขึ้นมาเองเด็ดขาด"""

class DisasterAuditorAgent:
    def __init__(self, predictor, explainer, benchmark, api_key: Optional[str] = None):
        self.predictor = predictor
        self.explainer = explainer
        self.benchmark = benchmark
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        
    def set_api_key(self, api_key: str):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python backend tools."""
        if name == "fetch_ground_truth_case":
            disaster_type = args.get("disaster_type")
            only_hard = args.get("only_hard_case", False)
            region = args.get("region")
            case = self.benchmark.get_random_case(
                disaster_type=disaster_type,
                only_incorrect=only_hard,
                region=region
            )
            if not case:
                return {"error": "ไม่พบเคสทดสอบที่ตรงกับเงื่อนไขที่ระบุ"}
            return case
            
        elif name == "run_model_prediction":
            features = args.get("features", {})
            return self.predictor.predict_sample(features)
            
        elif name == "explain_decision_with_shap":
            features = args.get("features", {})
            pred_id = args.get("predicted_class_id")
            df = self.predictor.predict_sample(features)["dataframe"]
            return self.explainer.explain_sample(df, target_class_idx=pred_id)
            
        return {"error": f"Unknown tool: {name}"}

    def run_ground_truth_challenge(self, case: Dict) -> Dict[str, Any]:
        """
        Direct evaluation pipeline for a Ground Truth case:
        Predicts, compares with ground truth, and computes SHAP explanations.
        """
        pred_res = self.predictor.predict_sample(case["features"])
        shap_res = self.explainer.explain_sample(
            pred_res["dataframe"],
            target_class_idx=pred_res["predicted_id"]
        )
        
        is_correct = (pred_res["predicted_class"] == case["ground_truth_class"])
        
        return {
            "case": case,
            "prediction": pred_res,
            "shap": shap_res,
            "is_correct": is_correct
        }

    def chat(self, user_message: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process user query via Claude API if available, 
        or via local Rule-based Auditor Engine if API Key is not set yet.
        """
        if self.client:
            return self._chat_with_claude(user_message, history or [])
        else:
            return self._chat_local_fallback(user_message)

    def _chat_with_claude(self, user_message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        messages = list(history)
        messages.append({"role": "user", "content": user_message})
        
        # 1. Send to Claude with Tools
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=CLAUDE_TOOLS,
            messages=messages
        )
        
        tool_results = []
        last_eval_data = None
        
        # 2. Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Add Claude's response (with tool_use blocks) to messages
            messages.append({"role": "assistant", "content": response.content})
            
            tool_outputs_for_claude = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    tool_res = self.execute_tool(tool_name, tool_args)
                    tool_results.append({"tool": tool_name, "args": tool_args, "result": tool_res})
                    
                    if tool_name == "fetch_ground_truth_case" and "case_id" in tool_res:
                        # Auto-evaluate with SHAP to pass complete context back
                        last_eval_data = self.run_ground_truth_challenge(tool_res)
                    
                    tool_outputs_for_claude.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_res, ensure_ascii=False)
                    })
                    
            # 3. Follow up with Claude to get final response
            second_response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=CLAUDE_TOOLS,
                messages=messages + [{"role": "user", "content": tool_outputs_for_claude}]
            )
            
            final_text = "".join([b.text for b in second_response.content if hasattr(b, "text")])
            return {
                "response": final_text,
                "tool_results": tool_results,
                "eval_data": last_eval_data,
                "mode": "claude_live"
            }
        else:
            final_text = "".join([b.text for b in response.content if hasattr(b, "text")])
            return {
                "response": final_text,
                "tool_results": [],
                "eval_data": None,
                "mode": "claude_live"
            }

    def _chat_local_fallback(self, user_message: str) -> Dict[str, Any]:
        """Local auditor when API key is pending."""
        msg = user_message.lower()
        
        # Check if user asked for a specific disaster type
        target_type = None
        for c in self.predictor.class_names:
            if c.lower() in msg:
                target_type = c
                break
                
        only_hard = ("ผิด" in msg or "ยาก" in msg or "hard" in msg or "จุดอ่อน" in msg)
        
        # Pull case
        case = self.benchmark.get_random_case(
            disaster_type=target_type,
            only_incorrect=only_hard
        )
        
        if not case:
            case = self.benchmark.get_random_case()
            
        eval_data = self.run_ground_truth_challenge(case)
        
        # Build local structured response
        status_icon = "🎯 **ถูกต้อง (CORRECT)**" if eval_data["is_correct"] else "❌ **ทายผิด (INCORRECT)**"
        gt = case["ground_truth_class"]
        pred = eval_data["prediction"]["predicted_class"]
        conf = eval_data["prediction"]["confidence"] * 100
        
        shap_points = "\n".join([f"- {p}" for p in eval_data["shap"]["human_explanations"][:4]])
        
        text = f"""### {status_icon} — ผลการตรวจสอบโมเดล (Ground Truth Audit)

* **รหัสเคสทดสอบ:** `{case['case_id']}`
* **สถานที่/ภูมิภาค:** {case['summary']['country']} ({case['summary']['region']})
* **เดือนที่เกิดเหตุ:** เดือนที่ {case['summary']['start_month']} | **หน่วยวัดขนาด:** {case['summary']['magnitude_scale']}
* **เฉลยจริง (Ground Truth):** **{gt}**
* **โมเดลทำนาย (Model Prediction):** **{pred}** (ความมั่นใจ **{conf:.2f}%**)

---
#### 🔬 การวิเคราะห์อิทธิพลตัวแปรด้วย SHAP (Explainable AI):
{shap_points}

*(หมายเหตุ: รันใน Local Model Auditor Mode เนื่องจากยังไม่ได้ระบุ Claude API Key หากใส่อาจารย์/Claude Key ระบบจะเพิ่มการวิเคราะห์เชิงลึกอัตโนมัติ)*
"""
        return {
            "response": text,
            "tool_results": [{"tool": "local_ground_truth_audit", "result": case}],
            "eval_data": eval_data,
            "mode": "local_auditor"
        }

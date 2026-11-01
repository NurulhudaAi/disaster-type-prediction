import json
import os
from typing import Dict, Any, List, Optional
import anthropic

CLAUDE_TOOLS = [
    {
        "name": "fetch_ground_truth_case",
        "description": "Fetch a real historical disaster case from the Ground Truth test bank (EM-DAT database) to evaluate and challenge the machine learning model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "disaster_type": {
                    "type": "string",
                    "description": "Filter by disaster category: Drought, Earthquake, Epidemic, Extreme Temperature, Flood, Landslide, Storm, Wildfire, or 'all'."
                },
                "only_hard_case": {
                    "type": "boolean",
                    "description": "If true, fetch only misclassified/hard cases to perform detailed error analysis and discover model limitations."
                },
                "region": {
                    "type": "string",
                    "description": "Geographic region to filter: Asia, Americas, Europe, Africa, Oceania."
                }
            }
        }
    },
    {
        "name": "run_model_prediction",
        "description": "Feed extracted event features into the pre-trained Random Forest Pipeline to classify the disaster type and compute confidence probabilities across all 8 classes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Dictionary of feature values representing the disaster event (start_month, duration, region, magnitude_scale, etc.)."
                }
            },
            "required": ["features"]
        }
    },
    {
        "name": "explain_decision_with_shap",
        "description": "Compute SHAP (Shapley Additive exPlanations) values via TreeExplainer to quantify the exact feature attributions driving the model's decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Dictionary of event features."
                },
                "predicted_class_id": {
                    "type": "integer",
                    "description": "Class index (0-7) predicted by the model."
                }
            },
            "required": ["features"]
        }
    }
]

SYSTEM_PROMPT = """You are the "Disaster Model Auditor Agent", an AI expert specializing strictly in disaster classification, machine learning evaluation, and Explainable AI (XAI).

STRICT SCOPE & DOMAIN GUARDRAILS:
1. You ONLY discuss topics related to natural disasters, EM-DAT historical disaster records, multi-class disaster classification, and the evaluation of the pre-trained Random Forest model.
2. If the user asks about unrelated topics (such as general programming, casual chitchat, cooking, entertainment, sports, or general advice), politely decline and redirect them back to disaster analysis and model auditing.
3. You must NEVER fabricate numbers, predictions, or accuracy metrics. Always invoke the provided tools (`fetch_ground_truth_case`, `run_model_prediction`, `explain_decision_with_shap`) to perform tests.

AUDITOR RESPONSIBILITIES:
- Always compare the Model's Prediction against the real Ground Truth.
- Clearly state whether the model was CORRECT or INCORRECT.
- Interpret the SHAP feature attributions in clear domain terms (e.g., explaining why magnitude scale, season, duration, or geography swayed the model).
- For misclassified cases (Hard Cases), provide constructive error analysis: explain the physical overlap between disaster categories (e.g., why a monsoon-season flood might share characteristics with a storm).
- Maintain an academic, professional, and rigorous tone suitable for a Senior Project defense.
- Communicate in clear, concise English."""

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
                return {"error": "No historical test case found matching the specified criteria."}
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
            messages.append({"role": "assistant", "content": response.content})
            
            tool_outputs_for_claude = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    tool_res = self.execute_tool(tool_name, tool_args)
                    tool_results.append({"tool": tool_name, "args": tool_args, "result": tool_res})
                    
                    if tool_name == "fetch_ground_truth_case" and "case_id" in tool_res:
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
        
        # Guardrail check for obvious out-of-domain queries
        out_of_domain_keywords = ["recipe", "cook", "bitcoin", "crypto", "dating", "movie", "song", "joke", "game"]
        if any(w in msg for w in out_of_domain_keywords):
            return {
                "response": "I am a specialized Disaster Model Auditor. I can only assist with natural disaster analysis, event classification, and evaluating the pre-trained Machine Learning model. Please ask a disaster-related question or request a model test.",
                "tool_results": [],
                "eval_data": None,
                "mode": "local_guardrail"
            }

        # Check if user asked for a specific disaster type
        target_type = None
        for c in self.predictor.class_names:
            if c.lower() in msg:
                target_type = c
                break
                
        only_hard = any(w in msg for w in ["hard", "wrong", "error", "misclassified", "weakness", "fail", "challenging"])
        
        # Pull case
        case = self.benchmark.get_random_case(
            disaster_type=target_type,
            only_incorrect=only_hard
        )
        
        if not case:
            case = self.benchmark.get_random_case()
            
        eval_data = self.run_ground_truth_challenge(case)
        
        # Build structured English response
        status_badge = "🎯 **CORRECT PREDICTION**" if eval_data["is_correct"] else "❌ **MISCLASSIFIED (HARD CASE)**"
        gt = case["ground_truth_class"]
        pred = eval_data["prediction"]["predicted_class"]
        conf = eval_data["prediction"]["confidence"] * 100
        
        shap_points = "\n".join([f"- {p}" for p in eval_data["shap"]["human_explanations"][:4]])
        
        text = f"""### {status_badge} — Ground Truth Audit Result

* **Case ID:** `{case['case_id']}`
* **Location:** {case['summary']['country']} ({case['summary']['region']})
* **Timing & Scale:** Month {case['summary']['start_month']} | Magnitude Scale: {case['summary']['magnitude_scale']}
* **Ground Truth (Actual):** **{gt}**
* **Model Prediction:** **{pred}** (Confidence: **{conf:.2f}%**)

---
#### 🔬 Explainable AI (SHAP Attributions):
{shap_points}

*(Running in Local Model Auditor mode. Connect your Claude API Key in the sidebar for full conversational reasoning and live agent orchestration).*
"""
        return {
            "response": text,
            "tool_results": [{"tool": "local_ground_truth_audit", "result": case}],
            "eval_data": eval_data,
            "mode": "local_auditor"
        }

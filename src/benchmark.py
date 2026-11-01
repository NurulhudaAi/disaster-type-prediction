import json
import random
from typing import Dict, List, Optional

class BenchmarkManager:
    """Manages the Ground Truth Test Bank for interactive evaluation."""
    
    def __init__(self, bank_path="models/ground_truth_bank.json"):
        with open(bank_path, "r", encoding="utf-8") as f:
            self.cases: List[Dict] = json.load(f)
            
        self.case_map = {c["case_id"]: c for c in self.cases}
        
    def get_summary_stats(self) -> Dict:
        total = len(self.cases)
        correct = sum(1 for c in self.cases if c["is_correct"])
        incorrect = total - correct
        
        class_dist = {}
        hard_by_class = {}
        for c in self.cases:
            gt = c["ground_truth_class"]
            class_dist[gt] = class_dist.get(gt, 0) + 1
            if not c["is_correct"]:
                hard_by_class[gt] = hard_by_class.get(gt, 0) + 1
                
        return {
            "total_cases": total,
            "correct_cases": correct,
            "incorrect_cases": incorrect,
            "test_accuracy": round(correct / total, 4) if total > 0 else 0.0,
            "class_distribution": class_dist,
            "hard_cases_by_class": hard_by_class
        }
        
    def get_random_case(
        self,
        disaster_type: Optional[str] = None,
        only_incorrect: bool = False,
        region: Optional[str] = None
    ) -> Optional[Dict]:
        filtered = self.cases
        
        if disaster_type and disaster_type.lower() != "all":
            filtered = [c for c in filtered if c["ground_truth_class"].lower() == disaster_type.lower()]
            
        if only_incorrect:
            filtered = [c for c in filtered if not c["is_correct"]]
            
        if region and region.lower() != "all":
            filtered = [c for c in filtered if c["summary"]["region"].lower() == region.lower()]
            
        if not filtered:
            return None
            
        return random.choice(filtered)
        
    def get_case_by_id(self, case_id: str) -> Optional[Dict]:
        return self.case_map.get(case_id)

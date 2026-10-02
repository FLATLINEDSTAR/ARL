import enum
from dataclasses import dataclass
from typing import List, Dict, Any

class CheckStatus(str, enum.Enum):
    """Status of a diagnostic check."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

@dataclass
class CheckResult:
    """Standardized data structure for a diagnostic check result."""
    category: str
    status: CheckStatus
    message: str
    remediation: str = ""

class SystemDoctor:
    """Central diagnostic engine for adaptive-rl doctor command."""
    
    def __init__(self):
        self.results: List[CheckResult] = []
        
    def run_all_checks(self) -> List[CheckResult]:
        """Runs all registered diagnostic checks.
        
        To be implemented in subsequent phases.
        """
        self.results.clear()
        # TODO: Implement checks (System, Hardware, Dependencies, OS, Git, etc.)
        return self.results
        
    def add_result(self, result: CheckResult) -> None:
        """Adds a diagnostic result to the internal list."""
        self.results.append(result)
        
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the diagnostic results to a dictionary structure."""
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == CheckStatus.PASS),
            "warnings": sum(1 for r in self.results if r.status == CheckStatus.WARN),
            "failed": sum(1 for r in self.results if r.status == CheckStatus.FAIL),
            "results": [
                {
                    "category": r.category,
                    "status": r.status.value,
                    "message": r.message,
                    "remediation": r.remediation
                }
                for r in self.results
            ]
        }

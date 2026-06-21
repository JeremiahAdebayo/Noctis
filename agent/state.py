from typing import TypedDict, Optional, Dict, List, Any, Annotated, operator
from agent.schemas import EngineerTask
class AgentState(TypedDict):
    # Issue metadata
    issue_id: str
    issue_title: str
    issue_body: str
    
    # Workspace Mapping
    repo_path: str
    test_file: Optional[str]
    file_registry: Dict[str, Any]
    pending_edits: List[Any]
    requirement_path: str
    
    # Planner outputs 
    engineer_tasks: List[EngineerTask]          # ← replaces target_file/target_functions/plan
    target_functions: Optional[str]
    dependency_manifest: Optional[Dict]
    
    # Execution State
    test_command: str
    iteration_count: int
    max_iterations: int
    is_resolved: bool
    
    # Feedback & Cache
    current_code: Optional[str]
    patches: Annotated[List[dict], operator.add]  # ← replaces current_code; reducer for fan-in
    test_output: Optional[str]
    failed_tasks: Optional[List[EngineerTask]]   # ← populated by critic on partial failure

    critic_feedback: Optional[str]
    plan: Optional[str]
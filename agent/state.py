from typing import TypedDict, Optional, Dict, List, Any

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
    target_file: List[Any]
    target_functions: Optional[str]
    dependency_manifest: Optional[Dict]
    
    # Execution State
    test_command: str
    iteration_count: int
    max_iterations: int
    is_resolved: bool
    
    # Feedback & Cache
    current_code: Optional[str]
    test_output: Optional[str]
    critic_feedback: Optional[str]
    plan: Optional[str]
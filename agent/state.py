from typing import TypedDict, Optional, Dict, List, Any

class Chunk(TypedDict):
    name: str           # e.g., "process_data"
    start_line: int     # AST line start
    end_line: int       # AST line end
    content: str        # The actual code block
    type: str           # "function", "class", "import", "constant"

class Patch(TypedDict):
    file_path: str
    target_name: str    # Function or class name to target
    new_content: str    # The new implementation
    rationale: str      # Why this change is needed (for the Reassembler log)

class FileRegistryEntry(TypedDict):
    chunks: List[Chunk]
    imports: List[str]
    abs_path: str

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
    
    # Planner outputs — THESE WERE MISSING
    target_file: Optional[str]
    target_function: Optional[str]
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
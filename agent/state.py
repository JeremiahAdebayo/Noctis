import operator
from typing import TypedDict, Optional, Dict, List, Any, Annotated
from agent.schemas import EngineerTask


def append_or_reset(existing: list, new: list | None) -> list:
    if new is None:
        return []
    return existing + new

class AgentState(TypedDict):
    # Issue metadata
    issue_id: str
    issue_title: str
    issue_body: str

    # Workspace Mapping
    repo_path: str
    file_registry: Dict[str, Any]
    requirement_path: str

    # Planner outputs
    engineer_tasks: List[EngineerTask]
    failed_tasks: Optional[List[EngineerTask]]

    # Test generation
    test_file: Optional[str]
    test_code: Optional[str]

    # Engineer outputs
    pending_edits: Annotated[List[Any], append_or_reset]

    # Execution state
    test_command: str
    iteration_count: int
    max_iterations: int
    is_resolved: bool

    # Feedback & cache
    test_output: Optional[str]
    critic_feedback: Optional[str]
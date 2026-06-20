# main.py
from agent.graph import app

issue = """
Choice.normalize_choice() supports case-insensitive matching via case_sensitive=False, but doesn't account for Unicode normalization. A Choice(["café", "naïve"], case_sensitive=False) will treat "café" typed with a precomposed accent (NFC) and "café" typed with a decomposed accent (NFD) as different values, even though they're visually and semantically identical — casefold() alone doesn't collapse different Unicode composition forms. Add Unicode NFC normalization as a step in normalize_choice, applied before the existing casefold step, so both forms map to the same normalized value.
"""


# In orch_test.py
initial_blackboard_state = {
    "issue_id" : "ISSUE-001",
    "issue_title" : "issue 001",
    "issue_body" : issue,
    "repo_path": "C:\\Users\\Jeremiah\\scripts\\Gem-asea\\tests",
    "target_file": "",
    #"orch_test.py",
    "target_function": None,
    "test_file": None,
    "original_code": None,
    "dependency_manifest": None,
    "file_registry": {},
    "pending_edits": [],
    "test_command": "pytest",
    "iteration_count": 0,
    "max_iterations": 5,
    "is_resolved": False,
    "current_code": None,
    "test_output": None,
    "critic_feedback": None,
    "plan": None,
}

if __name__ == "__main__":
    print("Kicking off autonomous multi-agent software engineering run...")
    final_state = app.invoke(initial_blackboard_state)
    print("\n--- RUN COMPLETION SUMMARY ---")
    print(f"Status Resolved: {final_state['is_resolved']}")
    print(f"Total Iterations: {final_state['iteration_count']}")
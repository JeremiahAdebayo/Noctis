# main.py
from agent.graph import app

issue = """
click.Path.convert() with executable=True checks executability against the wrong path when resolve_path=True is also set. The readable, writable, and existence checks all operate on rv (the path after resolution/realpath), but the executable check uses value (the original, unresolved argument). If a user passes a relative path or a symlink and has both resolve_path=True and executable=True set, the executable check can validate against a different filesystem location than every other check in the same method — potentially passing when it shouldn't, or failing when it shouldn't, depending on CWD or symlink target. All other checks in convert() are consistent in using rv; this one isn't.
"""


# In orch_test.py
initial_blackboard_state = {
    "issue_id" : "ISSUE-001",
    "issue_title" : "issue 001",
    "issue_body" : issue,
    "repo_path": "C:\\Users\\Jeremiah\\scripts\\Gem-asea\\tests",
    "target_file": "orch_test.py",
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
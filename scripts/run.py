# main.py
from agent.graph import app

# In orch_test.py
initial_blackboard_state = {
    "issue_id": None,
    "issue_title": None,
    "issue_body": None,
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
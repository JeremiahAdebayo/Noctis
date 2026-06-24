# main.py
from agent.graph import app

issue = """
## Bug: Cached score of 0 is treated as a cache miss

`UserService.get_user_score()` does not correctly return cached values when the cached score is `0`.

### Reproduction

```python
svc = UserService()

svc.get_user_score("new_123")
svc.get_user_score("new_123")
```

### Expected Behavior

The score should be computed once and returned from the cache on subsequent calls.

### Actual Behavior

The score is recomputed on every call for users whose score is `0`.

### Notes

The issue only affects cached values that evaluate to `False` in a boolean context (e.g. `0`). Other scores are returned from the cache as expected.
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
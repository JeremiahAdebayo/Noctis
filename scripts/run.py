# main.py
from agent.graph import app

issue = """
## Issue: Authenticated sessions are not preserved across API requests

### Description

After a successful login, requests made using the returned session token fail authentication. The same token that is returned by `login()` is rejected when passed to `profile()`, resulting in an unauthorized response.

### Steps to Reproduce

1. Create a new `API` instance.
2. Call `login("alice", "1234")`.
3. Pass the returned token to `profile(token)`.

### Expected Behavior

The profile endpoint should recognize the session and return the authenticated user's information.

### Actual Behavior

The profile endpoint responds with `401 Unauthorized`, as if the session does not exist.

### Notes

This appears to be a regression affecting session persistence between API calls. The issue is reproducible consistently and does not depend on the credentials used, provided the login succeeds.

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
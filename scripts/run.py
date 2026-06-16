# main.py
from agent.graph import app

issue = """Title: TaskEngine does not execute all dependent tasks

I'm seeing inconsistent behavior when running a dependency graph through TaskEngine.

For a graph where task D depends on B and C, and both B and C depend on A, sometimes D never appears in the completed set after calling run_all().

Example:

python
engine = TaskEngine()

engine.add_task(Task("A", task_a))
engine.add_task(Task("B", task_b, deps=["A"]))
engine.add_task(Task("C", task_c, deps=["A"]))
engine.add_task(Task("D", task_d, deps=["B", "C"]))

await engine.run_all()


Expected:

python
{"A", "B", "C", "D"}
```

Actual:

Sometimes only a subset of tasks are completed.

I haven't fully investigated, but it looks like tasks scheduled after dependency completion may not always finish before run_all() returns.

Can someone take a look?"""


# In orch_test.py
initial_blackboard_state = {
    "issue_id" : "ISSUE-001",
    "issue_title" : "TaskEngine circular dependency not detected",
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
# main.py
from agent.graph import app

issue = """# Bug: Users occasionally see incorrect order state after concurrent operations

## Description

We've received reports that order state can become inconsistent when multiple actions occur around the same time.

In production, users have occasionally observed one or more of the following:

* An order that was cancelled later appears as completed.
* A user is unable to create a new order because the system believes an active order already exists.
* The active order returned for a user appears to be stale or no longer valid.
* Rapid sequences of create, cancel, and complete operations sometimes produce unexpected results.

The issue is difficult to reproduce consistently and appears to occur only under concurrent usage.

## Expected Behavior

The system should maintain the following invariants:

1. A user should have at most one active order at a time.
2. Cancelled orders must never transition to completed status.
3. Completed orders must never transition to cancelled status.
4. Reads should observe a consistent view of order state.
5. Creating, cancelling, and completing orders concurrently should not violate business rules.

## Actual Behavior

Under concurrent workloads, the system occasionally violates one or more of the invariants above.

## Reproduction Notes

The issue is most frequently reported when:

* Multiple requests attempt to create orders for the same user simultaneously.
* An order is cancelled while another operation attempts to complete it.
* Reads occur while order state is actively changing.

## Task

Investigate the root cause and implement a fix.

The fix should:

* Preserve correctness under concurrent execution.
* Prevent invalid state transitions.
* Maintain the single-active-order guarantee.
* Include regression tests covering the failing scenarios.

Please ensure the solution addresses the underlying cause rather than only the observed symptoms.
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
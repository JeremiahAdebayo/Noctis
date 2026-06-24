# graph.py
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from agent.state import AgentState
from agent.schemas import (EngineerTask, RelatedFile)
from agent.nodes import (
    issue_parser_node,
    repo_reset_node,
    pre_planner_indexer_node,
    planner_node,
    engineer_node,
    critic_node, 
    resolver_node,
    reassembler_node,
    test_generator_node,
    test_executor_node
)

# In fan_out_to_engineers
def fan_out_to_engineers(state: AgentState):
    failed_task_paths = state.get("failed_tasks")
    print(f"[FAN OUT] failed_tasks value: {failed_task_paths}")
    print(f"[FAN OUT] failed_tasks type: {type(failed_task_paths)}")
    print(f"[FAN OUT] engineer_tasks count: {len(state.get('engineer_tasks', []))}")

    all_source_paths = [t.file_path for t in state["engineer_tasks"]]

    if failed_task_paths:
        # Safety check: filter out anything that isn't a known source file path
        # (catches critic hallucinating pytest node IDs or test file names)
        valid_failed_paths = [p for p in failed_task_paths if p in all_source_paths]

        if not valid_failed_paths:
            print("[FAN OUT WARNING]: failed_tasks contains no valid source file paths — retrying all tasks")
            tasks_to_run = state["engineer_tasks"]
        else:
            print(f"[FAN OUT] valid failed paths: {valid_failed_paths}")
            tasks_to_run = []
            for task in state["engineer_tasks"]:
                if task.file_path not in valid_failed_paths:
                    continue

                enriched_related = list(task.related_files)
                existing_related_paths = {rf.path for rf in enriched_related}

                for other_path in all_source_paths:
                    if other_path != task.file_path and other_path not in existing_related_paths:
                        enriched_related.append(RelatedFile(
                            path=other_path,
                            reason="Modified by a parallel engineer in the previous iteration — read current disk state for context."
                        ))

                tasks_to_run.append(EngineerTask(
                    file_path=task.file_path,
                    plan=task.plan,
                    target_functions=task.target_functions,
                    related_files=enriched_related,
                ))
    else:
        # First pass: run all tasks as planned
        tasks_to_run = state["engineer_tasks"]

    if not tasks_to_run:
        print("[FAN OUT CRITICAL]: tasks_to_run is empty — this will cause silent graph exit")

    return [
        Send("engineer", {
            "task": t,
            "repo_path": state["repo_path"],
            "issue_title": state["issue_title"],
            "issue_body": state["issue_body"],
            "test_code": state.get("test_code", ""),
            "critic_feedback": state.get("critic_feedback", ""),
        })
        for t in tasks_to_run
    ]
# 1. Define the Conditional Routing Gate Logic
def evaluate_critic_verdict(state: AgentState) -> str:
    """
    Inspects the blackboard memory to determine if the execution engine 
    should continue refining the patch or immediately conclude the cycle.
    """
    print("\n--- [ROUTING GATE] CHECKING EVALUATION STATUS ---")
    
    # Escape hatch: if the critic confirms success, exit the graph
    if state.get("is_resolved") is True:
        print("[Gate Decision]: Bug Verified Fixed. Terminating Workflow Successfully.")
        return "complete"
        
    # Guardrail: stop loop bleeding if execution exceeds the standard run limits
    if state.get("iteration_count", 0) >= 5:
        print("[Gate Decision]: Maximum iterations reached without clean pass. Forcing Exit.")
        return "complete"
        
    # Re-entry: if validation failed but we have attempts remaining, loop back to the coder
    print(f"[Gate Decision]: Fix Imperfect. Re-routing back to Engineer Node for Attempt {state['iteration_count'] + 1}.")
    return "retry"

def retry_router(state: AgentState):
    return {"pending_edits": None}

def debug_state_pass(state: AgentState):
    """Intercepts and prints raw framework state before the engineer runs."""
    print("\n=== [FRAMEWORK TELEMETRY] STATE CHANNELS LIVE AUDIT ===")
    print(f"Available State Keys: {list(state.keys())}")
    print(f"Raw 'plan' Channel Content: {repr(state.get('plan'))}")
    
    return {}  # <-- FIXED: Return an empty state delta dict, NOT a string node name

# 2. Build the Workflow State Graph Layout
workflow = StateGraph(AgentState)

# 3. Inject Nodes directly onto the execution grid
#workflow.add_node("issue_parser", issue_parser_node)
workflow.add_node("reset", repo_reset_node)
workflow.add_node("indexer", pre_planner_indexer_node)
workflow.add_node("planner", planner_node)
workflow.add_node("debug_agent", debug_state_pass)
workflow.add_node("test_generator", test_generator_node)
workflow.add_node("engineer", engineer_node)
workflow.add_node("reassembler", reassembler_node)
workflow.add_node("executor", test_executor_node)
workflow.add_node("critic", critic_node)
workflow.add_node("retry_router", retry_router)

workflow.set_entry_point("reset")
workflow.add_edge("reset", "indexer")
workflow.add_edge("indexer", "planner")
workflow.add_edge("planner", "debug_agent")
workflow.add_edge("debug_agent", "test_generator")

workflow.add_conditional_edges(
    "test_generator",
    fan_out_to_engineers,
    ["engineer"]
)

workflow.add_edge("engineer", "reassembler")
workflow.add_edge("reassembler", "executor")
workflow.add_edge("executor", "critic")

workflow.add_conditional_edges(
    "critic",
    evaluate_critic_verdict,
    {
        "retry": "planner",
        "complete": END
    }
)

workflow.add_conditional_edges(
    "retry_router",
    fan_out_to_engineers,
    ["engineer"]
)
app = workflow.compile()
print("[SYSTEM STATUS] : Workflow successfully completed")
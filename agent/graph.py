# graph.py
from langgraph.graph import StateGraph, END
from agent.state import AgentState
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

def debug_state_pass(state: AgentState):
    """Intercepts and prints raw framework state before the engineer runs."""
    print("\n=== [FRAMEWORK TELEMETRY] STATE CHANNELS LIVE AUDIT ===")
    print(f"Available State Keys: {list(state.keys())}")
    print(f"Raw 'plan' Channel Content: {repr(state.get('plan'))}")
    
    return {}  # <-- FIXED: Return an empty state delta dict, NOT a string node name

# 2. Build the Workflow State Graph Layout
workflow = StateGraph(AgentState)

# 3. Inject Nodes directly onto the execution grid
workflow.add_node("issue_parser", issue_parser_node)
workflow.add_node("reset", repo_reset_node)
workflow.add_node("indexer", pre_planner_indexer_node)
workflow.add_node("planner", planner_node)
workflow.add_node("debug_agent", debug_state_pass)
workflow.add_node("resolver", resolver_node)
workflow.add_node("test_generator", test_generator_node)
workflow.add_node("engineer", engineer_node)
workflow.add_node("reassembler", reassembler_node)
workflow.add_node("executor", test_executor_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("issue_parser")
workflow.add_edge("issue_parser", "reset")
workflow.add_edge("reset", "indexer")
workflow.add_edge("indexer", "planner")
workflow.add_edge("planner", "debug_agent")
workflow.add_edge("debug_agent", "resolver")
workflow.add_edge("resolver", "test_generator")
workflow.add_edge("test_generator", "engineer")
workflow.add_edge("engineer", "reassembler")
workflow.add_edge("reassembler", "executor")
workflow.add_edge("executor", "critic")

workflow.add_conditional_edges(
    "critic",
    evaluate_critic_verdict,
    {
        "retry": "engineer",
        "complete": END
    }
)

# 6. Compile the structural blueprints into an executable state machine
app = workflow.compile()
print("[System Status]: Gem-asea Graph Compilation Successful.")
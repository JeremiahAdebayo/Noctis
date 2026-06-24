# nodes.py
import os
import subprocess
import sys
from itertools import groupby
import xml.etree.ElementTree as ET
import libcst as cst
import ast
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from agent.schemas import (IssueParserOutput,
                           PlannerOutput,
                           EngineerOutput,
                           EngineerTask, 
                           CriticOutput, 
                           CodeMapVisitor, 
                           TestGeneratorOutput, 
                           apply_edit_to_source)
from agent.state import AgentState
from localizer.qdrant_utils import get_client, ensure_collection, get_embedder, index_chunks, clear_collection, retrieve_chunks
from localizer.lexical_index import LexicalIndex, ChunkRecord
from localizer.code_map_adapter import chunks_from_code_map
from localizer.fusion_rank import localize_full, get_chunk_summaries
from dotenv import load_dotenv


load_dotenv()
PROXY_URL = "http://localhost:4000/v1"
PROXY_KEY = os.getenv("LITELLM_MASTER_KEY")

planner_parser = PydanticOutputParser(pydantic_object=PlannerOutput)
coder_parser   = PydanticOutputParser(pydantic_object=EngineerOutput)
critic_parser  = PydanticOutputParser(pydantic_object=CriticOutput)

planner_llm = ChatOpenAI(base_url=PROXY_URL, model="gem-asea-planner", api_key=PROXY_KEY, temperature=0.1, extra_body={"reasoning_effort":"none"})
coder_llm   = ChatOpenAI(base_url=PROXY_URL, model="gem-asea-coder",   api_key=PROXY_KEY, temperature=0.1)
critic_llm  = ChatOpenAI(base_url=PROXY_URL, model="gem-asea-critic",  api_key=PROXY_KEY, temperature=0.1)

planner_prompt = ChatPromptTemplate.from_template(
    "You are an expert software architect analyzing a codebase to fix a bug.\n\n"
    "Issue: {issue_title}\n{issue_body}\n\n"
    "Relevant code retrieved from the repository, ranked by confidence "
    "(STACK_TRACE = directly implicated by the error, FUSED_RETRIEVAL = "
    "semantically/lexically relevant). Each chunk is labeled with its file path "
    "— multiple chunks may belong to the same file even if not adjacent in this list:\n"
    "{registry_summary}\n\n"
    "Previous attempt feedback (empty on first pass):\n{critic_feedback}\n\n"
    "Previous test output (empty on first pass):\n{test_output}\n\n"
    "Your Goal: Produce a complete list of per-file engineering tasks that together "
    "fully fix the issue. If critic feedback is present, it describes exactly what "
    "the previous attempt got wrong — use it to identify files that were missed or "
    "incorrectly patched and include them in your plan.\n\n"
    "Rules:\n"
    "1. Create ONE task per file that needs modification. If a file needs several "
    "changes, combine them into a single task's plan.\n"
    "2. 'file_path' must be the EXACT path as it appears in a chunk header above. "
    "Do not invent or guess paths.\n"
    "3. 'plan' must be specific to that file — describe exactly what needs to change "
    "and why, referencing the critic feedback if relevant.\n"
    "4. 'target_functions' must list ONLY function or class names that literally "
    "appear in that file's chunks above. Never invent names.\n"
    "5. If a task depends on another file you are NOT modifying, list it under "
    "'related_files' with a short reason.\n"
    "6. Only include files that require actual code changes.\n"
    "7. STACK_TRACE chunks indicate where the error occurred — weight these heavily.\n"
    "8. If critic feedback mentions a specific file or function that needs changing, "
    "you MUST include it as a task if it appears in the retrieved chunks above.\n\n"
    "Before finalizing: no two tasks share the same file_path, and every "
    "target_function name appears verbatim in the retrieved code for that file.\n"
)
# =========================================================================
# 1. SPECIALIZED MODEL INSTANTIATIONS
# =========================================================================

planner_chain = planner_prompt | planner_llm.with_structured_output(PlannerOutput)

# --- CODER CHAIN CONFIGURATION ---
coder_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an elite software engineer. You make precise, surgical edits to fix bugs.\n\n"
     "For each fix you must provide:\n"
     "- file_path: relative path to the file (must be one of the valid paths listed below)\n"
     "- node_type: one of 'function', 'class', or 'method'\n"
     "  * 'function' — a top-level function\n"
     "  * 'class' — an entire class definition\n"
     "  * 'method' — a method inside a class\n"
     "- node_path: the exact name of the node to replace\n"
     "  * For functions: just the name e.g. 'calculate_total'\n"
     "  * For classes: just the name e.g. 'PaymentProcessor'\n"
     "  * For methods: 'ClassName.method_name' e.g. 'PaymentProcessor.charge'\n"
     "- new_implementation: the complete, corrected implementation\n"
     "  * Include the full def or class signature\n"
     "  * Include all decorators the original had\n"
     "  * Use correct indentation as a top-level definition\n"
     "  * Do not truncate — write the entire node\n"
     "- add_imports: list of any new import lines needed\n"
     "  * Only include imports not already present in the file\n"
     "  * Leave empty if no new imports are needed\n"
     "- rationale: why this change fixes the bug\n\n"
     "Rules you must NEVER break:\n"
     "- Multiple edits allowed if the bug spans multiple functions, methods, or files\n"
     "- Prefer fixing the smallest node that contains the bug — method over class, function over module\n"
     "- Never guess a node name — use ONLY names visible in the source code provided\n"
     "- Never edit a file not listed under valid file paths\n"
     "- Any new dependency must be declared via add_imports only\n"
     "- If a fix requires changing a function signature, also patch callers if their source is provided\n"
     "- If critic feedback is provided, it describes exactly what went wrong — address it directly\n"
     "- Output ONLY valid JSON matching the required schema"
    ),
    ("human",
     "Issue Title: {issue_title}\n"
     "Issue Body:\n{issue_body}\n\n"
     "Execution Plan for this file:\n{plan}\n\n"
     "Target functions to modify: {target_functions}\n\n"
     "Source code of file to edit:\n{current_code}\n\n"
     "Related file context (read-only — do not edit these, but use them to understand dependencies):\n"
     "{related_context}\n\n"
     "Test file (for reference):\n{test_code}\n\n"
     "Critic feedback from previous attempt (address this directly if present):\n{critic_feedback}\n\n"
     "Valid file path for edits: {target_file}\n\n"
     "Produce all necessary patches now."
    )
])
structure_coder = coder_llm.with_structured_output(EngineerOutput)
coder_chain = coder_prompt | structure_coder

# --- CRITIC CHAIN CONFIGURATION ---
critic_prompt = ChatPromptTemplate.from_template(
    "You are a ruthless QA engine evaluating whether a bug fix attempt succeeded.\n\n"
    "Original Issue:\n{issue_body}\n\n"
    "Plans that were applied (one per file):\n{plans_applied}\n\n"
    "Test Execution Output:\n{test_output}\n\n"
    "Evaluate based on the test output alone:\n"
    "- is_resolved: true ONLY if all tests pass. false if any test fails or errors.\n"
    "- feedback: if tests failed, explain specifically what went wrong and what still "
    "needs to change to fix it. Reference the exact failure from the test output — "
    "function names, expected vs actual values, missing methods. Be specific enough "
    "that a planner can use this to identify which files need changing next.\n"
)
critic_chain = critic_prompt | critic_llm.with_structured_output(CriticOutput)

# =========================================================================
# 3. NODE IMPLEMENTATIONS
# =========================================================================


def pre_planner_indexer_node(state: AgentState) -> AgentState:
    print("\n--- [INDEXER] MAPPING REPOSITORY STRUCTURE (AST-BASED) ---")
    repo_path = state["repo_path"]
    state["file_registry"] = {}
    ignored_dirs = {'.git', '__pycache__', 'venv', '.venv', 'node_modules'}

    # Clear stale chunks from previous run
    clear_collection()

    all_qdrant_chunks = []
    lexical_index = LexicalIndex(db_path="lexical_repo.db")
    all_lexical_chunks = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            if not file.endswith(".py"):
                continue

            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, repo_path)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    source_lines = f.readlines()
                    source_text = "".join(source_lines)

                tree = ast.parse(source_text)
                visitor = CodeMapVisitor()
                visitor.visit(tree)

                state["file_registry"][relative_path] = {
                    "chunks": visitor.chunks,
                    "imports": visitor.imports,
                    "abs_path": full_path
                }

                # Attach source content to each chunk for Qdrant
                for chunk in visitor.chunks:
                    chunk_source = "".join(
                        source_lines[chunk["start_line"] - 1 : chunk["end_line"]]
                    )
                    all_qdrant_chunks.append({
                        "name": chunk["name"],
                        "type": chunk["type"],
                        "path": relative_path,
                        "source": chunk_source,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                    })
                
                # Also convert to ChunkRecords for lexical index
                lexical_chunks = chunks_from_code_map(visitor.chunks, source_text, relative_path)
                all_lexical_chunks.extend(lexical_chunks)

            except SyntaxError as e:
                print(f"[Indexer Warning]: Skipped {relative_path} due to syntax error: {e}")
                continue

    # Push everything to Qdrant in one batch
    if all_qdrant_chunks:
        index_chunks(all_qdrant_chunks)
    
    # Rebuild lexical index with all chunks
    if all_lexical_chunks:
        lexical_index.rebuild(all_lexical_chunks)
        print(f"[Indexer Status]: Indexed {len(all_lexical_chunks)} chunks to lexical index")

    print(f"[Indexer Status]: Mapped {len(state['file_registry'])} files, "
          f"indexed {len(all_qdrant_chunks)} chunks to Qdrant.")
    return state

def issue_parser_node(state: AgentState) -> AgentState:
    print("\n--- [ISSUE PARSER] ANALYZING REPOSITORY AND GENERATING ISSUE CONTEXT ---")
    
    repo_path = state["repo_path"]
    
    # Read all non-test Python source files
    repo_contents = {}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', '.venv', 'node_modules'}]
        for file in files:
            if file.endswith(".py") and not file.startswith("test_"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, repo_path)
                with open(full_path, "r", encoding="utf-8") as f:
                    repo_contents[relative_path] = f.read()

    if not repo_contents:
        print("[Issue Parser WARNING]: No source files found in repo.")
        return state

    repo_summary = "\n\n".join([
        f"=== {path} ===\n{content}"
        for path, content in repo_contents.items()
    ])

    issue_parser_llm = ChatOpenAI(
        base_url=PROXY_URL,
        model="gem-asea-planner",
        api_key=PROXY_KEY,
        temperature=0.1,
    ).with_structured_output(IssueParserOutput)

    issue_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert software engineer analyzing a codebase for bugs. "
         "Identify all logical bugs present in the source code and generate a clear issue report. "
         "Focus only on bugs that cause incorrect behavior — not style, formatting, or optimization issues."
         "CRITICAL: When referencing files, use ONLY the exact filenames shown in the === FILE === headers. "
         "Never infer, rename, or substitute filenames."),
        ("human",
         "Analyze this repository and identify all bugs present:\n\n"
         "{repo_summary}\n\n"
         "Generate a unique issue ID, concise issue title, and detailed issue body "
         "describing every bug found, which functions are affected, and what the correct behavior should be."
         "Also return a JSON array of dependency package names e.g. [\"requests\", \"numpy\"]."
        )
    ])

    chain = issue_prompt | issue_parser_llm
    result = chain.invoke({"repo_summary": repo_summary})

    print(f"[Issue Parser]: Generated issue ID — {result.issue_id}")
    print(f"[Issue Parser]: Title — {result.issue_title}")
    print(f"[Issue Parser]: Body — {result.issue_body}")

    requirement = os.path.join(repo_path, "requirements.txt")
    with open(requirement, "w", encoding="utf-8") as f:
        f.write("\n". join(result.imports))

    return {
        "issue_id": result.issue_id,
        "issue_title": result.issue_title,
        "issue_body": result.issue_body,
        "requirement_path": requirement
    }

def repo_reset_node(state: AgentState) -> AgentState:
    print("\n--- [RESET] RESTORING REPOSITORY TO CLEAN STATE ---")
    repo_path = state["repo_path"]
    
    try:
        # Reset all tracked file changes
        subprocess.run(
            ["git", "checkout", "."],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        # Remove any untracked files the agent generated (test files etc)
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_path,
            capture_output=True,
            check=True
        )
        print("[Reset]: Repository restored to clean git state.")
    except subprocess.CalledProcessError as e:
        print(f"[Reset WARNING]: Git reset failed — {e}. Proceeding anyway.")
    
    return state

from itertools import groupby

def planner_node(state: AgentState) -> AgentState:
    print("\n--- [PLANNER] ANALYZING REGISTRY & GENERATING STRATEGY ---")

    # Query always incorporates everything known so far
    # On first pass: issue text only
    # On retry: issue + critic feedback + test output
    # Planner doesn't need to know which pass it is — it just uses all available context
    query_parts = [state["issue_title"], state["issue_body"]]
    if state.get("critic_feedback"):
        query_parts.append(state["critic_feedback"])
    if state.get("test_output"):
        query_parts.append(state["test_output"])
    query = " ".join(query_parts)

    repo_path = state["repo_path"]

    candidates = localize_full(
        bug_report_text=state.get("issue_body", ""),
        query=query,
        repo_path=repo_path,
        lexical_db_path="lexical_repo.db",
        max_results=5,
        vector_top_k=10,
        lexical_top_k=10,
    )

    if not candidates:
        print("[Planner WARNING]: Fusion returned no candidates — falling back to registry summary")
        context = "\n".join([
            f"File: {path} | Components: {', '.join([c['name'] for c in meta['chunks']])}"
            for path, meta in state["file_registry"].items()
        ])
    else:
        seen_order = []
        by_file = {}
        for c in candidates:
            print(f"  {c.file_path} | {c.confidence_tier} | score: {c.score}")
            if c.file_path not in by_file:
                by_file[c.file_path] = []
                seen_order.append(c.file_path)
            by_file[c.file_path].append(c)

        grouped_candidates = [c for fp in seen_order for c in by_file[fp]]
        context = get_chunk_summaries(grouped_candidates)
        print(f"[Planner]: Retrieved {len(candidates)} candidates across {len(by_file)} files")

    # Pass full history to planner so it replans with complete awareness
    result: PlannerOutput = planner_chain.invoke({
        "issue_title": state["issue_title"],
        "issue_body": state["issue_body"],
        "registry_summary": context,
        "critic_feedback": state.get("critic_feedback", ""),
        "test_output": state.get("test_output", ""),
    })

    # Guardrail: catch duplicate file_path tasks
    paths = [t.file_path for t in result.engineer_tasks]
    if len(paths) != len(set(paths)):
        dupes = {p for p in paths if paths.count(p) > 1}
        print(f"[Planner WARNING]: Duplicate file_path tasks detected: {dupes}")

    for task in result.engineer_tasks:
        print(f"[Planner] file: {task.file_path} | target_functions: {task.target_functions}")

    return {
        "engineer_tasks": result.engineer_tasks,
        "pending_edits": [],
        "failed_tasks": None,
    }
# =========================================================================
# 3. NODE IMPLEMENTATIONS
# =========================================================================

def extract_target_functions(source: str, function_names: List[str]) -> str:
    """Extract only named functions/classes from source via AST — avoids sending whole files."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source[:1000]  # fallback if file has syntax errors

    extracted = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in function_names:
                segment = ast.get_source_segment(source, node)
                if segment:
                    extracted.append(segment)

    return "\n\n".join(extracted) if extracted else source[:1000]

def engineer_node(state: dict) -> dict:
    task: EngineerTask = state["task"]
    repo_path = state["repo_path"]

    # Read the target file from disk (post-reassembler state on retries)
    full_target_path = os.path.join(repo_path, task.file_path)
    with open(full_target_path, "r", encoding="utf-8") as f:
        disk_truth = f.read()

    print(f"[Engineer] target: {task.file_path}")
    print(f"[Engineer] target_functions: {task.target_functions}")

    # Build related context — extract only touched functions, not whole files
    related_context_parts = []
    for rf in task.related_files:
        full_path = os.path.join(repo_path, rf.path)
        if not os.path.exists(full_path):
            print(f"[Engineer WARNING]: related file not found on disk — {rf.path}")
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            related_source = f.read()

        if rf.relevant_functions:
            snippet = extract_target_functions(related_source, rf.relevant_functions)
        else:
            snippet = related_source[:1000]

        related_context_parts.append(
            f"### {rf.path}\n"
            f"# Reason: {rf.reason}\n"
            f"# Relevant functions: {', '.join(rf.relevant_functions) if rf.relevant_functions else 'general context'}\n\n"
            f"{snippet}"
        )

    related_context = "\n\n".join(related_context_parts) if related_context_parts else "No related files."

    result = coder_chain.invoke({
        "issue_title": state["issue_title"],
        "issue_body": state["issue_body"],
        "plan": task.plan,
        "current_code": disk_truth,
        "related_context": related_context,
        "test_code": state.get("test_code", "No test file available."),
        "target_functions": ", ".join(task.target_functions),
        "target_file": task.file_path,
        "critic_feedback": state.get("critic_feedback", "No feedback yet — first attempt."),
    })

    print(f"[Engineer]: Produced {len(result.edits)} edit(s) for {task.file_path}")

    return {
        "pending_edits": result.edits,
    }

def critic_node(state: AgentState) -> AgentState:
    print("\n--- [CRITIC] EVALUATING PATCH SET ---")

    # Summarize what was attempted — plans only, not full source
    plans_summary = "\n".join([
        f"- {task.file_path}: {task.plan}"
        for task in state["engineer_tasks"]
    ])

    result = critic_chain.invoke({
        "issue_body": state["issue_body"],
        "plans_applied": plans_summary,
        "test_output": state.get("test_output", "No test run data available."),
    })

    print(f"\n>>> [CRITIC FEEDBACK]: {result.feedback or 'No feedback provided'}")
    print(f">>> [CRITIC VERDICT]: {'RESOLVED' if result.is_resolved else 'NEEDS RETRY'}")

    return {
        "critic_feedback": result.feedback,
        "is_resolved": result.is_resolved,
        "iteration_count": state["iteration_count"] + 1,
    }

def install_dependencies(repo_path: str, python_executable: str):
    print("---INSTALLING DEPENDENCIES...---\n")

    result = subprocess.run(
        [python_executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"[Sandbox WARNING]: editable install failed:\n{result.stderr}")

    # Install test dependencies explicitly
    test_deps = ["pytest", "pytest-asyncio", "pytest-xdist", "pytest-mock"]
    result = subprocess.run(
        [python_executable, "-m", "pip", "install", *test_deps, "--quiet"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"[Sandbox WARNING]: test deps install failed:\n{result.stderr}")

    req_file = os.path.join(repo_path, "requirements.txt")
    pyproject = os.path.join(repo_path, "pyproject.toml")
    
    if os.path.exists(req_file):
        print("[Sandbox]: Installing from requirements.txt...")
        subprocess.run(
            [python_executable, "-m", "pip", "install", "-r", req_file, "-q"],
            cwd=repo_path,
            capture_output=True
        )
    elif os.path.exists(pyproject):
        print("[Sandbox]: Installing from pyproject.toml...")
        subprocess.run(
            [python_executable, "-m", "pip", "install", ".", "-q"],
            cwd=repo_path,
            capture_output=True
        )
    
    # Always ensure pytest itself is present regardless
    subprocess.run(
        [python_executable, "-m", "pip", "install", "pytest", "-q"],
        capture_output=True
    )

def resolve_module_to_path(module_name, registry_keys):
        # 1. Convert 'src.checkout.processor' -> 'src/checkout/processor'
        path_candidate = module_name.replace(".", os.sep)
        
        # 2. Look for exact matches in registry
        # We check keys that end with the candidate to allow for deeper imports
        for key in registry_keys:
            # Case A: Exact module file (e.g., src/utils/math.py)
            if key == f"{path_candidate}.py":
                return key
            # Case B: Module in a folder with __init__.py (e.g., src/utils/math/__init__.py)
            if key.startswith(path_candidate) and "__init__.py" in key:
                return key
        return None
def test_generator_node(state: AgentState) -> AgentState:
    print("\n--- [TEST GENERATOR] WRITING VERIFICATION SUITE ---")

    repo_path = state["repo_path"]
    engineer_tasks = state["engineer_tasks"]
    issue_id = state["issue_id"]

    if not engineer_tasks:
        print("[Test Generator WARNING]: No engineer tasks found — skipping test generation.")
        return {"test_file": None, "test_code": ""}

    # Build combined context from all files being changed
    file_contexts = []
    for task in engineer_tasks:
        full_path = os.path.join(repo_path, task.file_path)
        if not os.path.exists(full_path):
            print(f"[Test Generator WARNING]: {task.file_path} not found on disk — skipping.")
            continue
        with open(full_path, "r", encoding="utf-8") as f:
            source = f.read()
        file_contexts.append(
            f"### File: {task.file_path}\n"
            f"# Plan: {task.plan}\n"
            f"# Target functions: {', '.join(task.target_functions)}\n\n"
            f"{source}"
        )

    combined_context = "\n\n".join(file_contexts)
    # Read imports of each task's file and include dependency interfaces
    registry_keys = list(state["file_registry"].keys())
    dependency_context = []

    for task in engineer_tasks:
        meta = state["file_registry"].get(task.file_path, {})
        for imp in meta.get("imports", []):
            dep_path = resolve_module_to_path(imp, registry_keys)
            if dep_path and dep_path != task.file_path:
                abs_path = state["file_registry"][dep_path]["abs_path"]
                if os.path.exists(abs_path):
                    with open(abs_path, "r") as f:
                        dep_source = f.read()
                    dependency_context.append(f"### Dependency: {dep_path}\n{dep_source}")

    dependency_summary = "\n\n".join(dependency_context) if dependency_context else "No dependencies found."

    # Check if a test file already exists for this issue
    # Convention: test file is named after the primary (first) task's file
    test_file_name = f"test_issue_{issue_id}.py"
    test_file_path = os.path.join(repo_path, test_file_name)

    if os.path.exists(test_file_path):
        print(f"[Test Generator]: Existing test file found — using '{test_file_name}' as-is.")
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_code = f.read()
        return {"test_file": test_file_name, "test_code": test_code}

    test_generator_llm = ChatOpenAI(
        base_url=PROXY_URL,
        model="gem-asea-coder",
        api_key=PROXY_KEY,
        temperature=0.1,
    ).with_structured_output(TestGeneratorOutput)

    test_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior QA engineer writing a pytest verification suite for a bug fix.\n\n"
         "You will receive the source code of one or more files involved in fixing a single issue, "
         "along with the plan describing what each file needs to change.\n\n"
         "Your job: write ONE pytest test file that verifies the correct observable behavior "
         "after the fix is applied — not per-file unit tests, but behavioral tests that directly "
         "catch the bug described in the issue.\n\n"
         "Rules:\n"
         "- Tests must FAIL on the current buggy code and PASS after the fix.\n"
         "- Do NOT write tests that assert exceptions or reproduce the bug as expected behavior.\n"
         "- Assert what the code SHOULD do when working correctly.\n"
         "- Use only pytest, no unittest.\n"
         "- Imports must be correct and reference the actual module paths provided.\n"
         "- Write at least 7 test functions covering edge cases, not just the happy path.\n"
         "- Base tests on what the code actually does, not on file names."
         "Available dependency interfaces (use ONLY methods that exist here):\n{dependency_summary}\n\n"),
        ("human",
         "Issue: {issue_title}\n\n"
         "{issue_body}\n\n"
         "Files being changed (with plans and current source):\n\n"
         "{combined_context}\n\n"
         "Write a single complete pytest test file that verifies this issue is fixed.")
    ])

    chain = test_prompt | test_generator_llm
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        print(f"[Test Generator]: Generation attempt {attempt}/{max_attempts}")

        result = chain.invoke({
            "issue_title": state["issue_title"],
            "issue_body": state["issue_body"],
            "combined_context": combined_context,
            "dependency_summary": dependency_summary
        })

        # Validate syntax before writing to disk
        try:
            ast.parse(result.test_code)
        except SyntaxError as e:
            print(f"[Test Generator WARNING]: Syntax error on attempt {attempt}: {e}")
            continue

        # Write to disk
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(result.test_code)

        # Verify tests fail on current buggy code — if they pass now,
        # they won't catch anything after the fix either
        validation = subprocess.run(
            [sys.executable, "-m", "pytest", test_file_path, "-v", "--tb=short"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )

        if "failed" in validation.stdout or "error" in validation.stdout.lower():
            print(f"[Test Generator]: Tests correctly fail on buggy code — verified.")
            print(f"[Test Generator]: Written '{test_file_name}' with functions: {result.test_functions}")
            return {"test_file": test_file_name, "test_code": result.test_code}

        print(f"[Test Generator WARNING]: Tests pass on buggy code on attempt {attempt} — retrying.")
        os.remove(test_file_path)

    print(f"[Test Generator CRITICAL]: Could not generate valid failing tests after {max_attempts} attempts.")
    return {"test_file": None, "test_code": ""}

def test_executor_node(state: AgentState) -> AgentState:
    """
    Executes the workspace test suite, dumping results into an explicit 
    XML schema to avoid fragile text-scraping or regex.
    """
    print("\n--- [SANDBOX] RUNNING AUTOMATED VERIFICATION ---")
    repo_path = state["repo_path"]
    report_xml_path = os.path.join(repo_path, "junit_report.xml")
    install_dependencies(repo_path, sys.executable)
    
    try:
        cmd = f'"{sys.executable}" -m pytest --junitxml="{report_xml_path}"'
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            shell=True,
            capture_output=True,
            text=True,
            timeout=45
        )


        # Guardrail: pytest crashed before generating XML
        if not os.path.exists(report_xml_path):
            state["test_output"] = (
                f"CRITICAL: Test runner failed to execute.\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}\n"
                f"Return code: {result.returncode}"
            )
            state["is_resolved"] = False
            return state

        # Parse XML
        tree = ET.parse(report_xml_path)
        root = tree.getroot()
        failures_summary = []

        for testcase in root.findall(".//testcase"):
            failure_node = testcase.find("failure")
            error_node = testcase.find("error")
            problem_node = failure_node or error_node

            if problem_node is not None:
                test_name = testcase.get("name", "Unknown Test")
                classname = testcase.get("classname", "Unknown Class")
                problem_type = "FAILURE" if failure_node is not None else "ERROR"
                error_message = problem_node.get("message", "No message provided")
                traceback_text = problem_node.text or ""

                failures_summary.append(
                    f"{problem_type} — {classname}.{test_name}\n"
                    f"MESSAGE: {error_message}\n"
                    f"TRACEBACK:\n{traceback_text.strip()}\n"
                    f"{'='*50}"
                )

        os.remove(report_xml_path)

        # Extra guardrail: returncode non-zero but XML had no failure/error nodes
        # This catches cases like collection errors that slip through
        if len(failures_summary) == 0 and result.returncode != 0:
            state["test_output"] = (
                f"CRITICAL: Pytest exited with code {result.returncode} but no failures were parsed.\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
            state["is_resolved"] = False
            return state

        if len(failures_summary) == 0:
            print("[Sandbox Status]: All Tests Passed Cleanly.")
            state["test_output"] = "ALL TESTS PASSED CLEANLY."
            state["is_resolved"] = True
        else:
            print(f"[Sandbox Status]: Detected {len(failures_summary)} Failure(s)/Error(s).")
            state["test_output"] = "\n".join(failures_summary)
            state["is_resolved"] = False

    except subprocess.TimeoutExpired:
        state["test_output"] = "EXECUTION ERROR: Test suite timed out after 45 seconds."
        state["is_resolved"] = False
        if os.path.exists(report_xml_path):
            os.remove(report_xml_path)
    print(state["test_output"])
    return state

def resolver_node(state: AgentState) -> AgentState:
    print("\n--- [RESOLVER] MAPPING DEPENDENCIES WITH PATH NORMALIZATION ---")
    
    target_file = state.get("target_file")
    if not target_file:
        raise ValueError("Resolver triggered without a target_file.")

    manifest = {
    "target_file": target_file,
    "write_enabled": [target_file, f"test_{target_file}"],  # both are writable
    "read_only_context": {}
}

    # Helper: Normalize Import -> File Path

    # Retrieve from registry
    target_meta = state["file_registry"].get(target_file)
    registry_keys = list(state["file_registry"].keys())

    if target_meta:
        imports = target_meta.get("imports", [])
        
        for imp in imports:
            dep_path = resolve_module_to_path(imp, registry_keys)
            
            # If we find it, and it's not the target, add to read-only context
            if dep_path and dep_path != target_file:
                # Read content from disk
                full_path = state["file_registry"][dep_path]["abs_path"]
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8") as f:
                        manifest["read_only_context"][dep_path] = f.read()

    state["dependency_manifest"] = manifest
    print(f"[Resolver Status]: Resolved {len(manifest['read_only_context'])} dependencies.")
    
    return state

def reassembler_node(state: AgentState) -> AgentState:
    print("\n--- [REASSEMBLER] APPLYING LIBCST PATCHES ---")
    repo_path = state["repo_path"]
    patches = state.get("pending_edits", [])

    if not patches:
        print("[Reassembler]: No edits to apply.")
        return state

    patch_failed = False
    failed_feedback_parts = []
    touched_files = set()

    # Guardrail from earlier: fail loudly if two patches target the same file
    # in the same batch instead of silently overwriting one with the other.
    seen_paths = [p.file_path for p in patches]
    if len(seen_paths) != len(set(seen_paths)):
        dupes = {p for p in seen_paths if seen_paths.count(p) > 1}
        return {
            "pending_edits": [],
            "is_resolved": False,
            "critic_feedback": f"Multiple patches target the same file in one batch: {dupes}. Planner must partition by distinct file_path."
        }

    for patch in patches:
        print(patch)
        file_path = os.path.join(repo_path, patch.file_path)   # ← fixed

        if not os.path.exists(file_path):
            print(f"[Reassembler WARNING]: File not found — {patch.file_path}")
            patch_failed = True
            failed_feedback_parts.append(f"File '{patch.file_path}' does not exist.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            new_source = apply_edit_to_source(source, patch)
            ast.parse(new_source)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_source)

            touched_files.add(patch.file_path)
            print(f"[Reassembler]: Patched '{patch.node_path}' in '{patch.file_path}'")

        except ValueError as e:
            print(f"[Reassembler WARNING]: {e}")
            patch_failed = True
            failed_feedback_parts.append(
                f"Node '{patch.node_path}' not found in '{patch.file_path}'. "
                f"Double-check node_type is correct and node_path matches exactly "
                f"(for methods use 'ClassName.method_name')."
            )

        except cst.ParserSyntaxError as e:
            print(f"[Reassembler WARNING]: new_implementation has invalid syntax — {e}")
            patch_failed = True
            failed_feedback_parts.append(
                f"new_implementation for '{patch.node_path}' has a syntax error: {e}. Fix the syntax and resubmit."
            )

        except SyntaxError as e:
            print(f"[Reassembler WARNING]: Post-patch AST validation failed — {e}")
            patch_failed = True
            failed_feedback_parts.append(
                f"Patch for '{patch.node_path}' produced invalid Python after application: {e}."
            )

    if patch_failed:
        return {
            "pending_edits": [],
            "is_resolved": False,
            "critic_feedback": "\n".join(failed_feedback_parts)
        }

    # Re-index every file that was actually touched, not a single hardcoded one
    updated_registry = dict(state["file_registry"])
    for rel_path in touched_files:
        abs_path = os.path.join(repo_path, rel_path)
        with open(abs_path, "r") as f:
            source = f.read()
        try:
            tree = ast.parse(source)
            visitor = CodeMapVisitor()
            visitor.visit(tree)
            updated_registry[rel_path] = {
                "chunks": visitor.chunks,
                "imports": visitor.imports,
                "abs_path": abs_path
            }
            print(f"[Reassembler]: Re-indexed '{rel_path}'")
        except SyntaxError:
            pass  # keep stale registry entry for this file rather than crash the batch

    return {
        "pending_edits": None,
        "file_registry": updated_registry
    }
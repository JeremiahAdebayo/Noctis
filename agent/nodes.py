# nodes.py
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
import ast
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from agent.schemas import IssueParserOutput,PlannerOutput, EngineerOutput, CriticOutput, CodeMapVisitor, TestGeneratorOutput
from agent.state import AgentState
from dotenv import load_dotenv


load_dotenv()
PROXY_URL = "http://localhost:4000/v1"
PROXY_KEY = os.getenv("LITELLM_MASTER_KEY")

planner_parser = PydanticOutputParser(pydantic_object=PlannerOutput)
coder_parser   = PydanticOutputParser(pydantic_object=EngineerOutput)
critic_parser  = PydanticOutputParser(pydantic_object=CriticOutput)

planner_llm = ChatOpenAI(base_url=PROXY_URL, model="gem-asea-coder", api_key=PROXY_KEY, temperature=0.1)
coder_llm   = ChatOpenAI(base_url=PROXY_URL, model="gem-asea-coder",   api_key=PROXY_KEY, temperature=0.3)
critic_llm  = ChatOpenAI(base_url=PROXY_URL, model="gem-asea-critic",  api_key=PROXY_KEY, temperature=0.1)

planner_prompt = ChatPromptTemplate.from_template(
    "You are an expert software architect. You have access to a repository map.\n\n"
    "Issue: {issue_title}\n{issue_body}\n\n"
    "Available Repository Structure:\n{registry_summary}\n\n"
    "Your Goal: Create a precise execution plan. \n"
    "You must pick a 'target_function' from the registry that needs modification.\n"
)

# =========================================================================
# 1. SPECIALIZED MODEL INSTANTIATIONS
# =========================================================================

planner_chain = planner_prompt | planner_llm.with_structured_output(PlannerOutput)

# --- CODER CHAIN CONFIGURATION ---
coder_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an elite software engineer. You make precise surgical edits.\n"
     "For each fix provide:\n"
     "- file_path: relative path to the file\n"
     "- old_content: the EXACT text currently in the file including correct indentation and whitespace.\n"
     "  Copy it character-for-character from the source code provided.\n"
     "  Include enough surrounding lines to make it unambiguous — never just one short line.\n"
     "- new_content: the replacement text with correct indentation.\n"
     "- rationale: why this change was made.\n\n"
     "Don't infer what the code is supposed to do from the file name. Infer it after reading the code itself"
     "Rules:\n"
     "- Multiple edits allowed for multiple bugs in different locations.\n"
     "- If your fix changes a function signature, also patch the test file to match.\n"
     "- Never rewrite the entire file unless absolutely necessary.\n"
     "- Output ONLY valid JSON matching the required schema."
    ),
    ("human",
     "Issue Title: {issue_title}\n"
     "Issue Context:\n{issue_body}\n\n"
     "Master Execution Strategy:\n{plan}\n\n"
     "Target Function: {target_function}\n\n"
     "Current Source Code:\n{current_code}\n\n"
     "Current Test Code ({test_file_name}):\n{test_code}\n\n"
     "Previous Critic Feedback (if any):\n{critic_feedback}\n\n"
     "Produce all necessary patches now."
     "The ONLY valid file_path values for your edits are: {target_file} and {test_file_name}\n"
     "Do not use any other filenames.\n"
    )
])
structure_coder = coder_llm.with_structured_output(EngineerOutput)
coder_chain = coder_prompt | structure_coder

# --- CRITIC CHAIN CONFIGURATION ---
critic_prompt = ChatPromptTemplate.from_template(
    "You are a ruthless code review and QA automation engine.\n"
    "Original Issue:\n{issue_body}\n\n"
    "Proposed Code Patch:\n{patch_code}\n\n"
    "Automated Test Execution Output:\n{test_output}\n\n"
    "Evaluate if the issue is fully resolved. Output your verdict and feedback strictly following these formatting rules:\n"
    "{format_instructions}"
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

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if not file.endswith(".py"):
                continue
                
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, repo_path)
            
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                
                visitor = CodeMapVisitor()
                visitor.visit(tree)
                
                # STORE METADATA ONLY.
                # No content. No source lines. Just a map.
                state["file_registry"][relative_path] = {
                    "chunks": visitor.chunks,
                    "imports": visitor.imports,
                    "abs_path": full_path
                }
                
            except SyntaxError as e:
                print(f"[Indexer Warning]: Skipped {relative_path} due to syntax error: {e}")
                continue

    print(f"[Indexer Status]: Mapped {len(state['file_registry'])} files with dependency metadata.")
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
        model="gem-asea-coder",
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
        )
    ])

    chain = issue_prompt | issue_parser_llm
    result = chain.invoke({"repo_summary": repo_summary})

    print(f"[Issue Parser]: Generated issue ID — {result.issue_id}")
    print(f"[Issue Parser]: Title — {result.issue_title}")
    print(f"[Issue Parser]: Body — {result.issue_body}")

    return {
        "issue_id": result.issue_id,
        "issue_title": result.issue_title,
        "issue_body": result.issue_body
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

def planner_node(state: AgentState) -> AgentState:
    print("\n--- [PLANNER] ANALYZING REGISTRY & GENERATING STRATEGY ---")
    
    # Summary of map
    registry_summary = "\n".join([
    f"File: {path} | Components: {', '.join([c['name'] for c in meta['chunks']])}"
    for path, meta in state["file_registry"].items()
    ])
    
    # No format_instructions passed!
    result = planner_chain.invoke({
        "issue_title": state["issue_title"],
        "issue_body": state["issue_body"],
        "registry_summary": registry_summary
    })
    print(f"[Planner] target_file: {result.target_file}")
    print(f"[Planner] target_function: {result.target_function}")
    print(f"[Planner] rationale: {result.rationale}")
    return {
        "plan": result.plan,
        "target_file": result.target_file,
        "pending_edits": [],
        "target_function": result.target_function
    }

# =========================================================================
# 3. NODE IMPLEMENTATIONS
# =========================================================================

def engineer_node(state: AgentState) -> AgentState:
    print(f"\n--- [ENGINEER] GENERATING SURGICAL PATCHES (ATTEMPT {state['iteration_count']}) ---")

    target_file = state.get("target_file", "")
    target_function = state.get("target_function", "")
    full_target_path = os.path.join(state['repo_path'], target_file)

    with open(full_target_path, "r", encoding="utf-8") as f:
        disk_truth = f.read()

    # Pass test file contents if it exists
    test_file_name = state.get("test_file", f"test_{target_file}")
    test_file_path = os.path.join(state['repo_path'], test_file_name) if test_file_name else None
    test_truth = ""
    if test_file_path and os.path.exists(test_file_path):
        with open(test_file_path, "r", encoding="utf-8") as f:
            test_truth = f.read()

    result = coder_chain.invoke({
        "issue_title": state["issue_title"],
        "issue_body": state["issue_body"],
        "plan": state["plan"],
        "current_code": disk_truth,
        "test_code": test_truth,
        "critic_feedback": state.get("critic_feedback", "No feedback yet."),
        "target_function": target_function,
        "target_file": state.get("target_file", "Can't find target file"),
        "test_file_name": state.get("test_file_name", "")
    })

    return {
        "pending_edits": result.edits,
        "iteration_count": state["iteration_count"] + 1
    }

def critic_node(state: AgentState) -> AgentState:
    print("\n--- [GROQ LLAMA] EVALUATING OUTPUT AND SYSTEM INTEGRITY ---")
    
    # 1. Invoke the chain. 
    # Because your chain is configured for structured output, 'result' 
    # is already the Pydantic object (CriticOutput).
    result = critic_chain.invoke({
        "issue_body": state["issue_body"],
        "patch_code": state["current_code"],
        "test_output": state.get("test_output", "No run data available."),
        "format_instructions": critic_parser.get_format_instructions()
    })
    
    # 2. Access attributes directly from the Pydantic object.
    # No .content, no manual string parsing, no .get() methods.
    state["critic_feedback"] = result.feedback
    state["is_resolved"] = result.is_resolved
    
    print(f"\n>>> [CRITIC FEEDBACK]: {result.feedback or 'No feedback provided'}")
    
    return state

def install_dependencies(repo_path: str, python_executable: str):
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

def test_generator_node(state: AgentState) -> AgentState:
    print("\n--- [TEST GENERATOR] WRITING VERIFICATION SUITE ---")
    
    repo_path = state["repo_path"]
    target_file = state["target_file"]
    target_function = state["target_function"]

    # Check if a test file already exists for this target
    """existing_test_file = os.path.join(repo_path, f"test_{target_file}")
    if os.path.exists(existing_test_file):
        print(f"[Test Generator]: Existing test file found — using 'test_{target_file}' as-is.")
        return {"test_file": f"test_{target_file}"}"""

    # No existing tests — generate them
    full_target_path = os.path.join(repo_path, target_file)
    with open(full_target_path, "r", encoding="utf-8") as f:
        source = f.read()

    test_generator_llm = ChatOpenAI(
        base_url=PROXY_URL,
        model="gem-asea-coder",
        api_key=PROXY_KEY,
        temperature=0.1,
    ).with_structured_output(TestGeneratorOutput)

    test_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior QA engineer. Write pytest tests that define CORRECT expected behavior "
         "after the bug is fixed. Do NOT write tests that expect exceptions or reproduce the bug. "
         "Tests should assert what the function SHOULD return when working correctly. "
         "The test code you write should be inferred from what the main code is trying to do and not from the file name"
         "Use only pytest, no unittest."),
        ("human",
         "Issue: {issue_body}\n\n"
         "Target function: {target_function}\n"
         "Source file: {target_file}\n\n"
         "Source code:\n{source}\n\n"
         "Write a complete pytest test file that imports '{target_function}' "
         "from '{module_name}' and contains at least 7 test functions."
         "CRITICAL: Do not write test code based on the file name. Read the source code and write test code based on the function"
        )
    ])

    chain = test_prompt | test_generator_llm
    test_file_name = f"test_{target_file}"
    test_file_path = os.path.join(repo_path, test_file_name)
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        print(f"[Test Generator]: Generation attempt {attempt}/{max_attempts}")

        result = chain.invoke({
            "issue_body": state["issue_body"],
            "target_function": target_function,
            "target_file": target_file,
            "source": source,
            "module_name": target_file.replace(".py", "")
        })

        # Validate syntax
        try:
            ast.parse(result.test_code)
        except SyntaxError as e:
            print(f"[Test Generator WARNING]: Syntax error on attempt {attempt}: {e}")
            continue

        # Write to disk
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(result.test_code)

        # Validate tests fail on buggy code
        validation = subprocess.run(
            [sys.executable, "-m", "pytest", test_file_path, "-v", "--tb=short"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )

        if "failed" in validation.stdout or "error" in validation.stdout.lower():
            print(f"[Test Generator]: Tests correctly fail on buggy code — verified.")
            print(f"[Test Generator]: Written '{test_file_name}' with {len(result.test_functions)} test functions: {result.test_functions}")
            return {"test_file": test_file_name}

        print(f"[Test Generator WARNING]: Tests pass on buggy code on attempt {attempt} — retrying.")
        os.remove(test_file_path)

    # All attempts exhausted
    print(f"[Test Generator CRITICAL]: Could not generate valid tests after {max_attempts} attempts.")
    return {"test_file": None}

def test_executor_node(state: AgentState) -> AgentState:
    """
    Executes the workspace test suite, dumping results into an explicit 
    XML schema to avoid fragile text-scraping or regex.
    """
    print("\n--- [SANDBOX] RUNNING AUTOMATED VERIFICATION ---")
    repo_path = state["repo_path"]
    report_xml_path = os.path.join(repo_path, "junit_report.xml")
    install_dependencies(repo_path, sys.executable)
    
    # 1. Execute pytest with an explicit XML artifact directive
    try:
        # We use a single string and shell=True so Windows resolves 
        # the 'pytest' executable alias inside your virtual environment.
        cmd = f'"{sys.executable}" -m pytest --junitxml="{report_xml_path}"'
        
        result = subprocess.run(
            cmd, 
            cwd=repo_path,
            shell=True,             # Added: Required for Windows path resolution
            capture_output=True, 
            text=True, 
            timeout=45
        )
        
        # Guardrail: Check if the XML file was actually generated
        if not os.path.exists(report_xml_path):
            state["test_output"] = (
            f"CRITICAL: Test runner failed to execute.\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}\n"
            f"Return code: {result.returncode}"
            )
            print(f"[Sandbox DEBUG] stdout: {result.stdout}")
            print(f"[Sandbox DEBUG] stderr: {result.stderr}")
            print(f"[Sandbox DEBUG] returncode: {result.returncode}")
            state["test_output"] = f"CRITICAL: Test runner failed.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            return state

        # 2. Parse the machine-readable XML artifact
        tree = ET.parse(report_xml_path)
        root = tree.getroot()
        
        failures_summary = []
        
        # Iterate strictly through failing test cases
        for testcase in root.findall(".//testcase"):
            failure_node = testcase.find("failure")
            if failure_node is not None:
                test_name = testcase.get("name", "Unknown Test")
                classname = testcase.get("classname", "Unknown Class")
                error_message = failure_node.get("message", "No message provided")
                traceback_text = failure_node.text or ""
                
                # Format into a clean, structured schema block for the LLM
                failures_summary.append(
                    f"FAILING TEST: {classname}.{test_name}\n"
                    f"ERROR MESSAGE: {error_message}\n"
                    f"TRACEBACK:\n{traceback_text.strip()}\n"
                    f"{'='*50}"
                )
        
        # Clean up the artifact file immediately to ensure isolation between iterations
        os.remove(report_xml_path)
        
        # 3. Commit structured telemetry data back to the blackboard
        if len(failures_summary) == 0:
            print("[Sandbox Status]: All Tests Passed Cleanly.")
            state["test_output"] = "ALL TESTS PASSED CLEANLY."
            state["is_resolved"] = True
        else:
            print(f"[Sandbox Status]: Detected {len(failures_summary)} Explicit Failures.")
            state["test_output"] = "\n".join(failures_summary)
            state["is_resolved"] = False
            
    except subprocess.TimeoutExpired:
        state["test_output"] = "EXECUTION ERROR: Test suite timed out after 45 seconds."
        state["is_resolved"] = False
        if os.path.exists(report_xml_path):
            os.remove(report_xml_path)
            
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
    print("\n--- [REASSEMBLER] APPLYING SURGICAL PATCHES ---")
    repo_path = state["repo_path"]
    patches = state.get("pending_edits", [])
    file_path = state.get("target_file")

    if not patches:
        print("[Reassembler]: No edits to apply.")
        return state

    patch_failed = False

    for patch in patches:
        file_path = os.path.join(repo_path, file_path)
        print(file_path)

        if not os.path.exists(file_path):
            print(f"[Reassembler WARNING]: File not found — {patch.file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        original_source = source
        occurrences = source.count(patch.old_content)
        failed_feedback = ""
        if occurrences == 0:
            print(f"[Reassembler WARNING]: old_content not found in {file_path} — whitespace mismatch likely.")
            patch_failed = True
            failed_feedback= f"""Patch for '{file_path}' failed: old_content not found.  
                Reproduce old_content character-for-character from the source including exact indentation."""
            continue

        # Apply replacement
        new_source = source.replace(patch.old_content, patch.new_content, 1)

        # Validate result is valid Python
        try:
            ast.parse(new_source)
        except SyntaxError as e:
            print(f"[Reassembler WARNING]: Patch produces invalid Python — reverting: {e}")
            new_source = original_source
            patch_failed = True

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_source)

        if new_source != original_source:
            print(f"[Reassembler]: Applied patch to '{file_path}'")

    # If any patch failed, feed back to engineer with clear instructions
    if patch_failed:
        current_source = ""
        target_file = state["target_file"]
        target_abs_path = os.path.join(repo_path, target_file)
        with open(target_abs_path, "r", encoding="utf-8") as f:
            current_source = f.read()

        return {
            "pending_edits": [],
            "current_code": current_source,
            "file_registry": state["file_registry"],
            "is_resolved": False,
            "critic_feedback": failed_feedback
        }

    # Re-index target file
    target_file = state["target_file"]
    target_abs_path = os.path.join(repo_path, target_file)

    try:
        with open(target_abs_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        visitor = CodeMapVisitor()
        visitor.visit(tree)
        updated_registry = dict(state["file_registry"])
        updated_registry[target_file] = {
            "chunks": visitor.chunks,
            "imports": visitor.imports,
            "abs_path": target_abs_path
        }
        print(f"[Reassembler]: Re-indexed '{target_file}' — {len(visitor.chunks)} chunks mapped.")
    except SyntaxError as e:
        print(f"[Reassembler WARNING]: Re-index failed: {e}")
        updated_registry = state["file_registry"]

    return {
        "pending_edits": [],
        "current_code": source,
        "file_registry": updated_registry
    }
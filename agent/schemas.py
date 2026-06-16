# schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Union, Optional
import libcst as cst
import textwrap
import ast

class PlannerOutput(BaseModel):
    plan: str = Field(..., description="High-level execution plan describing exactly what changes to make.")
    target_functions: List[str] = Field(..., description="The EXACT functions or class names as it appears in the code above. Do not invent names.")
    rationale: str = Field(..., description="Why this specific location was chosen to fix the issue.")

class IssueParserOutput(BaseModel):
    issue_id: str = Field(description="A unique issue ID in the format ISSUE-XXX.")
    issue_title: str = Field(description="Concise title describing the bug(s) found.")
    issue_body: str = Field(description="Detailed description of the bugs, which functions are affected, and what the correct behavior should be.")
    imports : List[str] = Field(description = "List of names of depencencies to install")

class Edit(BaseModel):
    file_path: str = Field(
        description="Relative path to the file being edited."
    )
    node_type: Literal["function", "class", "method"] = Field(
        description=(
            "Type of node to replace. "
            "'function' for top-level functions. "
            "'class' for entire class definitions. "
            "'method' for a method inside a class."
        )
    )
    node_path: str = Field(
        description=(
            "For functions: just the name e.g. 'my_function'. "
            "For classes: just the name e.g. 'MyClass'. "
            "For methods: 'ClassName.method_name' e.g. 'MyClass.process'."
        )
    )
    new_implementation: str = Field(
        description=(
            "Complete new implementation of the node. "
            "Must be syntactically valid Python with correct indentation. "
            "Include decorators if the original had them."
        )
    )
    add_imports: List[str] = Field(
        default=[],
        description="New import lines to add if the fix requires new dependencies e.g. ['import re', 'from typing import Optional']."
    )
    rationale: str = Field(description="Why this change was made.")

class EngineerOutput(BaseModel):
    edits: List[Edit] = Field(description="List of surgical node-replacement patches.")

class TestGeneratorOutput(BaseModel):
    test_code: str = Field(description="Complete pytest test file contents. Raw Python only, no markdown.")
    test_functions: List[str] = Field(description="List of test function names included in the test file.")

class CriticOutput(BaseModel):
    is_resolved: bool = Field(
        description="Set to True only if the execution log explicitly shows all tests passed with a zero exit code."
    )
    feedback: str = Field(
        description="If tests failed, provide a merciless breakdown of the traceback and actionable advice for the next iteration. If passed, leave an approval note."
    )
    

class CodeMapVisitor(ast.NodeVisitor):
    def __init__(self):
        self.chunks = []
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.chunks.append({
            "name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "type": "function"
        })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.chunks.append({
            "name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "type": "async_function"
        })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.chunks.append({
            "name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "type": "class"
        })
        self.generic_visit(node)

class DependencyManifest(BaseModel):
    target_file: str
    read_only_files: Dict[str, str]  # Path: Content snippet (stubs/signatures)
    write_enabled_files: List[str]   # Paths the Engineer is allowed to edit

class FunctionReplacer(cst.CSTTransformer):
    """Replaces a top-level function or a method inside a specific class."""

    def __init__(self, target_name: str, new_code: str, class_name: Optional[str] = None):
        self.target_name = target_name
        self.new_code = new_code
        self.class_name = class_name
        self.found = False
        self._current_class: Optional[str] = None

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._current_class = node.name.value
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self._current_class = None
        return updated_node

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> Union[cst.FunctionDef, cst.BaseStatement]:
        name_matches = original_node.name.value == self.target_name

        if self.class_name:
            context_matches = self._current_class == self.class_name
        else:
            context_matches = self._current_class is None

        if name_matches and context_matches:
            self.found = True
            return cst.parse_statement(textwrap.dedent(self.new_code))

        return updated_node


class ClassReplacer(cst.CSTTransformer):
    """Replaces an entire top-level class definition."""

    def __init__(self, target_name: str, new_code: str):
        self.target_name = target_name
        self.new_code = new_code
        self.found = False

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> Union[cst.ClassDef, cst.BaseStatement]:
        if original_node.name.value == self.target_name:
            self.found = True
            return cst.parse_statement(textwrap.dedent(self.new_code))
        return updated_node


def apply_edit_to_source(source: str, edit: "Edit") -> str:
    """
    Applies a single Edit to source using libcst.
    Returns the modified source string.
    """
    module = cst.parse_module(source)

    if edit.node_type == "class":
        transformer = ClassReplacer(edit.node_path, edit.new_implementation)

    elif edit.node_type == "method":
        parts = edit.node_path.split(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"node_path for method must be 'ClassName.method_name', got '{edit.node_path}'"
            )
        class_name, method_name = parts
        transformer = FunctionReplacer(method_name, edit.new_implementation, class_name=class_name)

    else:  # function
        transformer = FunctionReplacer(edit.node_path, edit.new_implementation)

    new_module = module.visit(transformer)

    if not transformer.found:
        raise ValueError(
            f"Node '{edit.node_path}' (type: {edit.node_type}) not found in source."
        )

    # Handle imports
    result = new_module.code
    if edit.add_imports:
        result = _prepend_missing_imports(result, edit.add_imports)

    return result


def _prepend_missing_imports(source: str, imports: List[str]) -> str:
    """Adds import lines that aren't already present in the file."""
    existing = source.splitlines()
    to_add = [imp for imp in imports if imp not in existing]

    if not to_add:
        return source

    # Insert after the last existing import block
    last_import_line = 0
    for i, line in enumerate(existing):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_line = i + 1

    new_lines = existing[:last_import_line] + to_add + existing[last_import_line:]
    return "\n".join(new_lines)
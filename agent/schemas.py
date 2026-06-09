# schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict
import ast
class PlannerOutput(BaseModel):
    plan: str = Field(..., description="High-level execution plan.")
    target_file: str = Field(..., description="The path of the file to edit.")
    target_function: str = Field(..., description="The specific function or class name to target.")
    rationale: str = Field(..., description="Why this specific location was chosen.")

class IssueParserOutput(BaseModel):
    issue_id: str = Field(description="A unique issue ID in the format ISSUE-XXX.")
    issue_title: str = Field(description="Concise title describing the bug(s) found.")
    issue_body: str = Field(description="Detailed description of the bugs, which functions are affected, and what the correct behavior should be.")

class Edit(BaseModel):
    old_content: str = Field(
        description=(
            "The EXACT text currently in the file to be replaced. "
            "Must match character-for-character including indentation and whitespace. "
            "Include enough surrounding lines to be unambiguous — never just one short line."
        )
    )
    new_content: str = Field(
        description="The replacement text with correct indentation."
    )
    rationale: str = Field(
        description="Why this change was made."
    )

class EngineerOutput(BaseModel):
    explanation: str = Field(description="Concise summary of all fixes applied.")
    edits: List[Edit] = Field(description="List of surgical search/replace patches.")

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
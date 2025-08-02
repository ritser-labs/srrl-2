from typing import Dict, List, Any, Optional
import json
import subprocess
import tempfile
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import traceback
from datasets import load_dataset

def load_mbpp_dataset(split: str = "train", max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MBPP dataset from HuggingFace."""
    # Use the sanitized version which has cleaner test cases
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    
    problems = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
            
        problem = {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "code": item["code"],
            "test_list": item["test_list"],
            "test_imports": item.get("test_imports", [])
        }
        problems.append(problem)
    
    return problems


def execute_code_with_tests(code: str, test_list: List[str], test_imports: List[str] = None, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute code with test cases and return execution results.
    
    Returns:
        - success: bool - whether all tests passed
        - error: str - error message if execution failed
        - trace: str - execution trace/output
        - failed_tests: List[str] - list of failed test cases
    """
    if test_imports is None:
        test_imports = []
    
    # Create complete code with imports and tests
    full_code = "\n".join(test_imports) + "\n" if test_imports else ""
    full_code += code + "\n"
    
    # Add test cases
    test_code = []
    for test in test_list:
        test_code.append(f"try:\n    {test}\n    print(f'PASS: {test}')\nexcept Exception as e:\n    print(f'FAIL: {test} - {{str(e)}}')")
    
    full_code += "\n".join(test_code)
    
    # Execute in a temporary file for safety
    result = {
        "success": False,
        "error": "",
        "trace": "",
        "failed_tests": [],
        "passed_tests": []
    }
    
    try:
        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code in a restricted globals environment
            globals_dict = {
                "__builtins__": __builtins__,
                "math": __import__("math"),
                "re": __import__("re"),
                "itertools": __import__("itertools"),
                "collections": __import__("collections"),
                "heapq": __import__("heapq"),
                "operator": __import__("operator"),
            }
            
            exec(full_code, globals_dict)
        
        # Parse output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        result["trace"] = stdout_output + stderr_output
        
        # Parse test results
        for line in stdout_output.split('\n'):
            if line.startswith('PASS: '):
                test_case = line[6:]
                result["passed_tests"].append(test_case)
            elif line.startswith('FAIL: '):
                # Extract test case and error
                fail_part = line[6:]
                if ' - ' in fail_part:
                    test_case, error = fail_part.split(' - ', 1)
                    result["failed_tests"].append(f"{test_case}: {error}")
                else:
                    result["failed_tests"].append(fail_part)
        
        result["success"] = len(result["failed_tests"]) == 0 and len(result["passed_tests"]) > 0
        
        if stderr_output:
            result["error"] = stderr_output
            
    except Exception as e:
        result["error"] = str(e)
        result["trace"] = traceback.format_exc()
        result["failed_tests"] = test_list  # All tests failed due to execution error
    
    return result


def create_refinement_prompt(original_prompt: str, failed_code: str, execution_result: Dict[str, Any]) -> str:
    """Create a refined prompt with execution feedback for failed code attempts."""
    
    refinement_prompt = f"""The following code attempt failed to solve the problem correctly:

Original Problem:
{original_prompt}

Failed Code:
```python
{failed_code}
```

Execution Result:
"""
    
    if execution_result["error"]:
        refinement_prompt += f"Error: {execution_result['error']}\n"
    
    if execution_result["failed_tests"]:
        refinement_prompt += f"Failed Tests:\n"
        for test in execution_result["failed_tests"]:
            refinement_prompt += f"- {test}\n"
    
    if execution_result["trace"]:
        refinement_prompt += f"\nExecution Trace:\n{execution_result['trace']}\n"
    
    refinement_prompt += """
Please analyze the error and provide a corrected solution that fixes the issues and passes all test cases.

Write a corrected Python function that solves the problem:"""
    
    return refinement_prompt


def calculate_code_reward(execution_result: Dict[str, Any]) -> float:
    """Calculate reward based on code execution results."""
    if execution_result["success"]:
        return 1.0
    
    # Partial reward based on number of tests passed
    total_tests = len(execution_result["passed_tests"]) + len(execution_result["failed_tests"])
    if total_tests > 0:
        passed_tests = len(execution_result["passed_tests"])
        return passed_tests / total_tests
    
    # No tests passed, but code ran without syntax errors
    if not execution_result["error"] or "SyntaxError" not in execution_result["error"]:
        return 0.1
    
    # Syntax error or other execution failure
    return 0.0


def extract_code_from_completion(completion: str) -> str:
    """Extract Python code from model completion."""
    # Look for code blocks
    if "```python" in completion:
        start = completion.find("```python") + 9
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    
    if "```" in completion:
        start = completion.find("```") + 3
        end = completion.find("```", start)
        if end != -1:
            return completion[start:end].strip()
    
    # If no code blocks, try to extract function definition
    lines = completion.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
            code_lines.append(line)
        elif in_function:
            if line.strip() == "" or line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
            else:
                # Function likely ended
                break
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # Fallback: return the whole completion
    return completion.strip() 
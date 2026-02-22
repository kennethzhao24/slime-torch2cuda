import ast
import json
import time
import traceback
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional


def remove_func_from_pytorch_code(pytorch_code: str, func_name: str):
    """
    Remove the <func_name> in <pytorch_code>.
    Return: pytorch_code_after_removal, removed_pytorch_func
    
    <pytorch_code_after_removal> is the pytorch code after removing the function <func_name>
    <removed_pytorch_func> is the code that being removed from pytorch_code

    Args:
    """
    if not isinstance(pytorch_code, str):
        raise TypeError("pytorch_code must be a string")
    if not isinstance(func_name, str):
        raise TypeError("func_name must be a string")

    try:
        tree = ast.parse(pytorch_code)
    except SyntaxError as exc:
        raise ValueError(f"pytorch_code is not valid python code: {exc}") from exc

    target_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            if node.lineno is None or node.end_lineno is None:
                raise ValueError(f"Cannot determine source range for function {func_name}")
            target_ranges.append((node.lineno, node.end_lineno))

    if not target_ranges:
        raise ValueError(f"Function {func_name} not found in provided code")

    lines = pytorch_code.splitlines()
    had_trailing_newline = pytorch_code.endswith("\n")
    removed_blocks: list[str] = []

    # Remove from bottom-up to keep indexes stable; trim blank lines just above the target.
    for start, end in sorted(target_ranges, key=lambda r: r[0], reverse=True):
        start_idx = start - 1
        while start_idx > 0 and lines[start_idx - 1].strip() == "":
            start_idx -= 1
        removed_slice = lines[start_idx:end]
        removed_blocks.append("\n".join(removed_slice) + ("\n" if had_trailing_newline else ""))
        del lines[start_idx:end]

    new_code = "\n".join(lines)
    if had_trailing_newline and new_code:
        new_code += "\n"
    removed_pytorch_func = "".join(reversed(removed_blocks))
    return new_code, removed_pytorch_func
    

def extract_tag(content: str, tag_begin: str, tag_end: str) -> str:
    results = []
    start = 0
    while True:
        b = content.find(tag_begin, start)
        if b == -1:
            break
        b += len(tag_begin)
        e = content.find(tag_end, b)
        if e == -1:
            break
        results.append(content[b:e])
        start = e + len(tag_end)
    return ''.join(results)

def load_jsonl(fn: str) -> List[Dict]:
    with open(fn, "r") as fid:
        lines = fid.readlines()
    lines = [x.strip() for x in lines]
    data = [json.loads(x) for x in lines if len(x) > 0]
    return data

def save_jsonl(obj_list: List[Dict], fn: str):
    with open(fn, "w") as fid:
        for obj in obj_list:
            fid.write(json.dumps(obj) + "\n")
    

def set_debug_breakpoint():
    import debugpy, os    
    debug_port = int(os.environ.get("PYTHON_DEBUG_PORT", 15678))
    host = '0.0.0.0'
    try:
        debugpy.listen((host, debug_port))  # Attach to this port
        print(f"Debugger listening at {host}:{debug_port}, waiting for attach...")
        debugpy.wait_for_client()
    except:
        print(f"Port {debug_port} is already in use. Debugger will not listen.")
        time.sleep(100000)



def run_func_in_subprocess(func: Callable[..., Any], kwargs: Dict[str, Any], timeout: Optional[float] = None) -> Any:
    """
    Execute ``func(**kwargs)`` in a child process and return its result.

    The call blocks until the function returns or the optional ``timeout`` is reached.
    If the child raises, the stack trace is re-raised as a RuntimeError. If the timeout
    elapses, a TimeoutError is raised and the child is terminated.
    """
    if not callable(func):
        raise TypeError("func must be callable")
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dict")
    if timeout is not None:
        if not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number or None")
        if timeout <= 0:
            raise ValueError("timeout must be positive when provided")

    result_queue: Queue = Queue(maxsize=1)

    def _worker(target_func: Callable[..., Any], target_kwargs: Dict[str, Any]):
        try:
            result_queue.put(("result", target_func(**target_kwargs)))
        except Exception:
            result_queue.put(("error", traceback.format_exc()))

    process = Process(target=_worker, args=(func, kwargs))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"Function execution exceeded timeout of {timeout} seconds")
        return None

    if result_queue.empty():
        result_queue.close()
        result_queue.join_thread()
        print("Subprocess exited without returning a result")
        return None

    status, payload = result_queue.get()
    result_queue.close()
    result_queue.join_thread()

    if status == "result":
        return payload
    if status == "error":
        print(f"Function raised an exception in subprocess:\n{payload}")
        return None

    print(f"Unexpected status from subprocess: {status}")
    return None
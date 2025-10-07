import os
import sys
import subprocess
from ..scanner import NodeScanner

class FL_WF_Agent:
    """
    A node that uses Gemini AI to generate and execute JavaScript code for workflow manipulation
    """
    def __init__(self):
        # Initialize scanner
        self.scanner = NodeScanner(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        print("FL_WF_Agent: Scanner initialized")
        
        # Automatically run the scanner once at startup to ensure cache file exists
        try:
            print("FL_WF_Agent: Running initial node scan to create cache...")
            self.scanner.scan_nodes()
            print("FL_WF_Agent: Initial node scan completed")
        except Exception as e:
            print(f"FL_WF_Agent: Error during initial scan: {str(e)}")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "event": (["before_queued", "after_queued", "status", "progress", "executing",
                          "executed", "execution_start", "execution_error", "execution_success",
                          "execution_cached"], {"default": "before_queued"}),
                "code_prompt": ("STRING", {"multiline": True, "default": "Enter your code generation prompt here"}),
                "api_key": ("STRING", {"default": ""}),
                "javascript": ("STRING", {"default": "// Generated code will appear here", "multiline": True}),
            },
            "optional": {
                "scan_nodes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "exec_entrypoint"
    OUTPUT_NODE = True
    CATEGORY = "🏵️Fill Nodes/WIP"

    def exec_entrypoint(self, event, code_prompt, api_key, javascript, scan_nodes=False):
        try:
            if scan_nodes:
                print("\nFL_WF_Agent: Starting scanner execution...")
                
                # Get absolute path to scanner.py
                scanner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner.py")
                print(f"Scanner path: {scanner_path}")
                
                # Get cache directory and expected file path for the node definitions
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web', 'nodes')
                cache_file = os.path.join(cache_dir, 'node_definitions.txt')
                
                print(f"Expected cache directory: {cache_dir}")
                print(f"Expected cache file: {cache_file}")
                print(f"Cache directory exists: {os.path.exists(cache_dir)}")
                print(f"Cache file exists before scan: {os.path.exists(cache_file)}")
                
                # Execute scanner.py as subprocess with proper Python interpreter
                process = subprocess.Popen(
                    [sys.executable, scanner_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True  # Get string output instead of bytes
                )
                
                stdout, stderr = process.communicate()
                
                # Check if file exists after scan
                print(f"Cache file exists after scan: {os.path.exists(cache_file)}")
                if os.path.exists(cache_file):
                    print(f"Cache file size: {os.path.getsize(cache_file)} bytes")
                    print(f"Cache file absolute path: {os.path.abspath(cache_file)}")
                    # Get the first few lines to confirm content
                    try:
                        with open(cache_file, 'r') as f:
                            first_lines = [next(f) for _ in range(5)]
                        print(f"First few lines of cache file: {first_lines}")
                    except Exception as e:
                        print(f"Error reading cache file: {str(e)}")
                
                if process.returncode == 0:
                    print("Scanner completed successfully")
                    scan_feedback = {
                        "success": True,
                        "message": "Node scan completed successfully!",
                        "stdout": stdout,
                        "cache_path": os.path.abspath(cache_file),
                        "cache_exists": os.path.exists(cache_file),
                        "cache_size": os.path.getsize(cache_file) if os.path.exists(cache_file) else 0
                    }
                else:
                    print(f"Scanner failed with return code: {process.returncode}")
                    scan_feedback = {
                        "success": False,
                        "message": f"Scan failed with error: {stderr}",
                        "stderr": stderr,
                        "cache_path": os.path.abspath(cache_file) if os.path.exists(cache_file) else "N/A",
                        "cache_exists": os.path.exists(cache_file)
                    }
                
                # Print all output for debugging
                print("\nScanner stdout:")
                print(stdout)
                if stderr:
                    print("\nScanner stderr:")
                    print(stderr)
                
                return {"ui": {"scan_feedback": scan_feedback}}
            
            return ()
                
        except Exception as e:
            import traceback
            print(f"FL_WF_Agent ERROR: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return {"ui": {"scan_feedback": {
                "success": False,
                "message": f"Scan failed: {str(e)}"
            }}}
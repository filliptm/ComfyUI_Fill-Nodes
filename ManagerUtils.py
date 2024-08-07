import os
import json

def get_workflows_directory():
    # Change this to the desired location within your ComfyUI directory
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_WorkflowManager")

def get_workflow_path(name):
    workflows_dir = get_workflows_directory()
    return os.path.abspath(os.path.join(workflows_dir, f"{name}.json"))

def save_workflow(name, workflow_data):
    file_path = get_workflow_path(name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(workflow_data, f, indent=2)

def load_workflow(name):
    file_path = get_workflow_path(name)
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)
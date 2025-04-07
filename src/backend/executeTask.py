
import re

def handle_task(node_info, task_data):
    task_type = task_data.get("task_type")
    print(f"🧩 Received task: {task_type}")

    if task_type == "llm_training":
        # Simulate LLM training task
        model_name = task_data.get("model_name")
        hyperparameters = task_data.get("hyperparameters", {})
        data = task_data.get("data", {})

        print(f"🚀 Training {model_name} with hyperparameters {hyperparameters}")
        print(f"📦 Data: {data}")

        # Here, you would call your actual ML training function

        return {"status": "success", "message": f"Training {model_name} completed."}

    else:
        return {"status": "error", "message": "Unsupported task type."}

def validate_task_data(task_data):
    required_fields = ["task_type", "model_name", "data", "hyperparameters", "response_required"]
    missing_fields = [field for field in required_fields if field not in task_data]

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    # Optionally: Check types
    if not isinstance(task_data["task_type"], str):
        return False, "Invalid type for 'task_type'"
    if not isinstance(task_data["model_name"], str):
        return False, "Invalid type for 'model_name'"
    if not isinstance(task_data["data"], dict):
        return False, "Invalid type for 'data'"
    if not isinstance(task_data["hyperparameters"], dict):
        return False, "Invalid type for 'hyperparameters'"
    if not isinstance(task_data["response_required"], bool):
        return False, "Invalid type for 'response_required'"

    return True, None
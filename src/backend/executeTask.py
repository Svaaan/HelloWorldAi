

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

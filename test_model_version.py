from baseline_model import load_model_artifact


artifact = load_model_artifact()
metadata = artifact["metadata"]
print(metadata.get("version", "unknown"))
print(metadata.get("model_type", "unknown"))
print(metadata.get("trained_at", "unknown"))
print(metadata.get("threshold", "unknown"))
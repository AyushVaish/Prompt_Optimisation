import json
import datetime

def log_to_history(entry: dict):
    """
    Append a JSON-line to history.jsonl with a timestamp.
    """
    entry["timestamp"] = datetime.datetime.now().isoformat()
    with open("history.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
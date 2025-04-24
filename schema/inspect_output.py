# filepath: c:\Users\Linds\Repos\East_River\scripts\inspect_output.py
import json

fn = r"c:\Users\Linds\Repos\East_River\schema.json"
with open(fn) as f:
    s = json.load(f)

print(f"Found {len(s.get('properties', {}))} properties")
print("Sample keys:", list(s.get('properties', {}))[:10])
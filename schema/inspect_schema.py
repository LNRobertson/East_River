import pandas as pd
import json

DATA_PATH = r"c:\Users\Linds\Repos\East_River\data\training\east_river_training-v2.h5"
OUT_PATH  = r"c:\Users\Linds\Repos\East_River\schema\generated_schema.json"

df = pd.read_hdf(DATA_PATH, key="df", stop=1)

schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "East River Training v2 Input Schema",
    "description": "Each row in east_river_training-v1.h5 (key=\"df\").",
    "type": "object",
    "properties": {},
    "required": ["local_time","location"],
    "additionalProperties": False
}

for col, dt in df.dtypes.items():
    if col == "local_time":
        schema["properties"][col] = {
            "type":"string","format":"date-time",
            "description":"Observation timestamp"
        }
    elif col == "last_control_time":
        schema["properties"][col] = {
            "type":["string","null"],"format":"date-time",
            "description":"Last control action timestamp"
        }
    elif col == "location":
        schema["properties"][col] = {
            "type":"string","description":"Location identifier"
        }
    else:
        schema["properties"][col] = {
            "type":["number","null"],
            "description":f"{dt.name} feature '{col}'"
        }

# write & report
with open(OUT_PATH, "w") as f:
    json.dump(schema, f, indent=2)
print(f"Wrote {len(schema['properties'])} properties to {OUT_PATH}")
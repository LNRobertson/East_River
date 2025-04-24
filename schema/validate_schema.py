# filepath: c:\Users\Linds\Repos\East_River\scripts\validate_schema.py
import pandas as pd
import json
from jsonschema import validate, ValidationError

# 1) load one record
df = pd.read_hdf(
    r"c:\Users\Linds\Repos\East_River\data\training\east_river_training-v2.h5",
    key="df", stop=1
)
record = df.iloc[0].to_dict()

# 2) coerce to match schema
lt = record.get('local_time')
if pd.notna(lt):
    try:
        record['local_time'] = lt.isoformat()
    except AttributeError:
        # fallback for floats or numpy types
        record['local_time'] = pd.to_datetime(lt).isoformat()

lct = record.get('last_control_time')
if pd.notna(lct):
    try:
        record['last_control_time'] = lct.isoformat()
    except AttributeError:
        record['last_control_time'] = pd.to_datetime(lct).isoformat()

# force location → string
record['location'] = str(record['location'])

# 3) load schema
with open(r"c:\Users\Linds\Repos\East_River\schema\generated_schema.json") as f:
    schema = json.load(f)

# 4) validate
try:
    validate(instance=record, schema=schema)
    print("✅ Record is valid!")
except ValidationError as e:
    print("❌ Validation failed:", e)
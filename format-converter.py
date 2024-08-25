import csv
import json
import os

input_file = "spup01.jsonl"

base_name = os.path.splitext(input_file)[0]
json_output_file = f"{base_name}.json"
csv_output_file = f"{base_name}.csv"

data = []
with open(input_file, "r") as f:
    for line in f:
        data.append(json.loads(line))

with open(json_output_file, "w") as f:
    json.dump(data, f, indent=4)

with open(csv_output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "file_name"])
    writer.writeheader()
    writer.writerows(data)

for entry in data:
    text_content = entry["text"]
    file_name = entry["file_name"].replace(".jpg", ".txt")
    with open(file_name, "w") as f:
        f.write(text_content)

"""Send one example student to the running service and print the prediction.

Usage:
    python test_request.py                                   # uses examples/graduate.json, localhost:9696
    python test_request.py examples/dropout.json
    python test_request.py examples/dropout.json https://<your-service-url>/predict

Uses only the standard library, so it runs with any Python 3.
"""

from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen

path = sys.argv[1] if len(sys.argv) > 1 else "examples/graduate.json"
url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9696/predict"

with open(path, encoding="utf-8") as f:
    student = json.load(f)

request = Request(url, data=json.dumps(student).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
with urlopen(request) as response:
    result = json.load(response)

assert result["prediction"] in {"Dropout", "Enrolled", "Graduate"}
assert abs(sum(result["probabilities"].values()) - 1) < 0.01
print(json.dumps(result, indent=2))

import os
import re

PATTERNS = [
    (r"return\s+score\b", "Suspicious raw return of 'score'"),
    (r"return\s+reward\b", "Suspicious raw return of 'reward'"),
    (r"max\(0,\s*min\(1,", "Ad-hoc clamping logic found. Use to_open_unit_interval instead."),
    (r"round\(.*,\s*\d+\)", "Rounding used in logic? check if it's for display only."),
]

def audit():
    print("Auditing codebase for unsafe scoring paths...")
    found_issues = 0
    
    for root, dirs, files in os.walk("."):
        # Skip some dirs
        if any(d in root for d in [".git", "__pycache__", "venv", ".gemini"]):
            continue
            
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        for pattern, msg in PATTERNS:
                            if re.search(pattern, line):
                                # Exception for to_open_unit_interval implementation itself
                                if "to_open_unit_interval" in line and "def" in line:
                                    continue
                                # Exception for src/tasks.py specifically if it's expected
                                
                                print(f"  [{file}:{i+1}] {msg}")
                                print(f"    Line: {line.strip()}")
                                found_issues += 1
                                
    if found_issues == 0:
        print("No suspicious raw returns found.")
    else:
        print(f"\nAudit finished with {found_issues} potential issues.")

if __name__ == "__main__":
    audit()

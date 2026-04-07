import subprocess
import re
import sys

def verify_output():
    print("Running inference.py to verify validator output...")
    try:
        # Run inference.py and capture stdout
        result = subprocess.run(
            [sys.executable, "inference.py", "--level", "easy", "--seed", "42"],
            capture_output=True,
            text=True,
            timeout=60
        )
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"[ERROR] inference.py failed with exit code {result.returncode}")
            print(f"Stderr: {stderr}")
            return False

        # Find all [END] lines
        end_lines = [line for line in stdout.splitlines() if "[END]" in line]
        
        if not end_lines:
            print("[ERROR] No [END] lines found in output.")
            return False

        all_passed = True
        for line in end_lines:
            print(f"Checking line: {line}")
            # Extract score using regex
            match = re.search(r"score=([0-9.]+)", line)
            if not match:
                print(f"  [FAIL] Could not extract score from line.")
                all_passed = False
                continue
            
            score_str = match.group(1)
            score = float(score_str)
            
            print(f"  Extracted score: {score_str} ({score})")
            
            # Assertions
            is_in_range = 0.01 <= score <= 0.99
            is_strictly_open = 0.0 < score < 1.0
            
            if not is_strictly_open:
                print(f"  [FAIL] Score {score} is NOT strictly within (0, 1)!")
                all_passed = False
            elif not is_in_range:
                print(f"  [WARNING] Score {score} is outside the defensive [0.01, 0.99] range.")
                all_passed = False
            else:
                print(f"  [PASS] Score is safe.")

            # Check precision
            if len(score_str.split('.')[-1]) < 6:
                print(f"  [WARNING] Score {score_str} has less than 6 decimal places of precision.")
                # We won't fail strictly on this if the value is like 0.500000 -> 0.5, 
                # but our formatter should produce 0.500000.
                # Wait, f"{safe_final:.6f}" will produce "0.500000".
                if score_str != f"{score:.6f}":
                    print(f"  [FAIL] Expected precision .6f, got {score_str}")
                    all_passed = False

        if all_passed:
            print("\n[SUCCESS] All validator output requirements are met.")
            return True
        else:
            print("\n[FAILURE] Validator output requirements failed.")
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] inference.py timed out.")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if verify_output():
        sys.exit(0)
    else:
        sys.exit(1)

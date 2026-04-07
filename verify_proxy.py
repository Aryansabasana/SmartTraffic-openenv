import os
import sys

def verify_proxy():
    """Minimal test to verify OpenAI client initialization via proxy env vars."""
    print("--- LLM Proxy Configuration Check ---")
    
    api_url = os.environ.get("API_BASE_URL", "NOT_SET")
    api_key = os.environ.get("API_KEY", "NOT_SET")
    openai_key = os.environ.get("OPENAI_API_KEY", "NOT_SET")
    
    print(f"API_BASE_URL: {api_url}")
    print(f"API_KEY: {'[PRESENT]' if api_key != 'NOT_SET' else 'NOT_SET'}")
    print(f"OPENAI_API_KEY: {'[PRESENT]' if openai_key != 'NOT_SET' else 'NOT_SET'}")

    try:
        from openai import OpenAI
        print("OpenAI library: INSTALLED")
        
        # Priority mapping from the new agent.py logic
        final_key = api_key if api_key != 'NOT_SET' else (openai_key if openai_key != 'NOT_SET' else None)
        
        if not final_key:
            print("Status: FAIL (No API keys found in environment)")
            return False
            
        client = OpenAI(base_url=api_url, api_key=final_key)
        print(f"Status: PASS (Client initialized with base_url={api_url})")
        return True
        
    except ImportError:
        print("Status: FAIL (OpenAI library NOT found)")
        return False
    except Exception as e:
        print(f"Status: FAIL (Initialisation Error: {e})")
        return False

if __name__ == "__main__":
    verify_proxy()

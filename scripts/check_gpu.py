import subprocess
import time

def check_gpu_usage():
    """Check if GPU is being used (means training is active)"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        return "python" in result.stdout.lower()
    except:
        return None

def monitor_log_file(log_file, timeout=300):
    """Check if log file is being updated"""
    import os
    if not os.path.exists(log_file):
        return False
    
    initial_mtime = os.path.getmtime(log_file)
    time.sleep(timeout)
    current_mtime = os.path.getmtime(log_file)
    
    return current_mtime > initial_mtime

if __name__ == "__main__":
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/training.log"
    
    print(f"Monitoring {log_file}...")
    if not monitor_log_file(log_file, 60):
        print("⚠️  Training appears to be hung (no log updates in 60s)")
    else:
        print("✓ Training is active")

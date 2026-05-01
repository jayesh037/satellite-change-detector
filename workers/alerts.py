import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def trigger_alert(result_id: str, changed_area_km2: float, threshold_km2: float) -> bool:
    """
    Checks if the detected change area exceeds the threshold and triggers an alert if so.
    
    If the threshold is exceeded:
    - Logs the alert to outputs/alerts.log
    - Prints a formatted alert to the console
    
    Args:
        result_id (str): The ID of the DetectionResult.
        changed_area_km2 (float): The calculated changed area in square kilometers.
        threshold_km2 (float): The threshold area in square kilometers to trigger an alert.
        
    Returns:
        bool: True if an alert was triggered, False otherwise.
    """
    if changed_area_km2 <= threshold_km2:
        return False
        
    # Prepare the message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"ALERT: Significant change detected! Area: {changed_area_km2:.4f} km² (Exceeds threshold of {threshold_km2:.4f} km²). Result ID: {result_id}"
    
    # 1. Print formatted alert to console
    print(f"\n[{timestamp}] {message}\n")
    
    # 2. Log alert to outputs/alerts.log
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_file = outputs_dir / "alerts.log"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
        
    # 3. Add to in-memory alerts (importing here to avoid circular imports if needed, though TASKS_DB is in routes)
    # Actually, we will just rely on the log file as requested by the instructions "save results to outputs/ folder as JSON file instead of DB".
    # For alerts, returning True and logging is enough.
        
    return True
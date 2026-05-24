import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.inference import run_inference


def main() -> None:
    """
    Main function to parse arguments and execute the timeseries inference pipeline.
    
    Finds Sentinel-2 R10m directories for consecutive years and runs the change 
    detection inference model on each consecutive pair. Outputs results to 
    'outputs/timeseries' and writes a summary JSON.
    """
    parser = argparse.ArgumentParser(description="Run timeseries change detection inference.")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print pairs found without running inference."
    )
    parser.add_argument(
        "--pairs", 
        type=str, 
        help="Comma-separated list of pairs to run, e.g., 2021_2022,2022_2023"
    )
    args = parser.parse_args()

    root_dir = Path("data/ISRO")
    
    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist.")
        return

    # Find year subfolders
    years = sorted([d.name for d in root_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    r10m_paths: Dict[str, Path] = {}
    for year in years:
        paths = list(root_dir.glob(f"{year}/*/GRANULE/*/IMG_DATA/R10m"))
        if paths:
            r10m_paths[year] = paths[0]
        else:
            print(f"Warning: R10m directory not found for year {year}.")

    # Only keep years where R10m paths were found
    valid_years = sorted(list(r10m_paths.keys()))
    
    # Build consecutive pairs
    pairs = [(valid_years[i], valid_years[i+1]) for i in range(len(valid_years)-1)]
    
    # Filter pairs if --pairs flag is used
    if args.pairs:
        requested_pairs = args.pairs.split(",")
        filtered_pairs = []
        for p in requested_pairs:
            try:
                y1, y2 = p.split("_")
                if (y1, y2) in pairs:
                    filtered_pairs.append((y1, y2))
                elif y1 in r10m_paths and y2 in r10m_paths:
                    # Allow custom pairs if valid paths exist
                    filtered_pairs.append((y1, y2))
                else:
                    print(f"Warning: Requested pair {p} does not have valid paths.")
            except ValueError:
                print(f"Warning: Invalid pair format '{p}'. Expected 'YYYY_YYYY'.")
        pairs = filtered_pairs

    if args.dry_run:
        print("Dry run enabled. Found the following pairs:")
        for y1, y2 in pairs:
            print(f"  {y1} -> {y2}")
        return

    results_list: List[Dict[str, Any]] = []
    total = len(pairs)
    
    for i, (y1, y2) in enumerate(pairs):
        out_dir = Path("outputs/timeseries") / f"{y1}_{y2}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i+1}/{total}] Processing {y1} -> {y2}...")
        
        try:
            result = run_inference(
                t1_folder=str(r10m_paths[y1]),
                t2_folder=str(r10m_paths[y2]),
                checkpoint_path="checkpoints/best_model_finetuned.pth",
                output_dir=str(out_dir)
            )
            
            results_list.append({
                "period": f"{y1}_{y2}",
                "y1": y1,
                "y2": y2,
                "changed_area_km2": result["changed_area_km2"],
                "geojson_path": result["geojson_path"],
                "status": "success"
            })
            
            print(f"[{i+1}/{total}] {y1}→{y2}: {result['changed_area_km2']:.3f} km²")
            
        except Exception as e:
            print(f"[{i+1}/{total}] Error processing {y1} -> {y2}: {e}")
            results_list.append({
                "period": f"{y1}_{y2}",
                "y1": y1,
                "y2": y2,
                "status": "failed",
                "error": str(e)
            })

    # Write summary
    summary_path = Path("outputs/timeseries/summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "pairs_total": len(pairs),
        "results": results_list
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Timeseries inference complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

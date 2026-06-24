#!/usr/bin/env python3
import os
import argparse
import sys
import wfdb

# Mapping of dataset slugs to their descriptive names on PhysioNet
DATABASES = {
    "mitdb": "MIT-BIH Arrhythmia Database",
    "pwave": "MIT-BIH Arrhythmia Database P-Wave Annotations",
    "afdb": "MIT-BIH Atrial Fibrillation Database",
    "ltdb": "MIT-BIH Long-Term ECG Database",
    "svdb": "MIT-BIH Supraventricular Arrhythmia Database",
    "stdb": "MIT-BIH ST Change Database",
    "cdb": "MIT-BIH ECG Compression Database",
    "vfdb": "MIT-BIH Malignant Ventricular Ectopy Database",
    "nstdb": "MIT-BIH Noise Stress Test Database",
    "nsrdb": "MIT-BIH Normal Sinus Rhythm Database",
    "nsr2db": "Recordings excluded from MIT-BIH Normal Sinus Rhythm DB",
    "sddb": "Sudden Cardiac Death Holter Database",
    "adfecgdb": "Abdominal and Direct Fetal ECG Database",
    "nifecgdb": "Non-Invasive Fetal ECG Arrhythmia Database",
    "slpdb": "MIT-BIH Polysomnographic Database",
    "ecg-fragment-high-risk-label": "ECG Fragment Database for the Exploration of Dangerous Arrhythmia",
    "edb": "European ST-T Database"
}

def main():
    parser = argparse.ArgumentParser(
        description="Helper script to download MIT-BIH and other ECG datasets from PhysioNet using wfdb."
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported databases and their slugs."
    )
    
    parser.add_argument(
        "--db",
        type=str,
        help="The specific database slug to download (e.g., 'mitdb'). Pass 'all' to download everything."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("data", "raw"),
        help="Base directory where datasets will be saved (default: data/raw/)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the download directory."
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Databases on PhysioNet:")
        print("=" * 80)
        for slug, name in DATABASES.items():
            print(f"  {slug:<30} | {name}")
        print("=" * 80)
        return

    if not args.db:
        parser.print_help()
        print("\nError: Please specify a database to download via --db <slug> or list them via --list.")
        sys.exit(1)
        
    db_to_download = args.db.strip().lower()
    
    if db_to_download == "all":
        targets = list(DATABASES.keys())
        print(f"Preparing to download all {len(targets)} databases...")
    elif db_to_download in DATABASES:
        targets = [db_to_download]
    else:
        print(f"Error: Unknown database slug '{db_to_download}'.")
        print("Use --list to see the list of valid database slugs.")
        sys.exit(1)
        
    for slug in targets:
        name = DATABASES[slug]
        dl_path = os.path.join(args.output_dir, slug)
        os.makedirs(dl_path, exist_ok=True)
        
        print(f"\nDownloading: {name} ({slug})")
        print(f"Saving to: {os.path.abspath(dl_path)}")
        print("Downloading files (this may take a while depending on the database size)...")
        
        try:
            # wfdb.dl_database downloads the files from PhysioNet
            wfdb.dl_database(
                db_dir=slug,
                dl_dir=dl_path,
                overwrite=args.overwrite
            )
            print(f"Successfully downloaded {slug}!")
        except Exception as e:
            print(f"Error downloading {slug}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

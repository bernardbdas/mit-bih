"""
Database download utilities for PhysioNet ECG databases.

This module provides definitions for all supported databases on PhysioNet
and helper functions to download them using the wfdb library.
"""

import os
import wfdb

# Mapping of database slugs to their descriptive names on PhysioNet
DATABASES: dict[str, str] = {
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


def download_database(db_slug: str, output_dir: str, overwrite: bool = False) -> None:
    """
    Downloads an ECG database from PhysioNet using wfdb.

    Args:
        db_slug: The short identifier for the database (e.g., 'mitdb').
        output_dir: The directory where database files will be saved.
        overwrite: If True, existing files will be overwritten.

    Raises:
        ValueError: If the db_slug is not a known database in PhysioNet.
    """
    db_slug = db_slug.strip().lower()
    if db_slug not in DATABASES:
        raise ValueError(
            f"Unknown database slug '{db_slug}'. Supported slugs: {list(DATABASES.keys())}"
        )
        
    db_name = DATABASES[db_slug]
    dl_path = os.path.join(output_dir, db_slug)
    os.makedirs(dl_path, exist_ok=True)
    
    print(f"Downloading: {db_name} ({db_slug})")
    print(f"Saving to: {os.path.abspath(dl_path)}")
    
    # Trigger the PhysioNet download via wfdb
    wfdb.dl_database(
        db_dir=db_slug,
        dl_dir=dl_path,
        overwrite=overwrite
    )
    print(f"Successfully downloaded {db_slug}!")

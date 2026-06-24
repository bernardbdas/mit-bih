import argparse
import sys
from mit_bih.data import DATABASES, download_database

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIT-BIH Arrhythmia Project CLI"
    )
    
    # Optional subcommands
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    
    # download command
    dl_parser = subparsers.add_parser("download", help="Download datasets from PhysioNet")
    dl_parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="The specific database slug to download (e.g., 'mitdb'). Pass 'all' to download everything."
    )
    dl_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Base directory where datasets will be saved (default: data/raw/)"
    )
    dl_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the download directory."
    )
    dl_parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported databases and their slugs."
    )

    # Legacy options directly on the main command for backward compatibility
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
        default="data/raw",
        help="Base directory where datasets will be saved (default: data/raw/)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the download directory."
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list or (args.command == "download" and args.list):
        print("\nAvailable Databases on PhysioNet:")
        print("=" * 80)
        for slug, name in DATABASES.items():
            print(f"  {slug:<30} | {name}")
        print("=" * 80)
        return

    # Extract db parameter from either format
    db = args.db if args.db else (args.db if args.command == "download" else None)
    output_dir = args.output_dir
    overwrite = args.overwrite
    
    if not db:
        parser.print_help()
        sys.exit(1)
        
    db = db.strip().lower()
    if db == "all":
        targets = list(DATABASES.keys())
    elif db in DATABASES:
        targets = [db]
    else:
        print(f"Error: Unknown database slug '{db}'.")
        print("Use --list to see the list of valid database slugs.")
        sys.exit(1)
        
    for slug in targets:
        try:
            download_database(slug, output_dir, overwrite)
        except Exception as e:
            print(f"Error downloading {slug}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()

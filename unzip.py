import os
import zipfile
import argparse
from pathlib import Path
import shutil


def extract_all_zips(dataset_dir):
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"❌ Error: Directory '{dataset_dir}' not found!")
        return

    zip_files = list(dataset_path.glob("*.zip"))
    if not zip_files:
        print("⚠️ No .zip files found in the given directory.")
        return

    print(f"🔍 Found {len(zip_files)} zip files in '{dataset_dir}'")

    for zip_file in zip_files:
        print(f"📦 Extracting {zip_file.name} ...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract temporarily inside the same directory
                temp_dir = dataset_path / f"__temp_{zip_file.stem}"
                temp_dir.mkdir(exist_ok=True)

                zip_ref.extractall(temp_dir)

                # Move inner folders (like dataset1/, dataset2/) up one level
                for item in temp_dir.iterdir():
                    target_path = dataset_path / item.name
                    if target_path.exists():
                        print(f"⚠️ Skipping existing folder: {target_path}")
                        continue
                    shutil.move(str(item), str(target_path))

                # Delete temp folder after moving
                shutil.rmtree(temp_dir)

            print(f"✅ Done: {zip_file.name}")

        except zipfile.BadZipFile:
            print(f"❌ Skipped: {zip_file.name} is not a valid zip file!")

    print("\n🎉 All zip files extracted successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract all .zip files so their contents appear beside the zip files."
    )
    parser.add_argument("--path", required=True, help="Path to the folder containing .zip files")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.path)
    extract_all_zips(dataset_dir)


if __name__ == "__main__":
    main()

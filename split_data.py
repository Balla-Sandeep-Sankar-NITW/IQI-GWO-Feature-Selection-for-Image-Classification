import os
import shutil
from pathlib import Path
import argparse


def split_dataset(source_dir, split_info):
    source_path = Path(source_dir)
    categories = [d for d in source_path.iterdir() if d.is_dir()]

    # Create destination folders
    for split_name, _ in split_info:
        split_path = source_path / f"{split_name}_data"
        split_path.mkdir(exist_ok=True)

    print(f"Found {len(categories)} categories:")
    for category in categories:
        print(f"  - {category.name}")

    for category in categories:
        image_files = sorted([f for f in category.iterdir() if f.is_file()])
        total_images = len(image_files)

        split_indices = []
        cumulative = 0
        for _, ratio in split_info:
            cumulative += ratio
            split_indices.append(int(total_images * cumulative))

        prev_index = 0
        for i, (split_name, _) in enumerate(split_info):
            split_path = source_path / f"{split_name}_data" / category.name
            split_path.mkdir(parents=True, exist_ok=True)

            split_files = image_files[prev_index:split_indices[i]]
            for img in split_files:
                shutil.copy2(img, split_path / img.name)

            print(f"{category.name} -> {split_name}: {len(split_files)} images ({len(split_files)/total_images:.1%})")
            prev_index = split_indices[i]


def validate_splits(source_dir, split_info):
    print("\n" + "=" * 50)
    print("VALIDATING SPLITS")
    print("=" * 50)

    source_path = Path(source_dir)
    first_split = source_path / f"{split_info[0][0]}_data"
    categories = [d for d in first_split.iterdir() if d.is_dir()]

    for category in categories:
        print(f"\n{category.name}:")
        total = 0
        counts = {}

        for split_name, _ in split_info:
            split_dir = source_path / f"{split_name}_data" / category.name
            count = len(list(split_dir.iterdir()))
            counts[split_name] = count
            total += count

        for name, count in counts.items():
            print(f"  - {name}: {count} ({count/total:.1%})")
        print(f"  - Total: {total}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into multiple parts by ratio.")
    parser.add_argument("--path", required=True, help="Path to the dataset directory")
    parser.add_argument("--splits", type=int, required=True, help="Number of splits (e.g. 2 or 3)")
    parser.add_argument("--names", required=True, help="Comma-separated split names (e.g. train,val,test)")
    parser.add_argument("--ratios", required=True, help="Colon-separated ratios (e.g. 70:20:10 or 800/9:100/9)")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = args.path

    if not Path(source_dir).exists():
        print(f"Error: Directory '{source_dir}' not found!")
        return

    split_names = [n.strip() for n in args.names.split(",")]
    split_ratios = args.ratios.split(":")

    if len(split_names) != args.splits or len(split_ratios) != args.splits:
        print("Error: The number of split names or ratios doesn't match the number of splits.")
        return

    # Safely evaluate ratios like 800/9 or 70
    ratios = [eval(r) if "/" in r else float(r) for r in split_ratios]
    ratio_sum = sum(ratios)
    normalized_ratios = [r / ratio_sum for r in ratios]

    print("\nYour chosen configuration:")
    for name, r in zip(split_names, normalized_ratios):
        print(f"  - {name}: {r*100:.1f}%")

    split_info = list(zip(split_names, normalized_ratios))

    print("\nSplitting dataset...")
    split_dataset(source_dir, split_info)
    validate_splits(source_dir, split_info)

    print("\n✅ Dataset splitting completed successfully!")
    for name, _ in split_info:
        clean_name = name.strip()
        print(f"{clean_name.capitalize()} data: {Path(source_dir) / f'{clean_name}_data'}")


if __name__ == "__main__":
    main()

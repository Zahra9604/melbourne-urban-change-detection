# ============================================================
# split_dataset.py
#
# OSCD DATASET PREPARATION
#
# This script:
#
# 1. Reads OSCD all/train/test city lists
# 2. Splits original TRAIN into TRAIN + VALIDATION
# 3. Keeps original TEST unchanged
# 4. Creates Data directory
#
# 5. Creates:
#
#       Data/
#       ├── images/
#       │   ├── train/
#       │   ├── val/
#       │   └── test/
#       │
#       ├── labels/
#       │   ├── train/
#       │   ├── val/
#       │   └── test/
#       │
#       └── geojson/
#           ├── train/
#           ├── val/
#           └── test/
#
# 6. Finds and copies OSCD change-mask labels
#
# 7. Finds EXISTING OSCD GeoJSON:
#
#       C:\OSCD_dataset\Images\<city>\<city>.geojson
#
# 8. Finds EXISTING OSCD dates:
#
#       C:\OSCD_dataset\Images\<city>\dates.txt
#
# 9. Adds acquisition dates to the copied GeoJSON
#
# 10. Saves dates.txt together with each city's GeoJSON
#
# IMPORTANT:
#
# GeoJSON is NEVER created from the label TIFF.
#
# Existing OSCD GeoJSON is copied directly and only enriched
# with acquisition-date metadata.
#
# GEE downloading is NOT performed here.
# ============================================================


from pathlib import Path
import random
import shutil
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

OSCD_ROOT = Path(
    r"C:\OSCD_dataset"
)

IMAGES_ROOT = (
    OSCD_ROOT /
    "Images"
)

ALL_FILE = (
    IMAGES_ROOT /
    "all.txt"
)

TRAIN_FILE = (
    IMAGES_ROOT /
    "train.txt"
)

TEST_FILE = (
    IMAGES_ROOT /
    "test.txt"
)


# ============================================================
# LABEL ROOTS
# ============================================================

TRAIN_LABELS_ROOT = (
    OSCD_ROOT /
    "Train Labels"
)

TEST_LABELS_ROOT = (
    OSCD_ROOT /
    "Test Labels"
)


# ============================================================
# OUTPUT
# ============================================================

DATA_ROOT = Path(
    r"C:\earth-observation-change-detection\Data"
)

IMAGES_OUTPUT = (
    DATA_ROOT /
    "images"
)

LABELS_OUTPUT = (
    DATA_ROOT /
    "labels"
)

GEOJSON_OUTPUT = (
    DATA_ROOT /
    "geojson"
)


# ============================================================
# SETTINGS
# ============================================================

VAL_RATIO = 0.20

RANDOM_SEED = 42


# ============================================================
# READ CITY FILE
# ============================================================

def read_city_file(path):

    print()
    print("Reading city list:")
    print(f"  {path}")

    if not path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    cities = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Support:
            #
            # city1,city2,city3
            #
            # and:
            #
            # city1
            # city2
            # ------------------------------------------------

            parts = line.split(",")

            for city in parts:

                city = city.strip()

                if city:

                    cities.append(city)

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    cities = list(
        dict.fromkeys(cities)
    )

    print(
        f"  ✓ Cities found: {len(cities)}"
    )

    return cities


# ============================================================
# READ OSCD LISTS
# ============================================================

def read_oscd_lists():

    all_cities = read_city_file(
        ALL_FILE
    )

    original_train = read_city_file(
        TRAIN_FILE
    )

    original_test = read_city_file(
        TEST_FILE
    )

    return (
        all_cities,
        original_train,
        original_test
    )


# ============================================================
# CHECK DATASET LISTS
# ============================================================

def check_lists(
    all_cities,
    train_cities,
    test_cities
):

    all_set = set(
        all_cities
    )

    train_set = set(
        train_cities
    )

    test_set = set(
        test_cities
    )

    print("\n")
    print("=" * 70)
    print("OSCD DATASET CHECK")
    print("=" * 70)

    print(
        f"\nAll cities            : "
        f"{len(all_set)}"
    )

    print(
        f"Original train cities : "
        f"{len(train_set)}"
    )

    print(
        f"Original test cities  : "
        f"{len(test_set)}"
    )

    # --------------------------------------------------------
    # Train/test overlap
    # --------------------------------------------------------

    overlap = (
        train_set &
        test_set
    )

    if overlap:

        print(
            "\nWARNING: Train/test overlap:"
        )

        for city in sorted(overlap):

            print(
                f"    {city}"
            )

    else:

        print(
            "\n✓ No train/test overlap."
        )

    # --------------------------------------------------------
    # Missing from all.txt
    # --------------------------------------------------------

    missing = (
        train_set |
        test_set
    ) - all_set

    if missing:

        print(
            "\nWARNING: Cities missing "
            "from all.txt:"
        )

        for city in sorted(missing):

            print(
                f"    {city}"
            )

    else:

        print(
            "✓ all.txt contains "
            "all train/test cities."
        )


# ============================================================
# CREATE TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_split(
    original_train,
    original_test
):

    candidates = (
        original_train.copy()
    )

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        candidates
    )

    n_validation = max(
        1,
        round(
            len(candidates) *
            VAL_RATIO
        )
    )

    validation = (
        candidates[
            :n_validation
        ]
    )

    train = (
        candidates[
            n_validation:
        ]
    )

    # --------------------------------------------------------
    # Original OSCD test remains unchanged
    # --------------------------------------------------------

    test = (
        original_test.copy()
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    print("\n")
    print("=" * 70)
    print("CREATING DIRECTORIES")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test"
    ]:

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        (
            IMAGES_OUTPUT /
            split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        (
            LABELS_OUTPUT /
            split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # GeoJSON
        # ----------------------------------------------------

        (
            GEOJSON_OUTPUT /
            split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    print(
        "✓ Directories created."
    )


# ============================================================
# FIND LABEL
# ============================================================

def find_label(city):

    print(
        f"\nSearching label for: {city}"
    )

    possible_directories = [

        TRAIN_LABELS_ROOT /
        city /
        "cm",

        TEST_LABELS_ROOT /
        city /
        "cm"

    ]

    for directory in possible_directories:

        print(
            "  Checking:"
        )

        print(
            f"    {directory}"
        )

        if not directory.exists():

            print(
                "    Directory does not exist."
            )

            continue

        files = []

        for pattern in [
            "*.tif",
            "*.TIF",
            "*.tiff",
            "*.TIFF"
        ]:

            files.extend(
                directory.glob(pattern)
            )

        files = list(
            dict.fromkeys(files)
        )

        if not files:

            continue

        # ----------------------------------------------------
        # Prefer city-specific filename
        # ----------------------------------------------------

        city_matches = [

            file

            for file in files

            if city.lower()
            in file.name.lower()

        ]

        if city_matches:

            label = city_matches[0]

        else:

            label = files[0]

        print(
            "\n  ✓ Label found:"
        )

        print(
            f"    {label}"
        )

        return label

    print(
        "\n  ✗ No label found."
    )

    return None


# ============================================================
# COPY LABEL
# ============================================================

def copy_label(
    city,
    split,
    label_path
):

    destination = (

        LABELS_OUTPUT /
        split /
        f"{city}_cm.tif"

    )

    shutil.copy2(
        label_path,
        destination
    )

    print(
        "\n  ✓ Label copied:"
    )

    print(
        f"    {destination}"
    )

    return destination


# ============================================================
# FIND ORIGINAL OSCD GEOJSON
# ============================================================

def find_geojson(city):

    print(
        f"\nSearching original GeoJSON for: {city}"
    )

    city_directory = (
        IMAGES_ROOT /
        city
    )

    # --------------------------------------------------------
    # Expected:
    #
    # C:\OSCD_dataset\Images\abudhabi\abudhabi.geojson
    # --------------------------------------------------------

    expected_geojson = (
        city_directory /
        f"{city}.geojson"
    )

    print(
        "  Checking:"
    )

    print(
        f"    {expected_geojson}"
    )

    if expected_geojson.exists():

        print(
            "\n  ✓ GeoJSON found:"
        )

        print(
            f"    {expected_geojson}"
        )

        return expected_geojson

    # --------------------------------------------------------
    # Case-insensitive search
    # --------------------------------------------------------

    if city_directory.exists():

        geojson_files = []

        for pattern in [
            "*.geojson",
            "*.GeoJSON",
            "*.GEOJSON"
        ]:

            geojson_files.extend(
                city_directory.glob(pattern)
            )

        geojson_files = list(
            dict.fromkeys(
                geojson_files
            )
        )

        if geojson_files:

            city_matches = [

                file

                for file in geojson_files

                if file.stem.lower()
                == city.lower()

            ]

            if city_matches:

                geojson_path = (
                    city_matches[0]
                )

            else:

                geojson_path = (
                    geojson_files[0]
                )

            print(
                "\n  ✓ GeoJSON found:"
            )

            print(
                f"    {geojson_path}"
            )

            return geojson_path

    print(
        "\n  ✗ GeoJSON not found."
    )

    return None


# ============================================================
# FIND CITY DATES.TXT
# ============================================================

def find_dates_file(city):

    print(
        f"\nSearching acquisition dates for: {city}"
    )

    dates_path = (

        IMAGES_ROOT /
        city /
        "dates.txt"

    )

    print(
        "  Checking:"
    )

    print(
        f"    {dates_path}"
    )

    if dates_path.exists():

        print(
            "\n  ✓ dates.txt found:"
        )

        print(
            f"    {dates_path}"
        )

        return dates_path

    # --------------------------------------------------------
    # Case-insensitive fallback
    # --------------------------------------------------------

    city_directory = (
        IMAGES_ROOT /
        city
    )

    if city_directory.exists():

        for file in city_directory.iterdir():

            if (
                file.is_file()
                and
                file.name.lower()
                == "dates.txt"
            ):

                print(
                    "\n  ✓ dates.txt found:"
                )

                print(
                    f"    {file}"
                )

                return file

    print(
        "\n  ✗ dates.txt not found."
    )

    return None


# ============================================================
# READ DATES.TXT
# ============================================================

def read_city_dates(
    city,
    dates_path
):

    print(
        f"\nReading dates for: {city}"
    )

    with open(
        dates_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if len(lines) < 2:

        raise ValueError(
            f"\nExpected at least two "
            f"dates in:\n{dates_path}\n"
            f"Found: {len(lines)}"
        )

    # --------------------------------------------------------
    # OSCD dates.txt normally contains the two acquisition
    # dates. We take the first two non-empty lines.
    # --------------------------------------------------------

    date1 = lines[0]
    date2 = lines[1]

    # --------------------------------------------------------
    # Remove possible labels such as:
    #
    # date1: 2015-01-01
    # date2: 2017-01-01
    #
    # --------------------------------------------------------

    if ":" in date1:

        date1 = date1.split(
            ":",
            1
        )[1].strip()

    if ":" in date2:

        date2 = date2.split(
            ":",
            1
        )[1].strip()

    # --------------------------------------------------------
    # Remove possible commas
    # --------------------------------------------------------

    date1 = date1.strip().strip(",")
    date2 = date2.strip().strip(",")

    print(
        f"  Date 1: {date1}"
    )

    print(
        f"  Date 2: {date2}"
    )

    return {
        "date1": date1,
        "date2": date2
    }


# ============================================================
# LOAD GEOJSON
# ============================================================

def load_geojson(
    geojson_path
):

    with open(
        geojson_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# SAVE GEOJSON WITH DATES
# ============================================================

def save_geojson_with_dates(
    city,
    split,
    geojson_path,
    dates
):

    destination = (

        GEOJSON_OUTPUT /
        split /
        f"{city}.geojson"

    )

    # --------------------------------------------------------
    # Load EXISTING OSCD GeoJSON
    # --------------------------------------------------------

    geojson = load_geojson(
        geojson_path
    )

    # ========================================================
    # Add acquisition-date metadata
    #
    # This DOES NOT change the geometry.
    # This DOES NOT create polygons.
    # This DOES NOT use the label.
    # ========================================================

    geojson[
        "acquisition_dates"
    ] = {

        "date1":
            dates["date1"],

        "date2":
            dates["date2"]

    }

    # --------------------------------------------------------
    # Also add city metadata
    # --------------------------------------------------------

    geojson[
        "city"
    ] = city

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        destination,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            geojson,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\n  ✓ GeoJSON saved with dates:"
    )

    print(
        f"    {destination}"
    )

    return destination


# ============================================================
# COPY DATES.TXT
# ============================================================

def copy_dates_file(
    city,
    split,
    dates_path
):

    destination = (

        GEOJSON_OUTPUT /
        split /
        f"{city}_dates.txt"

    )

    shutil.copy2(
        dates_path,
        destination
    )

    print(
        "\n  ✓ dates.txt copied:"
    )

    print(
        f"    {destination}"
    )

    return destination


# ============================================================
# PROCESS ONE CITY
# ============================================================

def process_city(
    city,
    split
):

    print("\n")
    print("=" * 70)

    print(
        f"CITY: {city} | SPLIT: {split}"
    )

    print("=" * 70)

    # ========================================================
    # 1. FIND LABEL
    # ========================================================

    label_path = find_label(
        city
    )

    if label_path is None:

        print(
            f"\n✗ Cannot prepare {city}: "
            f"label missing."
        )

        return {
            "city": city,
            "split": split,
            "label": False,
            "geojson": False,
            "dates": False
        }

    # ========================================================
    # 2. COPY LABEL
    # ========================================================

    copy_label(
        city,
        split,
        label_path
    )

    # ========================================================
    # 3. FIND EXISTING GEOJSON
    # ========================================================

    geojson_path = find_geojson(
        city
    )

    if geojson_path is None:

        print(
            f"\n✗ GeoJSON not found for {city}"
        )

        return {
            "city": city,
            "split": split,
            "label": True,
            "geojson": False,
            "dates": False
        }

    # ========================================================
    # 4. FIND dates.txt
    # ========================================================

    dates_path = find_dates_file(
        city
    )

    if dates_path is None:

        print(
            f"\n✗ dates.txt not found for {city}"
        )

        return {
            "city": city,
            "split": split,
            "label": True,
            "geojson": False,
            "dates": False
        }

    # ========================================================
    # 5. READ DATES
    # ========================================================

    try:

        dates = read_city_dates(
            city,
            dates_path
        )

    except Exception as e:

        print(
            "\n✗ Could not read dates:"
        )

        print(
            f"  {e}"
        )

        return {
            "city": city,
            "split": split,
            "label": True,
            "geojson": False,
            "dates": False
        }

    # ========================================================
    # 6. SAVE EXISTING GEOJSON + DATES
    # ========================================================

    try:

        save_geojson_with_dates(
            city,
            split,
            geojson_path,
            dates
        )

    except Exception as e:

        print(
            "\n✗ Could not save GeoJSON:"
        )

        print(
            f"  {e}"
        )

        return {
            "city": city,
            "split": split,
            "label": True,
            "geojson": False,
            "dates": True
        }

    # ========================================================
    # 7. COPY dates.txt
    # ========================================================

    copy_dates_file(
        city,
        split,
        dates_path
    )

    # ========================================================
    # SUCCESS
    # ========================================================

    print(
        f"\n✓ {city} completed successfully."
    )

    return {
        "city": city,
        "split": split,
        "label": True,
        "geojson": True,
        "dates": True
    }


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    cities,
    split
):

    print("\n")
    print("=" * 70)

    print(
        f"PROCESSING {split.upper()}"
    )

    print("=" * 70)

    results = []

    for city in sorted(cities):

        result = process_city(
            city,
            split
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    successful = sum(
        1
        for r in results
        if (
            r["label"]
            and
            r["geojson"]
            and
            r["dates"]
        )
    )

    failed = (
        len(results) -
        successful
    )

    print("\n")
    print(
        f"{split.upper()} RESULT:"
    )

    print(
        f"  Successful: {successful}"
    )

    print(
        f"  Failed    : {failed}"
    )

    return results


# ============================================================
# SAVE CITY LIST
# ============================================================

def save_city_list(
    cities,
    filename
):

    path = (
        DATA_ROOT /
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for city in sorted(cities):

            f.write(
                city + "\n"
            )

    print(
        "\n✓ Saved:"
    )

    print(
        f"  {path}"
    )

    return path


# ============================================================
# SAVE SPLIT CSV
# ============================================================

def save_split_csv(
    train,
    validation,
    test
):

    records = []

    for city in train:

        records.append({
            "city": city,
            "split": "train"
        })

    for city in validation:

        records.append({
            "city": city,
            "split": "val"
        })

    for city in test:

        records.append({
            "city": city,
            "split": "test"
        })

    df = pd.DataFrame(
        records
    )

    df = df.sort_values(
        [
            "split",
            "city"
        ]
    )

    path = (
        DATA_ROOT /
        "city_split.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        "\n✓ Split CSV saved:"
    )

    print(
        f"  {path}"
    )

    return path


# ============================================================
# VERIFY OUTPUTS
# ============================================================

def verify_outputs(
    train,
    validation,
    test
):

    print("\n")
    print("=" * 70)
    print("OUTPUT VERIFICATION")
    print("=" * 70)

    all_results = []

    split_data = [
        ("train", train),
        ("val", validation),
        ("test", test)
    ]

    for split, cities in split_data:

        print("\n")
        print(
            split.upper()
        )

        print(
            "-" * 70
        )

        for city in sorted(cities):

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (

                LABELS_OUTPUT /
                split /
                f"{city}_cm.tif"

            )

            # ------------------------------------------------
            # GeoJSON
            # ------------------------------------------------

            geojson = (

                GEOJSON_OUTPUT /
                split /
                f"{city}.geojson"

            )

            # ------------------------------------------------
            # Dates
            # ------------------------------------------------

            dates = (

                GEOJSON_OUTPUT /
                split /
                f"{city}_dates.txt"

            )

            label_exists = (
                label.exists()
            )

            geojson_exists = (
                geojson.exists()
            )

            dates_exists = (
                dates.exists()
            )

            print(
                f"{city:<15}"
                f"Label: "
                f"{'✓' if label_exists else '✗'}   "
                f"GeoJSON: "
                f"{'✓' if geojson_exists else '✗'}   "
                f"Dates: "
                f"{'✓' if dates_exists else '✗'}"
            )

            all_results.append({

                "city":
                    city,

                "split":
                    split,

                "label":
                    label_exists,

                "geojson":
                    geojson_exists,

                "dates":
                    dates_exists

            })

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    total = len(
        all_results
    )

    labels = sum(
        r["label"]
        for r in all_results
    )

    geojson = sum(
        r["geojson"]
        for r in all_results
    )

    dates = sum(
        r["dates"]
        for r in all_results
    )

    complete = sum(

        r["label"]
        and r["geojson"]
        and r["dates"]

        for r in all_results

    )

    print("\n")
    print("=" * 70)
    print("OUTPUT SUMMARY")
    print("=" * 70)

    print(
        f"\nExpected cities : {total}"
    )

    print(
        f"Labels          : {labels}"
    )

    print(
        f"GeoJSON         : {geojson}"
    )

    print(
        f"Dates           : {dates}"
    )

    print(
        f"Complete        : {complete}"
    )

    return all_results


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("OSCD DATASET PREPARATION")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "GeoJSON files are copied from:"
    )

    print(
        r"  C:\OSCD_dataset\Images\<city>\<city>.geojson"
    )

    print(
        "\nDates are read from:"
    )

    print(
        r"  C:\OSCD_dataset\Images\<city>\dates.txt"
    )

    print(
        "\nGeoJSON is NOT generated from labels."
    )

    # ========================================================
    # Create Data directory
    # ========================================================

    DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Read city lists
    # ========================================================

    (
        all_cities,
        original_train,
        original_test
    ) = read_oscd_lists()

    # ========================================================
    # Check lists
    # ========================================================

    check_lists(
        all_cities,
        original_train,
        original_test
    )

    # ========================================================
    # Create split
    # ========================================================

    (
        train,
        validation,
        test
    ) = create_split(
        original_train,
        original_test
    )

    # ========================================================
    # Print final split
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    print(
        f"\nTRAIN: "
        f"{len(train)} cities"
    )

    print(
        "    " +
        ", ".join(
            sorted(train)
        )
    )

    print(
        f"\nVAL: "
        f"{len(validation)} cities"
    )

    print(
        "    " +
        ", ".join(
            sorted(validation)
        )
    )

    print(
        f"\nTEST: "
        f"{len(test)} cities"
    )

    print(
        "    " +
        ", ".join(
            sorted(test)
        )
    )

    # ========================================================
    # Create directories
    # ========================================================

    create_directories()

    # ========================================================
    # Save city lists
    # ========================================================

    train_path = save_city_list(
        train,
        "train_cities.txt"
    )

    val_path = save_city_list(
        validation,
        "val_cities.txt"
    )

    test_path = save_city_list(
        test,
        "test_cities.txt"
    )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = save_split_csv(
        train,
        validation,
        test
    )

    # ========================================================
    # Process TRAIN
    # ========================================================

    train_results = process_split(
        train,
        "train"
    )

    # ========================================================
    # Process VAL
    # ========================================================

    val_results = process_split(
        validation,
        "val"
    )

    # ========================================================
    # Process TEST
    # ========================================================

    test_results = process_split(
        test,
        "test"
    )

    # ========================================================
    # Verify
    # ========================================================

    results = verify_outputs(
        train,
        validation,
        test
    )

    # ========================================================
    # Final statistics
    # ========================================================

    train_success = sum(

        r["label"]
        and r["geojson"]
        and r["dates"]

        for r in train_results

    )

    val_success = sum(

        r["label"]
        and r["geojson"]
        and r["dates"]

        for r in val_results

    )

    test_success = sum(

        r["label"]
        and r["geojson"]
        and r["dates"]

        for r in test_results

    )

    train_failed = (
        len(train) -
        train_success
    )

    val_failed = (
        len(validation) -
        val_success
    )

    test_failed = (
        len(test) -
        test_success
    )

    # ========================================================
    # Final output
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATASET PREPARATION FINISHED")
    print("=" * 70)

    print(
        f"\nData:"
        f"\n{DATA_ROOT}"
    )

    print(
        f"\nTrain list:"
        f"\n{train_path}"
    )

    print(
        f"\nValidation list:"
        f"\n{val_path}"
    )

    print(
        f"\nTest list:"
        f"\n{test_path}"
    )

    print(
        f"\nCSV:"
        f"\n{csv_path}"
    )

    print("\n")
    print("=" * 70)

    print(
        "FINAL SUCCESS:"
    )

    print(
        f"  Train : {train_success}"
    )

    print(
        f"  Val   : {val_success}"
    )

    print(
        f"  Test  : {test_success}"
    )

    print("\n")
    print(
        "FINAL FAILED:"
    )

    print(
        f"  Train : {train_failed}"
    )

    print(
        f"  Val   : {val_failed}"
    )

    print(
        f"  Test  : {test_failed}"
    )

    print("=" * 70)

    if (

        train_failed == 0
        and
        val_failed == 0
        and
        test_failed == 0

    ):

        print(
            "\n✓ ALL CITIES PREPARED SUCCESSFULLY."
        )

        print(
            "\nExisting OSCD GeoJSON files were used."
        )

        print(
            "Acquisition dates were read from each "
            "city's dates.txt."
        )

        print(
            "Acquisition dates were added to each "
            "copied GeoJSON."
        )

    else:

        print(
            "\n⚠ SOME CITIES FAILED."
        )

        print(
            "Check the messages above."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
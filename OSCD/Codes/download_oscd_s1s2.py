# ============================================================
# GOOGLE EARTH ENGINE
# DOWNLOAD SENTINEL-1 AND SENTINEL-2 FOR OSCD DATASET
#
# IMPORTANT:
#
# OSCD labels are NOT georeferenced.
#
# Therefore:
#
#   GeoJSON = spatial footprint
#   Label   = target WIDTH / HEIGHT
#
# Every Sentinel image is downloaded using:
#
#   1. Same GeoJSON region
#   2. Same CRS
#   3. Same dimensions
#
# This guarantees:
#
#   Sentinel-1 T1 == Sentinel-2 T1 == Label
#   Sentinel-1 T2 == Sentinel-2 T2 == Label
#
# in terms of array dimensions and common spatial download grid.
#
# OUTPUT:
#
# Data/
# ├── geojson/
# │   ├── train/
# │   ├── val/
# │   └── test/
# │
# ├── labels/
# │   ├── train/
# │   ├── val/
# │   └── test/
# │
# └── images/
#     ├── train/
#     │   ├── sentinel1_city_t1.tif
#     │   ├── sentinel2_city_t1.tif
#     │   ├── sentinel1_city_t2.tif
#     │   └── sentinel2_city_t2.tif
#     │
#     ├── val/
#     └── test/
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import ee
import json
import time
import requests
import rasterio

from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# 2. GOOGLE EARTH ENGINE SERVICE ACCOUNT
# ============================================================

# ============================================================

print("=" * 70)
print('You have to set your own path key for  KEY_PATH parameter and also set your own service account for SERVICE_ACCOUNT parameter')
print("=" * 70)
SERVICE_ACCOUNT = "service account"
KEY_PATH = Path(r'please enter your path key file')


# ============================================================
# 3. INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

if not KEY_PATH.exists():

    raise FileNotFoundError(
        f"\nGEE service account key not found:\n"
        f"{KEY_PATH}\n"
    )


credentials = ee.ServiceAccountCredentials(
    SERVICE_ACCOUNT,
    str(KEY_PATH)
)

ee.Initialize(credentials)

print(
    "\nGoogle Earth Engine initialized successfully."
)


# ============================================================
# 4. BASE DIRECTORIES
# ============================================================

BASE_DIR = Path(
    r"C:\earth-observation-change-detection\Data"
)

GEOJSON_DIR = BASE_DIR / "geojson"

LABELS_DIR = BASE_DIR / "labels"

IMAGES_DIR = BASE_DIR / "images"


# ============================================================
# 5. SETTINGS
# ============================================================

# Search window around OSCD acquisition date.
SEARCH_DAYS = 30

# Maximum cloud percentage for Sentinel-2.
S2_MAX_CLOUD = 80

# ------------------------------------------------------------
# IMPORTANT
#
# Because the OSCD label has no CRS, we need to force all
# downloaded images to the same CRS.
#
# EPSG:4326 is used because the GeoJSON is normally WGS84.
# ------------------------------------------------------------

OUTPUT_CRS = "EPSG:4326"

# ------------------------------------------------------------
# Native Sentinel resolution.
#
# S1 VV/VH and S2 B2/B3/B4/B8 are approximately 10 m.
# The final array dimensions are controlled separately by
# the label dimensions.
# ------------------------------------------------------------

SCALE = 10

# Retry configuration.
DOWNLOAD_RETRIES = 3

# Pause between requests.
REQUEST_DELAY = 1


# ============================================================
# 6. SENTINEL COLLECTIONS
# ============================================================

S1_COLLECTION = (
    "COPERNICUS/S1_GRD"
)

S2_COLLECTION = (
    "COPERNICUS/S2_SR_HARMONIZED"
)


# ============================================================
# 7. SENTINEL-1 BANDS
# ============================================================

S1_BANDS = [
    "VV",
    "VH"
]


# ============================================================
# 8. SENTINEL-2 BANDS
# ============================================================

S2_BANDS = [
    "B2",       # Blue
    "B3",       # Green
    "B4",       # Red
    "B8"        # NIR
]


# ============================================================
# 9. FIND GEOJSON FILES
# ============================================================

def find_geojson_files():

    files = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        split_dir = GEOJSON_DIR / split

        if not split_dir.exists():

            print(
                f"WARNING: Missing GeoJSON folder:"
                f"\n    {split_dir}"
            )

            continue

        split_files = sorted(
            split_dir.glob("*.geojson")
        )

        for geojson_file in split_files:

            files.append(
                (
                    split,
                    geojson_file
                )
            )

    return files


# ============================================================
# 10. READ GEOJSON
# ============================================================

def read_geojson(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# 11. PARSE DATE
# ============================================================

def parse_date(date_string):

    return datetime.strptime(
        date_string,
        "%Y%m%d"
    )


# ============================================================
# 12. GET EE GEOMETRY
# ============================================================

def get_ee_geometry(data):

    if "features" not in data:

        raise ValueError(
            "GeoJSON does not contain 'features'."
        )

    if len(data["features"]) == 0:

        raise ValueError(
            "GeoJSON contains no features."
        )

    geometry = (
        data["features"][0]["geometry"]
    )

    return ee.Geometry(
        geometry
    )


# ============================================================
# 13. FIND LABEL FILE
# ============================================================

def find_label_file(
    split,
    city
):

    split_dir = (
        LABELS_DIR / split
    )

    if not split_dir.exists():

        raise FileNotFoundError(
            f"Label directory does not exist:\n"
            f"{split_dir}"
        )

    tif_files = sorted(
        split_dir.glob("*.tif")
    )

    if not tif_files:

        raise FileNotFoundError(
            f"No TIFF labels found in:\n"
            f"{split_dir}"
        )

    city_lower = city.lower()

    # --------------------------------------------------------
    # First: exact city substring match
    # --------------------------------------------------------

    for path in tif_files:

        if city_lower in path.stem.lower():

            return path

    # --------------------------------------------------------
    # Second: remove common suffixes
    # --------------------------------------------------------

    possible_names = [
        f"{city}_cm",
        f"{city}_label",
        f"{city}_labels",
        city
    ]

    for name in possible_names:

        candidate = (
            split_dir / f"{name}.tif"
        )

        if candidate.exists():

            return candidate

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    available = [
        p.name
        for p in tif_files
    ]

    raise FileNotFoundError(
        f"\nCould not find label for city:"
        f" {city}\n"
        f"Directory:"
        f" {split_dir}\n"
        f"Available files:"
        f"\n{available}"
    )


# ============================================================
# 14. GET LABEL DIMENSIONS
# ============================================================
#
# The OSCD labels are NOT georeferenced.
#
# We therefore ONLY use:
#
#     width
#     height
#
# from the label.
#
# We DO NOT use:
#
#     CRS
#     transform
#     bounds
#
# because they are not valid spatial information.
#
# ============================================================

def get_label_dimensions(
    label_path
):

    with rasterio.open(
        label_path
    ) as src:

        width = src.width

        height = src.height

        crs = src.crs

        transform = src.transform

    print()
    print(
        "Label information:"
    )

    print(
        f"    File       : {label_path.name}"
    )

    print(
        f"    Width      : {width}"
    )

    print(
        f"    Height     : {height}"
    )

    print(
        f"    CRS        : {crs}"
    )

    print(
        f"    Transform  : {transform}"
    )

    # --------------------------------------------------------
    # Warn if label is not georeferenced.
    # --------------------------------------------------------

    if crs is None:

        print(
            "    CRS status : None "
            "(expected for OSCD labels)"
        )

    return width, height


# ============================================================
# 15. GET SENTINEL-1 IMAGE
# ============================================================

def get_sentinel1_image(
    geometry,
    target_date
):

    target_datetime = parse_date(
        target_date
    )

    start_date = (
        target_datetime
        - timedelta(
            days=SEARCH_DAYS
        )
    ).strftime(
        "%Y-%m-%d"
    )

    end_date = (
        target_datetime
        + timedelta(
            days=SEARCH_DAYS + 1
        )
    ).strftime(
        "%Y-%m-%d"
    )

    print(
        f"    S1 search:"
        f" {start_date} → {end_date}"
    )

    collection = (
        ee.ImageCollection(
            S1_COLLECTION
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.eq(
                "instrumentMode",
                "IW"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH"
            )
        )
    )

    count = (
        collection
        .size()
        .getInfo()
    )

    print(
        f"    S1 images found:"
        f" {count}"
    )

    if count == 0:

        return None

    # --------------------------------------------------------
    # Target date in milliseconds.
    # --------------------------------------------------------

    target_millis = (
        ee.Date(
            target_datetime.strftime(
                "%Y-%m-%d"
            )
        )
        .millis()
    )

    # --------------------------------------------------------
    # Add temporal distance.
    # --------------------------------------------------------

    def add_time_difference(
        image
    ):

        difference = (
            ee.Number(
                image.date().millis()
            )
            .subtract(
                target_millis
            )
            .abs()
        )

        return image.set(
            "time_difference",
            difference
        )

    collection = collection.map(
        add_time_difference
    )

    # --------------------------------------------------------
    # Select temporally closest image.
    # --------------------------------------------------------

    image = ee.Image(
        collection
        .sort(
            "time_difference"
        )
        .first()
    )

    return image


# ============================================================
# 16. GET SENTINEL-2 IMAGE
# ============================================================

def get_sentinel2_image(
    geometry,
    target_date
):

    target_datetime = parse_date(
        target_date
    )

    start_date = (
        target_datetime
        - timedelta(
            days=SEARCH_DAYS
        )
    ).strftime(
        "%Y-%m-%d"
    )

    end_date = (
        target_datetime
        + timedelta(
            days=SEARCH_DAYS + 1
        )
    ).strftime(
        "%Y-%m-%d"
    )

    print(
        f"    S2 search:"
        f" {start_date} → {end_date}"
    )

    collection = (
        ee.ImageCollection(
            S2_COLLECTION
        )
        .filterBounds(
            geometry
        )
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                S2_MAX_CLOUD
            )
        )
    )

    count = (
        collection
        .size()
        .getInfo()
    )

    print(
        f"    S2 images found:"
        f" {count}"
    )

    if count == 0:

        return None

    # --------------------------------------------------------
    # Target date.
    # --------------------------------------------------------

    target_millis = (
        ee.Date(
            target_datetime.strftime(
                "%Y-%m-%d"
            )
        )
        .millis()
    )

    # --------------------------------------------------------
    # Add temporal distance.
    # --------------------------------------------------------

    def add_time_difference(
        image
    ):

        difference = (
            ee.Number(
                image.date().millis()
            )
            .subtract(
                target_millis
            )
            .abs()
        )

        return image.set(
            "time_difference",
            difference
        )

    collection = collection.map(
        add_time_difference
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Select closest date FIRST.
    #
    # The previous script sorted only by cloud percentage,
    # which could select an image far from the OSCD date.
    # --------------------------------------------------------

    collection = collection.sort(
        "time_difference"
    )

    image = ee.Image(
        collection.first()
    )

    return image


# ============================================================
# 17. PRINT IMAGE INFORMATION
# ============================================================

def print_image_information(
    image,
    satellite
):

    if image is None:

        return

    image_id = (
        image
        .get("system:id")
        .getInfo()
    )

    acquisition_date = (
        ee.Date(
            image.get(
                "system:time_start"
            )
        )
        .format(
            "YYYY-MM-dd HH:mm:ss"
        )
        .getInfo()
    )

    print()
    print(
        f"    {satellite}:"
    )

    print(
        f"        ID:"
        f" {image_id}"
    )

    print(
        f"        Acquisition:"
        f" {acquisition_date}"
    )

    # --------------------------------------------------------
    # Calculate temporal distance in days.
    # --------------------------------------------------------

    if satellite == "Sentinel-2":

        cloud = (
            image
            .get(
                "CLOUDY_PIXEL_PERCENTAGE"
            )
            .getInfo()
        )

        if cloud is not None:

            print(
                f"        Cloud:"
                f" {cloud:.2f}%"
            )

    # --------------------------------------------------------
    # Sentinel-1 metadata.
    # --------------------------------------------------------

    if satellite == "Sentinel-1":

        orbit_pass = (
            image
            .get(
                "orbitProperties_pass"
            )
            .getInfo()
        )

        relative_orbit = (
            image
            .get(
                "relativeOrbitNumber_start"
            )
            .getInfo()
        )

        print(
            f"        Orbit:"
            f" {orbit_pass}"
        )

        print(
            f"        Relative orbit:"
            f" {relative_orbit}"
        )


# ============================================================
# 18. DOWNLOAD EE IMAGE
# ============================================================
#
# IMPORTANT:
#
# `dimensions` forces the downloaded image to have exactly:
#
#     width × height
#
# The same dimensions are used for S1 and S2.
#
# `crs=EPSG:4326` ensures that both downloads use the same
# output coordinate reference system.
#
# ============================================================

def download_ee_image(
    image,
    bands,
    geometry,
    output_path,
    width,
    height,
    scale=10,
    crs="EPSG:4326"
):

    # --------------------------------------------------------
    # Select bands.
    # --------------------------------------------------------

    image = image.select(
        bands
    )

    # --------------------------------------------------------
    # Clip to GeoJSON footprint.
    # --------------------------------------------------------

    image = image.clip(
        geometry
    )

    # --------------------------------------------------------
    # Force exact output dimensions.
    # --------------------------------------------------------

    dimensions = (
        f"{width}x{height}"
    )

    params = {
        "region": geometry,
        "dimensions": dimensions,
        "crs": crs,
        "format": "GEO_TIFF",
        "filePerBand": False
    }

    # --------------------------------------------------------
    # Get download URL.
    # --------------------------------------------------------

    print()
    print(
        "    Requesting GEE download URL..."
    )

    url = image.getDownloadURL(
        params
    )

    print(
        "    Downloading:"
    )

    print(
        f"        {output_path.name}"
    )

    # --------------------------------------------------------
    # Download with retries.
    # --------------------------------------------------------

    last_error = None

    for attempt in range(
        1,
        DOWNLOAD_RETRIES + 1
    ):

        try:

            print(
                f"        Attempt:"
                f" {attempt}/{DOWNLOAD_RETRIES}"
            )

            response = requests.get(
                url,
                stream=True,
                timeout=600
            )

            response.raise_for_status()

            with open(
                output_path,
                "wb"
            ) as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:

                        f.write(chunk)

            print(
                "    Download complete."
            )

            break

        except Exception as e:

            last_error = e

            print(
                f"        Download failed:"
                f" {e}"
            )

            if attempt < DOWNLOAD_RETRIES:

                print(
                    "        Retrying..."
                )

                time.sleep(5)

    else:

        raise RuntimeError(
            f"Failed to download:"
            f" {output_path.name}\n"
            f"Last error:"
            f" {last_error}"
        )


# ============================================================
# 19. VERIFY RASTER DIMENSIONS
# ============================================================

def verify_raster(
    raster_path,
    expected_width,
    expected_height,
    expected_crs
):

    with rasterio.open(
        raster_path
    ) as src:

        width = src.width

        height = src.height

        count = src.count

        crs = src.crs

        transform = src.transform

    print()
    print(
        f"    Verification:"
        f" {raster_path.name}"
    )

    print(
        f"        Width:"
        f" {width}"
    )

    print(
        f"        Height:"
        f" {height}"
    )

    print(
        f"        Bands:"
        f" {count}"
    )

    print(
        f"        CRS:"
        f" {crs}"
    )

    # --------------------------------------------------------
    # Dimensions.
    # --------------------------------------------------------

    if width != expected_width:

        raise ValueError(
            f"\nWidth mismatch for:"
            f"\n{raster_path}"
            f"\nExpected:"
            f" {expected_width}"
            f"\nActual:"
            f" {width}"
        )

    if height != expected_height:

        raise ValueError(
            f"\nHeight mismatch for:"
            f"\n{raster_path}"
            f"\nExpected:"
            f" {expected_height}"
            f"\nActual:"
            f" {height}"
        )

    # --------------------------------------------------------
    # CRS.
    # --------------------------------------------------------

    if expected_crs is not None:

        if crs is None:

            raise ValueError(
                f"\nDownloaded raster has no CRS:"
                f"\n{raster_path}"
            )

        if str(crs) != str(expected_crs):

            raise ValueError(
                f"\nCRS mismatch:"
                f"\nExpected:"
                f" {expected_crs}"
                f"\nActual:"
                f" {crs}"
            )

    print(
        "        STATUS: OK"
    )

    return True


# ============================================================
# 20. VERIFY IMAGE AGAINST LABEL
# ============================================================

def verify_against_label(
    image_path,
    label_path
):

    with rasterio.open(
        image_path
    ) as image:

        image_width = image.width

        image_height = image.height

    with rasterio.open(
        label_path
    ) as label:

        label_width = label.width

        label_height = label.height

    if image_width != label_width:

        raise ValueError(
            f"\nWIDTH MISMATCH"
            f"\nImage:"
            f" {image_path}"
            f"\nImage width:"
            f" {image_width}"
            f"\nLabel width:"
            f" {label_width}"
        )

    if image_height != label_height:

        raise ValueError(
            f"\nHEIGHT MISMATCH"
            f"\nImage:"
            f" {image_path}"
            f"\nImage height:"
            f" {image_height}"
            f"\nLabel height:"
            f" {label_height}"
        )

    print(
        f"    ✓ Matches label:"
        f" {label_width} x {label_height}"
    )

    return True


# ============================================================
# 21. DOWNLOAD SENTINEL-1
# ============================================================

def download_sentinel1(
    geometry,
    city,
    target_date,
    time_label,
    output_dir,
    label_width,
    label_height,
    label_path
):

    output_path = (
        output_dir
        / f"sentinel1_{city}_{time_label}.tif"
    )

    # --------------------------------------------------------
    # Skip if already exists AND is valid.
    # --------------------------------------------------------

    if output_path.exists():

        print()
        print(
            f"    Existing file:"
            f" {output_path.name}"
        )

        try:

            verify_raster(
                output_path,
                label_width,
                label_height,
                OUTPUT_CRS
            )

            verify_against_label(
                output_path,
                label_path
            )

            print(
                "    Existing file is valid."
            )

            return

        except Exception:

            print(
                "    Existing file has incorrect"
                " dimensions/CRS."
            )

            print(
                "    Re-downloading..."
            )

            output_path.unlink()

    # --------------------------------------------------------
    # Search.
    # --------------------------------------------------------

    print()
    print(
        "    Searching Sentinel-1..."
    )

    image = get_sentinel1_image(
        geometry,
        target_date
    )

    if image is None:

        print(
            f"    WARNING:"
            f" No Sentinel-1 image found"
            f" for {target_date}"
        )

        return False

    # --------------------------------------------------------
    # Print metadata.
    # --------------------------------------------------------

    print_image_information(
        image,
        "Sentinel-1"
    )

    # --------------------------------------------------------
    # Download.
    # --------------------------------------------------------

    download_ee_image(
        image=image,
        bands=S1_BANDS,
        geometry=geometry,
        output_path=output_path,
        width=label_width,
        height=label_height,
        scale=SCALE,
        crs=OUTPUT_CRS
    )

    # --------------------------------------------------------
    # Verify.
    # --------------------------------------------------------

    verify_raster(
        output_path,
        label_width,
        label_height,
        OUTPUT_CRS
    )

    verify_against_label(
        output_path,
        label_path
    )

    return True


# ============================================================
# 22. DOWNLOAD SENTINEL-2
# ============================================================

def download_sentinel2(
    geometry,
    city,
    target_date,
    time_label,
    output_dir,
    label_width,
    label_height,
    label_path
):

    output_path = (
        output_dir
        / f"sentinel2_{city}_{time_label}.tif"
    )

    # --------------------------------------------------------
    # Skip if already exists AND is valid.
    # --------------------------------------------------------

    if output_path.exists():

        print()
        print(
            f"    Existing file:"
            f" {output_path.name}"
        )

        try:

            verify_raster(
                output_path,
                label_width,
                label_height,
                OUTPUT_CRS
            )

            verify_against_label(
                output_path,
                label_path
            )

            print(
                "    Existing file is valid."
            )

            return

        except Exception:

            print(
                "    Existing file has incorrect"
                " dimensions/CRS."
            )

            print(
                "    Re-downloading..."
            )

            output_path.unlink()

    # --------------------------------------------------------
    # Search.
    # --------------------------------------------------------

    print()
    print(
        "    Searching Sentinel-2..."
    )

    image = get_sentinel2_image(
        geometry,
        target_date
    )

    if image is None:

        print(
            f"    WARNING:"
            f" No Sentinel-2 image found"
            f" for {target_date}"
        )

        return False

    # --------------------------------------------------------
    # Print metadata.
    # --------------------------------------------------------

    print_image_information(
        image,
        "Sentinel-2"
    )

    # --------------------------------------------------------
    # Download.
    # --------------------------------------------------------

    download_ee_image(
        image=image,
        bands=S2_BANDS,
        geometry=geometry,
        output_path=output_path,
        width=label_width,
        height=label_height,
        scale=SCALE,
        crs=OUTPUT_CRS
    )

    # --------------------------------------------------------
    # Verify.
    # --------------------------------------------------------

    verify_raster(
        output_path,
        label_width,
        label_height,
        OUTPUT_CRS
    )

    verify_against_label(
        output_path,
        label_path
    )

    return True


# ============================================================
# 23. PROCESS ONE CITY
# ============================================================

def process_city(
    split,
    geojson_path
):

    print()
    print("=" * 75)

    print(
        f"Processing:"
        f" {split}/{geojson_path.name}"
    )

    print("=" * 75)

    # ========================================================
    # READ GEOJSON
    # ========================================================

    data = read_geojson(
        geojson_path
    )

    # ========================================================
    # CITY NAME
    # ========================================================

    city = data.get(
        "city",
        geojson_path.stem
    )

    city = str(
        city
    ).strip().lower()

    print(
        f"City:"
        f" {city}"
    )

    # ========================================================
    # ACQUISITION DATES
    # ========================================================

    acquisition_dates = data.get(
        "acquisition_dates",
        {}
    )

    date1 = acquisition_dates.get(
        "date1"
    )

    date2 = acquisition_dates.get(
        "date2"
    )

    if date1 is None:

        raise ValueError(
            f"date1 not found in:"
            f"\n{geojson_path}"
        )

    if date2 is None:

        raise ValueError(
            f"date2 not found in:"
            f"\n{geojson_path}"
        )

    print(
        f"Date 1:"
        f" {date1}"
    )

    print(
        f"Date 2:"
        f" {date2}"
    )

    # ========================================================
    # FIND LABEL
    # ========================================================

    label_path = find_label_file(
        split,
        city
    )

    print(
        f"Label:"
        f" {label_path}"
    )

    # ========================================================
    # LABEL DIMENSIONS
    # ========================================================

    (
        label_width,
        label_height
    ) = get_label_dimensions(
        label_path
    )

    print()
    print(
        "Target Sentinel dimensions:"
    )

    print(
        f"    Width:"
        f" {label_width}"
    )

    print(
        f"    Height:"
        f" {label_height}"
    )

    print(
        f"    CRS:"
        f" {OUTPUT_CRS}"
    )

    # ========================================================
    # EE GEOMETRY
    # ========================================================

    geometry = get_ee_geometry(
        data
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        IMAGES_DIR / split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # T1
    # ========================================================

    print()
    print(
        "-" * 70
    )

    print(
        f"T1 = {date1}"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Sentinel-1 T1
    # --------------------------------------------------------

    s1_t1_success = (
        download_sentinel1(
            geometry=geometry,
            city=city,
            target_date=date1,
            time_label="t1",
            output_dir=output_dir,
            label_width=label_width,
            label_height=label_height,
            label_path=label_path
        )
    )

    time.sleep(
        REQUEST_DELAY
    )

    # --------------------------------------------------------
    # Sentinel-2 T1
    # --------------------------------------------------------

    s2_t1_success = (
        download_sentinel2(
            geometry=geometry,
            city=city,
            target_date=date1,
            time_label="t1",
            output_dir=output_dir,
            label_width=label_width,
            label_height=label_height,
            label_path=label_path
        )
    )

    time.sleep(
        REQUEST_DELAY
    )

    # ========================================================
    # T2
    # ========================================================

    print()
    print(
        "-" * 70
    )

    print(
        f"T2 = {date2}"
    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Sentinel-1 T2
    # --------------------------------------------------------

    s1_t2_success = (
        download_sentinel1(
            geometry=geometry,
            city=city,
            target_date=date2,
            time_label="t2",
            output_dir=output_dir,
            label_width=label_width,
            label_height=label_height,
            label_path=label_path
        )
    )

    time.sleep(
        REQUEST_DELAY
    )

    # --------------------------------------------------------
    # Sentinel-2 T2
    # --------------------------------------------------------

    s2_t2_success = (
        download_sentinel2(
            geometry=geometry,
            city=city,
            target_date=date2,
            time_label="t2",
            output_dir=output_dir,
            label_width=label_width,
            label_height=label_height,
            label_path=label_path
        )
    )

    time.sleep(
        REQUEST_DELAY
    )

    # ========================================================
    # FINAL CITY VERIFICATION
    # ========================================================

    expected_files = [
        output_dir / f"sentinel1_{city}_t1.tif",
        output_dir / f"sentinel2_{city}_t1.tif",
        output_dir / f"sentinel1_{city}_t2.tif",
        output_dir / f"sentinel2_{city}_t2.tif"
    ]

    print()
    print(
        "Final verification:"
    )

    all_valid = True

    for path in expected_files:

        if not path.exists():

            print(
                f"    MISSING:"
                f" {path.name}"
            )

            all_valid = False

            continue

        try:

            verify_against_label(
                path,
                label_path
            )

        except Exception as e:

            print(
                f"    INVALID:"
                f" {path.name}"
            )

            print(
                f"        {e}"
            )

            all_valid = False

    if all_valid:

        print()
        print(
            f"✓ CITY COMPLETE:"
            f" {city}"
        )

        print(
            f"  Label:"
            f" {label_width} x {label_height}"
        )

        print(
            "  S1 T1: OK"
        )

        print(
            "  S2 T1: OK"
        )

        print(
            "  S1 T2: OK"
        )

        print(
            "  S2 T2: OK"
        )

    else:

        print()
        print(
            f"⚠ CITY INCOMPLETE:"
            f" {city}"
        )

    return all_valid


# ============================================================
# 24. MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "OSCD SENTINEL-1 / SENTINEL-2 DOWNLOADER"
    )
    print("=" * 75)

    print()
    print(
        "Base directory:"
    )

    print(
        f"    {BASE_DIR}"
    )

    print()
    print(
        "GeoJSON directory:"
    )

    print(
        f"    {GEOJSON_DIR}"
    )

    print()
    print(
        "Labels directory:"
    )

    print(
        f"    {LABELS_DIR}"
    )

    print()
    print(
        "Images directory:"
    )

    print(
        f"    {IMAGES_DIR}"
    )

    print()
    print(
        "Output CRS:"
    )

    print(
        f"    {OUTPUT_CRS}"
    )

    # ========================================================
    # FIND GEOJSON FILES
    # ========================================================

    geojson_files = (
        find_geojson_files()
    )

    print()
    print(
        f"Found:"
        f" {len(geojson_files)}"
        f" GeoJSON files."
    )

    if not geojson_files:

        print()
        print(
            "No GeoJSON files found."
        )

        return

    # ========================================================
    # PROCESS
    # ========================================================

    successful = 0

    failed = 0

    for index, (
        split,
        geojson_path
    ) in enumerate(
        geojson_files,
        start=1
    ):

        print()
        print(
            "#" * 75
        )

        print(
            f"City {index}/{len(geojson_files)}"
        )

        print(
            f"Split:"
            f" {split}"
        )

        print(
            f"File:"
            f" {geojson_path.name}"
        )

        print(
            "#" * 75
        )

        try:

            result = process_city(
                split=split,
                geojson_path=geojson_path
            )

            if result:

                successful += 1

            else:

                failed += 1

        except Exception as e:

            failed += 1

            print()
            print(
                "ERROR PROCESSING CITY"
            )

            print(
                f"    Split:"
                f" {split}"
            )

            print(
                f"    File:"
                f" {geojson_path.name}"
            )

            print(
                f"    Error:"
                f" {e}"
            )

            print()
            print(
                "Continuing with next city..."
            )

        # ----------------------------------------------------
        # Pause between cities.
        # ----------------------------------------------------

        if index < len(
            geojson_files
        ):

            time.sleep(2)

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print(
        "DOWNLOAD FINISHED"
    )
    print("=" * 75)

    print()
    print(
        f"Total GeoJSON files:"
        f" {len(geojson_files)}"
    )

    print(
        f"Successful:"
        f" {successful}"
    )

    print(
        f"Failed:"
        f" {failed}"
    )

    print()
    print(
        "Images saved in:"
    )

    print(
        f"    {IMAGES_DIR}"
    )

    print()
    print(
        "Expected structure:"
    )

    print(
        "    images/"
    )

    print(
        "      train/"
    )

    print(
        "      val/"
    )

    print(
        "      test/"
    )

    print()
    print(
        "Each city should contain:"
    )

    print(
        "    sentinel1_city_t1.tif"
    )

    print(
        "    sentinel2_city_t1.tif"
    )

    print(
        "    sentinel1_city_t2.tif"
    )

    print(
        "    sentinel2_city_t2.tif"
    )

    print("=" * 75)


# ============================================================
# 25. RUN
# ============================================================

if __name__ == "__main__":

    main()
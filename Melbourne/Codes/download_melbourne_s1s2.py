# ============================================================
# MELBOURNE URBAN CHANGE DETECTION DATASET
# ============================================================
#
# AOI:
#   Melbourne Western Growth Corridor
#
# Years:
#   2023
#   2025
#
# Sentinel-1:
#   VV
#   VH
#
# Sentinel-2:
#   B2
#   B3
#   B4
#   B8
#
# Resolution:
#   10 m
#
# Tile:
#   256 × 256 pixels
#
# Ground size:
#   2560 × 2560 m
#
# CRS:
#   EPSG:32755 (UTM Zone 55 South)
#
# IMPORTANT:
#   Every S1 and S2 tile uses exactly the same grid.
#
# Output:
#
# MelbourneData/
# │
# ├── images/
# │   ├── 2023/
# │   │   ├── sentinel1/
# │   │   └── sentinel2/
# │   │
# │   └── 2025/
# │       ├── sentinel1/
# │       └── sentinel2/
# │
# └── metadata/
#     └── melbourne_tiles.csv
#
# ============================================================


from pathlib import Path
import csv
import requests
import ee
from pyproj import Transformer


# ============================================================
# 1. CONFIGURATION
# ============================================================
print("=" * 70)
print('You have to set your own path key for  KEY_PATH parameter and also set your own service account for SERVICE_ACCOUNT parameter')
print("=" * 70)
SERVICE_ACCOUNT = "service account"
KEY_PATH = Path(r'please enter your path key file')



# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path(
    r"C:\melbourne-urban-change-detection\Melbourne"
    r"\MelbourneData"
)


# ============================================================
# 3. CHECK GEE KEY
# ============================================================

if not KEY_PATH.exists():

    raise FileNotFoundError(
        "\nGoogle Earth Engine service account key not found:\n"
        f"{KEY_PATH}\n"
    )


# ============================================================
# 4. INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

credentials = ee.ServiceAccountCredentials(
    SERVICE_ACCOUNT,
    str(KEY_PATH)
)

ee.Initialize(
    credentials=credentials
)

print(
    "\nGoogle Earth Engine initialized successfully."
)


# ============================================================
# 5. DIRECTORY STRUCTURE
# ============================================================

IMAGES_DIR = OUTPUT_DIR / "images"

IMAGES_2023_DIR = IMAGES_DIR / "2023"
IMAGES_2025_DIR = IMAGES_DIR / "2025"

S1_2023_DIR = IMAGES_2023_DIR / "sentinel1"
S2_2023_DIR = IMAGES_2023_DIR / "sentinel2"

S1_2025_DIR = IMAGES_2025_DIR / "sentinel1"
S2_2025_DIR = IMAGES_2025_DIR / "sentinel2"

METADATA_DIR = OUTPUT_DIR / "metadata"


for directory in [

    S1_2023_DIR,
    S2_2023_DIR,

    S1_2025_DIR,
    S2_2025_DIR,

    METADATA_DIR

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 6. MELBOURNE AOI
# ============================================================
#
# Western Melbourne growth corridor:
#
# Werribee
# Tarneit
# Truganina
# Mambourin
#
# WGS84
# ============================================================

WEST = 144.62
SOUTH = -37.90

EAST = 144.70
NORTH = -37.82


# ============================================================
# 7. TARGET CRS
# ============================================================

TARGET_CRS = "EPSG:32755"


# ============================================================
# 8. TILE PARAMETERS
# ============================================================

TILE_PIXELS = 256

PIXEL_SIZE = 10

TILE_SIZE_METERS = (
    TILE_PIXELS * PIXEL_SIZE
)


# ============================================================
# 9. DATE WINDOWS
# ============================================================
#
# Same seasonal period in both years.
#
# 2023:
#   1 March → 1 June
#
# 2025:
#   1 March → 1 June
#
# ============================================================

DATE_RANGES = {

    2023: (
        "2023-03-01",
        "2023-06-01"
    ),

    2025: (
        "2025-03-01",
        "2025-06-01"
    )
}


# ============================================================
# 10. SENTINEL-2 BANDS
# ============================================================

S2_BANDS = [

    "B2",   # Blue
    "B3",   # Green
    "B4",   # Red
    "B8"    # NIR

]


# ============================================================
# 11. SENTINEL-1 BANDS
# ============================================================

S1_BANDS = [

    "VV",
    "VH"

]


# ============================================================
# 12. CREATE EXACT UTM TILE GRID
# ============================================================

def create_tile_grid():

    print()
    print("=" * 70)
    print("CREATING EXACT 256 × 256 / 10 m TILE GRID")
    print("=" * 70)


    # --------------------------------------------------------
    # Coordinate transformers
    # --------------------------------------------------------

    transformer = Transformer.from_crs(
        "EPSG:4326",
        TARGET_CRS,
        always_xy=True
    )

    reverse_transformer = Transformer.from_crs(
        TARGET_CRS,
        "EPSG:4326",
        always_xy=True
    )


    # --------------------------------------------------------
    # Convert AOI corners to UTM
    # --------------------------------------------------------

    x_min, y_min = transformer.transform(
        WEST,
        SOUTH
    )

    x_max, y_max = transformer.transform(
        EAST,
        NORTH
    )


    print(
        f"\nAOI UTM X:"
        f" {x_min:.2f} → {x_max:.2f}"
    )

    print(
        f"AOI UTM Y:"
        f" {y_min:.2f} → {y_max:.2f}"
    )


    # --------------------------------------------------------
    # ALIGN AOI TO 2560 m GRID
    #
    # This is important.
    #
    # Every tile starts/ends exactly on the same 10 m grid.
    # --------------------------------------------------------

    import math


    grid_x_min = (
        math.floor(
            x_min / TILE_SIZE_METERS
        )
        * TILE_SIZE_METERS
    )

    grid_y_min = (
        math.floor(
            y_min / TILE_SIZE_METERS
        )
        * TILE_SIZE_METERS
    )

    grid_x_max = (
        math.ceil(
            x_max / TILE_SIZE_METERS
        )
        * TILE_SIZE_METERS
    )

    grid_y_max = (
        math.ceil(
            y_max / TILE_SIZE_METERS
        )
        * TILE_SIZE_METERS
    )


    # --------------------------------------------------------
    # Number of tiles
    # --------------------------------------------------------

    cols = int(
        (
            grid_x_max
            - grid_x_min
        )
        / TILE_SIZE_METERS
    )

    rows = int(
        (
            grid_y_max
            - grid_y_min
        )
        / TILE_SIZE_METERS
    )


    print(
        f"\nGrid X:"
        f" {grid_x_min:.2f} → {grid_x_max:.2f}"
    )

    print(
        f"Grid Y:"
        f" {grid_y_min:.2f} → {grid_y_max:.2f}"
    )

    print(
        f"\nTile size:"
        f" {TILE_PIXELS} × {TILE_PIXELS}"
    )

    print(
        f"Pixel size:"
        f" {PIXEL_SIZE} m"
    )

    print(
        f"Ground size:"
        f" {TILE_SIZE_METERS} × "
        f"{TILE_SIZE_METERS} m"
    )

    print(
        f"Columns: {cols}"
    )

    print(
        f"Rows: {rows}"
    )

    print(
        f"Total tiles: {rows * cols}"
    )


    # --------------------------------------------------------
    # Create tile list
    # --------------------------------------------------------

    tiles = []

    tile_id = 0


    for row in range(rows):

        for col in range(cols):

            # ------------------------------------------------
            # EXACT UTM BOUNDARIES
            # ------------------------------------------------

            tx_min = (
                grid_x_min
                + col * TILE_SIZE_METERS
            )

            tx_max = (
                tx_min
                + TILE_SIZE_METERS
            )

            ty_min = (
                grid_y_min
                + row * TILE_SIZE_METERS
            )

            ty_max = (
                ty_min
                + TILE_SIZE_METERS
            )


            # ------------------------------------------------
            # Convert corners to WGS84
            # ------------------------------------------------

            lon1, lat1 = (
                reverse_transformer.transform(
                    tx_min,
                    ty_min
                )
            )

            lon2, lat2 = (
                reverse_transformer.transform(
                    tx_max,
                    ty_max
                )
            )


            tile = {

                "tile_id":
                    f"melbourne_{tile_id:04d}",

                "row":
                    row,

                "col":
                    col,

                "xmin":
                    tx_min,

                "xmax":
                    tx_max,

                "ymin":
                    ty_min,

                "ymax":
                    ty_max,

                "lon_min":
                    min(lon1, lon2),

                "lon_max":
                    max(lon1, lon2),

                "lat_min":
                    min(lat1, lat2),

                "lat_max":
                    max(lat1, lat2)

            }


            tiles.append(tile)

            tile_id += 1


    return tiles


# ============================================================
# 13. SAVE TILE METADATA
# ============================================================

def save_metadata(tiles):

    metadata_path = (
        METADATA_DIR
        / "melbourne_tiles.csv"
    )


    fieldnames = [

        "tile_id",
        "row",
        "col",

        "xmin",
        "xmax",

        "ymin",
        "ymax",

        "lon_min",
        "lon_max",

        "lat_min",
        "lat_max"

    ]


    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for tile in tiles:

            writer.writerow(tile)


    print(
        f"\nMetadata saved:\n"
        f"{metadata_path}"
    )


# ============================================================
# 14. CREATE EXACT UTM TILE GEOMETRY
# ============================================================

def tile_geometry_utm(tile):

    return ee.Geometry.Rectangle(

        [

            tile["xmin"],
            tile["ymin"],

            tile["xmax"],
            tile["ymax"]

        ],

        proj=TARGET_CRS,

        geodesic=False

    )


# ============================================================
# 15. SENTINEL-2 CLOUD MASK
# ============================================================

def mask_sentinel2(image):

    scl = image.select(
        "SCL"
    )


    # --------------------------------------------------------
    # SCL classes removed:
    #
    # 1  = saturated / defective
    # 3  = cloud shadow
    # 7  = low probability cloud
    # 8  = medium probability cloud
    # 9  = high probability cloud
    # 10 = cirrus
    # --------------------------------------------------------

    mask = (

        scl.neq(1)

        .And(
            scl.neq(3)
        )

        .And(
            scl.neq(7)
        )

        .And(
            scl.neq(8)
        )

        .And(
            scl.neq(9)
        )

        .And(
            scl.neq(10)
        )

    )


    return (

        image

        .updateMask(mask)

        .select(S2_BANDS)

        .divide(10000)

        .copyProperties(
            image,
            [
                "system:time_start"
            ]
        )

    )


# ============================================================
# 16. GET SENTINEL-2 COMPOSITE
# ============================================================

def get_sentinel2(
    start_date,
    end_date,
    geometry
):

    collection = (

        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )

        .filterBounds(
            geometry
        )

        .filterDate(
            start_date,
            end_date
        )

        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                30
            )
        )

        .map(
            mask_sentinel2
        )

    )


    count = (
        collection
        .size()
        .getInfo()
    )


    print(
        f"S2 images found: {count}"
    )


    if count == 0:

        raise RuntimeError(
            "No Sentinel-2 images found."
        )


    return (

        collection

        .median()

        .select(
            S2_BANDS
        )

    )


# ============================================================
# 17. GET SENTINEL-1 COMPOSITE
# ============================================================

def get_sentinel1(
    start_date,
    end_date,
    geometry
):

    collection = (

        ee.ImageCollection(
            "COPERNICUS/S1_GRD"
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

        .select(
            S1_BANDS
        )

    )


    count = (
        collection
        .size()
        .getInfo()
    )


    print(
        f"S1 images found: {count}"
    )


    if count == 0:

        raise RuntimeError(
            "No Sentinel-1 images found."
        )


    return (

        collection

        .median()

        .select(
            S1_BANDS
        )

    )


# ============================================================
# 18. DOWNLOAD ONE TILE
# ============================================================

def download_tile(
    image,
    tile,
    output_path,
    bands
):

    # --------------------------------------------------------
    # Skip existing files
    # --------------------------------------------------------

    if output_path.exists():

        print(
            f"Already exists: "
            f"{output_path.name}"
        )

        return


    print(
        f"\nDownloading:"
        f" {output_path.name}"
    )


    # --------------------------------------------------------
    # EXACT UTM TILE
    # --------------------------------------------------------

    geometry = tile_geometry_utm(
        tile
    )


    # --------------------------------------------------------
    # EXACT 10 m PIXEL GRID
    #
    # Affine transform:
    #
    # [10, 0, xmin,
    #  0,-10, ymax]
    #
    # This gives:
    #
    # 256 × 256 pixels
    # 10 m pixels
    # exact tile alignment
    # --------------------------------------------------------

    crs_transform = [

        PIXEL_SIZE,
        0,
        tile["xmin"],

        0,
        -PIXEL_SIZE,
        tile["ymax"]

    ]


    # --------------------------------------------------------
    # Generate download URL
    # --------------------------------------------------------

    url = image.getDownloadURL(

        {

            "name":
                output_path.stem,

            "bands":
                bands,

            "region":
                geometry,

            "crs":
                TARGET_CRS,

            "crs_transform":
                crs_transform,

            "format":
                "GEO_TIFF",

            "filePerBand":
                False

        }

    )


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    response = requests.get(
        url,
        stream=True,
        timeout=600
    )


    response.raise_for_status()


    # --------------------------------------------------------
    # Save TIFF
    # --------------------------------------------------------

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
        f"Saved: "
        f"{output_path}"
    )


# ============================================================
# 19. DOWNLOAD ONE YEAR
# ============================================================

def download_year(
    year,
    tiles,
    full_aoi
):

    start_date, end_date = (
        DATE_RANGES[year]
    )


    print()
    print("=" * 70)

    print(
        f"PROCESSING YEAR {year}"
    )

    print("=" * 70)


    print(
        f"\nDate range:"
        f" {start_date} → {end_date}"
    )


    # ========================================================
    # SENTINEL-2
    # ========================================================

    print()
    print(
        "Creating Sentinel-2 composite..."
    )


    s2 = get_sentinel2(

        start_date,
        end_date,

        full_aoi

    )


    # ========================================================
    # SENTINEL-1
    # ========================================================

    print()
    print(
        "Creating Sentinel-1 composite..."
    )


    s1 = get_sentinel1(

        start_date,
        end_date,

        full_aoi

    )


    # ========================================================
    # SELECT DIRECTORIES
    # ========================================================

    if year == 2023:

        s1_dir = S1_2023_DIR

        s2_dir = S2_2023_DIR

    elif year == 2025:

        s1_dir = S1_2025_DIR

        s2_dir = S2_2025_DIR

    else:

        raise ValueError(
            f"Unsupported year: {year}"
        )


    # ========================================================
    # DOWNLOAD EACH TILE
    # ========================================================

    for index, tile in enumerate(
        tiles,
        start=1
    ):

        tile_id = tile[
            "tile_id"
        ]


        print()
        print(
            "-" * 70
        )

        print(
            f"{year} | "
            f"Tile {index}/{len(tiles)}"
        )

        print(
            f"Tile ID: {tile_id}"
        )

        print(
            f"Row: {tile['row']} | "
            f"Column: {tile['col']}"
        )


        # ====================================================
        # SENTINEL-2
        # ====================================================

        s2_path = (

            s2_dir
            /
            f"{tile_id}_s2_{year}.tif"

        )


        download_tile(

            s2,

            tile,

            s2_path,

            S2_BANDS

        )


        # ====================================================
        # SENTINEL-1
        # ====================================================

        s1_path = (

            s1_dir
            /
            f"{tile_id}_s1_{year}.tif"

        )


        download_tile(

            s1,

            tile,

            s1_path,

            S1_BANDS

        )


# ============================================================
# 20. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MELBOURNE URBAN CHANGE DETECTION DATASET")
    print("=" * 70)


    # ========================================================
    # CREATE TILE GRID
    # ========================================================

    tiles = create_tile_grid()


    print()
    print(
        f"\nCreated "
        f"{len(tiles)} tiles."
    )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    save_metadata(
        tiles
    )


    # ========================================================
    # CREATE FULL AOI
    # ========================================================

    full_aoi = ee.Geometry.Rectangle(

        [

            WEST,
            SOUTH,

            EAST,
            NORTH

        ],

        proj="EPSG:4326",

        geodesic=False

    )


    # ========================================================
    # DOWNLOAD 2023
    # ========================================================

    download_year(

        2023,

        tiles,

        full_aoi

    )


    # ========================================================
    # DOWNLOAD 2025
    # ========================================================

    download_year(

        2025,

        tiles,

        full_aoi

    )


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)


    print(
        f"\nTotal tiles: "
        f"{len(tiles)}"
    )


    print(
        "\nTile:"
    )

    print(
        f"  {TILE_PIXELS} × "
        f"{TILE_PIXELS} pixels"
    )


    print(
        "\nResolution:"
    )

    print(
        f"  {PIXEL_SIZE} m"
    )


    print(
        "\nGround coverage:"
    )

    print(
        f"  {TILE_SIZE_METERS} × "
        f"{TILE_SIZE_METERS} m"
    )


    print(
        "\nCRS:"
    )

    print(
        f"  {TARGET_CRS}"
    )


    print(
        "\nOutput:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
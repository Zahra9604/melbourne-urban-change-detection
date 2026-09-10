# ============================================================
# MELBOURNE DEA LAND COVER C3 DOWNLOADER
#
# Native DEA resolution: 30 m
#
# AOI:
#   144.62, -37.90
#   144.70, -37.82
#
# Years:
#   2023
#   2025
#
# CRS:
#   EPSG:32755
#
# Same tile boundaries as Sentinel-1 / Sentinel-2
#
# ============================================================


# ============================================================
# 1. AWS PUBLIC DATA SETTINGS
# ============================================================

import os

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_EC2_METADATA_DISABLED"] = "TRUE"


# ============================================================
# 2. IMPORTS
# ============================================================

from pathlib import Path

import csv
import math
import time

import numpy as np

import rasterio

from pyproj import Transformer

import pystac_client
import odc.stac


# ============================================================
# 3. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(
    r"C:\earth-observation-change-detection"
    r"\MelbourneData"
)


# ============================================================
# 4. MELBOURNE AOI
# ============================================================

WEST = 144.62
SOUTH = -37.90

EAST = 144.70
NORTH = -37.82


# ============================================================
# 5. TARGET CRS
# ============================================================

TARGET_CRS = "EPSG:32755"


# ============================================================
# 6. DEA
# ============================================================

DEA_PRODUCT = (
    "ga_ls_landcover_class_cyear_3"
)

DEA_BAND = "level4"

DEA_RESOLUTION = 30


# ============================================================
# 7. YEARS
# ============================================================

YEARS = [
    2023,
    2025
]


# ============================================================
# 8. TILE CONFIGURATION
#
# Same spatial tiles as Sentinel-1 / Sentinel-2
#
# 256 × 256 pixels at 10 m
# = 2560 × 2560 metres
#
# DEA is loaded at native 30 m.
# ============================================================

SENTINEL_TILE_PIXELS = 256

SENTINEL_PIXEL_SIZE = 10

TILE_SIZE_METERS = (
    SENTINEL_TILE_PIXELS
    * SENTINEL_PIXEL_SIZE
)


# ============================================================
# 9. OUTPUT DIRECTORIES
# ============================================================

DEA_DIR = (
    OUTPUT_DIR
    / "dea_landcover"
)

DEA_2023_DIR = (
    DEA_DIR
    / "2023"
)

DEA_2025_DIR = (
    DEA_DIR
    / "2025"
)

METADATA_DIR = (
    OUTPUT_DIR
    / "metadata"
)


for directory in [

    DEA_2023_DIR,
    DEA_2025_DIR,
    METADATA_DIR

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 10. DEA STAC CONNECTION
# ============================================================

print()
print("=" * 70)
print("CONNECTING TO DEA STAC")
print("=" * 70)


# ------------------------------------------------------------
# Use DEA public STAC API
# ------------------------------------------------------------

STAC_URL = (
    "https://explorer.dea.ga.gov.au/stac"
)


# ------------------------------------------------------------
# Retry connection
# ------------------------------------------------------------

catalog = None

MAX_RETRIES = 5


for attempt in range(
    1,
    MAX_RETRIES + 1
):

    try:

        print(
            f"\nSTAC connection "
            f"attempt {attempt}/{MAX_RETRIES}..."
        )


        catalog = (
            pystac_client.Client.open(
                STAC_URL
            )
        )


        print(
            "\nDEA STAC connected successfully."
        )

        break


    except Exception as e:

        print()
        print(
            f"STAC connection failed:"
        )

        print(
            str(e)
        )


        if attempt < MAX_RETRIES:

            wait_time = (
                attempt * 10
            )

            print(
                f"\nWaiting "
                f"{wait_time} seconds "
                f"before retry..."
            )

            time.sleep(
                wait_time
            )

        else:

            raise RuntimeError(
                "\nUnable to connect to "
                "DEA STAC after "
                f"{MAX_RETRIES} attempts.\n\n"
                "This is usually a temporary "
                "DEA server/network problem."
            )


# ============================================================
# 11. CREATE TILE GRID
# ============================================================

def create_tile_grid():

    print()
    print("=" * 70)
    print("CREATING MELBOURNE TILE GRID")
    print("=" * 70)


    # --------------------------------------------------------
    # WGS84 → UTM 55S
    # --------------------------------------------------------

    transformer = Transformer.from_crs(

        "EPSG:4326",

        TARGET_CRS,

        always_xy=True

    )


    # --------------------------------------------------------
    # UTM → WGS84
    # --------------------------------------------------------

    reverse_transformer = Transformer.from_crs(

        TARGET_CRS,

        "EPSG:4326",

        always_xy=True

    )


    # --------------------------------------------------------
    # AOI coordinates
    # --------------------------------------------------------

    x_min, y_min = transformer.transform(

        WEST,
        SOUTH

    )


    x_max, y_max = transformer.transform(

        EAST,
        NORTH

    )


    print()
    print(
        f"UTM X:"
    )

    print(
        f"{x_min:.2f} → {x_max:.2f}"
    )


    print()
    print(
        f"UTM Y:"
    )

    print(
        f"{y_min:.2f} → {y_max:.2f}"
    )


    # --------------------------------------------------------
    # Align tile grid
    # --------------------------------------------------------

    grid_x_min = (

        math.floor(
            x_min
            /
            TILE_SIZE_METERS
        )

        *
        TILE_SIZE_METERS

    )


    grid_y_min = (

        math.floor(
            y_min
            /
            TILE_SIZE_METERS
        )

        *
        TILE_SIZE_METERS

    )


    grid_x_max = (

        math.ceil(
            x_max
            /
            TILE_SIZE_METERS
        )

        *
        TILE_SIZE_METERS

    )


    grid_y_max = (

        math.ceil(
            y_max
            /
            TILE_SIZE_METERS
        )

        *
        TILE_SIZE_METERS

    )


    # --------------------------------------------------------
    # Rows / columns
    # --------------------------------------------------------

    cols = int(

        (
            grid_x_max
            -
            grid_x_min
        )
        /
        TILE_SIZE_METERS

    )


    rows = int(

        (
            grid_y_max
            -
            grid_y_min
        )
        /
        TILE_SIZE_METERS

    )


    print()
    print(
        f"Tile size:"
    )

    print(
        f"{TILE_SIZE_METERS} × "
        f"{TILE_SIZE_METERS} m"
    )


    print()
    print(
        f"Rows: {rows}"
    )

    print(
        f"Columns: {cols}"
    )

    print(
        f"Total tiles: "
        f"{rows * cols}"
    )


    tiles = []


    tile_id = 0


    # ========================================================
    # CREATE TILES
    # ========================================================

    for row in range(rows):

        for col in range(cols):

            tx_min = (

                grid_x_min
                +
                col
                *
                TILE_SIZE_METERS

            )


            tx_max = (

                tx_min
                +
                TILE_SIZE_METERS

            )


            ty_min = (

                grid_y_min
                +
                row
                *
                TILE_SIZE_METERS

            )


            ty_max = (

                ty_min
                +
                TILE_SIZE_METERS

            )


            # ------------------------------------------------
            # Convert to WGS84
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
                    min(
                        lon1,
                        lon2
                    ),

                "lon_max":
                    max(
                        lon1,
                        lon2
                    ),

                "lat_min":
                    min(
                        lat1,
                        lat2
                    ),

                "lat_max":
                    max(
                        lat1,
                        lat2
                    )

            }


            tiles.append(
                tile
            )


            tile_id += 1


    return tiles


# ============================================================
# 12. SAVE TILE METADATA
# ============================================================

def save_metadata(tiles):

    path = (

        METADATA_DIR
        /
        "melbourne_tiles.csv"

    )


    fields = [

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

        path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=fields

        )


        writer.writeheader()


        for tile in tiles:

            writer.writerow(
                tile
            )


    print()
    print(
        "Metadata saved:"
    )

    print(
        path
    )


# ============================================================
# 13. SEARCH DEA
# ============================================================

def search_dea(
    year
):

    print()
    print(
        f"Searching DEA Land Cover "
        f"for {year}..."
    )


    bbox = [

        WEST,
        SOUTH,
        EAST,
        NORTH

    ]


    start_date = (
        f"{year}-01-01"
    )

    end_date = (
        f"{year + 1}-01-01"
    )


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            search = catalog.search(

                bbox=bbox,

                collections=[
                    DEA_PRODUCT
                ],

                datetime=(

                    f"{start_date}/"
                    f"{end_date}"

                )

            )


            items = list(
                search.items()
            )


            print()
            print(
                f"DEA items found: "
                f"{len(items)}"
            )


            for item in items:

                print(
                    f"   {item.id}"
                )


            if len(items) == 0:

                raise RuntimeError(

                    f"No DEA Land Cover "
                    f"data found for {year}."

                )


            return items


        except Exception as e:

            print()
            print(
                f"DEA search failed "
                f"(attempt "
                f"{attempt}/{MAX_RETRIES})"
            )

            print(
                str(e)
            )


            if attempt < MAX_RETRIES:

                wait_time = (
                    attempt * 10
                )

                print(
                    f"Retrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

            else:

                raise


# ============================================================
# 14. LOAD DEA TILE
# ============================================================

def load_dea_tile(

    items,

    tile

):

    bbox = [

        tile["lon_min"],
        tile["lat_min"],

        tile["lon_max"],
        tile["lat_max"]

    ]


    print()
    print(
        "Loading DEA at native "
        f"{DEA_RESOLUTION} m..."
    )


    # --------------------------------------------------------
    # Explicit public AWS configuration
    # --------------------------------------------------------

    with rasterio.Env(

        AWS_NO_SIGN_REQUEST="YES",

        AWS_EC2_METADATA_DISABLED="TRUE"

    ):

        ds = odc.stac.load(

            items,

            bands=[
                DEA_BAND
            ],

            bbox=bbox,

            crs=TARGET_CRS,

            resolution=DEA_RESOLUTION,

            resampling="nearest",

            groupby="solar_day",

            chunks=None

        )


    if DEA_BAND not in ds:

        raise RuntimeError(
            f"Band {DEA_BAND} "
            "not found."
        )


    data = ds[
        DEA_BAND
    ]


    # --------------------------------------------------------
    # Remove time dimension
    # --------------------------------------------------------

    if "time" in data.dims:

        data = data.isel(
            time=0
        )


    return data


# ============================================================
# 15. SAVE DEA TILE
# ============================================================

def save_dea_tile(

    data,

    output_path

):

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    array = data.values


    if array.ndim != 2:

        raise RuntimeError(

            "Unexpected DEA shape: "
            f"{array.shape}"

        )


    # --------------------------------------------------------
    # DEA level4 is categorical
    # --------------------------------------------------------

    array = array.astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = (
        data.odc.geobox.transform
    )


    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    crs = (
        data.odc.geobox.crs
    )


    height, width = (
        array.shape
    )


    print()
    print(
        "DEA tile:"
    )

    print(
        f"  Width: "
        f"{width}"
    )

    print(
        f"  Height: "
        f"{height}"
    )

    print(
        f"  Resolution: "
        f"{DEA_RESOLUTION} m"
    )


    # --------------------------------------------------------
    # Write GeoTIFF
    # --------------------------------------------------------

    with rasterio.open(

        output_path,

        "w",

        driver="GTiff",

        height=height,

        width=width,

        count=1,

        dtype="uint8",

        crs=crs,

        transform=transform,

        nodata=255,

        compress="lzw"

    ) as dst:

        dst.write(

            array,

            1

        )


    print()
    print(
        f"Saved:"
    )

    print(
        output_path
    )


# ============================================================
# 16. PROCESS YEAR
# ============================================================

def process_year(

    year,

    tiles

):

    print()
    print("=" * 70)

    print(
        f"PROCESSING DEA LAND COVER "
        f"{year}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Search once for the whole AOI
    # --------------------------------------------------------

    items = search_dea(
        year
    )


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    if year == 2023:

        output_dir = (
            DEA_2023_DIR
        )

    else:

        output_dir = (
            DEA_2025_DIR
        )


    # ========================================================
    # TILE LOOP
    # ========================================================

    for index, tile in enumerate(

        tiles,

        start=1

    ):

        tile_id = (
            tile["tile_id"]
        )


        print()
        print(
            "-" * 70
        )

        print(
            f"{year} | "
            f"Tile "
            f"{index}/{len(tiles)}"
        )

        print(
            f"Tile ID: "
            f"{tile_id}"
        )


        output_path = (

            output_dir

            /

            f"{tile_id}_dea_{year}.tif"

        )


        # ----------------------------------------------------
        # Skip existing
        # ----------------------------------------------------

        if output_path.exists():

            print(
                "Already exists - "
                "skipping."
            )

            continue


        try:

            data = load_dea_tile(

                items,

                tile

            )


            save_dea_tile(

                data,

                output_path

            )


        except Exception as e:

            print()
            print(
                "ERROR:"
            )

            print(
                f"Tile: {tile_id}"
            )

            print(
                str(e)
            )

            print(
                "Continuing..."
            )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MELBOURNE DEA LAND COVER DOWNLOADER")
    print("=" * 70)


    print()
    print(
        "DEA product:"
    )

    print(
        DEA_PRODUCT
    )


    print()
    print(
        "Band:"
    )

    print(
        DEA_BAND
    )


    print()
    print(
        "Native resolution:"
    )

    print(
        f"{DEA_RESOLUTION} m"
    )


    print()
    print(
        "CRS:"
    )

    print(
        TARGET_CRS
    )


    # --------------------------------------------------------
    # Create tiles
    # --------------------------------------------------------

    tiles = create_tile_grid()


    print()
    print(
        f"Total tiles: "
        f"{len(tiles)}"
    )


    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        tiles
    )


    # --------------------------------------------------------
    # Process years
    # --------------------------------------------------------

    for year in YEARS:

        process_year(

            year,

            tiles

        )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("DEA DOWNLOAD COMPLETE")
    print("=" * 70)


    print()
    print(
        "Output:"
    )

    print(
        DEA_DIR
    )


    print()
    print(
        "Resolution:"
    )

    print(
        "Native DEA 30 m"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
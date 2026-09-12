
# See map values

# path = r"C:\earth-observation-change-detection\MelbourneData\dea_landcover\2023\melbourne_0000_dea_2023.tif"

# with rasterio.open(path) as src:

#     data = src.read(1)

#     values, counts = np.unique(
#         data[data != src.nodata],
#         return_counts=True
#     )

#     print("DEA class values:")
#     print("=" * 40)

#     for value, count in zip(values, counts):

#         print(
#             f"Class {value}: "
#             f"{count:,} pixels"
#         )
# ============================================================
# MELBOURNE URBAN CHANGE DETECTION
#
# DEA LAND COVER 30 m
#          ↓
# Resample to Sentinel-2 10 m grid
#          ↓
# Candidate Urban Change Map
#
# YEARS:
#     2023
#     2025
#
# DEA PRODUCT:
#     DEA Land Cover C3
#
# DEA BAND:
#     level4
#
# TARGET:
#     Sentinel-2 10 m grid
#
# CANDIDATE CHANGE:
#
#     2023 != Artificial Surface
#     AND
#     2025 == Artificial Surface
#
# OUTPUT LABEL:
#
#     0   = No candidate urban change
#     1   = Candidate urban change
#     255 = NoData
#
# ============================================================


import os
from pathlib import Path

import numpy as np
import rasterio

from rasterio.warp import reproject
from rasterio.enums import Resampling



ROOT = os.path.dirname(
    os.getcwd()
)


BASE_DIR = os.path.join(
    ROOT,
    "MelbourneData"
)


# ============================================================
# 2. INPUT DIRECTORIES
# ============================================================

# ------------------------------------------------------------
# DEA LAND COVER
# ------------------------------------------------------------

DEA_2023_DIR = os.path.join(
    BASE_DIR,
    "dea_landcover",
    "2023"
)


DEA_2025_DIR = os.path.join(
    BASE_DIR,
    "dea_landcover",
    "2025"
)


# ------------------------------------------------------------
# SENTINEL-2
#
# These are used as the 10 m reference grid.
# ------------------------------------------------------------

S2_2023_DIR = os.path.join(
    BASE_DIR,
    "images",
    "2023",
    "sentinel2"
)


S2_2025_DIR = os.path.join(
    BASE_DIR,
    "images",
    "2025",
    "sentinel2"
)


# ============================================================
# 3. OUTPUT DIRECTORIES
# ============================================================

# ------------------------------------------------------------
# General labels directory
# ------------------------------------------------------------

LABELS_DIR = os.path.join(
    BASE_DIR,
    "labels"
)


# ------------------------------------------------------------
# DEA 2023 resampled to 10 m
# ------------------------------------------------------------

DEA_10M_2023_DIR = os.path.join(
    BASE_DIR,
    "dea_10m_2023"
)


# ------------------------------------------------------------
# DEA 2025 resampled to 10 m
# ------------------------------------------------------------

DEA_10M_2025_DIR = os.path.join(
    BASE_DIR,
    "dea_10m_2025"
)


# ------------------------------------------------------------
# Candidate change labels
# ------------------------------------------------------------

CANDIDATE_DIR = os.path.join(
    BASE_DIR,
    "candidate_change_10m"
)


# ============================================================
# 4. CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [

    LABELS_DIR,

    DEA_10M_2023_DIR,

    DEA_10M_2025_DIR,

    CANDIDATE_DIR

]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# 5. DEA CLASS DEFINITIONS
# ============================================================

# DEA Land Cover C3 Level 4
#
# Class 93 = Artificial Surface
#
# This is the class we use to identify areas that are
# classified as artificial surface in DEA.
#
# Examples can include:
#
#     buildings
#     roads
#     other constructed/artificial surfaces
#
# IMPORTANT:
#
# This is NOT equivalent to "building only".
#
# It is an Artificial Surface class.
# ============================================================

ARTIFICIAL_CLASS = 93


# ============================================================
# 6. NODATA VALUE
# ============================================================

OUTPUT_NODATA = 255


# ============================================================
# 7. FIND SENTINEL-2 REFERENCE IMAGE
# ============================================================

def find_sentinel2_reference(tile_id):

    """
    Find the Sentinel-2 image that will be used
    as the 10 m reference grid.

    We prefer 2023 because the candidate change
    workflow is based on a common grid.

    If 2023 does not exist, use 2025.
    """

    candidates = [

        os.path.join(
            S2_2023_DIR,
            f"{tile_id}_s2_2023.tif"
        ),

        os.path.join(
            S2_2025_DIR,
            f"{tile_id}_s2_2025.tif"
        )

    ]


    for path in candidates:

        if os.path.exists(path):

            return path


    return None


# ============================================================
# 8. PRINT REFERENCE INFORMATION
# ============================================================

def print_reference_info(
    reference_path
):

    print()
    print(
        "Sentinel-2 reference:"
    )

    print(
        reference_path
    )


    with rasterio.open(
        reference_path
    ) as src:

        print()
        print(
            "Reference grid:"
        )

        print(
            f"  Width      : {src.width}"
        )

        print(
            f"  Height     : {src.height}"
        )

        print(
            f"  Bands      : {src.count}"
        )

        print(
            f"  CRS        : {src.crs}"
        )

        print(
            f"  Pixel width: {src.transform.a}"
        )

        print(
            f"  Pixel height: "
            f"{abs(src.transform.e)}"
        )

        print(
            f"  Bounds     : {src.bounds}"
        )


# ============================================================
# 9. RESAMPLE DEA 30 m → SENTINEL-2 10 m
# ============================================================

def resample_dea_to_sentinel(

    dea_path,

    reference_path,

    output_path

):

    print()
    print(
        "-" * 70
    )

    print(
        "RESAMPLING DEA 30 m → 10 m"
    )

    print(
        f"DEA:"
    )

    print(
        dea_path
    )

    print(
        f"Reference:"
    )

    print(
        reference_path
    )


    # ========================================================
    # OPEN DEA
    # ========================================================

    with rasterio.open(
        dea_path
    ) as dea:

        source = dea.read(
            1
        )

        source_transform = (
            dea.transform
        )

        source_crs = (
            dea.crs
        )

        source_nodata = (
            dea.nodata
        )


        print()
        print(
            "DEA properties:"
        )

        print(
            f"  Width  : {dea.width}"
        )

        print(
            f"  Height : {dea.height}"
        )

        print(
            f"  CRS    : {dea.crs}"
        )

        print(
            f"  Pixel  : "
            f"{abs(dea.transform.a)} m"
        )


        # ====================================================
        # OPEN SENTINEL-2
        # ====================================================

        with rasterio.open(
            reference_path
        ) as reference:

            target_width = (
                reference.width
            )

            target_height = (
                reference.height
            )

            target_transform = (
                reference.transform
            )

            target_crs = (
                reference.crs
            )


            print()
            print(
                "Target Sentinel-2 grid:"
            )

            print(
                f"  Width  : "
                f"{target_width}"
            )

            print(
                f"  Height : "
                f"{target_height}"
            )

            print(
                f"  CRS    : "
                f"{target_crs}"
            )

            print(
                f"  Pixel  : "
                f"{abs(target_transform.a)} m"
            )


            # =================================================
            # CREATE OUTPUT ARRAY
            # =================================================

            destination = np.full(

                (
                    target_height,
                    target_width
                ),

                OUTPUT_NODATA,

                dtype=np.uint8

            )


            # =================================================
            # DEA → SENTINEL GRID
            #
            # VERY IMPORTANT:
            #
            # DEA contains categorical classes.
            #
            # Therefore:
            #
            #     Resampling.nearest
            #
            # must be used.
            #
            # DO NOT use bilinear/cubic.
            # =================================================

            reproject(

                source=source,

                destination=destination,

                src_transform=(
                    source_transform
                ),

                src_crs=(
                    source_crs
                ),

                src_nodata=(
                    source_nodata
                    if source_nodata is not None
                    else OUTPUT_NODATA
                ),

                dst_transform=(
                    target_transform
                ),

                dst_crs=(
                    target_crs
                ),

                dst_nodata=(
                    OUTPUT_NODATA
                ),

                resampling=(
                    Resampling.nearest
                )

            )


            # =================================================
            # OUTPUT PROFILE
            # =================================================

            profile = {

                "driver":
                    "GTiff",

                "height":
                    target_height,

                "width":
                    target_width,

                "count":
                    1,

                "dtype":
                    "uint8",

                "crs":
                    target_crs,

                "transform":
                    target_transform,

                "nodata":
                    OUTPUT_NODATA,

                "compress":
                    "lzw"

            }


            # =================================================
            # SAVE
            # =================================================

            with rasterio.open(

                output_path,

                "w",

                **profile

            ) as dst:

                dst.write(

                    destination,

                    1

                )


    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )


# ============================================================
# 10. CREATE CANDIDATE CHANGE MAP
# ============================================================

def create_candidate_change(

    dea_2023_path,

    dea_2025_path,

    output_path

):

    print()
    print(
        "-" * 70
    )

    print(
        "CREATING CANDIDATE CHANGE MAP"
    )


    # ========================================================
    # READ DEA 2023
    # ========================================================

    with rasterio.open(
        dea_2023_path
    ) as src:

        data_2023 = src.read(
            1
        )

        profile = (
            src.profile.copy()
        )

        transform_2023 = (
            src.transform
        )

        crs_2023 = (
            src.crs
        )

        nodata_2023 = (
            src.nodata
        )


    # ========================================================
    # READ DEA 2025
    # ========================================================

    with rasterio.open(
        dea_2025_path
    ) as src:

        data_2025 = src.read(
            1
        )

        transform_2025 = (
            src.transform
        )

        crs_2025 = (
            src.crs
        )

        nodata_2025 = (
            src.nodata
        )


    # ========================================================
    # CHECK DIMENSIONS
    # ========================================================

    if data_2023.shape != data_2025.shape:

        raise RuntimeError(

            "\n2023 and 2025 DEA "
            "rasters have different "
            "dimensions.\n\n"

            f"2023: {data_2023.shape}\n"

            f"2025: {data_2025.shape}"

        )


    # ========================================================
    # CHECK CRS
    # ========================================================

    if crs_2023 != crs_2025:

        raise RuntimeError(

            "\n2023 and 2025 DEA "
            "CRS do not match.\n\n"

            f"2023: {crs_2023}\n"

            f"2025: {crs_2025}"

        )


    # ========================================================
    # CHECK TRANSFORM
    # ========================================================

    if transform_2023 != transform_2025:

        raise RuntimeError(

            "\n2023 and 2025 DEA "
            "grids do not match."

        )


    # ========================================================
    # VALID PIXELS
    # ========================================================

    valid_2023 = (

        data_2023
        !=
        OUTPUT_NODATA

    )


    valid_2025 = (

        data_2025
        !=
        OUTPUT_NODATA

    )


    # If source DEA has another nodata value,
    # remove it as well.

    if nodata_2023 is not None:

        valid_2023 &= (

            data_2023
            !=
            nodata_2023

        )


    if nodata_2025 is not None:

        valid_2025 &= (

            data_2025
            !=
            nodata_2025

        )


    valid = (

        valid_2023
        &
        valid_2025

    )


    # ========================================================
    # ARTIFICIAL SURFACE
    # ========================================================

    artificial_2023 = (

        data_2023
        ==
        ARTIFICIAL_CLASS

    )


    artificial_2025 = (

        data_2025
        ==
        ARTIFICIAL_CLASS

    )


    # ========================================================
    # CANDIDATE CHANGE
    #
    # 2023:
    #     NOT artificial surface
    #
    # 2025:
    #     artificial surface
    #
    # Therefore:
    #
    #     Non-artificial → Artificial
    #
    # ========================================================

    candidate = (

        valid

        &

        (~artificial_2023)

        &

        artificial_2025

    )


    # ========================================================
    # CREATE LABEL ARRAY
    # ========================================================

    label = np.zeros(

        data_2023.shape,

        dtype=np.uint8

    )


    # Candidate pixels = 1

    label[
        candidate
    ] = 1


    # Invalid pixels = 255

    label[
        ~valid
    ] = OUTPUT_NODATA


    # ========================================================
    # UPDATE PROFILE
    # ========================================================

    profile.update(

        driver="GTiff",

        dtype="uint8",

        count=1,

        nodata=OUTPUT_NODATA,

        compress="lzw"

    )


    # ========================================================
    # SAVE
    # ========================================================

    with rasterio.open(

        output_path,

        "w",

        **profile

    ) as dst:

        dst.write(

            label,

            1

        )


    # ========================================================
    # STATISTICS
    # ========================================================

    valid_pixels = int(
        np.sum(valid)
    )

    candidate_pixels = int(
        np.sum(candidate)
    )


    print()
    print(
        "Candidate change statistics:"
    )

    print(
        f"Valid pixels     : "
        f"{valid_pixels:,}"
    )

    print(
        f"Candidate pixels : "
        f"{candidate_pixels:,}"
    )


    if valid_pixels > 0:

        percentage = (

            candidate_pixels
            /
            valid_pixels
            *
            100

        )

        print(
            f"Candidate area   : "
            f"{percentage:.2f}%"
        )


    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )


# ============================================================
# 11. FIND TILE IDS
# ============================================================

def find_tile_ids():

    """
    Find tile IDs from DEA 2023 files.

    Expected names:

        melbourne_0000_dea_2023.tif
        melbourne_0001_dea_2023.tif
        ...

    """

    tile_ids = []


    if not os.path.exists(
        DEA_2023_DIR
    ):

        raise FileNotFoundError(

            "\nDEA 2023 directory "
            "does not exist:\n"

            f"{DEA_2023_DIR}"

        )


    for filename in os.listdir(
        DEA_2023_DIR
    ):

        if not filename.lower().endswith(
            ".tif"
        ):

            continue


        suffix = (
            "_dea_2023.tif"
        )


        if filename.endswith(
            suffix
        ):

            tile_id = (
                filename[
                    :-
                    len(suffix)
                ]
            )


            tile_ids.append(
                tile_id
            )


    return sorted(
        tile_ids
    )


# ============================================================
# 12. PROCESS ONE TILE
# ============================================================

def process_tile(
    tile_id
):

    print()
    print("=" * 70)

    print(
        f"PROCESSING TILE: "
        f"{tile_id}"
    )

    print("=" * 70)


    # ========================================================
    # INPUT DEA
    # ========================================================

    dea_2023_path = os.path.join(

        DEA_2023_DIR,

        f"{tile_id}_dea_2023.tif"

    )


    dea_2025_path = os.path.join(

        DEA_2025_DIR,

        f"{tile_id}_dea_2025.tif"

    )


    # ========================================================
    # FIND SENTINEL REFERENCE
    # ========================================================

    reference_path = (
        find_sentinel2_reference(
            tile_id
        )
    )


    # ========================================================
    # CHECK INPUTS
    # ========================================================

    missing = []


    if not os.path.exists(
        dea_2023_path
    ):

        missing.append(
            dea_2023_path
        )


    if not os.path.exists(
        dea_2025_path
    ):

        missing.append(
            dea_2025_path
        )


    if reference_path is None:

        missing.append(
            "Sentinel-2 reference image"
        )


    if missing:

        print()
        print(
            "MISSING INPUTS:"
        )


        for item in missing:

            print(
                f"  {item}"
            )


        return


    # ========================================================
    # SHOW REFERENCE
    # ========================================================

    print_reference_info(
        reference_path
    )


    # ========================================================
    # OUTPUT DEA 10 m
    # ========================================================

    dea_10m_2023_path = os.path.join(

        DEA_10M_2023_DIR,

        f"{tile_id}_dea10m_2023.tif"

    )


    dea_10m_2025_path = os.path.join(

        DEA_10M_2025_DIR,

        f"{tile_id}_dea10m_2025.tif"

    )


    # ========================================================
    # OUTPUT CANDIDATE
    # ========================================================

    candidate_path = os.path.join(

        CANDIDATE_DIR,

        f"{tile_id}_candidate_change_10m.tif"

    )


    # ========================================================
    # STEP 1
    #
    # DEA 2023 → 10 m
    # ========================================================

    if os.path.exists(
        dea_10m_2023_path
    ):

        print()
        print(
            "DEA 2023 10 m already exists:"
        )

        print(
            dea_10m_2023_path
        )

    else:

        resample_dea_to_sentinel(

            dea_2023_path,

            reference_path,

            dea_10m_2023_path

        )


    # ========================================================
    # STEP 2
    #
    # DEA 2025 → 10 m
    # ========================================================

    if os.path.exists(
        dea_10m_2025_path
    ):

        print()
        print(
            "DEA 2025 10 m already exists:"
        )

        print(
            dea_10m_2025_path
        )

    else:

        resample_dea_to_sentinel(

            dea_2025_path,

            reference_path,

            dea_10m_2025_path

        )


    # ========================================================
    # STEP 3
    #
    # CREATE CANDIDATE CHANGE
    # ========================================================

    if os.path.exists(
        candidate_path
    ):

        print()
        print(
            "Candidate map already exists:"
        )

        print(
            candidate_path
        )

    else:

        create_candidate_change(

            dea_10m_2023_path,

            dea_10m_2025_path,

            candidate_path

        )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "MELBOURNE DEA → 10 m "
        "CANDIDATE CHANGE PROCESSOR"
    )

    print("=" * 70)


    # ========================================================
    # SHOW DIRECTORIES
    # ========================================================

    print()
    print(
        "INPUT DIRECTORIES"
    )

    print(
        "-" * 70
    )

    print(
        f"DEA 2023:"
    )

    print(
        DEA_2023_DIR
    )

    print()

    print(
        f"DEA 2025:"
    )

    print(
        DEA_2025_DIR
    )

    print()

    print(
        f"S2 2023:"
    )

    print(
        S2_2023_DIR
    )

    print()

    print(
        f"S2 2025:"
    )

    print(
        S2_2025_DIR
    )


    print()
    print(
        "OUTPUT DIRECTORIES"
    )

    print(
        "-" * 70
    )

    print(
        f"DEA 10 m 2023:"
    )

    print(
        DEA_10M_2023_DIR
    )

    print()

    print(
        f"DEA 10 m 2025:"
    )

    print(
        DEA_10M_2025_DIR
    )

    print()

    print(
        f"Candidate change:"
    )

    print(
        CANDIDATE_DIR
    )


    # ========================================================
    # FIND TILES
    # ========================================================

    tile_ids = (
        find_tile_ids()
    )


    print()
    print(
        "=" * 70
    )

    print(
        f"DEA 2023 tiles found: "
        f"{len(tile_ids)}"
    )

    print(
        "=" * 70
    )


    if len(tile_ids) == 0:

        raise RuntimeError(

            "\nNo DEA 2023 tiles found.\n\n"

            "Expected files such as:\n"

            "melbourne_0000_dea_2023.tif\n"

            "melbourne_0001_dea_2023.tif\n"

            "melbourne_0002_dea_2023.tif\n"

        )


    # ========================================================
    # PROCESS ALL TILES
    # ========================================================

    successful = 0

    failed = 0


    for index, tile_id in enumerate(

        tile_ids,

        start=1

    ):

        print()
        print()
        print(
            "#" * 70
        )

        print(
            f"TILE {index}/{len(tile_ids)}"
        )

        print(
            f"{tile_id}"
        )

        print(
            "#" * 70
        )


        try:

            process_tile(
                tile_id
            )

            successful += 1


        except Exception as e:

            failed += 1


            print()
            print(
                "ERROR:"
            )

            print(
                str(e)
            )

            print()
            print(
                "Continuing with next tile..."
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()
    print("=" * 70)

    print(
        "PROCESSING COMPLETE"
    )

    print("=" * 70)


    print()
    print(
        f"Total tiles : "
        f"{len(tile_ids)}"
    )

    print(
        f"Successful  : "
        f"{successful}"
    )

    print(
        f"Failed      : "
        f"{failed}"
    )


    print()
    print(
        "DEA 10 m 2023:"
    )

    print(
        DEA_10M_2023_DIR
    )


    print()
    print(
        "DEA 10 m 2025:"
    )

    print(
        DEA_10M_2025_DIR
    )


    print()
    print(
        "Candidate change maps:"
    )

    print(
        CANDIDATE_DIR
    )


    print()
    print(
        "LABEL VALUES"
    )

    print(
        "  0   = No candidate urban change"
    )

    print(
        "  1   = Candidate urban change"
    )

    print(
        "  255 = NoData"
    )


    print()
    print(
        "ARTIFICIAL SURFACE CLASS"
    )

    print(
        f"  DEA class {ARTIFICIAL_CLASS}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
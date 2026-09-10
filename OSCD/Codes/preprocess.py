
# ============================================================
# OSCD LABEL GEOREFERENCING
#
# Purpose:
#
# OSCD labels may have NO CRS and NO geotransform.
#
# Example:
#
# Label:
#     CRS       = None
#     Transform = identity
#     Size      = 525 x 471
#
# Sentinel-2:
#     CRS       = EPSG:4326
#     Size      = 525 x 471
#     Transform = valid geographic transform
#
# If label dimensions are exactly equal to Sentinel-2 T1,
# we assume the label is already pixel-aligned.
#
# Therefore:
#
#     NO REPROJECTION
#
# We simply create a NEW GeoTIFF using:
#
#     label pixel values
#     Sentinel-2 CRS
#     Sentinel-2 transform
#     Sentinel-2 width
#     Sentinel-2 height
#
# This avoids copying potentially invalid TIFF metadata
# from the original OSCD label.
#
# Output:
#
# Data/
#   labels_crs_matched/
#       train/
#       val/
#       test/
#
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# DIRECTORIES
# ============================================================

PROJECT_ROOT = os.path.dirname(os.getcwd())

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "Data"
)

IMAGES_DIR = os.path.join(
    BASE_DIR,
    "images"
)

LABELS_DIR = os.path.join(
    BASE_DIR,
    "labels"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "labels_crs_matched"
)

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in SPLITS:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            split
        ),
        exist_ok=True
    )


# ============================================================
# CREATE GEOREFERENCED LABEL
# ============================================================

def create_georeferenced_label(
    label_path,
    sentinel2_path,
    output_path
):

    print()
    print("=" * 80)

    print("LABEL:")
    print(label_path)

    print()
    print("SENTINEL-2:")
    print(sentinel2_path)

    print("=" * 80)


    # ========================================================
    # READ SENTINEL-2
    # ========================================================

    with rasterio.open(
        sentinel2_path
    ) as s2:

        s2_crs = s2.crs
        s2_transform = s2.transform

        s2_width = s2.width
        s2_height = s2.height

        s2_bounds = s2.bounds

        print()
        print("Sentinel-2 information:")
        print(
            "   CRS       :",
            s2_crs
        )

        print(
            "   Width     :",
            s2_width
        )

        print(
            "   Height    :",
            s2_height
        )

        print(
            "   Transform :",
            s2_transform
        )

        print(
            "   Bounds    :",
            s2_bounds
        )


    # ========================================================
    # CHECK SENTINEL-2 CRS
    # ========================================================

    if s2_crs is None:

        raise RuntimeError(
            "Sentinel-2 has no CRS:\n"
            + sentinel2_path
        )


    # ========================================================
    # READ ORIGINAL LABEL
    # ========================================================

    with rasterio.open(
        label_path
    ) as label:

        label_width = label.width
        label_height = label.height
        label_count = label.count

        print()
        print("Original label information:")

        print(
            "   CRS       :",
            label.crs
        )

        print(
            "   Width     :",
            label_width
        )

        print(
            "   Height    :",
            label_height
        )

        print(
            "   Bands     :",
            label_count
        )

        print(
            "   Transform :",
            label.transform
        )

        # ----------------------------------------------------
        # Must contain exactly one band
        # ----------------------------------------------------

        if label_count != 1:

            raise RuntimeError(
                "Expected one label band but found "
                + str(label_count)
            )

        # ----------------------------------------------------
        # Read label pixels
        # ----------------------------------------------------

        label_data = label.read(1)


    # ========================================================
    # IMPORTANT:
    #
    # LABEL MUST HAVE SAME PIXEL DIMENSIONS
    # ========================================================

    if (
        label_width != s2_width
        or
        label_height != s2_height
    ):

        raise RuntimeError(

            "\nLabel dimensions do NOT match Sentinel-2.\n\n"

            "Label:\n"
            f"    {label_width} x {label_height}\n\n"

            "Sentinel-2:\n"
            f"    {s2_width} x {s2_height}\n\n"

            "Cannot safely assign Sentinel-2 georeferencing."
        )


    print()
    print(
        "Label dimensions match Sentinel-2."
    )

    print(
        "No reprojection is required."
    )


    # ========================================================
    # CHECK LABEL VALUES
    # ========================================================

    unique_values = np.unique(
        label_data
    )

    print()
    print(
        "Original label values:"
    )

    print(
        "   ",
        unique_values[:50]
    )


    # ========================================================
    # CONVERT LABEL TO UINT8
    #
    # OSCD change labels are categorical masks.
    #
    # Usually:
    #
    #     0 = no change
    #     1 = change
    #
    # uint8 is appropriate.
    # ========================================================

    label_data = label_data.astype(
        np.uint8
    )


    # ========================================================
    # REMOVE OLD OUTPUT IF IT EXISTS
    # ========================================================

    if os.path.exists(
        output_path
    ):

        try:

            os.remove(
                output_path
            )

        except PermissionError:

            raise RuntimeError(
                "\nCannot overwrite output file:\n"
                + output_path
                + "\n\n"
                "Close the file in QGIS/ArcGIS and "
                "run the script again."
            )


    # ========================================================
    # CREATE A COMPLETELY NEW TIFF PROFILE
    #
    # DO NOT COPY label.profile
    # ========================================================

    profile = {

        "driver": "GTiff",

        "dtype": "uint8",

        "width": s2_width,

        "height": s2_height,

        "count": 1,

        "crs": s2_crs,

        "transform": s2_transform,

        "nodata": 0,

        "compress": "lzw",

        "BIGTIFF": "IF_SAFER"
    }


    # ========================================================
    # WRITE NEW GEOTIFF
    # ========================================================

    print()
    print(
        "Creating new GeoTIFF..."
    )

    print(
        "Output:",
        output_path
    )


    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            label_data,
            1
        )


        # ----------------------------------------------------
        # Set useful metadata
        # ----------------------------------------------------

        dst.update_tags(
            DESCRIPTION="OSCD change detection label",
            SOURCE_LABEL=os.path.basename(label_path),
            REFERENCE_IMAGE=os.path.basename(sentinel2_path)
        )


    # ========================================================
    # REOPEN FILE
    #
    # This is VERY IMPORTANT.
    #
    # It verifies that GDAL/Rasterio can actually read
    # the newly created file.
    # ========================================================

    print()
    print(
        "Verifying output..."
    )


    try:

        with rasterio.open(
            output_path
        ) as check:

            print()
            print(
                "Output CRS:"
            )

            print(
                "   ",
                check.crs
            )

            print(
                "Output size:"
            )

            print(
                "   ",
                check.width,
                "x",
                check.height
            )

            print(
                "Output transform:"
            )

            print(
                "   ",
                check.transform
            )

            print(
                "Output bounds:"
            )

            print(
                "   ",
                check.bounds
            )

            print(
                "Output dtype:"
            )

            print(
                "   ",
                check.dtypes[0]
            )

            print(
                "Output bands:"
            )

            print(
                "   ",
                check.count
            )


            # ------------------------------------------------
            # Verify dimensions
            # ------------------------------------------------

            assert (
                check.width == s2_width
            )

            assert (
                check.height == s2_height
            )


            # ------------------------------------------------
            # Verify CRS
            # ------------------------------------------------

            assert (
                check.crs == s2_crs
            )


            # ------------------------------------------------
            # Verify transform
            # ------------------------------------------------

            assert (
                check.transform == s2_transform
            )


            # ------------------------------------------------
            # Read output pixels
            # ------------------------------------------------

            output_data = check.read(1)


    except Exception as e:

        raise RuntimeError(
            "\nThe output GeoTIFF could not be "
            "successfully reopened.\n\n"
            f"File:\n{output_path}\n\n"
            f"Error:\n{e}"
        )


    # ========================================================
    # VERIFY PIXEL VALUES WERE PRESERVED
    # ========================================================

    if not np.array_equal(
        label_data,
        output_data
    ):

        raise RuntimeError(
            "Label pixel values changed during processing."
        )


    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "SUCCESS"
    )
    print("=" * 80)

    print(
        "Created:"
    )

    print(
        output_path
    )

    print()
    print(
        "CRS match       : True"
    )

    print(
        "Size match      : True"
    )

    print(
        "Transform match : True"
    )

    print(
        "Pixels preserved: True"
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split
):

    print()
    print()
    print("#" * 80)

    print(
        "PROCESSING:",
        split.upper()
    )

    print("#" * 80)


    image_dir = os.path.join(
        IMAGES_DIR,
        split
    )

    label_dir = os.path.join(
        LABELS_DIR,
        split
    )

    output_dir = os.path.join(
        OUTPUT_DIR,
        split
    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # ========================================================
    # FIND SENTINEL-2 T1
    # ========================================================

    s2_files = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "sentinel2_*_t1.tif"
            )
        )
    )


    print()
    print(
        "Found",
        len(s2_files),
        "Sentinel-2 T1 images."
    )


    # ========================================================
    # PROCESS EACH CITY
    # ========================================================

    for s2_path in s2_files:

        filename = os.path.basename(
            s2_path
        )


        # ----------------------------------------------------
        # Extract city name
        #
        # sentinel2_aguasclaras_t1.tif
        #
        # -> aguasclaras
        # ----------------------------------------------------

        prefix = "sentinel2_"
        suffix = "_t1.tif"


        city = filename[
            len(prefix):-len(suffix)
        ]


        print()
        print(
            "-" * 80
        )

        print(
            "CITY:",
            city
        )

        print(
            "-" * 80
        )


        # ====================================================
        # PATHS
        # ====================================================

        label_path = os.path.join(
            label_dir,
            f"{city}_cm.tif"
        )


        output_path = os.path.join(
            output_dir,
            f"{city}_cm.tif"
        )


        # ====================================================
        # CHECK LABEL
        # ====================================================

        if not os.path.exists(
            label_path
        ):

            print(
                "WARNING: Label not found:"
            )

            print(
                label_path
            )

            print(
                "Skipping..."
            )

            continue


        # ====================================================
        # PROCESS
        # ====================================================

        try:

            create_georeferenced_label(

                label_path=label_path,

                sentinel2_path=s2_path,

                output_path=output_path
            )


        except Exception as e:

            print()
            print(
                "ERROR:"
            )

            print(
                str(e)
            )

            print()
            print(
                "Skipping this city..."
            )

            continue


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

print("OSCD project root:")
print(PROJECT_ROOT)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "Data"
)


# ============================================================
# LOAD SPLIT CSV
# ============================================================

def load_split_csv(split):
    """
    Load the CSV describing an OSCD dataset split.

    Expected files:
        Data/
            csv/
                train.csv
                val.csv
                test.csv
    """

    possible_paths = [

        os.path.join(
            DATA_DIR,
            f"{split}.csv"
        ),

        os.path.join(
            DATA_DIR,
            "csv",
            f"{split}.csv"
        ),

        os.path.join(
            PROJECT_ROOT,
            f"{split}.csv"
        ),
    ]


    csv_path = None


    for path in possible_paths:

        if os.path.isfile(path):

            csv_path = path

            break


    if csv_path is None:

        raise FileNotFoundError(

            f"\nCould not find CSV for split: "
            f"{split}\n\n"

            "Searched:\n"

            + "\n".join(
                possible_paths
            )
        )


    print(
        f"Loading {split} CSV:"
    )

    print(
        f"    {csv_path}"
    )


    dataframe = pd.read_csv(
        csv_path
    )


    print(
        f"    Samples: "
        f"{len(dataframe)}"
    )


    print(
        f"    Columns: "
        f"{list(dataframe.columns)}"
    )


    return dataframe
# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)

    print(
        "OSCD LABEL GEOREFERENCING"
    )

    print("=" * 80)

    print()
    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()
    print(
        "Input labels:"
    )

    print(
        LABELS_DIR
    )

    print()
    print(
        "Sentinel-2 images:"
    )

    print(
        IMAGES_DIR
    )

    print()
    print(
        "Output labels:"
    )

    print(
        OUTPUT_DIR
    )


    # ========================================================
    # PROCESS TRAIN / VAL / TEST
    # ========================================================

    for split in SPLITS:

        process_split(
            split
        )


    print()
    print()
    print("=" * 80)

    print(
        "ALL PROCESSING FINISHED"
    )

    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

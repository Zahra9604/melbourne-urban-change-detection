# ============================================================
# MELBOURNE 5-PANEL URBAN CHANGE LABELING GUI
# ============================================================
#
# PANEL 1:
#     Sentinel-2 2023 RGB
#
# PANEL 2:
#     Sentinel-2 2025 RGB
#
# PANEL 3:
#     DEA Candidate Change Map
#
# PANEL 4:
#     OSCD-Trained Model Change Map
#
# PANEL 5:
#     MANUAL CHANGE LABEL - EDITABLE
#
# ============================================================
#
# FEATURES
#
# 1. Existing manual labels are automatically loaded.
#
# 2. Manual labels are displayed in Panel 5.
#
# 3. Existing manual labels are shown as red overlays
#    on Sentinel-2 2023 and 2025.
#
# 4. Draw polygons on:
#       - Sentinel-2 2023
#       - Sentinel-2 2025
#       - Manual label
#
# 5. Delete Last Point
#
# 6. Delete Last Polygon
#
# 7. Save immediately after each polygon.
#
# 8. Zoom level is preserved after drawing.
#
# 9. Pan position is preserved after drawing.
#
# 10. Previous / Next automatically save the current tile.
#
# ============================================================
#
# MANUAL LABEL
#
#     0 = NO CHANGE
#     1 = CHANGE
#
# Saved to:
#
# C:\earth-observation-change-detection\
# Melbourne\MelbourneData\change_labels_10m\
#
# Example:
#
#     melbourne_0000_change_label_10m.tif
#
# ============================================================
#
# MOUSE CONTROLS
#
# Left click:
#     Add polygon point
#
# Double left click:
#     Finish polygon
#
# Right click:
#     Cancel current polygon
#
# Middle mouse:
#     Pan
#
# Mouse wheel:
#     Synchronized zoom
#
# ============================================================
#
# BUTTONS
#
# Save Label
#     Save current manual label
#
# Delete Last Polygon
#     Undo the most recently completed polygon
#     created during the current GUI session
#
# Delete Last Point
#     Remove the most recently added point from
#     the polygon currently being drawn
#
# Reset View
#     Return to full-tile view
#
# Drawing ON/OFF
#     Enable/disable polygon drawing
#
# ============================================================


import os
import re
import traceback
import tkinter as tk
from tkinter import messagebox

import numpy as np
import rasterio

from rasterio.features import rasterize

from shapely.geometry import Polygon

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.patches import Polygon as MplPolygon


# ============================================================
# PATHS
# ============================================================

root = os.getcwd()
PROJECT_ROOT = os.path.dirname(root)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "MelbourneData"
)


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


DEA_CANDIDATE_DIR = os.path.join(
    BASE_DIR,
    "candidate_change_10m"
)


MODEL_CHANGE_DIR = os.path.join(
    BASE_DIR,
    "change_maps"
)


# ============================================================
# MANUAL LABEL DIRECTORY
# ============================================================

LABEL_DIR = os.path.join(
    BASE_DIR,
    "change_labels_10m"
)

os.makedirs(
    LABEL_DIR,
    exist_ok=True
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

PERCENTILE_LOW = 2

PERCENTILE_HIGH = 98

MAX_DISPLAY_SIZE = 1800


# ============================================================
# LIST TIFF FILES
# ============================================================

def list_tif_files(folder):

    if not os.path.exists(folder):

        return []

    return sorted(
        [
            os.path.join(
                folder,
                filename
            )
            for filename in os.listdir(folder)
            if filename.lower().endswith(
                (
                    ".tif",
                    ".tiff"
                )
            )
        ]
    )


# ============================================================
# EXTRACT TILE ID
# ============================================================

def extract_tile_id(filename):

    name = os.path.basename(
        filename
    ).lower()

    match = re.search(
        r"(melbourne_\d+)",
        name
    )

    if match:

        return match.group(1)

    return os.path.splitext(
        name
    )[0]


# ============================================================
# MANUAL LABEL PATH
# ============================================================

def get_label_path(tile_id):

    return os.path.join(
        LABEL_DIR,
        f"{tile_id}_change_label_10m.tif"
    )


# ============================================================
# FIND MODEL CHANGE MAP
# ============================================================

def find_model_change_for_tile(
    tile_id,
    files
):

    matches = []

    for path in files:

        name = os.path.basename(
            path
        ).lower()

        if tile_id.lower() in name:

            matches.append(path)

    if not matches:

        return None

    preferred = [
        path
        for path in matches
        if "change" in os.path.basename(
            path
        ).lower()
    ]

    if preferred:

        return sorted(
            preferred
        )[0]

    return sorted(
        matches
    )[0]


# ============================================================
# BUILD TILE PAIRS
# ============================================================

def build_pairs():

    files_2023 = list_tif_files(
        S2_2023_DIR
    )

    files_2025 = list_tif_files(
        S2_2025_DIR
    )

    dea_files = list_tif_files(
        DEA_CANDIDATE_DIR
    )

    model_files = list_tif_files(
        MODEL_CHANGE_DIR
    )

    print()
    print("=" * 90)
    print("MELBOURNE CHANGE LABELING - TILE PAIRING")
    print("=" * 90)

    print()
    print(
        "2023 Sentinel-2:",
        len(files_2023)
    )

    print(
        "2025 Sentinel-2:",
        len(files_2025)
    )

    print(
        "DEA candidate maps:",
        len(dea_files)
    )

    print(
        "OSCD model change maps:",
        len(model_files)
    )

    # --------------------------------------------------------
    # Dictionaries
    # --------------------------------------------------------

    d2023 = {
        extract_tile_id(path): path
        for path in files_2023
    }

    d2025 = {
        extract_tile_id(path): path
        for path in files_2025
    }

    ddea = {
        extract_tile_id(path): path
        for path in dea_files
    }

    # --------------------------------------------------------
    # Require:
    #
    # S2 2023
    # S2 2025
    # DEA
    #
    # Model optional.
    # --------------------------------------------------------

    common_ids = sorted(
        set(d2023.keys())
        &
        set(d2025.keys())
        &
        set(ddea.keys())
    )

    pairs = []

    for tile_id in common_ids:

        model_path = find_model_change_for_tile(
            tile_id,
            model_files
        )

        pair = {

            "tile_id":
                tile_id,

            "s2_2023":
                d2023[tile_id],

            "s2_2025":
                d2025[tile_id],

            "dea":
                ddea[tile_id],

            "model":
                model_path
        }

        pairs.append(
            pair
        )

        label_path = get_label_path(
            tile_id
        )

        print()
        print("-" * 80)

        print(
            "TILE:",
            tile_id
        )

        print(
            "2023:",
            os.path.basename(
                pair["s2_2023"]
            )
        )

        print(
            "2025:",
            os.path.basename(
                pair["s2_2025"]
            )
        )

        print(
            "DEA:",
            os.path.basename(
                pair["dea"]
            )
        )

        if model_path:

            print(
                "MODEL:",
                os.path.basename(
                    model_path
                )
            )

        else:

            print(
                "MODEL: NOT FOUND"
            )

        if os.path.exists(
            label_path
        ):

            print(
                "MANUAL LABEL: EXISTS"
            )

        else:

            print(
                "MANUAL LABEL: NEW"
            )

    print()
    print("=" * 90)

    print(
        "Found",
        len(pairs),
        "matching tiles."
    )

    print("=" * 90)

    return pairs


# ============================================================
# RGB PERCENTILE STRETCH
# ============================================================

def percentile_stretch(rgb):

    rgb = rgb.astype(
        np.float32
    )

    output = np.zeros_like(
        rgb,
        dtype=np.float32
    )

    for i in range(3):

        band = rgb[:, :, i]

        valid = np.isfinite(
            band
        )

        if not np.any(valid):

            continue

        values = band[valid]

        low = np.percentile(
            values,
            PERCENTILE_LOW
        )

        high = np.percentile(
            values,
            PERCENTILE_HIGH
        )

        if high <= low:

            high = low + 1.0

        stretched = (
            band - low
        ) / (
            high - low
        )

        output[:, :, i] = np.clip(
            stretched,
            0,
            1
        )

    output = np.nan_to_num(
        output,
        nan=0,
        posinf=1,
        neginf=0
    )

    return output


# ============================================================
# READ SENTINEL RGB
# ============================================================

def read_sentinel_rgb(path):

    print()
    print("Reading Sentinel-2:")
    print(path)

    with rasterio.open(path) as src:

        data = src.read()

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        width = src.width

        height = src.height

        nodata = src.nodata

    if data.shape[0] < 3:

        raise ValueError(
            "Sentinel-2 image must contain at least 3 bands."
        )

    # --------------------------------------------------------
    # Expected:
    #
    # Band 1 = B2
    # Band 2 = B3
    # Band 3 = B4
    # Band 4 = B8
    #
    # RGB = B4, B3, B2
    # --------------------------------------------------------

    blue = data[0]

    green = data[1]

    red = data[2]

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=2
    ).astype(
        np.float32
    )

    if nodata is not None:

        invalid = (
            data[0] == nodata
        )

        rgb[invalid] = np.nan

    rgb = percentile_stretch(
        rgb
    )

    return {

        "rgb":
            rgb,

        "transform":
            transform,

        "crs":
            crs,

        "width":
            width,

        "height":
            height,

        "profile":
            profile
    }


# ============================================================
# PREPARE DISPLAY RGB
# ============================================================

def prepare_display_rgb(
    rgb,
    max_size=MAX_DISPLAY_SIZE
):

    height, width = (
        rgb.shape[:2]
    )

    scale = min(
        1.0,
        max_size / max(
            height,
            width
        )
    )

    if scale >= 1:

        return rgb

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    rows = np.linspace(
        0,
        height - 1,
        new_height
    ).astype(
        np.int32
    )

    cols = np.linspace(
        0,
        width - 1,
        new_width
    ).astype(
        np.int32
    )

    return rgb[
        rows[:, None],
        cols[None, :]
    ]


# ============================================================
# READ CHANGE MAP
# ============================================================

def read_change_map(path):

    if path is None:

        return None

    print()
    print("Reading change map:")
    print(path)

    with rasterio.open(path) as src:

        data = src.read(1)

        transform = src.transform

        crs = src.crs

        width = src.width

        height = src.height

    data = np.where(
        data > 0,
        1,
        0
    ).astype(
        np.uint8
    )

    print(
        "Shape:",
        data.shape
    )

    print(
        "CRS:",
        crs
    )

    print(
        "Unique values:",
        np.unique(data)
    )

    print(
        "Change pixels:",
        int(
            np.sum(data == 1)
        )
    )

    return {

        "data":
            data,

        "transform":
            transform,

        "crs":
            crs,

        "width":
            width,

        "height":
            height
    }


# ============================================================
# RASTER EXTENT
# ============================================================

def raster_extent(
    transform,
    width,
    height
):

    left = transform.c

    top = transform.f

    right = (
        left +
        transform.a * width
    )

    bottom = (
        top +
        transform.e * height
    )

    return [
        min(left, right),
        max(left, right),
        min(bottom, top),
        max(bottom, top)
    ]


# ============================================================
# CREATE EMPTY LABEL
# ============================================================

def create_empty_label(s2):

    return np.zeros(
        (
            s2["height"],
            s2["width"]
        ),
        dtype=np.uint8
    )


# ============================================================
# LOAD EXISTING MANUAL LABEL
# ============================================================

def load_existing_label(
    tile_id,
    s2
):

    path = get_label_path(
        tile_id
    )

    print()
    print("=" * 80)
    print(
        "CHECKING MANUAL LABEL:",
        tile_id
    )
    print("=" * 80)

    print(
        "Expected path:"
    )

    print(
        path
    )

    # --------------------------------------------------------
    # LABEL DOES NOT EXIST
    # --------------------------------------------------------

    if not os.path.exists(path):

        print()
        print(
            "No existing label."
        )

        print(
            "Creating empty label."
        )

        return create_empty_label(
            s2
        )

    # --------------------------------------------------------
    # LABEL EXISTS
    # --------------------------------------------------------

    print()
    print(
        "EXISTING LABEL FOUND!"
    )

    try:

        with rasterio.open(
            path
        ) as src:

            print(
                "Label:",
                path
            )

            print(
                "Label size:",
                src.width,
                "x",
                src.height
            )

            print(
                "Label CRS:",
                src.crs
            )

            print(
                "S2 CRS:",
                s2["crs"]
            )

            print(
                "Label transform:",
                src.transform
            )

            print(
                "S2 transform:",
                s2["transform"]
            )

            # ------------------------------------------------
            # Check dimensions
            # ------------------------------------------------

            if (
                src.width != s2["width"]
                or
                src.height != s2["height"]
            ):

                raise ValueError(
                    "Existing label dimensions do not "
                    "match Sentinel-2 2023."
                )

            # ------------------------------------------------
            # Check CRS
            # ------------------------------------------------

            if src.crs != s2["crs"]:

                raise ValueError(
                    "Existing label CRS does not "
                    "match Sentinel-2 2023."
                )

            # ------------------------------------------------
            # Check transform
            # ------------------------------------------------

            if src.transform != s2["transform"]:

                raise ValueError(
                    "Existing label transform does not "
                    "match Sentinel-2 2023."
                )

            data = src.read(1)

        data = np.where(
            data > 0,
            1,
            0
        ).astype(
            np.uint8
        )

        print()
        print(
            "Existing label successfully loaded."
        )

        print(
            "Unique values:",
            np.unique(data)
        )

        print(
            "Existing change pixels:",
            int(
                np.sum(data == 1)
            )
        )

        return data

    except Exception as e:

        traceback.print_exc()

        messagebox.showwarning(
            "Manual label problem",
            f"{tile_id}\n\n"
            f"Existing manual label could not be loaded.\n\n"
            f"{str(e)}\n\n"
            f"A new empty label will be created."
        )

        return create_empty_label(
            s2
        )


# ============================================================
# SAVE MANUAL LABEL
# ============================================================

def save_label(
    tile_id,
    label,
    s2
):

    path = get_label_path(
        tile_id
    )

    label = np.where(
        label > 0,
        1,
        0
    ).astype(
        np.uint8
    )

    profile = s2["profile"].copy()

    profile.update(
        {

            "driver":
                "GTiff",

            "height":
                s2["height"],

            "width":
                s2["width"],

            "count":
                1,

            "dtype":
                "uint8",

            "crs":
                s2["crs"],

            "transform":
                s2["transform"],

            "nodata":
                None,

            "compress":
                "lzw",

            "BIGTIFF":
                "IF_SAFER"
        }
    )

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            label,
            1
        )

    print()
    print("=" * 80)
    print(
        "MANUAL LABEL SAVED"
    )
    print("=" * 80)

    print(
        "File:",
        path
    )

    print(
        "Shape:",
        label.shape
    )

    print(
        "CRS:",
        s2["crs"]
    )

    print(
        "Unique values:",
        np.unique(label)
    )

    print(
        "Change pixels:",
        int(
            np.sum(label == 1)
        )
    )

    return path


# ============================================================
# GUI CLASS
# ============================================================

class MelbourneChangeGUI:

    def __init__(
        self,
        root,
        pairs
    ):

        self.root = root

        self.pairs = pairs

        # ----------------------------------------------------
        # ALWAYS START FROM FIRST TILE
        # ----------------------------------------------------

        self.index = 0

        # ----------------------------------------------------
        # Raster data
        # ----------------------------------------------------

        self.s2_2023 = None

        self.s2_2025 = None

        self.dea = None

        self.model = None

        self.label = None

        self.tile_id = None

        self.extent = None

        # ----------------------------------------------------
        # Drawing
        # ----------------------------------------------------

        self.drawing = True

        self.current_polygon = []

        self.polygon_artists = []

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Each time a polygon is completed we store a copy
        # of the label BEFORE that polygon was added.
        #
        # This allows:
        #
        #     Delete Last Polygon
        #
        # ----------------------------------------------------

        self.label_history = []

        # ----------------------------------------------------
        # Temporary polygon artists
        # ----------------------------------------------------

        self.temp_polygon_artist_2023 = None

        self.temp_polygon_artist_2025 = None

        self.temp_polygon_artist_manual = None

        # ----------------------------------------------------
        # Pan
        # ----------------------------------------------------

        self.panning = False

        self.pan_start = None

        self.pan_start_xlim = None

        self.pan_start_ylim = None

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        self.manual_image = None

        # ----------------------------------------------------
        # GUI
        # ----------------------------------------------------

        self.create_gui()

        # ----------------------------------------------------
        # Load first tile
        # ----------------------------------------------------

        self.load_tile(0)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )


    # ========================================================
    # CREATE GUI
    # ========================================================

    def create_gui(self):

        self.root.title(
            "Melbourne Urban Change Labeling - 5 Panels"
        )

        self.root.geometry(
            "1800x1050"
        )

        # ----------------------------------------------------
        # TOOLBAR
        # ----------------------------------------------------

        toolbar = tk.Frame(
            self.root,
            bg="#222222"
        )

        toolbar.pack(
            side=tk.TOP,
            fill=tk.X
        )

        tk.Label(
            toolbar,
            text="MELBOURNE CHANGE LABELING",
            fg="white",
            bg="#222222",
            font=(
                "Arial",
                14,
                "bold"
            )
        ).pack(
            side=tk.LEFT,
            padx=10,
            pady=8
        )

        self.previous_button = tk.Button(
            toolbar,
            text="◀ Previous",
            command=self.previous_tile,
            width=11
        )

        self.previous_button.pack(
            side=tk.LEFT,
            padx=3
        )

        self.next_button = tk.Button(
            toolbar,
            text="Next ▶",
            command=self.next_tile,
            width=11
        )

        self.next_button.pack(
            side=tk.LEFT,
            padx=3
        )

        tk.Button(
            toolbar,
            text="Save Label",
            command=self.save_current_label,
            width=12
        ).pack(
            side=tk.LEFT,
            padx=3
        )

        # ----------------------------------------------------
        # NEW:
        # DELETE LAST POLYGON
        # ----------------------------------------------------

        tk.Button(
            toolbar,
            text="Delete Last Polygon",
            command=self.delete_last_polygon,
            width=18
        ).pack(
            side=tk.LEFT,
            padx=3
        )

        # ----------------------------------------------------
        # NEW:
        # DELETE LAST POINT
        # ----------------------------------------------------

        tk.Button(
            toolbar,
            text="Delete Last Point",
            command=self.delete_last_point,
            width=16
        ).pack(
            side=tk.LEFT,
            padx=3
        )

        tk.Button(
            toolbar,
            text="Reset View",
            command=self.reset_view,
            width=12
        ).pack(
            side=tk.LEFT,
            padx=3
        )

        self.draw_button = tk.Button(
            toolbar,
            text="Drawing ON",
            command=self.toggle_drawing,
            width=13
        )

        self.draw_button.pack(
            side=tk.LEFT,
            padx=15
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status = tk.StringVar()

        tk.Label(
            self.root,
            textvariable=self.status,
            anchor="w",
            font=(
                "Arial",
                10
            )
        ).pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=3
        )

        # ----------------------------------------------------
        # HELP
        # ----------------------------------------------------

        tk.Label(
            self.root,
            text=(
                "Left click = polygon point | "
                "Double-click = finish polygon | "
                "Right click = cancel | "
                "Mouse wheel = synchronized zoom | "
                "Middle mouse = synchronized pan | "
                "Delete Last Point = remove current point | "
                "Delete Last Polygon = undo last polygon"
            ),
            anchor="w",
            font=(
                "Arial",
                9
            )
        ).pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=2
        )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        self.figure = Figure(
            figsize=(
                18,
                10
            ),
            dpi=100
        )

        gs = self.figure.add_gridspec(
            2,
            4,
            height_ratios=[
                1,
                1.35
            ],
            hspace=0.15,
            wspace=0.08
        )

        self.ax_2023 = self.figure.add_subplot(
            gs[0, 0]
        )

        self.ax_2025 = self.figure.add_subplot(
            gs[0, 1]
        )

        self.ax_dea = self.figure.add_subplot(
            gs[0, 2]
        )

        self.ax_model = self.figure.add_subplot(
            gs[0, 3]
        )

        self.ax_manual = self.figure.add_subplot(
            gs[1, :]
        )

        self.figure.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.96,
            bottom=0.04,
            hspace=0.12,
            wspace=0.06
        )

        # ----------------------------------------------------
        # CANVAS
        # ----------------------------------------------------

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.root
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # MOUSE EVENTS
        # ----------------------------------------------------

        self.canvas.mpl_connect(
            "button_press_event",
            self.mouse_press
        )

        self.canvas.mpl_connect(
            "button_release_event",
            self.mouse_release
        )

        self.canvas.mpl_connect(
            "motion_notify_event",
            self.mouse_move
        )

        self.canvas.mpl_connect(
            "scroll_event",
            self.mouse_scroll
        )

        # ----------------------------------------------------
        # KEYBOARD SHORTCUTS
        # ----------------------------------------------------

        self.root.bind(
            "<Left>",
            lambda event:
            self.previous_tile()
        )

        self.root.bind(
            "<Right>",
            lambda event:
            self.next_tile()
        )

        self.root.bind(
            "r",
            lambda event:
            self.reset_view()
        )

        # Backspace = Delete Last Point
        self.root.bind(
            "<BackSpace>",
            lambda event:
            self.delete_last_point()
        )

        # Ctrl+Z = Delete Last Polygon
        self.root.bind(
            "<Control-z>",
            lambda event:
            self.delete_last_polygon()
        )


    # ========================================================
    # LOAD TILE
    # ========================================================

    def load_tile(
        self,
        index
    ):

        # ----------------------------------------------------
        # Save current label before changing tile
        # ----------------------------------------------------

        if (
            self.label is not None
            and
            self.tile_id is not None
        ):

            self.save_current_label()

        self.index = index

        pair = self.pairs[
            self.index
        ]

        self.tile_id = pair[
            "tile_id"
        ]

        self.status.set(
            f"Loading {self.tile_id}..."
        )

        self.root.update_idletasks()

        try:

            # ------------------------------------------------
            # Load Sentinel-2 2023
            # ------------------------------------------------

            self.s2_2023 = read_sentinel_rgb(
                pair["s2_2023"]
            )

            # ------------------------------------------------
            # Load Sentinel-2 2025
            # ------------------------------------------------

            self.s2_2025 = read_sentinel_rgb(
                pair["s2_2025"]
            )

            # ------------------------------------------------
            # Load DEA
            # ------------------------------------------------

            self.dea = read_change_map(
                pair["dea"]
            )

            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            if pair["model"]:

                self.model = read_change_map(
                    pair["model"]
                )

            else:

                self.model = None

            # ------------------------------------------------
            # Load existing manual label
            # ------------------------------------------------

            self.label = load_existing_label(
                self.tile_id,
                self.s2_2023
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # New tile = new undo history.
            #
            # Existing saved label becomes the baseline.
            # ------------------------------------------------

            self.label_history = []

            # ------------------------------------------------
            # Check alignment
            # ------------------------------------------------

            self.check_alignment()

            # ------------------------------------------------
            # Reset drawing
            # ------------------------------------------------

            self.current_polygon = []

            self.clear_polygon_artists()

            self.drawing = True

            self.update_draw_button()

            # ------------------------------------------------
            # Display tile
            #
            # preserve_view=False because this is a new tile.
            # ------------------------------------------------

            self.display_tile(
                preserve_view=False
            )

            self.reset_view()

            self.update_status()

        except Exception as e:

            traceback.print_exc()

            messagebox.showerror(
                "Error loading tile",
                str(e)
            )


    # ========================================================
    # CHECK ALIGNMENT
    # ========================================================

    def check_alignment(self):

        print()
        print("=" * 80)
        print(
            "GRID CHECK:",
            self.tile_id
        )
        print("=" * 80)

        print(
            "S2 2023:",
            self.s2_2023["width"],
            "x",
            self.s2_2023["height"]
        )

        print(
            "S2 2025:",
            self.s2_2025["width"],
            "x",
            self.s2_2025["height"]
        )

        if self.dea:

            print(
                "DEA:",
                self.dea["width"],
                "x",
                self.dea["height"]
            )

        if self.model:

            print(
                "MODEL:",
                self.model["width"],
                "x",
                self.model["height"]
            )

        print(
            "MANUAL:",
            self.label.shape
        )

        print(
            "Manual unique values:",
            np.unique(self.label)
        )


    # ========================================================
    # SAVE CURRENT VIEW
    #
    # This is the key fix for the zoom problem.
    #
    # Before display_tile() clears the axes, we save:
    #
    #     xlim
    #     ylim
    #
    # After recreating the images, we restore them.
    # ========================================================

    def get_current_view(self):

        try:

            xlim = self.ax_2023.get_xlim()

            ylim = self.ax_2023.get_ylim()

            return (
                xlim,
                ylim
            )

        except Exception:

            return None


    # ========================================================
    # RESTORE VIEW
    # ========================================================

    def restore_view(
        self,
        view
    ):

        if view is None:

            return

        xlim, ylim = view

        axes = [
            self.ax_2023,
            self.ax_2025,
            self.ax_dea,
            self.ax_model,
            self.ax_manual
        ]

        for ax in axes:

            ax.set_xlim(
                xlim
            )

            ax.set_ylim(
                ylim
            )


    # ========================================================
    # DISPLAY TILE
    # ========================================================

    def display_tile(
        self,
        preserve_view=True
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Save current zoom BEFORE clearing the axes.
        # ----------------------------------------------------

        previous_view = None

        if preserve_view:

            previous_view = self.get_current_view()

        # ----------------------------------------------------
        # Clear axes
        # ----------------------------------------------------

        self.ax_2023.clear()

        self.ax_2025.clear()

        self.ax_dea.clear()

        self.ax_model.clear()

        self.ax_manual.clear()

        # ----------------------------------------------------
        # Sentinel extent
        # ----------------------------------------------------

        self.extent = raster_extent(
            self.s2_2023["transform"],
            self.s2_2023["width"],
            self.s2_2023["height"]
        )

        # ----------------------------------------------------
        # Prepare RGB
        # ----------------------------------------------------

        rgb2023 = prepare_display_rgb(
            self.s2_2023["rgb"]
        )

        rgb2025 = prepare_display_rgb(
            self.s2_2025["rgb"]
        )

        # ====================================================
        # PANEL 1 - 2023
        # ====================================================

        self.ax_2023.imshow(
            rgb2023,
            extent=self.extent,
            origin="upper",
            interpolation="nearest"
        )

        self.ax_2023.set_title(
            "Sentinel-2 2023 RGB",
            fontsize=11,
            fontweight="bold"
        )

        # ====================================================
        # PANEL 2 - 2025
        # ====================================================

        self.ax_2025.imshow(
            rgb2025,
            extent=self.extent,
            origin="upper",
            interpolation="nearest"
        )

        self.ax_2025.set_title(
            "Sentinel-2 2025 RGB",
            fontsize=11,
            fontweight="bold"
        )

        # ====================================================
        # MANUAL LABEL OVERLAY
        # ====================================================

        label_mask = np.ma.masked_where(
            self.label == 0,
            self.label
        )

        self.ax_2023.imshow(
            label_mask,
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=0.45,
            extent=self.extent,
            origin="upper",
            interpolation="nearest",
            zorder=20
        )

        self.ax_2025.imshow(
            label_mask,
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=0.45,
            extent=self.extent,
            origin="upper",
            interpolation="nearest",
            zorder=20
        )

        # ====================================================
        # PANEL 3 - DEA
        # ====================================================

        if self.dea:

            dea_extent = raster_extent(
                self.dea["transform"],
                self.dea["width"],
                self.dea["height"]
            )

            self.ax_dea.imshow(
                self.dea["data"],
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=dea_extent,
                origin="upper",
                interpolation="nearest"
            )

        self.ax_dea.set_title(
            "DEA Candidate Change",
            fontsize=11,
            fontweight="bold"
        )

        # ====================================================
        # PANEL 4 - MODEL
        # ====================================================

        if self.model:

            model_extent = raster_extent(
                self.model["transform"],
                self.model["width"],
                self.model["height"]
            )

            self.ax_model.imshow(
                self.model["data"],
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=model_extent,
                origin="upper",
                interpolation="nearest"
            )

            model_title = (
                "OSCD Model Change"
            )

        else:

            model_title = (
                "OSCD Model Change\n"
                "NOT FOUND"
            )

        self.ax_model.set_title(
            model_title,
            fontsize=11,
            fontweight="bold"
        )

        # ====================================================
        # PANEL 5 - MANUAL LABEL
        # ====================================================

        self.manual_image = self.ax_manual.imshow(
            self.label,
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=self.extent,
            origin="upper",
            interpolation="nearest"
        )

        self.ax_manual.set_title(
            "MANUAL CHANGE LABEL — EDITABLE",
            fontsize=14,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        change_pixels = int(
            np.sum(
                self.label == 1
            )
        )

        total_pixels = (
            self.label.shape[0]
            *
            self.label.shape[1]
        )

        if total_pixels > 0:

            change_percent = (
                change_pixels
                /
                total_pixels
                *
                100
            )

        else:

            change_percent = 0

        # ----------------------------------------------------
        # Information box
        # ----------------------------------------------------

        self.ax_manual.text(
            0.01,
            0.02,
            (
                f"0 = NO CHANGE   |   "
                f"1 = CHANGE   |   "
                f"Change pixels: {change_pixels:,}   |   "
                f"Change: {change_percent:.2f}%"
            ),
            transform=self.ax_manual.transAxes,
            fontsize=10,
            color="yellow",
            backgroundcolor="black",
            zorder=100
        )

        # ----------------------------------------------------
        # Existing label indicator
        # ----------------------------------------------------

        label_path = get_label_path(
            self.tile_id
        )

        if os.path.exists(label_path):

            label_text = (
                "EXISTING LABEL LOADED"
            )

            label_color = "green"

        else:

            label_text = (
                "NEW EMPTY LABEL"
            )

            label_color = "black"

        self.ax_manual.text(
            0.99,
            0.98,
            label_text,
            transform=self.ax_manual.transAxes,
            fontsize=10,
            color="white",
            backgroundcolor=label_color,
            horizontalalignment="right",
            verticalalignment="top",
            zorder=100
        )

        # ====================================================
        # HIDE COORDINATES
        # ====================================================

        for ax in [
            self.ax_2023,
            self.ax_2025,
            self.ax_dea,
            self.ax_model,
            self.ax_manual
        ]:

            ax.set_xlabel("")

            ax.set_ylabel("")

            ax.set_xticklabels([])

            ax.set_yticklabels([])

            ax.tick_params(
                axis="both",
                which="both",
                length=0
            )

        # ====================================================
        # IMPORTANT:
        #
        # Restore previous zoom AFTER imshow().
        #
        # This prevents Matplotlib from zooming out.
        # ====================================================

        if preserve_view:

            self.restore_view(
                previous_view
            )

        self.canvas.draw_idle()


    # ========================================================
    # TOGGLE DRAWING
    # ========================================================

    def toggle_drawing(self):

        self.drawing = not self.drawing

        if not self.drawing:

            self.cancel_polygon()

            self.status.set(
                "Drawing OFF."
            )

        else:

            self.status.set(
                "Drawing ON — "
                "click 2023, 2025 or Manual Label."
            )

        self.update_draw_button()


    # ========================================================
    # UPDATE DRAW BUTTON
    # ========================================================

    def update_draw_button(self):

        if self.drawing:

            self.draw_button.config(
                text="Drawing ON",
                relief=tk.SUNKEN
            )

        else:

            self.draw_button.config(
                text="Drawing OFF",
                relief=tk.RAISED
            )


    # ========================================================
    # MOUSE PRESS
    # ========================================================

    def mouse_press(
        self,
        event
    ):

        # ====================================================
        # RIGHT CLICK
        # ====================================================

        if event.button == 3:

            self.cancel_polygon()

            return

        # ====================================================
        # MIDDLE CLICK - PAN
        # ====================================================

        if event.button == 2:

            valid_axes = [
                self.ax_2023,
                self.ax_2025,
                self.ax_dea,
                self.ax_model,
                self.ax_manual
            ]

            if event.inaxes not in valid_axes:

                return

            if (
                event.xdata is None
                or
                event.ydata is None
            ):

                return

            self.panning = True

            self.pan_start = (
                event.xdata,
                event.ydata
            )

            self.pan_start_xlim = (
                event.inaxes.get_xlim()
            )

            self.pan_start_ylim = (
                event.inaxes.get_ylim()
            )

            return

        # ====================================================
        # LEFT CLICK
        # ====================================================

        if event.button != 1:

            return

        if not self.drawing:

            return

        drawable_axes = [
            self.ax_2023,
            self.ax_2025,
            self.ax_manual
        ]

        if event.inaxes not in drawable_axes:

            return

        if (
            event.xdata is None
            or
            event.ydata is None
        ):

            return

        # ====================================================
        # DOUBLE CLICK = FINISH POLYGON
        # ====================================================

        if event.dblclick:

            if len(
                self.current_polygon
            ) >= 2:

                self.current_polygon.append(
                    (
                        event.xdata,
                        event.ydata
                    )
                )

                self.finish_polygon()

            return

        # ====================================================
        # ADD POINT
        # ====================================================

        self.current_polygon.append(
            (
                event.xdata,
                event.ydata
            )
        )

        self.update_temp_polygon()


    # ========================================================
    # MOUSE RELEASE
    # ========================================================

    def mouse_release(
        self,
        event
    ):

        if event.button == 2:

            self.panning = False

            self.pan_start = None


    # ========================================================
    # TEMPORARY POLYGON
    # ========================================================

    def update_temp_polygon(self):

        self.remove_temp_polygon_artists()

        if len(
            self.current_polygon
        ) < 2:

            self.canvas.draw_idle()

            return

        # ----------------------------------------------------
        # 2023
        # ----------------------------------------------------

        self.temp_polygon_artist_2023 = MplPolygon(
            self.current_polygon,
            closed=False,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
            zorder=100
        )

        self.ax_2023.add_patch(
            self.temp_polygon_artist_2023
        )

        # ----------------------------------------------------
        # 2025
        # ----------------------------------------------------

        self.temp_polygon_artist_2025 = MplPolygon(
            self.current_polygon,
            closed=False,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
            zorder=100
        )

        self.ax_2025.add_patch(
            self.temp_polygon_artist_2025
        )

        # ----------------------------------------------------
        # Manual
        # ----------------------------------------------------

        self.temp_polygon_artist_manual = MplPolygon(
            self.current_polygon,
            closed=False,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
            zorder=100
        )

        self.ax_manual.add_patch(
            self.temp_polygon_artist_manual
        )

        self.canvas.draw_idle()


    # ========================================================
    # DELETE LAST POINT
    # ========================================================
    #
    # Removes the last point from the polygon currently being
    # drawn.
    #
    # It does NOT modify the saved manual raster because the
    # polygon has not been finished yet.
    #
    # Keyboard:
    #
    #     Backspace
    #
    # ========================================================

    def delete_last_point(self):

        if not self.current_polygon:

            self.status.set(
                "No current polygon points to delete."
            )

            return

        removed_point = (
            self.current_polygon.pop()
        )

        print()
        print(
            "Deleted last point:",
            removed_point
        )

        self.remove_temp_polygon_artists()

        if len(
            self.current_polygon
        ) >= 2:

            self.update_temp_polygon()

        else:

            self.canvas.draw_idle()

        self.status.set(
            f"Deleted last point. "
            f"Current polygon points: "
            f"{len(self.current_polygon)}"
        )


    # ========================================================
    # FINISH POLYGON
    # ========================================================

    def finish_polygon():

        pass


    # ========================================================
    # FINISH POLYGON
    # ========================================================

    def finish_polygon(
        self
    ):

        if len(
            self.current_polygon
        ) < 3:

            self.cancel_polygon()

            return

        points = list(
            self.current_polygon
        )

        if points[0] != points[-1]:

            points.append(
                points[0]
            )

        try:

            polygon = Polygon(
                points
            )

            if not polygon.is_valid:

                polygon = polygon.buffer(
                    0
                )

            if polygon.is_empty:

                raise ValueError(
                    "Polygon is empty."
                )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Save a copy BEFORE adding this polygon.
            #
            # This is used by:
            #
            #     Delete Last Polygon
            # ------------------------------------------------

            previous_label = (
                self.label.copy()
            )

            self.label_history.append(
                previous_label
            )

            # ------------------------------------------------
            # Rasterize
            # ------------------------------------------------

            self.rasterize_polygon(
                polygon
            )

            # ------------------------------------------------
            # SAVE IMMEDIATELY
            # ------------------------------------------------

            self.save_current_label()

            # ------------------------------------------------
            # Remove temporary polygon
            # ------------------------------------------------

            self.remove_temp_polygon_artists()

            # ------------------------------------------------
            # Clear current polygon
            # ------------------------------------------------

            self.current_polygon = []

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Redraw the label but PRESERVE the current
            # zoom and pan.
            # ------------------------------------------------

            self.display_tile(
                preserve_view=True
            )

            # ------------------------------------------------
            # Draw finished polygon outlines
            # ------------------------------------------------

            for ax in [
                self.ax_2023,
                self.ax_2025,
                self.ax_manual
            ]:

                artist = MplPolygon(
                    points,
                    closed=True,
                    facecolor="red",
                    edgecolor="yellow",
                    alpha=0.35,
                    linewidth=1.5,
                    zorder=100
                )

                ax.add_patch(
                    artist
                )

                self.polygon_artists.append(
                    artist
                )

            self.update_status()

            self.canvas.draw_idle()

        except Exception as e:

            traceback.print_exc()

            messagebox.showerror(
                "Polygon error",
                str(e)
            )


    # ========================================================
    # DELETE LAST POLYGON
    # ========================================================
    #
    # Restores the label to the state BEFORE the most recently
    # completed polygon.
    #
    # IMPORTANT:
    #
    # This works for polygons created during the current GUI
    # session.
    #
    # If the program was closed and reopened, the GeoTIFF only
    # contains the combined raster label, so the individual
    # previous polygon cannot be reconstructed.
    #
    # Keyboard:
    #
    #     Ctrl + Z
    #
    # ========================================================

    def delete_last_polygon(self):

        # ----------------------------------------------------
        # If a polygon is currently being drawn:
        #
        # Do not delete a completed polygon.
        # Instead remove the last point.
        # ----------------------------------------------------

        if self.current_polygon:

            self.delete_last_point()

            return

        # ----------------------------------------------------
        # Nothing to undo
        # ----------------------------------------------------

        if not self.label_history:

            self.status.set(
                "No polygon available to delete."
            )

            messagebox.showinfo(
                "Delete Last Polygon",
                "There is no polygon from the current "
                "session that can be undone."
            )

            return

        # ----------------------------------------------------
        # Restore previous label
        # ----------------------------------------------------

        self.label = (
            self.label_history.pop()
        )

        # ----------------------------------------------------
        # Remove last polygon artist
        # ----------------------------------------------------

        if self.polygon_artists:

            artist = (
                self.polygon_artists.pop()
            )

            try:

                artist.remove()

            except Exception:

                pass

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Save immediately after undo.
        # ----------------------------------------------------

        self.save_current_label()

        # ----------------------------------------------------
        # Redraw while preserving zoom.
        # ----------------------------------------------------

        self.display_tile(
            preserve_view=True
        )

        self.update_status()

        self.canvas.draw_idle()

        print()
        print(
            "Last polygon deleted."
        )

        print(
            "Remaining undo states:",
            len(self.label_history)
        )


    # ========================================================
    # RASTERIZE POLYGON
    # ========================================================

    def rasterize_polygon(
        self,
        polygon
    ):

        shape = (
            self.s2_2023["height"],
            self.s2_2023["width"]
        )

        transform = (
            self.s2_2023["transform"]
        )

        polygon_mask = rasterize(
            [
                (
                    polygon,
                    1
                )
            ],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

        # ----------------------------------------------------
        # Add change pixels
        # ----------------------------------------------------

        self.label[
            polygon_mask == 1
        ] = 1

        self.label = np.where(
            self.label == 1,
            1,
            0
        ).astype(
            np.uint8
        )

        print()
        print(
            "Polygon added."
        )

        print(
            "Current change pixels:",
            int(
                np.sum(
                    self.label == 1
                )
            )
        )


    # ========================================================
    # REMOVE TEMP POLYGONS
    # ========================================================

    def remove_temp_polygon_artists(self):

        for attr in [
            "temp_polygon_artist_2023",
            "temp_polygon_artist_2025",
            "temp_polygon_artist_manual"
        ]:

            artist = getattr(
                self,
                attr,
                None
            )

            if artist:

                try:

                    artist.remove()

                except Exception:

                    pass

                setattr(
                    self,
                    attr,
                    None
                )


    # ========================================================
    # CANCEL POLYGON
    # ========================================================

    def cancel_polygon(self):

        self.current_polygon = []

        self.remove_temp_polygon_artists()

        self.canvas.draw_idle()


    # ========================================================
    # CLEAR POLYGON ARTISTS
    # ========================================================

    def clear_polygon_artists(self):

        for artist in self.polygon_artists:

            try:

                artist.remove()

            except Exception:

                pass

        self.polygon_artists = []

        self.cancel_polygon()


    # ========================================================
    # SAVE CURRENT LABEL
    # ========================================================

    def save_current_label(self):

        if (
            self.label is None
            or
            self.s2_2023 is None
            or
            self.tile_id is None
        ):

            return

        try:

            path = save_label(
                self.tile_id,
                self.label,
                self.s2_2023
            )

            change_pixels = int(
                np.sum(
                    self.label == 1
                )
            )

            self.status.set(
                f"{self.tile_id} | "
                f"SAVED | "
                f"Change pixels: "
                f"{change_pixels:,}"
            )

        except Exception as e:

            traceback.print_exc()

            messagebox.showerror(
                "Save error",
                str(e)
            )


    # ========================================================
    # NEXT TILE
    # ========================================================

    def next_tile(self):

        self.save_current_label()

        if self.index >= (
            len(self.pairs) - 1
        ):

            messagebox.showinfo(
                "Last tile",
                "This is the last tile."
            )

            return

        self.load_tile(
            self.index + 1
        )


    # ========================================================
    # PREVIOUS TILE
    # ========================================================

    def previous_tile(self):

        self.save_current_label()

        if self.index <= 0:

            messagebox.showinfo(
                "First tile",
                "This is the first tile."
            )

            return

        self.load_tile(
            self.index - 1
        )


    # ========================================================
    # RESET VIEW
    # ========================================================

    def reset_view(self):

        if self.extent is None:

            return

        xmin, xmax, ymin, ymax = (
            self.extent
        )

        for ax in [
            self.ax_2023,
            self.ax_2025,
            self.ax_dea,
            self.ax_model,
            self.ax_manual
        ]:

            ax.set_xlim(
                xmin,
                xmax
            )

            ax.set_ylim(
                ymin,
                ymax
            )

        self.canvas.draw_idle()


    # ========================================================
    # SYNCHRONIZED ZOOM
    # ========================================================

    def mouse_scroll(
        self,
        event
    ):

        valid_axes = [
            self.ax_2023,
            self.ax_2025,
            self.ax_dea,
            self.ax_model,
            self.ax_manual
        ]

        if event.inaxes not in valid_axes:

            return

        if (
            event.xdata is None
            or
            event.ydata is None
        ):

            return

        ax = event.inaxes

        old_xlim = ax.get_xlim()

        old_ylim = ax.get_ylim()

        x = event.xdata

        y = event.ydata

        old_width = (
            old_xlim[1]
            -
            old_xlim[0]
        )

        old_height = (
            old_ylim[1]
            -
            old_ylim[0]
        )

        if event.button == "up":

            scale = 0.75

        else:

            scale = 1.333333333

        new_width = (
            old_width * scale
        )

        new_height = (
            old_height * scale
        )

        rx = (
            x
            -
            old_xlim[0]
        ) / old_width

        ry = (
            y
            -
            old_ylim[0]
        ) / old_height

        new_xmin = (
            x
            -
            rx * new_width
        )

        new_xmax = (
            x
            +
            (1 - rx)
            * new_width
        )

        new_ymin = (
            y
            -
            ry * new_height
        )

        new_ymax = (
            y
            +
            (1 - ry)
            * new_height
        )

        for panel in valid_axes:

            panel.set_xlim(
                new_xmin,
                new_xmax
            )

            panel.set_ylim(
                new_ymin,
                new_ymax
            )

        self.canvas.draw_idle()


    # ========================================================
    # SYNCHRONIZED PAN
    # ========================================================

    def mouse_move(
        self,
        event
    ):

        if not self.panning:

            return

        valid_axes = [
            self.ax_2023,
            self.ax_2025,
            self.ax_dea,
            self.ax_model,
            self.ax_manual
        ]

        if event.inaxes not in valid_axes:

            return

        if (
            event.xdata is None
            or
            event.ydata is None
        ):

            return

        start_x, start_y = (
            self.pan_start
        )

        dx = (
            event.xdata
            -
            start_x
        )

        dy = (
            event.ydata
            -
            start_y
        )

        new_xlim = (
            self.pan_start_xlim[0] - dx,
            self.pan_start_xlim[1] - dx
        )

        new_ylim = (
            self.pan_start_ylim[0] - dy,
            self.pan_start_ylim[1] - dy
        )

        for ax in valid_axes:

            ax.set_xlim(
                new_xlim
            )

            ax.set_ylim(
                new_ylim
            )

        self.canvas.draw_idle()


    # ========================================================
    # STATUS
    # ========================================================

    def update_status(self):

        if self.label is None:

            return

        change_pixels = int(
            np.sum(
                self.label == 1
            )
        )

        no_change_pixels = int(
            np.sum(
                self.label == 0
            )
        )

        label_path = get_label_path(
            self.tile_id
        )

        if os.path.exists(
            label_path
        ):

            label_status = (
                "LABEL SAVED"
            )

        else:

            label_status = (
                "NEW LABEL"
            )

        self.status.set(
            f"Tile {self.index + 1}/"
            f"{len(self.pairs)}   |   "
            f"{self.tile_id}   |   "
            f"{label_status}   |   "
            f"CHANGE: {change_pixels:,}   |   "
            f"NO CHANGE: {no_change_pixels:,}   |   "
            f"Undo polygons: {len(self.label_history)}"
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            self.save_current_label()

            print()
            print(
                "Current tile saved before closing."
            )

        except Exception:

            traceback.print_exc()

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 90)
    print(
        "MELBOURNE 5-PANEL URBAN CHANGE LABELING GUI"
    )
    print("=" * 90)

    print()
    print(
        "2023 Sentinel-2:"
    )

    print(
        S2_2023_DIR
    )

    print()
    print(
        "2025 Sentinel-2:"
    )

    print(
        S2_2025_DIR
    )

    print()
    print(
        "DEA candidate:"
    )

    print(
        DEA_CANDIDATE_DIR
    )

    print()
    print(
        "OSCD model:"
    )

    print(
        MODEL_CHANGE_DIR
    )

    print()
    print(
        "MANUAL LABELS:"
    )

    print(
        LABEL_DIR
    )

    # --------------------------------------------------------
    # Required directories
    # --------------------------------------------------------

    required_dirs = [

        S2_2023_DIR,

        S2_2025_DIR,

        DEA_CANDIDATE_DIR,

        MODEL_CHANGE_DIR
    ]

    for directory in required_dirs:

        if not os.path.exists(
            directory
        ):

            print()
            print(
                "ERROR: Directory not found:"
            )

            print(
                directory
            )

            return

    # --------------------------------------------------------
    # Existing labels
    # --------------------------------------------------------

    existing_labels = list_tif_files(
        LABEL_DIR
    )

    print()
    print(
        "Existing manual labels:",
        len(existing_labels)
    )

    for path in existing_labels:

        print(
            "   ",
            os.path.basename(path)
        )

    # --------------------------------------------------------
    # Build pairs
    # --------------------------------------------------------

    pairs = build_pairs()

    if not pairs:

        print()
        print(
            "ERROR:"
        )

        print(
            "No matching tiles were found."
        )

        return

    # --------------------------------------------------------
    # Start GUI
    # --------------------------------------------------------

    root = tk.Tk()

    app = MelbourneChangeGUI(
        root,
        pairs
    )

    root.mainloop()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
#  Author: Kyle Tranfaglia
#  Title: Main Program for City of Cambridge Flood Analysis Tool
#  Last updated: 03/07/25
#  Description: This program is a flood analysis tool for the City of Cambridge
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QStackedWidget, QListWidget, QHBoxLayout, QLabel, QSlider, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QComboBox
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import random
import sys
import os
import folium
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import box
import numpy as np


# 3D elevation flood level simulator class
class Elevation(QWidget):
    def __init__(self):
        super().__init__()

        # Layout setup
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(self.layout)

        # Title Label
        self.title = QLabel("3D Elevation Flood Simulation Model")
        self.title.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                                 "border-radius: 8px; background-color: #444444; padding: 10px;")
        self.layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Matplotlib Figure
        self.figure = plt.figure(figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Load initial DEM data
        self.current_elevation = None
        self.base_elevation = None
        self.load_dem_data()

        # Controls Layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        # Year label and slider
        self.year_label = QLabel("Year: 2025")
        self.year_label.setStyleSheet("font-size: 20px;")
        controls_layout.addWidget(self.year_label)

        # Slider (water level factor)
        self.year_slider = QSlider(Qt.Orientation.Horizontal)
        self.year_slider.setRange(0, 100)
        self.year_slider.setValue(0)
        self.year_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.year_slider.valueChanged.connect(self.update_year_label)
        controls_layout.addWidget(self.year_slider)

        # Hurricane Category (label and combo box grouped)
        hurricane_layout = QHBoxLayout()
        hurricane_layout.setSpacing(5)
        hurricane_label = QLabel("Hurricane:")
        hurricane_label.setStyleSheet("font-size: 22px;")
        hurricane_layout.addWidget(hurricane_label)

        self.category_combo = QComboBox()
        for i in range(1, 6):
            self.category_combo.addItem(f"Category {i}")
        self.category_combo.setStyleSheet("font-size: 22px;")
        self.category_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        hurricane_layout.addWidget(self.category_combo)
        controls_layout.addLayout(hurricane_layout)

        # Simulation Button
        self.simulate_button = QPushButton("Simulate Flooding")
        self.simulate_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.simulate_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.simulate_button.clicked.connect(self.simulate_flooding)
        controls_layout.addWidget(self.simulate_button)

        self.layout.addLayout(controls_layout)

        # Button Layout for additional controls
        button_layout = QHBoxLayout()

        # Reset Button
        self.reset_button = QPushButton("Reset Model")
        self.reset_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.reset_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.reset_button.clicked.connect(self.reset_model)
        button_layout.addWidget(self.reset_button)

        # Save Button (Export PNG)
        self.save_button = QPushButton("Save Model as PNG")
        self.save_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.save_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.save_button.clicked.connect(self.save_3d_model)
        button_layout.addWidget(self.save_button)

        # Add button layout to main layout
        self.layout.addLayout(button_layout)

        # Initial plot
        self.plot_3d_terrain(flood_level=None)

    # Load DEM data and filter out NoData values
    def load_dem_data(self):
        dem_path = "../Data/Dorchester_DEM/dorc2015_m/"  # Adjust path as needed

        try:
            with rasterio.open(dem_path) as dataset:
                print("DEM file opened successfully.")

                # Convert Cambridge BBox to DEM CRS
                cambridge_bbox = [-76.13, 38.53, -76.04, 38.61]  # Lat/Lon format
                minx, miny, maxx, maxy = transform_bounds("EPSG:4326", dataset.crs, *cambridge_bbox)
                geom = [box(minx, miny, maxx, maxy)]  # Create bounding box in correct CRS

                # Crop the DEM dataset
                out_image, out_transform = mask(dataset, geom, crop=True)
                elevation_data = out_image[0]  # Extract elevation array

                # Replace NoData values
                nodata_value = dataset.nodata  # Get the NoData value (-3.4028234663852886e+38)
                if nodata_value is not None:
                    elevation_data[elevation_data == nodata_value] = np.nan  # Convert to NaN

                min_valid = np.nanmin(elevation_data)  # Smallest valid value
                elevation_data = np.nan_to_num(elevation_data, nan=min_valid)  # Replace NaNs

                self.base_elevation = elevation_data
                self.current_elevation = np.copy(elevation_data)

                print(f"Loaded DEM: min={np.nanmin(self.base_elevation)}, max={np.nanmax(self.base_elevation)}")

        except Exception as e:
            print("Error loading DEM data:", e)
            self.base_elevation = None
            self.current_elevation = None

    # Update the year label as the slider changes
    def update_year_label(self):
        year = self.year_slider.value() + 2025
        self.year_label.setText(f"Year: {year}")

    # Generate an interactive 3D elevation plot with a color legend and info box
    def plot_3d_terrain(self, flood_level=None):
        if self.base_elevation is None:
            return

        self.figure.clear()  # Clear previous plot
        ax = self.figure.add_subplot(111, projection='3d')

        # Create X, Y grid
        height, width = self.base_elevation.shape
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        X, Y = np.meshgrid(x, y)

        # Start with base elevation
        elevation = np.copy(self.base_elevation)

        # Define exact dataset min/max values
        min_elev = -0.82  # Lowest elevation in dataset
        max_elev = 11.42  # Highest elevation in dataset

        # Ensure z-axis has proper range
        z_min = min_elev
        z_max = max_elev
        if flood_level is not None:
            z_max = max(z_max, flood_level)

        ax.set_zlim(z_min, z_max)

        # Plot terrain
        terrain = ax.plot_surface(X, Y, elevation, cmap="terrain", edgecolor='none', alpha=0.8)

        # Keep the elevation color scale legend always visible
        cbar = self.figure.colorbar(terrain, shrink=0.7, aspect=20, pad=0.1)
        cbar.set_label("Elevation (feet)", fontsize=12)

        # If flooding, overlay a water surface
        if flood_level is not None and flood_level > 0:
            flood_mask = elevation < flood_level
            water_surface = np.ones_like(elevation) * flood_level

            # Masked arrays to plot only flooded areas
            water_X = np.ma.masked_array(X, ~flood_mask)
            water_Y = np.ma.masked_array(Y, ~flood_mask)
            water_Z = np.ma.masked_array(water_surface, ~flood_mask)

            # Plot water surface
            water = ax.plot_surface(water_X, water_Y, water_Z, color='blue', alpha=0.5, edgecolor='none')

        # Labels
        ax.set_xlabel("X (Longitude Approx.)")
        ax.set_ylabel("Y (Latitude Approx.)")
        ax.set_zlabel("Elevation (feet)")

        # **Keep the title intact**
        title_text = f"Cambridge, MD - 3D Elevation Model"
        if flood_level is not None:
            title_text = f"Cambridge, MD - 3D Flood Simulation (Water Level: {flood_level:.2f} ft)"
        ax.set_title(title_text, fontsize=14, fontweight='bold')

        # **Place elevation info box on the left to mirror the color legend**
        left_box = self.figure.add_axes([0.16, 0.36, 0.12, 0.16])  # Left side placement
        left_box.axis("off")  # Hide axis

        # Add text inside the left box
        left_text = f"Elevation Range:\n{min_elev:.2f} to {max_elev:.2f} ft"
        if flood_level is not None:
            left_text += f"\nCurrent Water Level:\n{flood_level:.2f} ft"

        left_box.text(0.5, 0.5, left_text, fontsize=10, ha='center', va='center',
                      bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

        # Keep interactive controls
        ax.view_init(elev=30, azim=135)

        # Update canvas
        self.canvas.draw()

    # Apply flooding based on year slider and hurricane category
    def simulate_flooding(self):
        if self.base_elevation is None:
            print("No elevation data loaded")
            return

        # Get simulation parameters
        year_factor = self.year_slider.value() / 100.0  # 0 to 1

        category_text = self.category_combo.currentText().strip()
        try:
            hurricane_category = int(category_text.split()[1])  # Category 1-5
        except Exception:
            hurricane_category = 1

        # Get the elevation range of our terrain
        min_elev = np.min(self.base_elevation)
        max_elev = np.max(self.base_elevation)
        terrain_range = max_elev - min_elev

        # Calculate sea level rise based on year (linear model)
        base_water_level = min_elev + (year_factor * terrain_range * 0.25)

        # Hurricane intensity effect (exponential impact)
        hurricane_multiplier = 1.0 + (hurricane_category ** 1.5) * 0.1

        # Calculate flood level
        flood_level = base_water_level * hurricane_multiplier

        # Make sure flood level is at least at the minimum elevation to show some effect
        flood_level = max(flood_level, min_elev)

        # Apply the flood level to the terrain
        self.plot_3d_terrain(flood_level=flood_level)

        # Debug information
        print(f"Simulating flooding for Year {2025 + self.year_slider.value()}, "
              f"Hurricane {hurricane_category}, Flood Level: {flood_level:.2f} feet")
        print(f"Terrain elevation range: {min_elev:.2f} to {max_elev:.2f} feet")

    # Reset the model to base elevation without flooding
    def reset_model(self):
        self.year_slider.setValue(0)
        self.category_combo.setCurrentIndex(0)
        self.plot_3d_terrain(flood_level=None)

    # Save the current 3D model as a PNG image
    def save_3d_model(self):
        # Get user's home directory
        home_dir = os.path.expanduser("~")

        # Find the Downloads folder
        downloads_dir = os.path.join(home_dir, "Downloads")

        # Ensure the Downloads folder exists
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)

        # Define the output file path
        file_path = os.path.join(downloads_dir, "Cambridge_Flood_Model.png")

        # Save the figure
        self.figure.savefig(file_path, dpi=300)
        print(f"Flood Model saved as {file_path}")


# Interactive flood risk analysis map class
class InteractiveMap(QWidget):
    def __init__(self):
        super().__init__()

        # Page layout with title
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        label = QLabel("Interactive Flood Analysis Map")
        label.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                            "border-radius: 8px; background-color: #444444; padding: 10px;")
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.web_view = QWebEngineView()
        self.web_view.setMinimumSize(800, 600)
        self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.web_view)

        # Add a recenter button below the map
        self.recenter_button = QPushButton("Re-center Map")
        self.recenter_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.recenter_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.recenter_button.clicked.connect(self.recenter_map)
        layout.addWidget(self.recenter_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Set the layout and create the interactive map
        self.setLayout(layout)
        self.create_map()

    def create_map(self):
        # Create a Folium map centered on Cambridge, MD.
        map_center = [38.572, -76.078]
        m = folium.Map(location=map_center, zoom_start=15, )
        # Optionally add a marker at the center for reference.
        folium.Marker(map_center, popup="Cambridge, MD").add_to(m)

        # Define a focused bounding box around the center where flood info will be plotted
        lat_min = map_center[0] - 0.012
        lat_max = map_center[0] + 0.012
        lon_min = map_center[1] - 0.012
        lon_max = map_center[1] + 0.012

        # Plot 10 random flood risk markers.
        num_markers = 10
        for i in range(num_markers):
            lat = random.uniform(lat_min, lat_max)
            lon = random.uniform(lon_min, lon_max)
            # Randomly assign a flood risk level.
            risk = random.choice(["High", "Moderate", "Low"])
            if risk == "High":
                color = "red"
                popup_text = "High Flood Risk<br>Estimated damage: $200k"
                radius = 24 + (i * 2)
            elif risk == "Moderate":
                color = "orange"
                popup_text = "Moderate Flood Risk<br>Estimated damage: $100k"
                radius = 16 + (i * 2)
            else:
                color = "yellow"
                popup_text = "Low Flood Risk<br>Estimated damage: $50k"
                radius = 12 + (i * 2)

            # Add a CircleMarker with the risk info
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=popup_text
            ).add_to(m)

        # **Add a legend using HTML and CSS**
        legend_html = """
                            <div style="
                                position: fixed; 
                                bottom: 20px; right: 20px; 
                                background-color: white; 
                                padding: 10px; 
                                border: 2px solid black; 
                                border-radius: 5px;
                                font-size: 16px;
                                z-index: 1000;
                            ">
                                <b>Flood Risk Legend</b><br>
                                <span style="display:inline-block; width: 15px; height: 15px; background-color: red;"></span> High Risk<br>
                                <span style="display:inline-block; width: 15px; height: 15px; background-color: orange;"></span> Moderate Risk<br>
                                <span style="display:inline-block; width: 15px; height: 15px; background-color: yellow;"></span> Low Risk
                            </div>
                            """

        # Add the legend as a child of the map
        m.get_root().html.add_child(folium.Element(legend_html))

        # Render the map to HTML and load it into the QWebEngineView
        html_data = m.get_root().render()
        self.web_view.setHtml(html_data, baseUrl=QUrl("http://localhost/"))

    # Re-create the map to re-center on Cambridge
    def recenter_map(self):
        self.create_map()


# Simulates flood regions over a static map of Cambridge given certain criteria
class StreetView(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(5)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Title Label
        self.title = QLabel("Street View Flood Simulation")
        self.title.setStyleSheet(
            "font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
            "border-radius: 8px; background-color: #444444; padding: 10px;"
        )
        self.layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # QWebEngineView to display the Folium map
        self.web_view = QWebEngineView()
        self.web_view.setMinimumSize(800, 600)
        self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.web_view)

        # Controls Layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        # Year label
        self.year_label = QLabel("Year: 2025")
        self.year_label.setStyleSheet("font-size: 20px;")
        controls_layout.addWidget(self.year_label)

        # Slider (water level factor)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.slider.valueChanged.connect(self.update_year_label)

        controls_layout.addWidget(self.slider)

        # Hurricane Category (label and combo box grouped)
        hurricane_layout = QHBoxLayout()
        hurricane_layout.setSpacing(5)
        hurricane_label = QLabel("Hurricane:")
        hurricane_label.setStyleSheet("font-size: 22px;")
        hurricane_layout.addWidget(hurricane_label)
        self.category_combo = QComboBox()
        for i in range(1, 6):
            self.category_combo.addItem(f"Category {i}")
        self.category_combo.setStyleSheet("font-size: 22px;")
        self.category_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        hurricane_layout.addWidget(self.category_combo)
        controls_layout.addLayout(hurricane_layout)

        # Simulation Button
        self.sim_button = QPushButton("Simulate")
        self.sim_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.sim_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.sim_button.clicked.connect(self.simulate)
        controls_layout.addWidget(self.sim_button)

        self.layout.addLayout(controls_layout)
        self.setLayout(self.layout)

        # Set map center for Cambridge, MD
        self.map_center = [38.575, -76.078]

        # Create an initial map
        self.init_map()

    def init_map(self):
        # Create a locked Folium map centered on Cambridge, MD

        # Create a Folium map centered on Cambridge, MD
        m = folium.Map(location=self.map_center,
                       zoom_start=15,
                       min_zoom=15,
                       max_zoom=15,
                       zoom_control=False,
                       dragging=False,
                       control_scale=True)

        # Add a reference marker at the center
        folium.Marker([38.572, -76.078], popup="Cambridge, MD").add_to(m)

        # Render the map in the QWebEngineView
        html_data = m.get_root().render()
        self.web_view.setHtml(html_data, baseUrl=QUrl("http://localhost/"))

    def update_year_label(self):
        year = self.slider.value() + 2025
        self.year_label.setText(f"Year: {year}")

    def simulate(self):
        category_text = self.category_combo.currentText().strip()
        try:
            category = int(category_text.split()[1])
        except Exception:
            category = 1
        water_factor = self.slider.value() / 100.0

        # Re-create the Folium map for simulation (locked in place)
        m = folium.Map(location=self.map_center,
                       zoom_start=15,
                       min_zoom=15,
                       max_zoom=15,
                       zoom_control=False,
                       dragging=False,
                       control_scale=True)

        # Add a reference marker at the center
        folium.Marker([38.572, -76.078], popup="Cambridge, MD").add_to(m)

        # **Add a legend using HTML and CSS**
        legend_html = """
                    <div style="
                        position: fixed; 
                        bottom: 20px; right: 20px; 
                        background-color: white; 
                        padding: 10px; 
                        border: 2px solid black; 
                        border-radius: 5px;
                        font-size: 16px;
                        z-index: 1000;
                    ">
                        <b>Flood Estimate Legend</b><br>
                        <span style="display:inline-block; width: 15px; height: 15px; background-color: red;"></span> Severe Flooding<br>
                        <span style="display:inline-block; width: 15px; height: 15px; background-color: orange;"></span> Moderate Flooding<br>
                        <span style="display:inline-block; width: 15px; height: 15px; background-color: yellow;"></span> Minor Flooding
                    </div>
                    """

        # Add the legend as a child of the map
        m.get_root().html.add_child(folium.Element(legend_html))

        # Define a bounding box around the center for placing flood markers
        lat_min = self.map_center[0] - 0.012
        lat_max = self.map_center[0] + 0.012
        lon_min = self.map_center[1] - 0.012
        lon_max = self.map_center[1] + 0.012

        # Restrict markers vertically
        lat_range = lat_max - lat_min
        effective_lat_min = lat_min + 0.2 * lat_range
        effective_lat_max = lat_max - 0.2 * lat_range

        # Use fewer markers for a dramatic impact region
        num_markers = 10
        for _ in range(num_markers):
            lat = random.uniform(effective_lat_min, effective_lat_max)
            lon = random.uniform(lon_min, lon_max)

            # Compute a base radius that increases with water factor and category
            base_radius = 80 + water_factor * 200 * (category / 5.0)  # in meters

            # Determine number of concentric circles: more rings for higher categories.
            num_circles = 3 + (category - 1)  # e.g., cat1 -> 3 rings, cat5 -> 7 rings

            # For each ring, we interpolate both the color and opacity
            f = (6 - category) / 5.0
            for i in range(num_circles):
                t = i / (num_circles - 1) if num_circles > 1 else 0
                effective_t = min(t * (f + 0.25), 1.0)

                # Increase radius per ring (each ring 30% larger)
                radius = base_radius * (1 + 0.25 * i)

                # Interpolate opacity: use a base that decreases with category.
                base_opacity = 0.7 - (category - 1) * 0.1
                opacity = base_opacity - effective_t * (base_opacity - 0.3)

                # Interpolate color from red (#FF0000) to yellow (#FFFF00) using effective_t.
                r = 255
                g = int(effective_t * 255)
                b = 0
                color = f"#{r:02X}{g:02X}{b:02X}"

                folium.Circle(
                    location=[lat, lon],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=opacity
                ).add_to(m)

        # Render the map to an HTML string and load it in the web view
        html_data = m.get_root().render()
        self.web_view.setHtml(html_data, baseUrl=QUrl("http://localhost/"))


# Main application window with sidebar navigation and multiple pages
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cambridge Flood Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Main container for sidebar and content
        self.main_container = QWidget()
        self.setCentralWidget(self.main_container)
        self.layout = QHBoxLayout()
        self.main_container.setLayout(self.layout)

        # Sidebar setup with fixed width
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(275)
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)

        # QTreeWidget for hierarchical navigation
        self.sidebar_tree = QTreeWidget()
        self.sidebar_tree.setHeaderHidden(True)
        self.sidebar_tree = QTreeWidget()
        self.sidebar_tree.setHeaderHidden(True)
        self.sidebar_tree.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.sidebar_tree.setStyleSheet("QTreeWidget { font-size: 23px; padding: 4px; }")

        # Create top-level and sub-items
        home_item = QTreeWidgetItem(["Home"])
        settings_item = QTreeWidgetItem(["Settings"])
        flood_item = QTreeWidgetItem(["Flood Simulators"])
        street_view_item = QTreeWidgetItem(["-- Street View"])
        model_elevation_item = QTreeWidgetItem(["-- 3D Elevation"])
        interactive_map_item = QTreeWidgetItem(["-- Interactive Map"])
        # Add subcategories under Flood Simulators
        flood_item.addChildren([street_view_item, model_elevation_item, interactive_map_item])
        insurance_item = QTreeWidgetItem(["Insurance Projections"])
        damage_item = QTreeWidgetItem(["Damage Estimator"])

        # Add all top-level items to the tree
        self.sidebar_tree.addTopLevelItem(home_item)
        self.sidebar_tree.addTopLevelItem(settings_item)
        self.sidebar_tree.addTopLevelItem(flood_item)
        self.sidebar_tree.addTopLevelItem(insurance_item)
        self.sidebar_tree.addTopLevelItem(damage_item)

        # Connect click signal
        self.sidebar_tree.itemClicked.connect(self.handle_tree_item_click)
        self.sidebar_layout.addWidget(self.sidebar_tree)

        # Create a container (wrapper) for the sidebar and the toggle button
        self.sidebar_wrapper = QWidget()
        self.sidebar_wrapper_layout = QHBoxLayout()
        self.sidebar_wrapper.setLayout(self.sidebar_wrapper_layout)
        self.sidebar_wrapper_layout.setSpacing(0)
        self.sidebar_wrapper_layout.addWidget(self.sidebar)

        # Toggle button for sidebar, placed outside the sidebar widget
        self.toggle_button = QPushButton("◀")
        self.toggle_button.setStyleSheet("font-size: 19px;")
        self.toggle_button.setFixedWidth(25)
        self.toggle_button.setFixedHeight(40)
        self.toggle_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.toggle_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_wrapper_layout.addWidget(self.toggle_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.layout.addWidget(self.sidebar_wrapper)

        # Main content area as a stacked widget
        self.central_widget = QStackedWidget()
        self.layout.addWidget(self.central_widget, 1)

        # Home page
        home_page = QWidget()
        home_layout = QVBoxLayout()
        home_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        home_label = QLabel("Welcome to the Cambridge Flood Analysis Tool")
        home_label.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                                 "border-radius: 8px; background-color: #444444; padding: 10px;")
        home_layout.addWidget(home_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        image_label = QLabel()
        pixmap = QPixmap("Cambridge_logo.png")
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)
        home_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        home_page.setLayout(home_layout)

        # Create pages for each sidebar selection
        self.pages = {
            "Home": home_page,
            "Settings": QWidget(),
            "-- Street View": StreetView(),
            "-- 3D Elevation": Elevation(),
            "-- Interactive Map": InteractiveMap(),
            "Insurance Projections": QWidget(),
            "Damage Estimator": QWidget()
        }

        # Add all pages to the stacked widget
        for page in self.pages.values():
            self.central_widget.addWidget(page)

    # Handle expansion and collapse of sub categories and switching pages
    def handle_tree_item_click(self, item, column):
        # If the item has children, toggle expansion/collapse without switching pages.
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())
        else:
            page_name = item.text(0)
            if page_name in self.pages:
                self.central_widget.setCurrentWidget(self.pages[page_name])

    # Toggle the side navigation bar
    def toggle_sidebar(self):
        if self.sidebar.isVisible():
            self.sidebar.hide()
            self.toggle_button.setText("▶")
        else:
            self.sidebar.show()
            self.toggle_button.setText("◀")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.showMaximized()  # Ensure full window display
    sys.exit(app.exec())

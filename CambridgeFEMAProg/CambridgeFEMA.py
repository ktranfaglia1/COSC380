#  Author: Kyle Tranfaglia
#  Title: Main Program for City of Cambridge Flood Analysis Tool
#  Last updated: 03/15/25
#  Description: This program is a flood analysis tool for the City of Cambridge
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QStackedWidget, QHBoxLayout, QLabel, QSlider, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QLineEdit, QFrame,
    QTableWidget, QTableWidgetItem, QComboBox, QFileDialog,
    QAbstractItemView, QMessageBox, QTabWidget, QGroupBox,
    QSpinBox, QHeaderView, QScrollArea
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QCursor, QFont
from PyQt6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random
import sys
import os
import folium
import rasterio
import zipfile
import io
import datetime
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderQueryError
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import box
import numpy as np
import pandas as pd
import pickle


class LidarSurface(QWidget):
    def __init__(self):
        super().__init__()

        # Layout setup
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(self.layout)

        # Title Label
        self.title = QLabel("3D LiDAR Surface Model")
        self.title.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                                 "border-radius: 8px; background-color: #444444; padding: 10px;")
        self.layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Removed: Subtitle label explaining LiDAR

        # Matplotlib Figure
        self.figure = plt.figure(figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Lidar data properties
        self.lidar_data = None
        self.current_file_index = 0
        self.lidar_files = self.find_lidar_files()

        # Controls Layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        # Year label and slider
        self.year_label = QLabel("Year: 2025")
        self.year_label.setStyleSheet("font-size: 22px;")
        controls_layout.addWidget(self.year_label)

        # Slider (water level factor)
        self.year_slider = QSlider(Qt.Orientation.Horizontal)
        self.year_slider.setRange(0, 100)
        self.year_slider.setValue(0)
        self.year_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.year_slider.valueChanged.connect(self.update_year_label)
        controls_layout.addWidget(self.year_slider)

        # File selection dropdown
        file_layout = QHBoxLayout()
        file_layout.setSpacing(5)
        file_label = QLabel("LiDAR Data:")
        file_label.setStyleSheet("font-size: 22px;")
        file_layout.addWidget(file_label)

        self.file_combo = QComboBox()
        self.file_combo.addItems([os.path.basename(f) for f in self.lidar_files])
        self.file_combo.setStyleSheet("font-size: 20px;")
        self.file_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.file_combo.currentIndexChanged.connect(self.change_lidar_file)
        file_layout.addWidget(self.file_combo)
        controls_layout.addLayout(file_layout)

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

        # View Controls Button - removed since controls will always be visible
        # self.view_button = QPushButton("View Controls")
        # self.view_button.setStyleSheet("font-size: 22px; padding: 8px;")
        # self.view_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        # self.view_button.clicked.connect(self.show_view_controls)
        # button_layout.addWidget(self.view_button)

        # Add button layout to main layout
        self.layout.addLayout(button_layout)

        # Create view control panel (always visible now)
        self.view_controls = QWidget()
        view_controls_layout = QHBoxLayout()
        self.view_controls.setLayout(view_controls_layout)

        # Add elevation exaggeration control
        elevation_layout = QVBoxLayout()
        elevation_label = QLabel("Elevation Exaggeration")
        # Updated font size to match other labels
        elevation_label.setStyleSheet("font-size: 22px;")
        elevation_layout.addWidget(elevation_label)

        self.elevation_slider = QSlider(Qt.Orientation.Horizontal)
        self.elevation_slider.setRange(10, 30)  # 1x to 3x exaggeration
        self.elevation_slider.setValue(15)  # 1.5x default
        self.elevation_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.elevation_slider.setTickInterval(5)
        self.elevation_slider.valueChanged.connect(lambda: self.update_3d_view())
        elevation_layout.addWidget(self.elevation_slider)
        view_controls_layout.addLayout(elevation_layout)

        # Add view angle controls
        angle_layout = QVBoxLayout()
        angle_label = QLabel("View Angle")
        # Updated font size to match other labels
        angle_label.setStyleSheet("font-size: 22px;")
        angle_layout.addWidget(angle_label)

        angle_buttons_layout = QHBoxLayout()
        for angle in ["Top", "Side", "Front", "Isometric"]:
            angle_button = QPushButton(angle)
            # Made buttons bigger with more padding and larger font
            angle_button.setStyleSheet("font-size: 18px; padding: 8px;")
            # Added pointing hand cursor
            angle_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            angle_button.clicked.connect(lambda checked, a=angle: self.change_view_angle(a))
            angle_buttons_layout.addWidget(angle_button)

        angle_layout.addLayout(angle_buttons_layout)
        view_controls_layout.addLayout(angle_layout)

        # Show view controls by default
        self.layout.addWidget(self.view_controls)

        # Loading status label - Will be hidden after operations instead of showing messages
        self.status_label = QLabel("Loading LiDAR data...")
        self.status_label.setStyleSheet("font-size: 18px; color: #444444;")
        self.layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Load initial data
        QApplication.processEvents()  # Ensure UI updates before potentially long loading process
        self.load_lidar_data()

    def find_lidar_files(self):
        """Find all LiDAR .sid files in the data directory"""
        lidar_files = []
        data_dir = "../Data/Dorchester_LiDAR/"

        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if "Cambridge" in file and file.endswith(".sid"):
                    lidar_files.append(os.path.join(data_dir, file))

        # If no files found in standard directories, add sample file paths for testing
        # Modified to remove "Cambridge_reduced_" prefix
        if not lidar_files:
            for i in range(1, 10):
                lidar_files.append(f"cloud_{i}.sid")
                lidar_files.append(f"dem_{i}.sid")

        return sorted(lidar_files)

    def change_lidar_file(self, index):
        """Change LiDAR file when user selects a different one from the dropdown"""
        if 0 <= index < len(self.lidar_files):
            self.current_file_index = index
            self.load_lidar_data()

    def load_lidar_data(self):
        """Load LiDAR data from selected file"""
        self.status_label.setText(
            f"Loading LiDAR data from {os.path.basename(self.lidar_files[self.current_file_index])}...")
        QApplication.processEvents()  # Update UI

        try:
            # Attempt to open the file with rasterio
            with rasterio.open(self.lidar_files[self.current_file_index]) as dataset:
                # Extract the data
                self.lidar_data = dataset.read(1)

                # Replace NoData values with NaN
                nodata_value = dataset.nodata
                if nodata_value is not None:
                    self.lidar_data[self.lidar_data == nodata_value] = np.nan

                # Get min/max for display
                self.min_elev = np.nanmin(self.lidar_data)
                self.max_elev = np.nanmax(self.lidar_data)

                # Hide status label after successful loading
                self.status_label.hide()

                # Plot the initial surface
                self.plot_3d_lidar(flood_level=None)

        except Exception as e:
            # Hide status label after error (instead of showing error message)
            self.status_label.hide()

            # Generate synthetic data for demo/testing if file can't be loaded
            self.generate_synthetic_data()
            self.plot_3d_lidar(flood_level=None)

    def generate_synthetic_data(self):
        """Generate synthetic data including terrain, buildings and vegetation for demonstration"""
        # Create a grid
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)

        # Create a terrain-like surface using sine functions for the base terrain
        Z = 3 * np.sin(X / 10) * np.cos(Y / 10) + np.random.rand(100, 100) * 0.5

        # Add a few hills and depressions to the terrain
        for _ in range(5):
            cx, cy = np.random.randint(0, 100, 2)
            r = np.random.randint(10, 30)
            height = np.random.uniform(1, 3)
            mask = ((X - cx) ** 2 + (Y - cy) ** 2) < r ** 2
            Z[mask] += height * np.exp(-((X[mask] - cx) ** 2 + (Y[mask] - cy) ** 2) / (r ** 2))

        # Add buildings (rectangular blocks)
        for _ in range(30):  # Add 30 buildings
            # Random building location and size
            bx = np.random.randint(10, 90)
            by = np.random.randint(10, 90)
            width = np.random.randint(3, 8)
            length = np.random.randint(3, 8)
            height = np.random.uniform(5, 15)  # Taller than terrain features

            # Create building mask
            b_mask = (np.abs(X - bx) < width / 2) & (np.abs(Y - by) < length / 2)

            # Add building with flat top at its maximum height
            if np.any(b_mask):
                base_height = np.max(Z[b_mask])  # Get the terrain height at building location
                Z[b_mask] = base_height + height  # Set building height

        # Add vegetation (trees and bushes)
        for _ in range(100):  # Add 100 vegetation elements
            vx = np.random.randint(5, 95)
            vy = np.random.randint(5, 95)

            # Vegetation size and height
            size = np.random.randint(1, 4)  # Size of vegetation (1=bush, 2-3=small tree, 4=large tree)
            v_height = np.random.uniform(1, 8) * size / 2  # Height based on size

            # Create vegetation mask (circular)
            radius = size
            v_mask = ((X - vx) ** 2 + (Y - vy) ** 2) < radius ** 2

            # Only add vegetation if it's not on a building
            if np.any(v_mask):
                # Get current heights
                current_heights = Z[v_mask]
                # Only modify points that aren't already buildings (assuming buildings are higher)
                terrain_mask = current_heights < np.mean(Z) + 5
                if np.any(terrain_mask):
                    # Get terrain mask for vegetation area
                    combined_mask = np.zeros_like(Z, dtype=bool)
                    combined_mask[v_mask] = terrain_mask

                    # Add vegetation with conical shape (higher in center)
                    center_dist = np.sqrt((X[combined_mask] - vx) ** 2 + (Y[combined_mask] - vy) ** 2)
                    max_dist = np.max(center_dist)
                    if max_dist > 0:
                        # Calculate height factor (1 at center, 0 at edge)
                        height_factor = (1 - center_dist / max_dist)
                        base_heights = Z[combined_mask]  # Get terrain heights
                        Z[combined_mask] = base_heights + v_height * height_factor  # Add vegetation

        self.lidar_data = Z
        self.min_elev = np.min(Z)
        self.max_elev = np.max(Z)

        # Hide status label after generating data
        self.status_label.hide()

    def update_year_label(self):
        """Update the year label as the slider changes"""
        year = self.year_slider.value() + 2025
        self.year_label.setText(f"Year: {year}")

    def plot_3d_lidar(self, flood_level=None):
        """Plot the 3D LiDAR surface with enhanced realism for buildings and vegetation"""
        if self.lidar_data is None:
            self.status_label.hide()
            return

        self.figure.clear()  # Clear previous plot
        ax = self.figure.add_subplot(111, projection='3d')

        # Create X, Y grid
        height, width = self.lidar_data.shape
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        X, Y = np.meshgrid(x, y)

        # Adaptive downsampling based on data complexity
        # Less downsampling for areas with high variation (buildings, trees)
        gradient_x = np.gradient(self.lidar_data, axis=0)
        gradient_y = np.gradient(self.lidar_data, axis=1)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Base downsample factor (higher for better performance)
        base_downsample = 3

        # For complex areas (buildings, vegetation with high gradient)
        high_detail_mask = gradient_magnitude > np.percentile(gradient_magnitude, 70)

        # Separate terrain, vegetation and buildings based on height and gradient
        height_percentiles = np.percentile(self.lidar_data, [30, 70, 90])
        terrain_mask = self.lidar_data < height_percentiles[0]
        vegetation_mask = (self.lidar_data >= height_percentiles[0]) & (self.lidar_data < height_percentiles[2])
        building_mask = self.lidar_data >= height_percentiles[2]

        # Ensure z-axis has proper range
        z_min = self.min_elev
        z_max = self.max_elev
        if flood_level is not None:
            z_max = max(z_max, flood_level)

        ax.set_zlim(z_min, z_max)

        # Enhanced rendering approach - split by feature type for better visualization

        # 1. Plot base terrain with earthy colormap
        X_terrain = X[::base_downsample, ::base_downsample]
        Y_terrain = Y[::base_downsample, ::base_downsample]
        Z_terrain = self.lidar_data[::base_downsample, ::base_downsample]

        # Create mask for downsampled terrain
        terrain_ds_mask = terrain_mask[::base_downsample, ::base_downsample]

        # Create masked arrays for terrain
        X_terrain_masked = np.ma.masked_array(X_terrain, ~terrain_ds_mask)
        Y_terrain_masked = np.ma.masked_array(Y_terrain, ~terrain_ds_mask)
        Z_terrain_masked = np.ma.masked_array(Z_terrain, ~terrain_ds_mask)

        # Plot terrain with specific style
        terrain_plot = ax.plot_surface(X_terrain_masked, Y_terrain_masked, Z_terrain_masked,
                                       cmap="YlOrBr", alpha=0.9, shade=True,
                                       rstride=1, cstride=1, linewidth=0,
                                       vmin=z_min, vmax=z_max)

        # 2. Plot vegetation with green colormap and less downsampling for details
        veg_downsample = max(1, base_downsample - 1)  # Less downsampling for vegetation
        X_veg = X[::veg_downsample, ::veg_downsample]
        Y_veg = Y[::veg_downsample, ::veg_downsample]
        Z_veg = self.lidar_data[::veg_downsample, ::veg_downsample]

        # Create mask for downsampled vegetation
        veg_ds_mask = vegetation_mask[::veg_downsample, ::veg_downsample]

        # Create masked arrays for vegetation
        X_veg_masked = np.ma.masked_array(X_veg, ~veg_ds_mask)
        Y_veg_masked = np.ma.masked_array(Y_veg, ~veg_ds_mask)
        Z_veg_masked = np.ma.masked_array(Z_veg, ~veg_ds_mask)

        # Plot vegetation with specific style
        veg_plot = ax.plot_surface(X_veg_masked, Y_veg_masked, Z_veg_masked,
                                   cmap="Greens", alpha=0.9, shade=True,
                                   rstride=1, cstride=1, linewidth=0,
                                   vmin=z_min, vmax=z_max)

        # 3. Plot buildings with minimal downsampling for sharp edges
        building_downsample = max(1, base_downsample - 2)  # Minimal downsampling for buildings
        X_building = X[::building_downsample, ::building_downsample]
        Y_building = Y[::building_downsample, ::building_downsample]
        Z_building = self.lidar_data[::building_downsample, ::building_downsample]

        # Create mask for downsampled buildings
        building_ds_mask = building_mask[::building_downsample, ::building_downsample]

        # Create masked arrays for buildings
        X_building_masked = np.ma.masked_array(X_building, ~building_ds_mask)
        Y_building_masked = np.ma.masked_array(Y_building, ~building_ds_mask)
        Z_building_masked = np.ma.masked_array(Z_building, ~building_ds_mask)

        # Plot buildings with specific style (grayscale for urban appearance)
        building_plot = ax.plot_surface(X_building_masked, Y_building_masked, Z_building_masked,
                                        cmap="Greys", alpha=0.95, shade=True,
                                        rstride=1, cstride=1, linewidth=0,
                                        vmin=z_min, vmax=z_max)

        # For color legend, use the terrain plot as reference
        terrain = terrain_plot

        # Create a custom legend instead of using the colorbar
        # This provides better visual explanation of the different features
        legend_box = self.figure.add_axes([0.7, 0.15, 0.2, 0.2])  # Position at bottom right
        legend_box.axis("off")

        # Create colored patches for different elements
        legend_items = [
            plt.Rectangle((0, 0), 1, 1, fc='#555555', ec='black', alpha=0.9),  # Buildings (gray)
            plt.Rectangle((0, 0), 1, 1, fc='#2ca02c', ec='black', alpha=0.9),  # Vegetation (green)
            plt.Rectangle((0, 0), 1, 1, fc='#d8b365', ec='black', alpha=0.9),  # Terrain (tan)
        ]

        # Add water to legend if flooding is active
        if flood_level is not None:
            legend_items.append(plt.Rectangle((0, 0), 1, 1, fc='#5EB1FF', ec='black', alpha=0.7))
            legend_text = ['Buildings', 'Vegetation', 'Terrain', 'Flood Water']
        else:
            legend_text = ['Buildings', 'Vegetation', 'Terrain']

        # Add the legend
        legend_box.legend(legend_items, legend_text, loc='center',
                          fontsize=10, frameon=True, framealpha=0.8)

        # Remove elevation scale bar (as requested)

        # If flooding, overlay a water surface with enhanced realism
        if flood_level is not None and flood_level > z_min:
            # Use the maximum downsampling for water (for performance)
            water_downsample = base_downsample
            X_water = X[::water_downsample, ::water_downsample]
            Y_water = Y[::water_downsample, ::water_downsample]

            # Create a water surface mesh
            water_surface = np.ones_like(self.lidar_data[::water_downsample, ::water_downsample]) * flood_level

            # Find areas where terrain is below flood level
            flood_mask = self.lidar_data[::water_downsample, ::water_downsample] < flood_level

            # Create masked arrays to plot only flooded areas
            water_X = np.ma.masked_array(X_water, ~flood_mask)
            water_Y = np.ma.masked_array(Y_water, ~flood_mask)
            water_Z = np.ma.masked_array(water_surface, ~flood_mask)

            # Plot water surface with enhanced appearance
            water = ax.plot_surface(water_X, water_Y, water_Z,
                                    color='#5EB1FF',  # Light blue for better water appearance
                                    alpha=0.6,
                                    linewidth=0,
                                    edgecolor='none',
                                    shade=True)

            # Add water edge effect (for more realism where water meets land)
            # Find the boundary of the flood
            flood_boundary = np.zeros_like(flood_mask)
            x_max, y_max = flood_mask.shape

            # Simple boundary detection
            for x in range(1, x_max - 1):
                for y in range(1, y_max - 1):
                    if flood_mask[x, y] and not all([
                        flood_mask[x - 1, y],
                        flood_mask[x + 1, y],
                        flood_mask[x, y - 1],
                        flood_mask[x, y + 1]
                    ]):
                        flood_boundary[x, y] = True

            # Extract boundary coordinates
            boundary_points = np.where(flood_boundary)
            if len(boundary_points[0]) > 0:
                X_boundary = X_water[boundary_points]
                Y_boundary = Y_water[boundary_points]
                Z_boundary = np.ones_like(X_boundary) * flood_level

                # Plot water boundary with slightly darker blue
                ax.scatter(X_boundary, Y_boundary, Z_boundary, color='#007ACC', s=1, alpha=0.8)

        # Enhanced labels
        ax.set_xlabel("X (Longitude Approx.)", fontsize=10, labelpad=10)
        ax.set_ylabel("Y (Latitude Approx.)", fontsize=10, labelpad=10)
        ax.set_zlabel("Elevation (feet)", fontsize=10, labelpad=10)

        # Set optimal viewing angle
        ax.view_init(elev=25, azim=235)  # Better angle to see structures

        # Remove tick labels for cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add grid for better spatial reference
        ax.grid(True, alpha=0.3, linestyle='--')

        # Title
        title_text = f"Cambridge, MD - 3D LiDAR Surface Model"
        if flood_level is not None:
            title_text = f"Cambridge, MD - 3D LiDAR Surface Flood Simulation (Water Level: {flood_level:.2f} ft)"
        ax.set_title(title_text, fontsize=14, fontweight='bold')

        # Add info box
        left_box = self.figure.add_axes([0.16, 0.36, 0.12, 0.18])
        left_box.axis("off")

        # Add text with data metrics and feature explanation
        left_text = f"LiDAR Data:\n{os.path.basename(self.lidar_files[self.current_file_index])}\n\n"
        left_text += f"Elevation Range:\n{self.min_elev:.2f} to {self.max_elev:.2f} ft\n\n"
        left_text += "Features Visible:\n• Buildings\n• Vegetation\n• Terrain"

        if flood_level is not None:
            left_text += f"\n\nWater Level:\n{flood_level:.2f} ft"
            # Calculate flood coverage percentage
            flood_coverage = np.sum(self.lidar_data < flood_level) / self.lidar_data.size * 100
            left_text += f"\nFlooded Area: {flood_coverage:.1f}%"
            # Add vulnerability metrics
            building_threshold = np.percentile(self.lidar_data, 90)  # Assume tallest 10% are buildings
            vegetation_threshold = np.percentile(self.lidar_data, 70)  # Assume 70-90% are vegetation
            buildings_flooded = np.sum((self.lidar_data > building_threshold) &
                                       (self.lidar_data < flood_level)) / np.sum(
                self.lidar_data > building_threshold) * 100
            vegetation_flooded = np.sum((self.lidar_data > vegetation_threshold) &
                                        (self.lidar_data < building_threshold) &
                                        (self.lidar_data < flood_level)) / np.sum(
                (self.lidar_data > vegetation_threshold) &
                (self.lidar_data < building_threshold)) * 100
            if buildings_flooded > 0:
                left_text += f"\nBuildings Affected: {buildings_flooded:.1f}%"
            if vegetation_flooded > 0:
                left_text += f"\nVegetation Affected: {vegetation_flooded:.1f}%"

        left_box.text(0.5, 0.5, left_text, fontsize=10, ha='center', va='center',
                      bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

        # Keep interactive controls
        ax.view_init(elev=30, azim=135)

        # Update canvas
        self.canvas.draw()

    def simulate_flooding(self):
        """Apply flooding based on year slider and hurricane category"""
        if self.lidar_data is None:
            # Hide status label instead of showing error message
            self.status_label.hide()
            return

        # Get simulation parameters
        year_factor = self.year_slider.value() / 100.0  # 0 to 1

        category_text = self.category_combo.currentText().strip()
        try:
            hurricane_category = int(category_text.split()[1])  # Category 1-5
        except Exception:
            hurricane_category = 1

        # Calculate sea level rise based on year (linear model)
        terrain_range = self.max_elev - self.min_elev
        base_water_level = self.min_elev + (year_factor * terrain_range * 0.25)

        # Hurricane intensity effect (exponential impact)
        hurricane_multiplier = 1.0 + (hurricane_category ** 1.5) * 0.1

        # Calculate flood level
        flood_level = base_water_level * hurricane_multiplier

        # Make sure flood level is at least at the minimum elevation to show some effect
        flood_level = max(flood_level, self.min_elev)

        # Hide status label instead of updating text
        self.status_label.hide()

        # Apply the flood level to the terrain
        self.plot_3d_lidar(flood_level=flood_level)

    def reset_model(self):
        """Reset the model to default state without flooding"""
        self.year_slider.setValue(0)
        self.category_combo.setCurrentIndex(0)
        # Hide status label instead of showing message
        self.status_label.hide()
        self.plot_3d_lidar(flood_level=None)

    # Removed: show_view_controls method since controls are always visible

    def change_view_angle(self, angle_preset):
        """Change the view angle based on preset"""
        if angle_preset == "Top":
            elev, azim = 90, 0  # Directly from top
        elif angle_preset == "Side":
            elev, azim = 0, 180  # Side view
        elif angle_preset == "Front":
            elev, azim = 0, 270  # Front view
        elif angle_preset == "Isometric":
            elev, azim = 30, 225  # Three-quarter view
        else:
            return

        # Get the figure's axes
        if hasattr(self, 'figure') and self.figure.axes:
            for ax in self.figure.axes:
                if hasattr(ax, 'view_init'):  # Check if it's a 3D axis
                    ax.view_init(elev=elev, azim=azim)
                    self.canvas.draw()
                    break

        # Removed status message
        # self.status_label.setText(f"View changed to {angle_preset} perspective")
        self.status_label.hide()

    def update_3d_view(self):
        """Update the 3D view with current elevation exaggeration"""
        # Get the current axes
        if hasattr(self, 'figure') and self.figure.axes:
            for ax in self.figure.axes:
                if hasattr(ax, 'get_zlim'):  # Check if it's a 3D axis
                    # Get current z limits
                    z_min, z_max = ax.get_zlim()

                    # Calculate new range based on exaggeration factor
                    exaggeration = self.elevation_slider.value() / 10.0  # Convert 10-30 to 1.0-3.0
                    z_range = self.max_elev - self.min_elev
                    new_range = z_range * exaggeration

                    # Set new limits maintaining the minimum
                    ax.set_zlim(self.min_elev, self.min_elev + new_range)

                    # Update the plot
                    self.canvas.draw()
                    break

        # Removed status message
        # self.status_label.setText(f"Elevation exaggeration set to {self.elevation_slider.value() / 10.0}x")
        self.status_label.hide()

    def save_3d_model(self):
        """Save the current 3D model as a PNG image"""
        # Get user's home directory
        home_dir = os.path.expanduser("~")

        # Find the Downloads folder
        downloads_dir = os.path.join(home_dir, "Downloads")

        # Ensure the Downloads folder exists
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)

        # Define the output file path
        file_path = os.path.join(downloads_dir, "Cambridge_LiDAR_Surface_Model.png")

        # Allow user to choose a different location/filename
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LiDAR Surface Model",
            file_path,
            "PNG Files (*.png)"
        )

        if file_path:
            # Save the figure
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')

            # Hide status label instead of showing message
            self.status_label.hide()


class DamageEstimator(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize class variables that will be set later
        self.building_sliders = {}
        self.impact_sliders = {}
        self.year_slider = None
        self.year_label = None
        self.hurricane_combo = None
        self.property_value_input = None
        self.recovery_spinner = None
        self.results_table = None
        self.economic_table = None
        self.figure = None
        self.canvas = None
        self.economic_figure = None
        self.economic_canvas = None
        self.download_damage_button = None
        self.download_economic_button = None
        self.analyze_button = None
        self.economic_analyze_button = None

        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Title
        self.title = QLabel("Damage Estimation Tool")
        self.title.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                                 "border-radius: 8px; background-color: #444444; padding: 10px;")
        self.layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { font-size: 14px; height: 30px; }")
        self.tabs.tabBar().setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # Tab 1: Property Damage Estimation
        self.property_tab = QWidget()
        self.property_layout = QVBoxLayout()
        self.property_tab.setLayout(self.property_layout)

        # Tab 2: Economic Impact Analysis
        self.economic_tab = QWidget()
        self.economic_layout = QVBoxLayout()
        self.economic_tab.setLayout(self.economic_layout)

        # Add tabs to the widget
        self.tabs.addTab(self.property_tab, "Property Damage")
        self.tabs.addTab(self.economic_tab, "Economic Impact")

        # Add the tabs to the main layout
        self.layout.addWidget(self.tabs)

        # Setup controls for both tabs
        self.setup_property_tab()
        self.setup_economic_tab()

        # Set the main layout
        self.setLayout(self.layout)

        # Initialize data
        self.damage_data = self.initialize_damage_data()
        self.vulnerability_curves = self.initialize_vulnerability_curves()

    def setup_property_tab(self):
        """Set up the property damage estimation tab"""

        # Parameter controls group (existing code)
        controls_group = QGroupBox("Scenario Parameters")
        # Change 1: Update the label size
        controls_group.setStyleSheet("QGroupBox { font-size: 16px; }")
        controls_layout = QVBoxLayout()

        # Year slider (existing code)
        year_layout = QHBoxLayout()
        self.year_label = QLabel("Year: 2025")
        self.year_label.setStyleSheet("font-size: 18px;")
        self.year_slider = QSlider(Qt.Orientation.Horizontal)
        self.year_slider.setRange(0, 100)  # 0 = 2025, 100 = 2125
        self.year_slider.setValue(0)
        self.year_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.year_slider.setTickInterval(10)
        self.year_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.year_slider.valueChanged.connect(self.update_year_label)
        year_layout.addWidget(self.year_label)
        year_layout.addWidget(self.year_slider)
        controls_layout.addLayout(year_layout)

        # Hurricane category dropdown (existing code)
        hurricane_layout = QHBoxLayout()
        hurricane_label = QLabel("Hurricane Category:")
        hurricane_label.setStyleSheet("font-size: 18px;")
        self.hurricane_combo = QComboBox()
        self.hurricane_combo.addItems(["Category 1", "Category 2", "Category 3", "Category 4", "Category 5"])
        self.hurricane_combo.setStyleSheet("font-size: 18px;")
        self.hurricane_combo.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        hurricane_layout.addWidget(hurricane_label)
        hurricane_layout.addWidget(self.hurricane_combo)
        hurricane_layout.addStretch()
        controls_layout.addLayout(hurricane_layout)

        # Property value input
        value_layout = QHBoxLayout()
        value_label = QLabel("Total Property Value ($):")
        value_label.setStyleSheet("font-size: 18px;")
        self.property_value_input = QLineEdit("250000000")  # Default: $250M
        self.property_value_input.setStyleSheet("font-size: 18px;")
        self.property_value_input.editingFinished.connect(self.validate_property_value)
        value_layout.addWidget(value_label)
        value_layout.addWidget(self.property_value_input)
        controls_layout.addLayout(value_layout)

        # Building distribution controls (existing code)
        buildings_group = QGroupBox("Building Types")
        # Change 1: Update the label size
        buildings_group.setStyleSheet("QGroupBox { font-size: 16px; }")
        buildings_layout = QVBoxLayout()

        # Create sliders for building type distribution (existing code)
        self.building_sliders = {}
        building_types = {
            "Residential": 65,
            "Commercial": 20,
            "Industrial": 10,
            "Government/Public": 5
        }

        for btype, default_pct in building_types.items():
            slider_layout = QHBoxLayout()
            label = QLabel(f"{btype}: {default_pct}%")
            label.setStyleSheet("font-size: 16px;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default_pct)
            slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            slider.valueChanged.connect(lambda val, lbl=label, bt=btype: self.update_building_label(val, lbl, bt))
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            buildings_layout.addLayout(slider_layout)
            self.building_sliders[btype] = slider

        buildings_group.setLayout(buildings_layout)

        # Add controls to layout
        controls_group.setLayout(controls_layout)
        self.property_layout.addWidget(controls_group)
        self.property_layout.addWidget(buildings_group)

        # Analysis button & download button
        analyze_button_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze Damage")
        self.analyze_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.analyze_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.analyze_button.clicked.connect(self.analyze_damage)
        analyze_button_layout.addWidget(self.analyze_button)

        self.download_damage_button = QPushButton("Download Damage Analysis")
        self.download_damage_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.download_damage_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.download_damage_button.clicked.connect(self.download_property_results)
        self.download_damage_button.setEnabled(False)
        analyze_button_layout.addWidget(self.download_damage_button)
        self.property_layout.addLayout(analyze_button_layout)

        # ---- SEPARATE RESULTS SECTIONS ----

        # 1. Table Results Section
        table_section = QGroupBox("Damage Breakdown")
        table_section.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        table_section.setFixedHeight(125)
        table_layout = QVBoxLayout()

        # Table for damage breakdown
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Damage Category", "Amount ($)", "% of Total", "Notes"])
        self.results_table.setColumnWidth(0, 200)  # Damage Category column
        self.results_table.setColumnWidth(1, 150)  # Amount column
        self.results_table.setColumnWidth(2, 150)  # % of Total column
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.results_table.setRowCount(0)  # Start with no rows
        self.results_table.setFixedHeight(75)

        table_layout.addWidget(self.results_table)
        table_section.setLayout(table_layout)
        self.property_layout.addWidget(table_section)

        # 2. Chart Results Section
        chart_section = QGroupBox("Damage Visualization")
        chart_section.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        chart_layout = QVBoxLayout()

        # Change 3: Use a scroll area to maintain plot size
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        chart_widget = QWidget()
        chart_inner_layout = QVBoxLayout(chart_widget)

        # Chart for visualization
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        chart_inner_layout.addWidget(self.canvas)

        scroll_area.setWidget(chart_widget)
        chart_layout.addWidget(scroll_area)

        chart_section.setLayout(chart_layout)
        self.property_layout.addWidget(chart_section)

    def validate_property_value(self):
        """Enforces minimum and maximum values for the property value input"""
        try:
            # Get current value, removing any commas
            current_value = float(self.property_value_input.text().replace(',', ''))

            # Set min/max constraints
            MIN_VALUE = 1000000  # $1M
            MAX_VALUE = 10000000000  # $10B

            # Enforce limits
            if current_value < MIN_VALUE:
                self.property_value_input.setText(f"{MIN_VALUE:,.0f}")
            elif current_value > MAX_VALUE:
                self.property_value_input.setText(f"{MAX_VALUE:,.0f}")
            else:
                self.property_value_input.setText(f"{current_value:,.0f}")
        except ValueError:
            # If conversion fails, reset to default
            self.property_value_input.setText("250000000")
            QMessageBox.warning(self, "Invalid Input",
                                "Please enter a valid number. The value has been reset to the default.")

    def setup_economic_tab(self):
        """Set up the economic impact analysis tab"""

        # Economic impact controls (existing code)
        controls_group = QGroupBox("Economic Impact Parameters")
        # Change 1: Update the label size
        controls_group.setStyleSheet("QGroupBox { font-size: 16px; }")
        controls_layout = QVBoxLayout()

        # Recovery period (existing code)
        recovery_layout = QHBoxLayout()
        recovery_label = QLabel("Recovery Period (months):")
        recovery_label.setStyleSheet("font-size: 18px;")
        self.recovery_spinner = QSpinBox()
        self.recovery_spinner.setRange(1, 60)
        self.recovery_spinner.setValue(12)  # Default: 1 year
        self.recovery_spinner.setStyleSheet("font-size: 18px;")
        self.recovery_spinner.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        recovery_layout.addWidget(recovery_label)
        recovery_layout.addWidget(self.recovery_spinner)
        recovery_layout.addStretch()
        controls_layout.addLayout(recovery_layout)

        # Economic impacts to include (existing code)
        impacts_group = QGroupBox("Impact Categories")
        # Change 1: Update the label size
        impacts_group.setStyleSheet("QGroupBox { font-size: 16px; }")
        impacts_layout = QVBoxLayout()

        # Economic impact categories with default values (existing code)
        self.impact_sliders = {}
        impact_categories = {
            "Business Interruption": 30,
            "Tourism Reduction": 20,
            "Infrastructure Repair": 25,
            "Public Services Cost": 15,
            "Environmental Remediation": 10
        }

        for impact, default_pct in impact_categories.items():
            slider_layout = QHBoxLayout()
            label = QLabel(f"{impact}: {default_pct}%")
            label.setStyleSheet("font-size: 16px;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default_pct)
            slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            slider.valueChanged.connect(lambda val, lbl=label, ic=impact: self.update_impact_label(val, lbl, ic))
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)
            impacts_layout.addLayout(slider_layout)
            self.impact_sliders[impact] = slider

        impacts_group.setLayout(impacts_layout)

        # Add controls to layout
        controls_group.setLayout(controls_layout)
        self.economic_layout.addWidget(controls_group)
        self.economic_layout.addWidget(impacts_group)

        # Analysis button
        economic_analyze_button_layout = QHBoxLayout()
        self.economic_analyze_button = QPushButton("Analyze Economic Impact")
        self.economic_analyze_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.economic_analyze_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.economic_analyze_button.clicked.connect(self.analyze_economic_impact)
        economic_analyze_button_layout.addWidget(self.economic_analyze_button)
        self.download_economic_button = QPushButton("Download Economic Analysis")
        self.download_economic_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.download_economic_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.download_economic_button.clicked.connect(self.download_economic_results)
        self.download_economic_button.setEnabled(False)
        economic_analyze_button_layout.addWidget(self.download_economic_button)
        self.economic_layout.addLayout(economic_analyze_button_layout)

        # ---- SEPARATE RESULTS SECTIONS ----

        # 1. Table Results Section
        economic_table_section = QGroupBox("Economic Impact Breakdown")
        economic_table_section.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        economic_table_section.setFixedHeight(125)
        economic_table_layout = QVBoxLayout()

        # Table for economic impact breakdown
        self.economic_table = QTableWidget()
        self.economic_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.economic_table.setColumnCount(4)
        self.economic_table.setHorizontalHeaderLabels(
            ["Impact Category", "Short-term ($)", "Long-term ($)", "Total ($)"])
        # Set specific widths for each column
        self.economic_table.setColumnWidth(0, 200)  # Impact Category column
        self.economic_table.setColumnWidth(1, 150)  # Short-term column
        self.economic_table.setColumnWidth(2, 150)  # Long-term column
        self.economic_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.economic_table.setRowCount(0)  # Start with no rows
        self.economic_table.setFixedHeight(75)

        economic_table_layout.addWidget(self.economic_table)
        economic_table_section.setLayout(economic_table_layout)
        self.economic_layout.addWidget(economic_table_section)

        # 2. Chart Results Section
        economic_chart_section = QGroupBox("Economic Impact Visualization")
        economic_chart_section.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        economic_chart_layout = QVBoxLayout()

        # Change 3: Use a scroll area to maintain plot size
        economic_scroll_area = QScrollArea()
        economic_scroll_area.setWidgetResizable(True)
        economic_chart_widget = QWidget()
        economic_chart_inner_layout = QVBoxLayout(economic_chart_widget)

        # Chart for visualization
        self.economic_figure = Figure(figsize=(10, 6))
        self.economic_canvas = FigureCanvas(self.economic_figure)
        self.economic_canvas.setMinimumHeight(400)
        economic_chart_inner_layout.addWidget(self.economic_canvas)

        economic_scroll_area.setWidget(economic_chart_widget)
        economic_chart_layout.addWidget(economic_scroll_area)

        economic_chart_section.setLayout(economic_chart_layout)
        self.economic_layout.addWidget(economic_chart_section)

    def update_year_label(self):
        """Update the year label as the slider changes"""
        year = self.year_slider.value() + 2025
        self.year_label.setText(f"Year: {year}")

    def update_building_label(self, value, label, building_type):
        """Update building type percentage label"""
        label.setText(f"{building_type}: {value}%")

        # Ensure total is 100%
        total = sum(slider.value() for slider in self.building_sliders.values())

        # Reset all labels to normal or red depending on total
        for lbl in [widget for widget in self.findChildren(QLabel) if
                    any(bt in widget.text() for bt in self.building_sliders.keys())]:
            if total != 100:
                lbl.setStyleSheet("font-size: 16px; color: red;")
            else:
                lbl.setStyleSheet("font-size: 16px;")

    def update_impact_label(self, value, label, impact_category):
        """Update economic impact category percentage label"""
        label.setText(f"{impact_category}: {value}%")

        # Ensure total is 100%
        total = sum(slider.value() for slider in self.impact_sliders.values())

        # Reset all labels to normal or red depending on total
        for lbl in [widget for widget in self.findChildren(QLabel) if
                    any(ic in widget.text() for ic in self.impact_sliders.keys())]:
            if total != 100:
                lbl.setStyleSheet("font-size: 16px; color: red;")
            else:
                lbl.setStyleSheet("font-size: 16px;")

    def initialize_damage_data(self):
        """Initialize damage modeling data based on HAZUS-MH methodology"""
        # Damage factors for different hurricane categories
        damage_factors = {
            "cat1": {
                "buildings": 0.05,  # 5% damage to buildings
                "contents": 0.10,  # 10% damage to contents
                "displacement": 0.15,  # 15% of population displaced
                "debris": 0.05,  # 5% debris generation factor
                "infrastructure": 0.03,  # 3% damage to infrastructure
                "cleanup": 0.02  # 2% of total property value for cleanup
            },
            "cat2": {
                "buildings": 0.15,
                "contents": 0.25,
                "displacement": 0.30,
                "debris": 0.12,
                "infrastructure": 0.10,
                "cleanup": 0.05
            },
            "cat3": {
                "buildings": 0.30,
                "contents": 0.45,
                "displacement": 0.50,
                "debris": 0.25,
                "infrastructure": 0.20,
                "cleanup": 0.10
            },
            "cat4": {
                "buildings": 0.50,
                "contents": 0.70,
                "displacement": 0.75,
                "debris": 0.40,
                "infrastructure": 0.35,
                "cleanup": 0.18
            },
            "cat5": {
                "buildings": 0.75,
                "contents": 0.90,
                "displacement": 0.95,
                "debris": 0.65,
                "infrastructure": 0.60,
                "cleanup": 0.30
            }
        }

        # Climate change multipliers (increase damage based on future year)
        year_multipliers = {}
        base_year = 2025
        for year in range(base_year, base_year + 101):  # 2025 to 2125
            # Calculate multiplier: 0% increase in 2025, up to 50% increase by 2125
            year_factor = (year - base_year) / 100.0
            multiplier = 1.0 + (0.5 * year_factor)
            year_multipliers[year] = multiplier

        return {
            "damage_factors": damage_factors,
            "year_multipliers": year_multipliers
        }

    def initialize_vulnerability_curves(self):
        """Initialize vulnerability curves based on HAZUS-MH flood methodology"""
        # Simplified vulnerability curves for different building types
        vulnerability_data = {
            "Residential": {
                "depth_damage_ratio": [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90],  # Damage ratio by depth (ft)
                "contents_factor": 1.2,  # Contents damage is 1.2x structure damage
                "displacement_days": [0, 5, 15, 45, 90, 180, 365]  # Days displaced by depth
            },
            "Commercial": {
                "depth_damage_ratio": [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85],
                "contents_factor": 1.5,
                "displacement_days": [0, 3, 10, 30, 60, 120, 240]
            },
            "Industrial": {
                "depth_damage_ratio": [0.0, 0.05, 0.20, 0.35, 0.50, 0.65, 0.80],
                "contents_factor": 1.8,
                "displacement_days": [0, 7, 20, 60, 120, 180, 270]
            },
            "Government/Public": {
                "depth_damage_ratio": [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80],
                "contents_factor": 1.0,
                "displacement_days": [0, 7, 15, 30, 90, 180, 365]
            }
        }

        return vulnerability_data

    def get_current_parameters(self):
        """Get common parameters used in both analysis methods"""
        # Get parameters
        year = self.year_slider.value() + 2025
        hurricane_category = self.hurricane_combo.currentIndex() + 1  # 1-5

        # Get total property value
        try:
            total_property_value = float(self.property_value_input.text().replace(',', ''))
        except ValueError:
            raise ValueError("Please enter a valid property value.")

        # Get damage factors based on hurricane category
        cat_key = f"cat{hurricane_category}"
        damage_factors = self.damage_data["damage_factors"][cat_key]

        # Apply year multiplier for climate change effects
        year_multiplier = self.damage_data["year_multipliers"][year]

        return {
            "year": year,
            "hurricane_category": hurricane_category,
            "total_property_value": total_property_value,
            "damage_factors": damage_factors,
            "year_multiplier": year_multiplier
        }

    def analyze_damage(self):
        """Analyze property damage based on input parameters"""
        try:
            self.download_damage_button.setEnabled(True)

            # Get common parameters
            params = self.get_current_parameters()

            # Check if building distribution adds up to 100%
            total_building_pct = sum(slider.value() for slider in self.building_sliders.values())
            if total_building_pct != 100:
                QMessageBox.warning(self, "Input Error", "Building type percentages must sum to 100%.")
                return

            # Calculate base damage values
            building_damage = params["total_property_value"] * params["damage_factors"]["buildings"] * params[
                "year_multiplier"]
            contents_damage = params["total_property_value"] * params["damage_factors"]["contents"] * params[
                "year_multiplier"]
            infrastructure_damage = params["total_property_value"] * params["damage_factors"]["infrastructure"] * \
                                    params["year_multiplier"]
            cleanup_cost = params["total_property_value"] * params["damage_factors"]["cleanup"] * params[
                "year_multiplier"]
            debris_removal = params["total_property_value"] * params["damage_factors"]["debris"] * params[
                "year_multiplier"]

            # Estimate displacement and relocation costs
            avg_household_value = 250000  # Average home value in Cambridge
            num_households = (params["total_property_value"] * 0.65) / avg_household_value  # Assume 65% residential
            displaced_households = num_households * params["damage_factors"]["displacement"] * params["year_multiplier"]
            persons_displaced = displaced_households * 2.5  # Average household size

            # Calculate displacement duration based on hurricane category
            displacement_days = {1: 30, 2: 60, 3: 120, 4: 180, 5: 365}[params["hurricane_category"]]
            relocation_cost = persons_displaced * 100 * displacement_days

            # Calculate emergency response costs (based on FEMA guidelines)
            emergency_response = params["total_property_value"] * 0.05 * params[
                "year_multiplier"]  # 5% of total property value

            # Calculate total direct damages
            total_direct_damage = building_damage + contents_damage + infrastructure_damage

            # Calculate total cost (direct + indirect)
            total_cost = total_direct_damage + cleanup_cost + debris_removal + relocation_cost + emergency_response

            # Calculate economic multiplication factor
            economic_multiplier = 1.5 + (params["hurricane_category"] * 0.3)  # Ranges from 1.8 to 3.0
            total_economic_impact = total_direct_damage * economic_multiplier

            # Update results table
            self.populate_results_table(
                building_damage,
                contents_damage,
                infrastructure_damage,
                cleanup_cost,
                debris_removal,
                relocation_cost,
                emergency_response,
                total_direct_damage,
                total_economic_impact,
                total_cost,
                params["total_property_value"]
            )

            # Update chart
            self.update_damage_chart(
                building_damage,
                contents_damage,
                infrastructure_damage,
                cleanup_cost,
                debris_removal,
                relocation_cost,
                emergency_response
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during analysis: {str(e)}")

    def populate_results_table(self, building_damage, contents_damage, infrastructure_damage,
                               cleanup_cost, debris_removal, relocation_cost, emergency_response,
                               total_direct, total_economic, total_cost, total_property_value):
        """Populate the results table with damage estimates"""
        # Clear the table and set row count
        self.results_table.clearContents()
        self.results_table.setRowCount(10)  # We need exactly 10 rows

        # Set data
        data = [
            ["Building Structural Damage", building_damage, building_damage / total_cost * 100,
             "Direct damage to building structures"],
            ["Contents Damage", contents_damage, contents_damage / total_cost * 100,
             "Damage to furniture, equipment and personal property"],
            ["Infrastructure Damage", infrastructure_damage, infrastructure_damage / total_cost * 100,
             "Roads, utilities, public facilities"],
            ["Cleanup Costs", cleanup_cost, cleanup_cost / total_cost * 100,
             "Post-disaster cleanup operations"],
            ["Debris Removal", debris_removal, debris_removal / total_cost * 100,
             "Removal and disposal of disaster debris"],
            ["Relocation Costs", relocation_cost, relocation_cost / total_cost * 100,
             "Temporary housing and relocation expenses"],
            ["Emergency Response", emergency_response, emergency_response / total_cost * 100,
             "First responders, emergency services"],
            ["Total Direct Damage", total_direct, total_direct / total_cost * 100,
             "Sum of direct physical damages"],
            ["Total Economic Impact", total_economic, total_economic / total_cost * 100,
             "Long-term economic effects (multiplier effect)"],
            ["Total Cost", total_cost, 100.0,
             f"{(total_cost / total_property_value * 100):.1f}% of total property value"]
        ]

        # Format and insert data
        for row, (category, amount, percent, note) in enumerate(data):
            # Create new items for each cell
            category_item = QTableWidgetItem(category)
            amount_item = QTableWidgetItem(f"${amount:,.2f}")
            percent_item = QTableWidgetItem(f"{percent:.1f}%")
            note_item = QTableWidgetItem(note)

            # Set the items in the table
            self.results_table.setItem(row, 0, category_item)
            self.results_table.setItem(row, 1, amount_item)
            self.results_table.setItem(row, 2, percent_item)
            self.results_table.setItem(row, 3, note_item)

        # Make sure the final row is bold for total
        for col in range(4):
            if self.results_table.item(9, col):
                self.results_table.item(9, col).setFont(QFont("Arial", weight=QFont.Weight.Bold))

        # Resize rows to fit content and make sure the table refreshes
        self.results_table.resizeRowsToContents()
        self.results_table.update()

    def update_damage_chart(self, building_damage, contents_damage, infrastructure_damage,
                            cleanup_cost, debris_removal, relocation_cost, emergency_response):
        """Update the damage breakdown chart"""
        # Clear the figure
        self.figure.clear()

        # Data for charts
        labels = ['Building', 'Contents', 'Infrastructure',
                  'Cleanup', 'Debris', 'Relocation', 'Emergency']
        sizes = [building_damage, contents_damage, infrastructure_damage,
                 cleanup_cost, debris_removal, relocation_cost, emergency_response]

        # Create pie chart
        ax = self.figure.add_subplot(121)
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title('Damage Cost Breakdown')

        # Create bar chart for comparison
        ax2 = self.figure.add_subplot(122)
        y_pos = np.arange(len(labels))

        # Convert values to hundred thousands for display
        sizes_in_hundred_thousands = [size / 100000 for size in sizes]

        bars = ax2.barh(y_pos, sizes_in_hundred_thousands, align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Damage Amount ($ in Hundred Thousands)')
        ax2.set_title('Damage Category Comparison')

        # Add value labels to the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            # Format the value in hundred thousands with a dollar sign and commas
            value_text = f'${sizes_in_hundred_thousands[i]:,.1f}'
            ax2.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                     value_text, ha='left', va='center', fontsize=9)

        # Format x-axis as currency
        ax2.xaxis.set_major_formatter('${x:,.0f}')

        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw()

    def analyze_economic_impact(self):
        """Analyze economic impact based on input parameters"""
        try:
            self.download_economic_button.setEnabled(True)

            # Check if impact categories add up to 100%
            total_impact_pct = sum(slider.value() for slider in self.impact_sliders.values())
            if total_impact_pct != 100:
                QMessageBox.warning(self, "Input Error", "Impact category percentages must sum to 100%.")
                return

            # Get common parameters
            params = self.get_current_parameters()
            recovery_months = self.recovery_spinner.value()

            # Calculate base economic factors
            # Cambridge, MD estimated GDP (simplified for demonstration)
            estimated_gdp = params["total_property_value"] * 0.2  # Annual GDP estimate

            # Calculate impact percentages
            short_term_factors = {
                "Business Interruption": 0.1 * params["hurricane_category"] * params["year_multiplier"],
                "Tourism Reduction": 0.05 * params["hurricane_category"] * params["year_multiplier"],
                "Infrastructure Repair": 0.08 * params["hurricane_category"] * params["year_multiplier"],
                "Public Services Cost": 0.06 * params["hurricane_category"] * params["year_multiplier"],
                "Environmental Remediation": 0.04 * params["hurricane_category"] * params["year_multiplier"],
                "Tax Revenue Loss": 0.07 * params["hurricane_category"] * params["year_multiplier"],
                "Employment Impact": 0.09 * params["hurricane_category"] * params["year_multiplier"]
            }

            # Adjust based on recovery period (longer recovery = more impact)
            recovery_factor = min(recovery_months / 12.0, 5.0)  # Cap at 5 years equivalent

            # Calculate impacts
            economic_impacts = {}
            for category, factor in short_term_factors.items():
                # Short-term impact (first year)
                short_term = estimated_gdp * factor

                # Long-term impact (subsequent years, diminishing)
                if recovery_factor <= 1.0:
                    long_term = 0  # Less than a year recovery
                else:
                    # Use a diminishing return formula for long-term impacts
                    long_term = short_term * (recovery_factor - 1) * 0.7  # 70% of first year impact for remaining time

                # Total impact
                total_impact = short_term + long_term

                economic_impacts[category] = {
                    "short_term": short_term,
                    "long_term": long_term,
                    "total": total_impact
                }

            # Calculate total economic impact
            total_short_term = sum(impact["short_term"] for impact in economic_impacts.values())
            total_long_term = sum(impact["long_term"] for impact in economic_impacts.values())
            total_impact = total_short_term + total_long_term

            # Add totals to the impacts dictionary
            economic_impacts["Total Economic Impact"] = {
                "short_term": total_short_term,
                "long_term": total_long_term,
                "total": total_impact
            }

            # Update economic table
            self.populate_economic_table(economic_impacts)

            # Update economic chart
            self.update_economic_chart(economic_impacts)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during analysis: {str(e)}")

    def populate_economic_table(self, economic_impacts):
        """Populate the economic impact table"""
        # Clear the table and set row count
        self.economic_table.clearContents()
        self.economic_table.setRowCount(len(economic_impacts))

        # Populate table
        for row, (category, impacts) in enumerate(economic_impacts.items()):
            # Create new items for each cell
            category_item = QTableWidgetItem(category)
            short_term_item = QTableWidgetItem(f"${impacts['short_term']:,.2f}")
            long_term_item = QTableWidgetItem(f"${impacts['long_term']:,.2f}")
            total_item = QTableWidgetItem(f"${impacts['total']:,.2f}")

            # Set the items in the table
            self.economic_table.setItem(row, 0, category_item)
            self.economic_table.setItem(row, 1, short_term_item)
            self.economic_table.setItem(row, 2, long_term_item)
            self.economic_table.setItem(row, 3, total_item)

            # Bold the total row
            if category == "Total Economic Impact":
                for col in range(4):
                    if self.economic_table.item(row, col):
                        self.economic_table.item(row, col).setFont(QFont("Arial", weight=QFont.Weight.Bold))

        # Resize rows to fit content and make sure the table refreshes
        self.economic_table.resizeRowsToContents()
        self.economic_table.update()

    def update_economic_chart(self, economic_impacts):
        """Update the economic impact visualization chart"""
        # Clear the figure
        self.economic_figure.clear()

        # Create subplot for the stacked bar chart
        ax1 = self.economic_figure.add_subplot(121)

        # Prepare data for stacked bar chart (excluding the total)
        categories = [cat for cat in economic_impacts.keys() if cat != "Total Economic Impact"]
        short_term_values = [economic_impacts[cat]["short_term"] for cat in categories]
        long_term_values = [economic_impacts[cat]["long_term"] for cat in categories]
        total_values = [short_term_values[i] + long_term_values[i] for i in range(len(categories))]

        # Create stacked bar chart
        x = np.arange(len(categories))
        width = 0.8

        # Convert to hundred thousands for better display
        short_term_hundred_k = [val / 100000 for val in short_term_values]
        long_term_hundred_k = [val / 100000 for val in long_term_values]
        total_hundred_k = [val / 100000 for val in total_values]

        # Create the stacked bars
        short_term_bars = ax1.bar(x, short_term_hundred_k, width, label='Short-term Impact')
        long_term_bars = ax1.bar(x, long_term_hundred_k, width, bottom=short_term_hundred_k, label='Long-term Impact')

        # Add labels and title
        ax1.set_ylabel('Economic Impact ($ in Hundred Thousands)')
        ax1.set_title('Economic Impact by Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=30, ha='right')
        ax1.legend()

        # Add value labels to the bars
        for i, (short_bar, long_bar) in enumerate(zip(short_term_bars, long_term_bars)):
            # Short-term value label (at the middle of its portion)
            if short_term_hundred_k[i] > 0:
                height = short_bar.get_height()
                ax1.text(i, height / 2, f'${short_term_hundred_k[i]:,.1f}',
                         ha='center', va='center', fontsize=8, color='white', fontweight='bold')

            # Long-term value label (if it exists, at the middle of its portion)
            if long_term_hundred_k[i] > 0:
                height = long_bar.get_height()
                ypos = short_term_hundred_k[i] + height / 2
                ax1.text(i, ypos, f'${long_term_hundred_k[i]:,.1f}',
                         ha='center', va='center', fontsize=8, color='white', fontweight='bold')

            # Total value label at the top
            total_height = short_term_hundred_k[i] + long_term_hundred_k[i]
            ax1.text(i, total_height + 0.5, f'${total_hundred_k[i]:,.1f}',
                     ha='center', va='bottom', fontsize=9)

        # Create pie chart for total impact distribution
        ax2 = self.economic_figure.add_subplot(122)

        # Create pie chart
        ax2.pie(total_values, labels=categories, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax2.set_title('Total Impact Distribution')

        # Update the canvas
        self.economic_figure.tight_layout()
        self.economic_canvas.draw()

    def download_property_results(self):
        """Download property damage analysis results with user-selected path"""
        try:
            download_dir = str(Path.home() / "Downloads")
            default_zip_filename = os.path.join(download_dir, "Cambridge_Damage_Analysis.zip")

            # Ask user for save location (default to Downloads)
            zip_filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Damage Analysis",
                default_zip_filename,
                "ZIP Files (*.zip)"
            )

            # Check if user canceled
            if not zip_filename:
                return

            # Create a zip file
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                # 1. Export table data to CSV
                csv_data = io.StringIO()

                # Write header row
                headers = ["Damage Category", "Amount ($)", "% of Total", "Notes"]
                csv_data.write(",".join(f'"{h}"' for h in headers) + "\n")

                # Write data rows
                for row in range(self.results_table.rowCount()):
                    row_data = []
                    for col in range(self.results_table.columnCount()):
                        item = self.results_table.item(row, col)
                        if item:
                            # Quote the cell data to handle commas and special characters
                            row_data.append(f'"{item.text()}"')
                        else:
                            row_data.append('""')
                    csv_data.write(",".join(row_data) + "\n")

                # Add CSV to zip file
                zipf.writestr("Cambridge_Damage_Analysis_Table.csv", csv_data.getvalue())

                # 2. Save the plot as PNG
                plot_buffer = io.BytesIO()
                self.figure.savefig(plot_buffer, format='png', dpi=300)
                plot_buffer.seek(0)
                zipf.writestr("Cambridge_Damage_Visualization.png", plot_buffer.getvalue())

                # 3. Add a metadata file
                meta_data = io.StringIO()
                meta_data.write("# Cambridge Flood Damage Analysis\n")
                meta_data.write("# Generated by Cambridge Flood Analysis Tool\n\n")
                meta_data.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                meta_data.write(f"Scenario Year: {self.year_slider.value() + 2025}\n")
                meta_data.write(f"Hurricane Category: Category {self.hurricane_combo.currentIndex() + 1}\n")
                meta_data.write(f"Total Property Value: {self.property_value_input.text()}\n")

                zipf.writestr("Cambridge_Analysis_Metadata.txt", meta_data.getvalue())

        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Error downloading results: {str(e)}")

    def download_economic_results(self):
        """Download economic impact analysis results with user-selected path"""
        try:
            download_dir = str(Path.home() / "Downloads")
            default_zip_filename = os.path.join(download_dir, "Cambridge_Economic_Analysis.zip")

            # Ask user for save location (default to Downloads)
            zip_filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Economic Analysis",
                default_zip_filename,
                "ZIP Files (*.zip)"
            )

            # Check if user canceled
            if not zip_filename:
                return

            # Create a zip file
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                # 1. Export table data to CSV
                csv_data = io.StringIO()

                # Write header row
                headers = ["Impact Category", "Short-term ($)", "Long-term ($)", "Total ($)"]
                csv_data.write(",".join(f'"{h}"' for h in headers) + "\n")

                # Write data rows
                for row in range(self.economic_table.rowCount()):
                    row_data = []
                    for col in range(self.economic_table.columnCount()):
                        item = self.economic_table.item(row, col)
                        if item:
                            # Quote the cell data to handle commas and special characters
                            row_data.append(f'"{item.text()}"')
                        else:
                            row_data.append('""')
                    csv_data.write(",".join(row_data) + "\n")

                # Add CSV to zip file
                zipf.writestr("Cambridge_Economic_Analysis_Table.csv", csv_data.getvalue())

                # 2. Save the plot as PNG
                plot_buffer = io.BytesIO()
                self.economic_figure.savefig(plot_buffer, format='png', dpi=300)
                plot_buffer.seek(0)
                zipf.writestr("Cambridge_Economic_Visualization.png", plot_buffer.getvalue())

                # 3. Add a metadata file
                meta_data = io.StringIO()
                meta_data.write("# Cambridge Economic Impact Analysis\n")
                meta_data.write("# Generated by Cambridge Flood Analysis Tool\n\n")
                meta_data.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                meta_data.write(f"Scenario Year: {self.year_slider.value() + 2025}\n")
                meta_data.write(f"Hurricane Category: Category {self.hurricane_combo.currentIndex() + 1}\n")
                meta_data.write(f"Recovery Period: {self.recovery_spinner.value()} months\n")
                meta_data.write(f"Total Property Value: {self.property_value_input.text()}\n")

                zipf.writestr("Cambridge_Analysis_Metadata.txt", meta_data.getvalue())

        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Error downloading economic analysis: {str(e)}")


class AddressValidator(QThread):
    result_signal = pyqtSignal(bool)

    def __init__(self, address):
        super().__init__()
        self.address = address

    def run(self):
        geolocator = Nominatim(user_agent="insurance_app", timeout=5)
        try:
            print(f"Validating address: {self.address}")
            location = geolocator.geocode(self.address, exactly_one=True)
            if location:
                print(f"Found location: {location.address}")
                reverse_location = geolocator.reverse((location.latitude, location.longitude), exactly_one=True)
                address_details = reverse_location.raw.get("address", {})
                print(f"Address details: {address_details}")

                # Check for Cambridge in either city or town fields
                city = address_details.get("city", "").lower()
                town = address_details.get("town", "").lower()
                state = address_details.get("state", "").lower()

                # The address details show 'town' instead of 'city' for Cambridge
                if ("cambridge" in town.lower() or "cambridge" in city.lower()) and state.lower() in ["md", "maryland"]:
                    print("Address is valid!")
                    self.result_signal.emit(True)
                    return
                print("Address is not in Cambridge, MD")
            else:
                print("Location not found")
            self.result_signal.emit(False)
        except (GeocoderTimedOut, GeocoderServiceError, GeocoderQueryError) as e:
            print(f"Geocoding error: {e}")
            self.result_signal.emit(False)


class InsuranceProjections(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Title
        self.title = QLabel("Home Insurance Projections")
        self.title.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                                 "border-radius: 8px; background-color: #444444; padding: 10px;")
        self.layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Form Layout for User Input
        form_container = QVBoxLayout()
        form_container.setAlignment(Qt.AlignmentFlag.AlignHCenter)  # Center align all contents

        # Function to center label and input in an HBoxLayout
        def create_centered_row(label_text, input_widget, w_width=695, l_width=125):
            input_widget.setStyleSheet("font-size: 16px; padding: 8px;")
            input_widget.setFixedWidth(w_width)

            label = QLabel(label_text)
            label.setFixedWidth(l_width)
            label.setStyleSheet("font-size: 16px; border: 2px solid #444444; border-radius: 8px;")

            row_layout = QHBoxLayout()  # Horizontal layout for centering
            row_layout.addStretch()  # Push content to center
            row_layout.addWidget(label)
            row_layout.addWidget(input_widget)
            row_layout.addStretch()  # Push content to center

            return row_layout

        self.address_input = QLineEdit()
        self.value_input = QLineEdit()
        self.year_input = QLineEdit()
        self.address_input.setText("307 Gay St, Cambridge, MD 21613")
        self.value_input.setText("250000")
        self.year_input.setText("2025")

        # Connect validation signals
        self.value_input.editingFinished.connect(self.validate_home_value)
        self.year_input.editingFinished.connect(self.validate_projection_year)

        # Add centered rows to the form container
        form_container.addLayout(create_centered_row("Address:", self.address_input))
        form_container.addLayout(create_centered_row("Home Value ($):", self.value_input))
        form_container.addLayout(create_centered_row("Projection Year:", self.year_input))

        self.validator_thread = None
        self.layout.addLayout(form_container)

        # Submit Button
        self.submit_button = QPushButton("Generate Projection")
        self.submit_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.submit_button.setFixedWidth(925)
        self.submit_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.submit_button.clicked.connect(self.validate_and_generate_projection)
        self.layout.addWidget(self.submit_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Horizontal Layout for Table and Graph with balanced widths
        self.results_layout = QHBoxLayout()
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_layout.addWidget(self.table, 1)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Create a frame to wrap the graph
        self.graph_frame = QFrame()
        self.graph_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #444444;
                border-radius: 8px;
                background-color: white;
            }
        """)

        # Apply padding around the graph by using a QVBoxLayout inside the frame
        graph_layout = QVBoxLayout()
        graph_layout.setContentsMargins(10, 10, 10, 10)  # Add padding for better spacing
        graph_layout.addWidget(self.canvas)

        # Set the layout to the frame
        self.graph_frame.setLayout(graph_layout)

        # Replace direct graph addition with the styled frame
        self.results_layout.addWidget(self.graph_frame, 1)

        self.layout.addLayout(self.results_layout)

        # Download Buttons
        self.download_layout = QHBoxLayout()
        self.download_table_button = QPushButton("Download Table")
        self.download_table_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.download_table_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.download_table_button.clicked.connect(self.download_table)
        self.download_table_button.setEnabled(False)
        self.download_graph_button = QPushButton("Download Graph")
        self.download_graph_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.download_graph_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.download_graph_button.clicked.connect(self.download_graph)
        self.download_graph_button.setEnabled(False)

        self.download_layout.addWidget(self.download_table_button)
        self.download_layout.addWidget(self.download_graph_button)
        self.layout.addLayout(self.download_layout)

        self.setLayout(self.layout)

    # Update validate_and_generate_projection to handle empty address differently
    def validate_and_generate_projection(self):
        """Validate all inputs and generate projection if valid"""
        address = self.address_input.text().strip()

        if not address:
            # Reset to default address
            default_address = "307 Gay St, Cambridge, MD 21613"
            self.address_input.setText(default_address)

            # Show error message
            QMessageBox.warning(self, "Invalid Address",
                                "Please enter a valid Cambridge address. The address has been reset to the default.")
            return

        # Validate inputs
        self.validate_home_value()
        self.validate_projection_year()

        self.submit_button.setEnabled(False)

        # Create and start the validator thread
        self.validator_thread = AddressValidator(address)
        self.validator_thread.result_signal.connect(self.handle_address_validation_and_generate)
        self.validator_thread.start()

    def handle_address_validation_and_generate(self, is_valid):
        if is_valid:
            self.address_input.setStyleSheet("font-size: 16px; padding: 8px;")
            print("Address validated, generating projection...")
            self.generate_projection()
        else:
            # Reset to default address
            default_address = "307 Gay St, Cambridge, MD 21613"
            self.address_input.setText(default_address)

            # Reset styling to normal
            self.address_input.setStyleSheet("font-size: 16px; padding: 8px;")

            # Show error message
            QMessageBox.warning(self, "Invalid Address",
                                "Not a valid Cambridge address. The address has been reset to the default.")

            print("Address validation failed")

        self.submit_button.setEnabled(True)

    def generate_projection(self):
        try:
            self.download_table_button.setEnabled(True)
            self.download_graph_button.setEnabled(True)

            # Get home value (commas removed)
            home_value = int(self.value_input.text().replace(',', ''))

            # Get year
            year = int(self.year_input.text())

            print(f"Generating projection for: Value=${home_value}, Year={year}")

            # base rate was calculated by Cambridge avg home insurance / Cambridge avg home price
            base_rate = 0.006  # Starting insurance rate (0.6% of home value per year)
            inflation_factor = 1.02  # Annual price increase due to inflation
            start_year = 2025
            end_year = 2125

            hurricane_rate_2025 = 1 / 2.5  # Hurricanes per year in 2025
            hurricane_rate_2125 = 1 / 1.5  # Hurricanes per year in 2125

            # Compute yearly rate change using linear interpolation
            rate_increase_per_year = (hurricane_rate_2125 - hurricane_rate_2025) / (end_year - start_year)

            # Hurricane intensity model based on NOAA projections
            initial_proportion_cat4_plus = 0.30  # 30% in 2025
            target_increase = 0.20  # 50% in 2125
            proportion_increase_per_year = target_increase / (end_year - start_year)

            # Sensitivity factor for how hurricanes impact insurance rates
            sensitivity_factor = 0.7

            years = list(range(start_year, year + 1))
            insurance_costs = []
            cat3_or_lower_counts = []
            cat4_or_greater_counts = []
            cumulative_hurricanes = []
            cumulative_insurance_increase = []
            cumulative_cat3 = 0
            cumulative_cat4 = 0

            # Initial insurance cost for baseline comparison
            base_insurance_cost = home_value * base_rate

            for y in years:
                # Inflation-adjusted insurance cost
                inflation_factor_yearly = (inflation_factor ** (y - start_year))
                insurance_cost = home_value * base_rate * inflation_factor_yearly

                # Hurricane occurrence calculation
                current_hurricane_rate = hurricane_rate_2025 + (y - start_year) * rate_increase_per_year
                total_hurricanes = np.random.poisson(current_hurricane_rate)

                # Determine proportion of Category 4+ hurricanes for this year
                cat4_proportion = initial_proportion_cat4_plus + proportion_increase_per_year * (y - start_year)
                cat4_proportion = min(cat4_proportion, 1.0)  # Ensure it never exceeds 100%

                # Split hurricanes into categories
                cat4_hurricanes = int(round(total_hurricanes * cat4_proportion))
                cat3_or_lower_hurricanes = total_hurricanes - cat4_hurricanes

                # Accumulate hurricane counts
                cumulative_cat3 += cat3_or_lower_hurricanes
                cumulative_cat4 += cat4_hurricanes
                total_cumulative_hurricanes = cumulative_cat3 + cumulative_cat4

                # Adjust insurance cost based on hurricane intensity
                insurance_adjustment = 1 + sensitivity_factor * (cat4_proportion - initial_proportion_cat4_plus)
                insurance_cost *= insurance_adjustment

                # Calculate cumulative insurance increase percentage
                insurance_increase_percent = ((insurance_cost - base_insurance_cost) / base_insurance_cost) * 100

                # Store results
                insurance_costs.append(insurance_cost)
                cat3_or_lower_counts.append(cat3_or_lower_hurricanes)
                cat4_or_greater_counts.append(cat4_hurricanes)
                cumulative_hurricanes.append(total_cumulative_hurricanes)
                cumulative_insurance_increase.append(insurance_increase_percent)

            # Display updated results
            self.display_results(years, insurance_costs, cat3_or_lower_counts, cat4_or_greater_counts,
                                 cumulative_hurricanes, cumulative_insurance_increase)

        except ValueError as e:
            print(f"Projection error: {e}")
            return

    def display_results(self, years, costs, cat3_counts, cat4_counts, cumulative_hurricanes,
                        cumulative_insurance_increase):
        """Update the table and graph with new projections"""

        # Update Table
        self.table.setRowCount(len(years))
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Year", "Estimated Yearly Insurance ($)", "Cat 3 or Lower Hurricanes",
            "Cat 4+ Hurricanes", "Cumulative Hurricanes", "Cumulative Price Increase (%)"
        ])

        self.table.setColumnWidth(0, 65)
        self.table.setColumnWidth(1, 170)
        self.table.setColumnWidth(2, 170)
        self.table.setColumnWidth(3, 170)
        self.table.setColumnWidth(4, 170)
        self.table.setColumnWidth(5, 170)

        for i, (year, cost, cat3, cat4, cum_hurr, cum_insur) in enumerate(
                zip(years, costs, cat3_counts, cat4_counts, cumulative_hurricanes, cumulative_insurance_increase)):
            self.table.setItem(i, 0, QTableWidgetItem(str(year)))
            self.table.setItem(i, 1, QTableWidgetItem(f"${cost:,.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(str(cat3)))
            self.table.setItem(i, 3, QTableWidgetItem(str(cat4)))
            self.table.setItem(i, 4, QTableWidgetItem(str(cum_hurr)))
            self.table.setItem(i, 5, QTableWidgetItem(f"{cum_insur:.2f}%"))

        # Update Graph
        self.ax.clear()
        self.ax.plot(years, costs, marker='o', linestyle='-', color='b', label="Insurance Cost ($)")
        self.ax.set_xlabel("Year")
        self.ax.set_ylabel("Insurance Cost ($)")
        self.ax.set_title(f"Projected Home Insurance Costs for {self.address_input.text().strip()}")
        self.ax.legend()
        self.canvas.draw()

    def validate_home_value(self):
        """Enforces minimum and maximum values for the home value input"""
        try:
            # Get current value, removing any commas
            current_value = float(self.value_input.text().replace(',', ''))

            # Set min/max constraints
            MIN_VALUE = 10000  # $10K
            MAX_VALUE = 50000000  # $50M

            # Enforce limits
            if current_value < MIN_VALUE:
                self.value_input.setText(f"{MIN_VALUE:,.0f}")
            elif current_value > MAX_VALUE:
                self.value_input.setText(f"{MAX_VALUE:,.0f}")
            else:
                self.value_input.setText(f"{current_value:,.0f}")
        except ValueError:
            # If conversion fails, reset to default
            self.value_input.setText("250000")
            QMessageBox.warning(self, "Invalid Input",
                                "Please enter a valid number. The value has been reset to the default.")

    def validate_projection_year(self):
        """Enforces minimum and maximum values for the projection year input"""
        try:
            # Get current value
            current_year = int(self.year_input.text())

            # Set min/max constraints
            MIN_YEAR = 2025
            MAX_YEAR = 2125

            # Enforce limits
            if current_year < MIN_YEAR:
                self.year_input.setText(str(MIN_YEAR))
            elif current_year > MAX_YEAR:
                self.year_input.setText(str(MAX_YEAR))
        except ValueError:
            # If conversion fails, reset to default
            self.year_input.setText("2025")
            QMessageBox.warning(self, "Invalid Input",
                                "Please enter a valid year. The value has been reset to the default.")

    def download_table(self):
        # Get user's home directory and Downloads folder
        home_dir = os.path.expanduser("~")
        downloads_dir = os.path.join(home_dir, "Downloads")

        # Ensure the Downloads folder exists
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)

        # Define default file path
        file_path = os.path.join(downloads_dir, "Cambridge_Insurance_Projection_Data.csv")

        # Allow user to choose the final location (pre-set to Downloads)
        path, _ = QFileDialog.getSaveFileName(self, "Save Table", file_path, "CSV Files (*.csv)")

        # Check if user canceled the dialog
        if not path:
            return  # Exit the function if no path was selected

        try:
            data = []
            for row in range(self.table.rowCount()):
                data.append([self.table.item(row, col).text() for col in range(self.table.columnCount())])

            df = pd.DataFrame(data,
                              columns=["Year", "Estimated Yearly Insurance ($)", "Cat 3 or Lower Hurricanes",
                                       "Cat 4+ Hurricanes", "Cumulative Hurricanes", "Cumulative Price Increase (%)"])
            df.to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Error downloading insurance projection table: {str(e)}")

    def download_graph(self):
        # Get user's home directory and Downloads folder
        home_dir = os.path.expanduser("~")
        downloads_dir = os.path.join(home_dir, "Downloads")

        # Ensure the Downloads folder exists
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)

        # Define default file path
        file_path = os.path.join(downloads_dir, "Cambridge_Insurance_Projection_Graph.png")

        # Allow user to choose the final location (pre-set to Downloads)
        path, _ = QFileDialog.getSaveFileName(self, "Save Graph", file_path, "PNG Files (*.png)")

        # Check if user canceled the dialog
        if not path:
            return  # Exit the function if no path was selected

        try:
            self.figure.savefig(path)
        except Exception as e:
            QMessageBox.critical(self, "Download Error", f"Error downloading insurance projection graph: {str(e)}")


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

        # Loading message
        self.loading_label = QLabel("Click 'Load Model' to generate the elevation model")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 24px; color: #444444; margin: 40px;")
        self.layout.addWidget(self.loading_label)

        # Load Model Button
        self.load_button = QPushButton("Load Model")
        self.load_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.load_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.load_button.clicked.connect(self.initialize_model)
        self.layout.addWidget(self.load_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Matplotlib Figure - initially hidden
        self.figure = plt.figure(figsize=(16, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.hide()  # Hide until data is loaded
        self.layout.addWidget(self.canvas)

        # Controls Layout - initially hidden
        self.controls_widget = QWidget()
        controls_layout = QHBoxLayout(self.controls_widget)
        controls_layout.setSpacing(15)

        # Year label and slider
        self.year_label = QLabel("Year: 2025")
        self.year_label.setStyleSheet("font-size: 22px;")
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

        self.controls_widget.hide()  # Hide until data is loaded
        self.layout.addWidget(self.controls_widget)

        # Button Layout for additional controls - initially hidden
        self.buttons_widget = QWidget()
        button_layout = QHBoxLayout(self.buttons_widget)

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

        self.buttons_widget.hide()  # Hide until data is loaded
        self.layout.addWidget(self.buttons_widget)

        # Initialize variables
        self.current_elevation = None
        self.base_elevation = None
        self.data_loaded = False
        self.cache_file = os.path.join(os.path.expanduser("~"), ".cambridge_dem_cache.pkl")
        self.data_thread = None

    def initialize_model(self):
        """Start loading the model data when requested"""
        if self.data_loaded:
            return

        # Update UI to show loading state
        self.loading_label.setText("Loading elevation data... Please wait.")
        self.load_button.setEnabled(False)
        QApplication.processEvents()  # Force UI update

        # Create and start the loading thread
        self.data_thread = DataLoadingThread(self.cache_file)
        self.data_thread.finished.connect(self.on_data_loaded)
        self.data_thread.start()

    def on_data_loaded(self):
        """Handle the completion of data loading"""
        if self.data_thread and self.data_thread.loaded_data is not None:
            # Get data from thread
            self.base_elevation = self.data_thread.loaded_data
            self.current_elevation = np.copy(self.base_elevation)
            self.data_loaded = True

            # Update UI to show model is ready
            self.loading_label.hide()
            self.load_button.hide()
            self.canvas.show()
            self.controls_widget.show()
            self.buttons_widget.show()

            # Plot initial model
            self.plot_3d_terrain(flood_level=None)
        else:
            # Handle loading failure
            self.loading_label.setText("Failed to load elevation data. Please try again.")
            self.load_button.setEnabled(True)

    # Update the year label as the slider changes
    def update_year_label(self):
        year = self.year_slider.value() + 2025
        self.year_label.setText(f"Year: {year}")

    def plot_3d_terrain(self, flood_level=None):
        """Generate an interactive 3D elevation plot with a color legend and info box"""
        if self.base_elevation is None:
            return

        self.figure.clear()  # Clear previous plot
        ax = self.figure.add_subplot(111, projection='3d')

        # Create X, Y grid with proper coordinates for Cambridge, MD
        height, width = self.base_elevation.shape

        # Define approximate lat/lon bounds for Cambridge, MD
        # These coordinates define the bounding box for Cambridge
        lat_min, lat_max = 38.53, 38.61  # Latitude range for Cambridge
        lon_min, lon_max = -76.13, -76.04  # Longitude range for Cambridge

        # Create coordinate grids that map to real-world coordinates
        lon_coords = np.linspace(lon_min, lon_max, width)
        lat_coords = np.linspace(lat_min, lat_max, height)
        X, Y = np.meshgrid(lon_coords, lat_coords)

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

        # Downsample for performance - take every Nth point
        downsample_factor = 4  # Adjust based on performance needs
        X_ds = X[::downsample_factor, ::downsample_factor]
        Y_ds = Y[::downsample_factor, ::downsample_factor]
        elevation_ds = elevation[::downsample_factor, ::downsample_factor]

        # Plot terrain
        terrain = ax.plot_surface(X_ds, Y_ds, elevation_ds, cmap="terrain", edgecolor='none', alpha=0.8)

        # Keep the elevation color scale legend always visible
        cbar = self.figure.colorbar(terrain, shrink=0.7, aspect=20, pad=0.1)
        cbar.set_label("Elevation (feet)", fontsize=12)

        # If flooding, overlay a water surface
        if flood_level is not None and flood_level > 0:
            flood_mask = elevation_ds < flood_level
            water_surface = np.ones_like(elevation_ds) * flood_level

            # Masked arrays to plot only flooded areas
            water_X = np.ma.masked_array(X_ds, ~flood_mask)
            water_Y = np.ma.masked_array(Y_ds, ~flood_mask)
            water_Z = np.ma.masked_array(water_surface, ~flood_mask)

            # Plot water surface
            water = ax.plot_surface(water_X, water_Y, water_Z, color='blue', alpha=0.5, edgecolor='none')

        # Labels with more accurate coordinate descriptions
        ax.set_xlabel("Longitude (°W)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_zlabel("Elevation (feet)")

        # Format axes with degree symbols
        ax.xaxis.set_major_formatter(lambda x, pos: f"{abs(x):.2f}°W")
        ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.2f}°N")

        # Keep the title intact
        title_text = f"Cambridge, MD - 3D Elevation Model"
        if flood_level is not None:
            title_text = f"Cambridge, MD - 3D Flood Simulation (Water Level: {flood_level:.2f} ft)"
        ax.set_title(title_text, fontsize=14, fontweight='bold')

        # Place elevation info box on the left to mirror the color legend
        left_box = self.figure.add_axes([0.16, 0.36, 0.12, 0.16])  # Left side placement
        left_box.axis("off")  # Hide axis

        # Add text inside the left box
        left_text = f"Elevation Range:\n{min_elev:.2f} to {max_elev:.2f} ft"
        if flood_level is not None:
            left_text += f"\nCurrent Water Level:\n{flood_level:.2f} ft"

        # Add coordinate range info
        left_text += f"\n\nLocation:\nCambridge, MD\n{lat_min:.2f}°N to {lat_max:.2f}°N\n{abs(lon_min):.2f}°W to {abs(lon_max):.2f}°W"

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
        file_path = os.path.join(downloads_dir, "Cambridge_DEM_Elevation_Model.png")

        # Allow user to choose a different location/filename
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Elevation Model",
            file_path,
            "PNG Files (*.png)"
        )

        if file_path:
            # Save the figure
            self.figure.savefig(file_path, dpi=300)
            print(f"Flood Model saved as {file_path}")


class DataLoadingThread(QThread):
    """Thread for loading elevation data with caching support"""

    def __init__(self, cache_file):
        super().__init__()
        self.cache_file = cache_file
        self.loaded_data = None

    def run(self):
        """Load DEM data, using cache if available"""
        # First try to load from cache
        if self.try_load_from_cache():
            print("Loaded DEM data from cache")
            return

        # If no cache, load from source
        self.load_dem_from_source()

        # Save to cache if successful
        if self.loaded_data is not None:
            self.save_to_cache()

    def try_load_from_cache(self):
        """Attempt to load data from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.loaded_data = pickle.load(f)
                return True
        except Exception as e:
            print(f"Error loading from cache: {e}")
        return False

    def save_to_cache(self):
        """Save loaded data to cache file"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.loaded_data, f)
            print(f"Saved DEM data to cache: {self.cache_file}")
        except Exception as e:
            print(f"Error saving to cache: {e}")

    def load_dem_from_source(self):
        """Load DEM data from the original source"""
        try:
            dem_path = "../Data/Dorchester_DEM/dorc2015_m/"  # Adjust path as needed

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

                self.loaded_data = elevation_data
                print(f"Loaded DEM: min={np.nanmin(self.loaded_data)}, max={np.nanmax(self.loaded_data)}")

        except Exception as e:
            print(f"Error loading DEM data: {e}")
            # If there's an error, generate synthetic data
            self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """Generate synthetic DEM data when the real data can't be loaded"""
        print("Generating synthetic DEM data")
        # Create a 200x200 synthetic terrain using sine functions and random noise
        x = np.linspace(0, 10, 200)
        y = np.linspace(0, 10, 200)
        X, Y = np.meshgrid(x, y)

        # Base terrain with some hills and valleys
        Z = (np.sin(X) * np.cos(Y) +
             np.sin(2 * X) * np.cos(2 * Y) / 2 +
             np.sin(3 * X) * np.cos(3 * Y) / 3)

        # Add some random noise for realistic terrain
        Z += np.random.rand(200, 200) * 0.2

        # Scale to the appropriate elevation range for Cambridge
        Z = (Z - Z.min()) * 12 - 0.82  # Scale to range from -0.82 to ~11.42 feet

        self.loaded_data = Z


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
        flood_item = QTreeWidgetItem(["Flood Simulators"])
        street_view_item = QTreeWidgetItem(["-- Street View"])
        model_elevation_item = QTreeWidgetItem(["-- 3D Elevation"])
        model_surface_item = QTreeWidgetItem(["-- 3D Surface"])
        interactive_map_item = QTreeWidgetItem(["-- Interactive Map"])
        # Add subcategories under Flood Simulators
        flood_item.addChildren([street_view_item, model_elevation_item, model_surface_item, interactive_map_item])
        insurance_item = QTreeWidgetItem(["Insurance Projections"])
        damage_item = QTreeWidgetItem(["Damage Estimator"])

        # Add all top-level items to the tree
        self.sidebar_tree.addTopLevelItem(home_item)
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
            "-- Street View": StreetView(),
            "-- 3D Elevation": Elevation(),
            "-- 3D Surface": LidarSurface(),
            "-- Interactive Map": InteractiveMap(),
            "Insurance Projections": InsuranceProjections(),
            "Damage Estimator": DamageEstimator()
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

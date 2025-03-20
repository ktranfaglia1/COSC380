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
    QSpinBox, QHeaderView
)
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random
import sys
import os
import folium
import rasterio
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderQueryError
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import box
import numpy as np
import pandas as pd


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

        # Parameter controls group
        controls_group = QGroupBox("Scenario Parameters")
        controls_layout = QVBoxLayout()

        # Year slider
        year_layout = QHBoxLayout()
        self.year_label = QLabel("Year: 2025")
        self.year_label.setStyleSheet("font-size: 18px;")
        self.year_slider = QSlider(Qt.Orientation.Horizontal)
        self.year_slider.setRange(0, 100)  # 0 = 2025, 100 = 2125
        self.year_slider.setValue(0)
        self.year_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.year_slider.setTickInterval(10)
        self.year_slider.valueChanged.connect(self.update_year_label)
        year_layout.addWidget(self.year_label)
        year_layout.addWidget(self.year_slider)
        controls_layout.addLayout(year_layout)

        # Hurricane category dropdown
        hurricane_layout = QHBoxLayout()
        hurricane_label = QLabel("Hurricane Category:")
        hurricane_label.setStyleSheet("font-size: 18px;")
        self.hurricane_combo = QComboBox()
        self.hurricane_combo.addItems(["Category 1", "Category 2", "Category 3", "Category 4", "Category 5"])
        self.hurricane_combo.setStyleSheet("font-size: 18px;")
        hurricane_layout.addWidget(hurricane_label)
        hurricane_layout.addWidget(self.hurricane_combo)
        controls_layout.addLayout(hurricane_layout)

        # Property value input
        value_layout = QHBoxLayout()
        value_label = QLabel("Total Property Value ($):")
        value_label.setStyleSheet("font-size: 18px;")
        self.property_value_input = QLineEdit("250000000")  # Default: $250M
        self.property_value_input.setStyleSheet("font-size: 18px;")
        value_layout.addWidget(value_label)
        value_layout.addWidget(self.property_value_input)
        controls_layout.addLayout(value_layout)

        # Building distribution controls
        buildings_group = QGroupBox("Building Types")
        buildings_layout = QVBoxLayout()

        # Create sliders for building type distribution
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
            # Use a lambda that stores current values to avoid late binding issues
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

        # Analysis button
        analyze_button = QPushButton("Analyze Damage")
        analyze_button.setStyleSheet("font-size: 22px; padding: 8px;")
        analyze_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        analyze_button.clicked.connect(self.analyze_damage)
        self.property_layout.addWidget(analyze_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Results display
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.Shape.StyledPanel)
        results_frame.setStyleSheet("background-color: white;")
        results_layout = QVBoxLayout()

        # Table for damage breakdown
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Make table read-only
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Damage Category", "Amount ($)", "% of Total", "Notes"])
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self.results_table)

        # Chart for visualization
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)

        results_frame.setLayout(results_layout)
        self.property_layout.addWidget(results_frame)

        # Download button
        download_button = QPushButton("Export Results")
        download_button.setStyleSheet("font-size: 22px; padding: 8px;")
        download_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        download_button.clicked.connect(self.export_results)
        self.property_layout.addWidget(download_button, alignment=Qt.AlignmentFlag.AlignHCenter)

    def setup_economic_tab(self):
        """Set up the economic impact analysis tab"""

        # Economic impact controls
        controls_group = QGroupBox("Economic Impact Parameters")
        controls_layout = QVBoxLayout()

        # Recovery period
        recovery_layout = QHBoxLayout()
        recovery_label = QLabel("Recovery Period (months):")
        recovery_label.setStyleSheet("font-size: 18px;")
        self.recovery_spinner = QSpinBox()
        self.recovery_spinner.setRange(1, 60)
        self.recovery_spinner.setValue(12)  # Default: 1 year
        self.recovery_spinner.setStyleSheet("font-size: 18px;")
        recovery_layout.addWidget(recovery_label)
        recovery_layout.addWidget(self.recovery_spinner)
        controls_layout.addLayout(recovery_layout)

        # Economic impacts to include
        impacts_group = QGroupBox("Impact Categories")
        impacts_layout = QVBoxLayout()

        # Economic impact categories with default values
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
            # Use a lambda that stores current values to avoid late binding issues
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
        economic_analyze_button = QPushButton("Analyze Economic Impact")
        economic_analyze_button.setStyleSheet("font-size: 22px; padding: 8px;")
        economic_analyze_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        economic_analyze_button.clicked.connect(self.analyze_economic_impact)
        self.economic_layout.addWidget(economic_analyze_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Results display
        economic_results_frame = QFrame()
        economic_results_frame.setFrameShape(QFrame.Shape.StyledPanel)
        economic_results_frame.setStyleSheet("background-color: white;")
        economic_results_layout = QVBoxLayout()

        # Table for economic impact breakdown
        self.economic_table = QTableWidget()
        self.economic_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Make table read-only
        self.economic_table.setColumnCount(4)
        self.economic_table.setHorizontalHeaderLabels(
            ["Impact Category", "Short-term ($)", "Long-term ($)", "Total ($)"])
        self.economic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        economic_results_layout.addWidget(self.economic_table)

        # Chart for visualization
        self.economic_figure = Figure(figsize=(10, 6))
        self.economic_canvas = FigureCanvas(self.economic_figure)
        economic_results_layout.addWidget(self.economic_canvas)

        economic_results_frame.setLayout(economic_results_layout)
        self.economic_layout.addWidget(economic_results_frame)

        # Download button
        economic_download_button = QPushButton("Export Economic Analysis")
        economic_download_button.setStyleSheet("font-size: 22px; padding: 8px;")
        economic_download_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        economic_download_button.clicked.connect(self.export_economic_results)
        self.economic_layout.addWidget(economic_download_button, alignment=Qt.AlignmentFlag.AlignHCenter)

    def update_year_label(self):
        """Update the year label as the slider changes"""
        year = self.year_slider.value() + 2025
        self.year_label.setText(f"Year: {year}")

    def update_building_label(self, value, label, building_type):
        """Update building type percentage label"""
        label.setText(f"{building_type}: {value}%")

        # Ensure total is 100%
        total = sum(slider.value() for slider in self.building_sliders.values())
        if total != 100:
            # Highlight if not 100%
            label.setStyleSheet("font-size: 16px; color: red;")
        else:
            label.setStyleSheet("font-size: 16px;")

    def update_impact_label(self, value, label, impact_category):
        """Update economic impact category percentage label"""
        label.setText(f"{impact_category}: {value}%")

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
        # Clear the table
        self.results_table.clearContents()

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
            self.results_table.setItem(row, 0, QTableWidgetItem(category))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"${amount:,.2f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{percent:.1f}%"))
            self.results_table.setItem(row, 3, QTableWidgetItem(note))

        # Resize rows to fit content
        self.results_table.resizeRowsToContents()

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

        ax2.barh(y_pos, sizes, align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Damage Amount ($)')
        ax2.set_title('Damage Category Comparison')

        # Format x-axis as currency
        ax2.xaxis.set_major_formatter('${x:,.0f}')

        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw()

    def analyze_economic_impact(self):
        """Analyze economic impact based on input parameters"""
        try:
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
        # Clear the table
        self.economic_table.clearContents()

        # Populate table
        for row, (category, impacts) in enumerate(economic_impacts.items()):
            self.economic_table.setItem(row, 0, QTableWidgetItem(category))
            self.economic_table.setItem(row, 1, QTableWidgetItem(f"${impacts['short_term']:,.2f}"))
            self.economic_table.setItem(row, 2, QTableWidgetItem(f"${impacts['long_term']:,.2f}"))
            self.economic_table.setItem(row, 3, QTableWidgetItem(f"${impacts['total']:,.2f}"))

        # Resize rows to fit content
        self.economic_table.resizeRowsToContents()

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

        # Create stacked bar chart
        x = np.arange(len(categories))
        width = 0.8

        ax1.bar(x, short_term_values, width, label='Short-term Impact')
        ax1.bar(x, long_term_values, width, bottom=short_term_values, label='Long-term Impact')

        # Add labels and title
        ax1.set_ylabel('Economic Impact ($)')
        ax1.set_title('Economic Impact by Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.legend()

        # Format y-axis as currency
        ax1.yaxis.set_major_formatter('${x:,.0f}')

        # Create pie chart for total impact distribution
        ax2 = self.economic_figure.add_subplot(122)

        # Prepare data for pie chart (only using total values, excluding the total row)
        total_values = [economic_impacts[cat]["total"] for cat in categories]

        # Create pie chart
        ax2.pie(total_values, labels=categories, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax2.set_title('Total Impact Distribution')

        # Update the canvas
        self.economic_figure.tight_layout()
        self.economic_canvas.draw()

    def export_results(self):
        """Export property damage analysis results to a CSV file"""
        try:
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Property Damage Analysis", "", "CSV Files (*.csv);;All Files (*)"
            )

            if not file_path:
                return  # User canceled

            # Ensure file has .csv extension
            if not file_path.endswith('.csv'):
                file_path += '.csv'

            # Create data for export
            data = []
            for row in range(self.results_table.rowCount()):
                row_data = []
                for col in range(self.results_table.columnCount()):
                    item = self.results_table.item(row, col)
                    if item:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                if any(row_data):  # Only add non-empty rows
                    data.append(row_data)

            # Convert to DataFrame for easy CSV export
            headers = ["Damage Category", "Amount ($)", "% of Total", "Notes"]
            df = pd.DataFrame(data, columns=headers)

            # Add metadata
            year = self.year_slider.value() + 2025
            hurricane_category = self.hurricane_combo.currentIndex() + 1
            metadata = pd.DataFrame([
                ["Analysis Date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Scenario Year", str(year)],
                ["Hurricane Category", f"Category {hurricane_category}"],
                ["Total Property Value", self.property_value_input.text()]
            ], columns=["Parameter", "Value"])

            # Export to CSV
            with open(file_path, 'w', newline='') as f:
                f.write("# Cambridge Flood Damage Analysis\n")
                f.write("# Generated by Cambridge Flood Analysis Tool\n\n")

                # Write metadata
                metadata.to_csv(f, index=False)
                f.write("\n\n")

                # Write damage data
                df.to_csv(f, index=False)

            QMessageBox.information(self, "Export Successful",
                                    f"Results successfully exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results: {str(e)}")

    def export_economic_results(self):
        """Export economic impact analysis results to a CSV file"""
        try:
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Economic Impact Analysis", "", "CSV Files (*.csv);;All Files (*)"
            )

            if not file_path:
                return  # User canceled

            # Ensure file has .csv extension
            if not file_path.endswith('.csv'):
                file_path += '.csv'

            # Create data for export
            data = []
            for row in range(self.economic_table.rowCount()):
                row_data = []
                for col in range(self.economic_table.columnCount()):
                    item = self.economic_table.item(row, col)
                    if item:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                if any(row_data):  # Only add non-empty rows
                    data.append(row_data)

            # Convert to DataFrame for easy CSV export
            headers = ["Impact Category", "Short-term ($)", "Long-term ($)", "Total ($)"]
            df = pd.DataFrame(data, columns=headers)

            # Add metadata
            year = self.year_slider.value() + 2025
            hurricane_category = self.hurricane_combo.currentIndex() + 1
            recovery_months = self.recovery_spinner.value()
            metadata = pd.DataFrame([
                ["Analysis Date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Scenario Year", str(year)],
                ["Hurricane Category", f"Category {hurricane_category}"],
                ["Recovery Period", f"{recovery_months} months"],
                ["Total Property Value", self.property_value_input.text()]
            ], columns=["Parameter", "Value"])

            # Export to CSV
            with open(file_path, 'w', newline='') as f:
                f.write("# Cambridge Economic Impact Analysis\n")
                f.write("# Generated by Cambridge Flood Analysis Tool\n\n")

                # Write metadata
                metadata.to_csv(f, index=False)
                f.write("\n\n")

                # Write impact data
                df.to_csv(f, index=False)

            QMessageBox.information(self, "Export Successful",
                                    f"Economic analysis successfully exported to {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting economic analysis: {str(e)}")


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
            input_widget.returnPressed.connect(self.validate_and_generate_projection)
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
        self.download_graph_button = QPushButton("Download Graph")
        self.download_graph_button.setStyleSheet("font-size: 22px; padding: 8px;")
        self.download_graph_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.download_graph_button.clicked.connect(self.download_graph)

        self.download_layout.addWidget(self.download_table_button)
        self.download_layout.addWidget(self.download_graph_button)
        self.layout.addLayout(self.download_layout)

        self.setLayout(self.layout)

    def validate_and_generate_projection(self):
        address = self.address_input.text().strip()

        if not address:
            self.address_input.setStyleSheet("background-color: rgba(248, 215, 218, 0.3);")  # Subtle red
            print("No Address")
            return

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
            self.address_input.setStyleSheet("font-size: 16px; padding: 8px; "
                                             "background-color: rgba(255, 100, 100, 0.3);")  # Subtle red
            print("Address validation failed")

        self.submit_button.setEnabled(True)

    def generate_projection(self):
        try:
            # Get home value and validate range
            try:
                home_value = int(self.value_input.text())
                home_value = max(10000, min(home_value, 10000000))  # Clamp between 10k and 10M
                self.value_input.setText(str(int(home_value)))
            except ValueError:
                home_value = 500000  # Default
                self.value_input.setText(str(int(home_value)))

            # Get year and validate range
            try:
                year = int(self.year_input.text())
                year = max(2025, min(year, 2125))  # Clamp between 2025 and 2100
                self.year_input.setText(str(year))
            except ValueError:
                year = 2025
                self.year_input.setText(str(year))

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

    def download_table(self):
        # Get user's home directory and Downloads folder
        home_dir = os.path.expanduser("~")
        downloads_dir = os.path.join(home_dir, "Downloads")

        # Ensure the Downloads folder exists
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)

        # Define default file path
        file_path = os.path.join(downloads_dir, "Insurance_Projection_Data.csv")

        # Allow user to choose the final location (pre-set to Downloads)
        path, _ = QFileDialog.getSaveFileName(self, "Save Table", file_path, "CSV Files (*.csv)")

        if path:
            data = []
            for row in range(self.table.rowCount()):
                data.append([self.table.item(row, col).text() for col in range(self.table.columnCount())])

            df = pd.DataFrame(data,
                              columns=["Year", "Estimated Yearly Insurance ($)", "Cat 3 or Lower Hurricanes",
                                       "Cat 4+ Hurricanes", "Cumulative Hurricanes", "Cumulative Price Increase (%)"])
            df.to_csv(path, index=False)

    def download_graph(self):
        # Get user's home directory and Downloads folder
        home_dir = os.path.expanduser("~")
        downloads_dir = os.path.join(home_dir, "Downloads")

        # Ensure the Downloads folder exists
        if not os.path.exists(downloads_dir):
            os.makedirs(downloads_dir)

        # Define default file path
        file_path = os.path.join(downloads_dir, "Insurance_Projection_Graph.png")

        # Allow user to choose the final location (pre-set to Downloads)
        path, _ = QFileDialog.getSaveFileName(self, "Save Graph", file_path, "PNG Files (*.png)")

        if path:
            self.figure.savefig(path)


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
            "-- 3D Elevation": QWidget(),
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

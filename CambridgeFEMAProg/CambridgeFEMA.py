#  Author: Kyle Tranfaglia
#  Title: Main Program for City of Cambridge Flood Analysis Tool
#  Last updated: 02/28/25
#  Description: This program uses PyQt6 and *** packages to build a flood analysis tool for the City of Cambridge
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QStackedWidget, QListWidget, QHBoxLayout, QLabel, QSlider, QSizePolicy,
    QTreeWidget, QTreeWidgetItem, QComboBox
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtWebEngineWidgets import QWebEngineView
import random
import sys
import folium


class InteractiveMap(QWidget):
    def __init__(self):
        super().__init__()
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

        self.setLayout(layout)
        self.create_map()

    def create_map(self):
        # Create a Folium map centered on Cambridge, MD.
        map_center = [38.572, -76.078]
        m = folium.Map(location=map_center, zoom_start=15,)
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
        m = folium.Map(location=self.map_center,
                       zoom_start=15,
                       min_zoom=15,
                       max_zoom=15,
                       zoom_control=False,
                       dragging=False,
                       control_scale=True)
        folium.Marker([38.572, -76.078], popup="Cambridge, MD").add_to(m)
        html_data = m.get_root().render()
        self.web_view.setHtml(html_data, baseUrl=QUrl("http://localhost/"))

    def simulate(self):
        # Get simulation parameters
        slider_value = self.slider.value()
        year = slider_value + 2025
        self.year_label.setText(f"Year: {year}")
        category_text = self.category_combo.currentText().strip()
        try:
            category = int(category_text.split()[1])
        except Exception:
            category = 1
        water_factor = slider_value / 100.0

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
                effective_t = min(t * (f + 0.3), 1.0)

                # Increase radius per ring (each ring 30% larger)
                radius = base_radius * (1 + 0.3 * i)

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
        print(f"Simulate pressed: {category_text}, Year: {year}")


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
        model_3d_item = QTreeWidgetItem(["-- 3D Model"])
        interactive_map_item = QTreeWidgetItem(["-- Interactive Map"])
        # Add subcategories under Flood Simulators
        flood_item.addChildren([street_view_item, model_3d_item, interactive_map_item])
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
        pixmap = QPixmap("Cambridge_logo.png")  # Adjust path as needed
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)
        home_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        home_page.setLayout(home_layout)

        # Create pages for each sidebar selection
        self.pages = {
            "Home": home_page,
            "Settings": QWidget(),
            "-- Street View": StreetView(),
            "-- Topographic": QWidget(),
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

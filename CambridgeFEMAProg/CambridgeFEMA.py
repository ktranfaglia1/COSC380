#  Author: Kyle Tranfaglia
#  Title: Main Program for City of Cambridge Flood Analysis Tool
#  Last updated: 02/21/25
#  Description: This program uses PyQt6 and *** packages to build a flood analysis tool for the City of Cambridge
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QStackedWidget, QListWidget, QHBoxLayout, QLabel, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import sys


# Simulates water level rise over a static map of Cambridge using a slider
class WaterLevelSimulator(QWidget):
    def __init__(self):
        super().__init__()

        # Layout setup
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Title Label
        self.title = QLabel("Water Level Simulation")
        self.title.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                                 "border-radius: 8px; background-color: #444444; padding: 10px;")
        self.layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Map Display (Make sure to show the entire map)
        self.map_label = QLabel()
        pixmap = QPixmap("Cambridge_map.png")
        self.map_label.setPixmap(pixmap)
        self.map_label.setScaledContents(True)
        self.map_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.map_label.setMinimumSize(600, 400)
        self.layout.addWidget(self.map_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Flood Overlay (Transparent Blue)
        # Start with no water (overlay height = 0)
        self.flood_overlay = QLabel(self.map_label)
        self.flood_overlay.setStyleSheet("background-color: rgba(0, 0, 255, 100);")
        self.flood_overlay.setGeometry(0, 0, self.map_label.width(), 0)

        # Water Level Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)  # 0: no water, 100: full map
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_flood_level)
        self.layout.addWidget(self.slider)

        # Water Level Label
        self.level_label = QLabel("Year: 2025")
        self.layout.addWidget(self.level_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.setLayout(self.layout)

    # Updates the flood overlay height based on the slider value
    def update_flood_level(self, value):
        # Get the current displayed size of the map
        map_width = self.map_label.width()
        map_height = self.map_label.height()

        # Compute the overlay height as a fraction of the map's height
        water_height = int((value / 100.0) * map_height)

        # The overlay starts at the top (y = 0) and its height increases with the slider
        self.flood_overlay.setGeometry(0, 0, map_width, water_height)
        self.level_label.setText(f"Year: {value + 2025}")


# Main application window with sidebar navigation and multiple pages
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cambridge Flood Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Create a container widget for sidebar and content
        self.main_container = QWidget()
        self.setCentralWidget(self.main_container)
        self.layout = QHBoxLayout()
        self.main_container.setLayout(self.layout)

        # Create the sidebar with a fixed width
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)  # Adjust width for optimal display
        self.sidebar_layout = QVBoxLayout()
        self.sidebar.setLayout(self.sidebar_layout)
        self.sidebar_list = QListWidget()
        self.sidebar_list.setSpacing(5)
        self.sidebar_list.addItems(["Home", "Water Level Simulator", "Insurance Projections",
                                    "Damage Estimator", "Settings"])
        self.sidebar_list.setStyleSheet("font-size: 21px; font-family: 'Roboto';")
        self.sidebar_layout.addWidget(self.sidebar_list)

        # Create a container for sidebar and toggle button
        self.sidebar_wrapper = QWidget()
        self.sidebar_wrapper_layout = QHBoxLayout()
        self.sidebar_wrapper.setLayout(self.sidebar_wrapper_layout)
        self.sidebar_wrapper_layout.setSpacing(0)
        self.sidebar_wrapper_layout.addWidget(self.sidebar)

        # Create the toggle button, attaching it directly to the sidebar
        self.toggle_button = QPushButton("◀")
        self.toggle_button.setFixedWidth(25)
        self.toggle_button.setFixedHeight(40)
        self.toggle_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_wrapper_layout.addWidget(self.toggle_button, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.layout.addWidget(self.sidebar_wrapper)

        # Create the main content area
        self.central_widget = QStackedWidget()
        self.layout.addWidget(self.central_widget, 1)

        # Home page
        home_page = QWidget()
        home_layout = QVBoxLayout()
        home_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Header Label (Title)
        home_label = QLabel("Welcome to the Cambridge Flood Analysis Tool")
        home_label.setStyleSheet("font-size: 42px; font-family: 'Roboto'; border: 2px solid black; "
                                 "border-radius: 8px; background-color: #444444; padding: 10px;")
        home_layout.addWidget(home_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Image Label
        image_label = QLabel()
        pixmap = QPixmap("Cambridge_logo.png")  # Replace with your actual image path
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)  # Allow scaling

        # Add image to layout
        home_layout.addWidget(image_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        home_page.setLayout(home_layout)

        # --- CREATE PAGES ---
        self.pages = {
            "Home": home_page,
            "Water Level Simulator": WaterLevelSimulator(),
            "Insurance Projections": QWidget(),
            "Damage Estimator": QWidget(),
            "Settings": QWidget()
        }

        # Add all pages to the stacked widget
        for page in self.pages.values():
            self.central_widget.addWidget(page)

        # Handle navigation
        self.sidebar_list.currentRowChanged.connect(self.switch_page)

    # Switches the displayed page based on the sidebar selection
    def switch_page(self, index):
        self.central_widget.setCurrentIndex(index)

    # Toggles the visibility of the sidebar
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

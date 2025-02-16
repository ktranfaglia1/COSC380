#  Author: Kyle Tranfaglia
#  Title: Test Program for the City of Cambridge GIS Tool
#  Last updated: 02/15/25
#  Description: This program tests folium features integrated with PyQt corresponding web engine compatibility
#  to generate a layered and featured web-based interactive map accessible in a Python window

import sys
import folium
import pandas as pd
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os


# Main class to display an interactive web based map using a PyQt window
class MapWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup window
        self.setWindowTitle("Advanced Folium Map")
        self.setGeometry(100, 100, 800, 600)

        # Create the map using Folium
        self.map = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles=None)
        folium.TileLayer('cartodbdark_matter').add_to(self.map)  # Dark Matter theme

        # Add a Tile Layer (using Stamen Terrain)
        folium.TileLayer(
            'Stamen Terrain',
            attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
        ).add_to(self.map)

        # Add a Marker with a Popup
        folium.Marker(
            [37.7749, -122.4194],
            popup=folium.Popup("<b>San Francisco</b><br>Iconic city in California", max_width=250)
        ).add_to(self.map)

        # Add a Circle with Popup
        folium.Circle(radius=500, location=[37.7749, -122.4194], color='blue', fill=True,
                      fill_opacity=0.4, popup="Larger Circle").add_to(self.map)

        # Add Polygon (draw a triangle)
        folium.Polygon(
            locations=[
                [37.7749, -122.4194],
                [37.7799, -122.4194],
                [37.7799, -122.4144]
            ],
            color='red', fill=True, fill_opacity=0.6, popup="Bigger Polygon"
        ).add_to(self.map)

        # Example GeoJSON data (typically load this from a file)
        geo_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-122.45, 37.75],
                                [-122.45, 37.77],
                                [-122.42, 37.77],
                                [-122.42, 37.75],
                                [-122.45, 37.75]
                            ]
                        ]
                    },
                    "properties": {
                        "name": "Region 1",
                        "value": 10
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-122.42, 37.75],
                                [-122.42, 37.77],
                                [-122.39, 37.77],
                                [-122.39, 37.75],
                                [-122.42, 37.75]
                            ]
                        ]
                    },
                    "properties": {
                        "name": "Region 2",
                        "value": 20
                    }
                }
            ]
        }

        # Example Data (matching the GeoJSON 'name' property)
        data = pd.DataFrame({
            'name': ['Region 1', 'Region 2'],
            'value': [10, 20]
        })

        # Create a Choropleth layer with correct formatting
        folium.Choropleth(
            geo_data=geo_data,  # GeoJSON data
            data=data,  # Data frame containing data to color
            key_on="feature.properties.name",  # Property to match in GeoJSON (change to 'name')
            columns=['name', 'value'],  # Specify the columns for mapping in the DataFrame
            fill_color="YlGnBu",  # Color scheme for the choropleth
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Value"
        ).add_to(self.map)

        # Save the map as an HTML file
        map_file_path = "folium_map.html"
        self.map.save(map_file_path)

        # Check if the file exists and print the path
        if not os.path.exists(map_file_path):
            print(f"Error: {map_file_path} does not exist.")
            return  # Exit if the map file is not found

        # Setting up PyQt Web Engine to display Folium map
        self.web_view = QWebEngineView()

        # Use absolute file path to load the map
        url = QUrl.fromLocalFile(os.path.abspath(map_file_path))
        self.web_view.setUrl(url)

        # Set layout to display the map
        layout = QVBoxLayout()
        layout.addWidget(self.web_view)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapWindow()
    sys.exit(app.exec_())

"""
Page for utilities
Çeşitli fonksiyonların tanımlandığı yer
"""

import json
import os
import tempfile
import uuid
import xml.etree.ElementTree as et
import zipfile

import ee
import folium
import geemap.foliumap as geemap
import numpy as np
import pandas as pd
import streamlit as st
from lxml import etree
import plotly.graph_objects as go
from matplotlib import colors

delta_nbr_colors = {
    "Veri Yok": "ffffff",
    "Yüksek yeniden büyüme": "7a8737",
    "Düşük yeniden büyüme": "acbe4d",
    "Yanmamış": "0ae042",
    "Düşük Tahribat": "fff70b",
    "Orta-Düşük tahribat": "ffaf38",
    "Orta-yüksek tahribat": "ff641b",
    "Yüksek tahribat": "a41fd6",
}

ndvi_colors = {
    "min":-1, "max":1,
    "palette": [
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#d9ef8b",
        "#a6d96a",
        "#66bd63",
        "#1a9850",
    ]
}

ndvi_colors2 = {
    "min":-1, "max":1,
    "palette": [
        "#af002d", # -1, -0.9
        "#af002d", # -0.9, -0.8
        "#af002d", # -0.8, -0.7
        "#af002d", # -0.7, -0.6
        "#af002d", # -0.6, -0.5
        "#af002d", # -0.5, -0.4
        "#af002d", # -0.4, -0.3
        "#af002d", # -0.3, -0.2
        "#af002d", # -0.2, -0.1
        "#af002d", # -0.1, 0
        "#af002d", # 0, 0.1
        "#ee4c41", # 0.1, 0.2
        "#fd8958", # 0.2, 0.3
        "#ffbf7b", # 0.3, 0.4
        "#ffedab", # 0.4, 0.5
        "#e6f8a6", # 0.5, 0.6
        "#b9e284", # 0.6, 0.7
        "#66cc66", # 0.7, 0.8
        "#009966", # 0.8, 0.9
        "#006633", # 0.9, 1
    ]
}

ndvi_colors2_legend = {
    "0.9-1.0": "006633",
    "0.8-0.9": "009966",
    "0.7-0.8": "66cc66",
    "0.6-0.7": "b9e284",
    "0.5-0.6": "e6f8a6",
    "0.4-0.5": "ffedab",
    "0.3-0.4": "ffbf7b",
    "0.2-0.3": "fd8958",
    "0.1-0.2": "ee4c41",
    "-1.0-0.1 ": "af002d",
}

ndvi_colors2_hist = {
    "0.90": "#006633",
    "0.80": "#009966",
    "0.70": "#66cc66",
    "0.60": "#b9e284",
    "0.50": "#e6f8a6",
    "0.40": "#ffedab",
    "0.30": "#ffbf7b",
    "0.20": "#fd8958",
    "0.10": "#ee4c41",
    "0.00": "#af002d"
}

ndvi_colors2_imshow = colors.ListedColormap([
    "#af002d",
    "#ee4c41",
    "#fd8958",
    "#ffbf7b",
    "#ffedab",
    "#e6f8a6",
    "#b9e284",
    "#66cc66",
    "#009966",
    "#006633"
    ])

scale = [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
ndvi_colors2_norm=colors.BoundaryNorm(scale, 10)

mask_seq_imshow = colors.ListedColormap([
    "white",
    "darkred",
    "red",
    "orange"
    ])

mask_seq_colors = {
    "min":0, "max": 3,
    "palette": [
        "#ffffff", # 0
        "#8b0000", # 0
        "#ff0000", # 1
        "#ffa500", # 2
    ]
}
mask_scale = [0, 1, 2, 3]
mask_colors_norm=colors.BoundaryNorm(mask_scale, 3)

evi_colors = {
    # "min":0.1, "max":0.8,
    "min":0, "max":1,
    "palette": [
        'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
        '74A901', '66A000', '529400', '3E8601', '207401', '056201',
        '004C00', '023B01', '012E01', '011D01', '011301'
    ]
}

evi_scale = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
evi_colors_norm=colors.BoundaryNorm(evi_scale, 8)

evi_colors_imshow = colors.ListedColormap([
    "#FFFFFF",
    "#CE7E45",
    "#DF923D",
    "#F1B555",
    "#FCD163",
    "#99B718",
    "#74A901",
    "#66A000",
    "#529400",
    "#3E8601",
    "#207401",
    "#056201",
    "#004C00",
    "#023B01",
    "#012E01",
    "#011D01",
    "#011301"
    ])

mbi_colors = {
    "min":-0.3, "max":0.6,
    "palette": [
        'FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
        '74A901', '66A000', '529400', '3E8601', '207401', '056201',
        '004C00', '023B01', '012E01', '011D01', '011301'
    ]
}

mbi_scale = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
mbi_colors_norm=colors.BoundaryNorm(mbi_scale, 9)

mbi_colors_imshow = colors.ListedColormap([
    "#FFFFFF",
    "#CE7E45",
    "#DF923D",
    "#F1B555",
    "#FCD163",
    "#99B718",
    "#74A901",
    "#66A000",
    "#529400",
    "#3E8601",
    "#207401",
    "#056201",
    "#004C00",
    "#023B01",
    "#012E01",
    "#011D01",
    "#011301"
    ])

NAMES = [
    "Veri Yok",
    "Yüksek tahribat",
    "Orta-yüksek tahribat",
    "Orta-Düşük tahribat",
    "Düşük Tahribat",
    "Yanmamış",
    "Düşük yeniden büyüme",
    "Yüksek yeniden büyüme",
]



def calculate_dnbr_dataframe(number_of_pixels):
    """
    The function to calculate the DNBR dataframe.
    """
    names = NAMES
    values = np.array(number_of_pixels)  # pixel numbers
    hectares = values * 900 / 10000  # convert to hectares
    percenteges = hectares / np.sum(hectares) * 100  # calculate percenteges

    dataframe = pd.DataFrame(
        {"dNBR sınıfları": names, "hektar": hectares, "yüzde": percenteges}
    )
    dataframe = dataframe.style.format(precision=2)
    return dataframe.to_html()



def get_plotly_charts(number_of_pixels):
    """
    The function to generate the plotly charts.
    """
    fig = go.Figure(
        data=[
            go.Pie(
                labels=NAMES,
                values=list(number_of_pixels),
                sort=False,
                marker=dict(
                    colors=[
                        "ffffff",
                        "a41fd6",
                        "ff641b",
                        "ffaf38",
                        "fff70b",
                        "0ae042",
                    ]
                ),
            )
        ],
    )

    return fig


def get_pixel_counts(image, geometry):
    """
    The function to get the pixel counts of classes in an dNBR image.
    """
    # pylint: disable=no-member
    thresholds = ee.Image([-1000, -251, -101, 99, 269, 439, 659, 2000])
    classified = image.lt(thresholds).reduce("sum").toInt()
    allpix = classified.updateMask(classified)
    pixstats = allpix.reduceRegion(
        maxPixels= 10000000000,
        reducer=ee.Reducer.count(),  # count pixels in a single class
        geometry=geometry,
        scale=30,
    )

    allpixels = ee.Number(pixstats.get("sum"))  # extract pixel count as a number
    allpixels.getInfo()
    results = []

    results = []
    for i in range(8):
        single_mask = classified.updateMask(classified.eq(i))  # mask a single class
        stats = single_mask.reduceRegion(
            maxPixels= 10000000000,
            reducer=ee.Reducer.count(),  # count pixels in a single class
            geometry=geometry,
            scale=30,
        )
        pix = ee.Number(stats.get("sum"))

        results.append(pix.getInfo())
    return results


def map_search(folium_map: geemap.Map) -> None:  # sourcery skip: use-named-expression
    """
    The function to generate the search box above the map.
    """
    keyword = st.text_input("Bölge arayın:", "")
    if keyword:
        locations = geemap.geocode(keyword)
        if locations is not None and len(locations) > 0:
            str_locations = [str(g)[1:-1] for g in locations]
            location = st.selectbox("Bölge seçin:", str_locations)
            loc_index = str_locations.index(location)
            selected_loc = locations[loc_index]
            lat, lng = selected_loc.lat, selected_loc.lng
            folium.Marker(location=[lat, lng], popup=location).add_to(folium_map)
            folium_map.set_center(lng, lat, 12)
            st.session_state["zoom_level"] = 12


def kml_geometry_export(file_path):
    """
    The function to export the geometry of a KML file.
    """
    # pylint: disable=c-extension-no-member
    root = etree.parse(file_path)

    for i in root.iter():
        path = root.getelementpath(i).split("}")[0] + "}"

    tree = et.parse(file_path)
    root = tree.getroot()

    name = root.find(f".//*{path}coordinates")
    geolist = name.text.strip().split(" ")

    geometry = []

    for i in geolist:
        current = i.split(",")
        # en az 3 point içermesi lazım yoksa EEException error veriyor.
        geometry.append([float(current[0]), float(current[1])])

    return ee.Geometry.Polygon(geometry)



def uploaded_file_to_gdf(data):
    """
    The function to convert uploaded file to geodataframe.
    """
    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    # gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"

    if file_path.lower().endswith(".kml"):

        return kml_geometry_export(file_path)

    if file_path.lower().endswith(".kmz"):
        # unzip it to get kml file
        in_kmz = os.path.abspath(file_path)
        out_dir = os.path.dirname(in_kmz)
        out_kml = os.path.join(out_dir, "doc.kml")
        with zipfile.ZipFile(in_kmz, "r") as zip_ref:
            zip_ref.extractall(out_dir)

        return kml_geometry_export(out_kml)

    if file_path.lower().endswith(".geojson"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.loads(file.read())

        return ee.Geometry.Polygon(data["features"][0]["geometry"]["coordinates"][0])

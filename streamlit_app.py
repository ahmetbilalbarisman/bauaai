"""
Streamlit App
"""
# pylint: disable=wrong-import-order

import streamlit as st
from apps import fire_analysis, home, timelapse, about, change_detection
from streamlit_option_menu import option_menu
from PIL import Image

from typing import Callable


st.set_page_config(page_title="Yangın Analizi", page_icon="🔥", layout="wide")

#Uygulamanın sayfa başlıkları
apps = [
    {"func": home.app, "title": "Ana Sayfa", "icon": "house"},
    {"func": fire_analysis.app, "title": "Yangın Analizi", "icon": "geo-alt"},
    {"func": timelapse.app, "title": "Timelapse", "icon": "hourglass-split"},
    {"func": change_detection.app, "title": "Değişim Analizi", "icon": "geo-alt"},
    {"func": about.app, "title": "Hakkında", "icon": "info"},
]

titles = [app["title"] for app in apps]
icons = [app["icon"] for app in apps]

params = st.experimental_get_query_params()

if "page" in params:
    DEFAULT_INDEX = titles.index(params["page"][0].lower())
else:
    DEFAULT_INDEX = 0

#TEMA logosu
with st.sidebar:
    logo = Image.open("assets/tema-logo.jpg")
    st.image(logo, use_column_width=True)

    selected = option_menu(
        "TEMA",
        options=titles,
        icons=icons,
        menu_icon="list",
        default_index=DEFAULT_INDEX,
    )


for app in apps:
    if app["title"] == selected:
        page_func: Callable = app["func"]
        page_func()
        break

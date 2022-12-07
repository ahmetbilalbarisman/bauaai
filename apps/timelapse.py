"""
The page for create timelapse
"""

import datetime
from datetime import date
import os

import geemap.foliumap as geemap
import streamlit as st

from . import rois, satellite_params, utils

CRS = "epsg:4326"  # Coordinate Reference System
DAY_WINDOW = datetime.timedelta(days=6)
INITIAL_DATE_WINDOW = 6


def app():
    """
    The main app that streamlit will render for create timelapse page.
    """

    col1, col2 = st.columns([2, 1])

    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4

    main_map = geemap.Map(
        basemap="ROADMAP",
        plugin_Draw=True,
        Draw_export=True,
        locate_control=True,
        plugin_LatLngPopup=False,
    )

    st.session_state["vis_params"] = None
    st.session_state["satellite"] = None
    st.session_state["roi"] = None
    timelapse_title = None
    pre_fire = date.today() - 2 * DAY_WINDOW
    post_fire = date.today() - DAY_WINDOW

    with col2:  # right column
        data = st.file_uploader(
            "ROI olarak kullanmak için şekil dosyası ekleyin 😇👇",
            type=["geojson", "kml", "kmz"],
        )

        selected_roi = st.selectbox(
            "Çalışılacak ROI'yi seçin veya şekil dosyası yükleyin.",
            ["Yüklenilen dosyayı seç"] + list(rois.fire_cases.keys()),
            index=0,
        )

        if selected_roi != "Yüklenilen dosyayı seç":  # rois coming from fire_cases
            st.session_state["roi"] = rois.fire_cases[selected_roi]["region"]
            pre_fire = date.fromisoformat(
                rois.fire_cases[selected_roi]["date_range"][0]
            )
            post_fire = date.fromisoformat(
                rois.fire_cases[selected_roi]["date_range"][1]
            )

        elif data:  # if rois coming from users
            st.session_state["roi"] = utils.uploaded_file_to_gdf(data)

        selected_satellite = st.selectbox(
            "Çalışılacak uyduyu seçin", list(satellite_params.satellite.keys())
        )

        if selected_satellite == "sentinel-2":
            sensor_start_year = 2015
            timelapse_title = "Sentinel-2 Timelapse"
            timelapse_speed = 5

        elif selected_satellite == "landsat-8":
            sensor_start_year = 1984
            timelapse_title = "Landsat Timelapse"
            timelapse_speed = 5

        with st.form("submit_landsat_form"):

            roi = None
            if st.session_state["roi"] is not None:
                roi = st.session_state["roi"]
            out_gif = geemap.temp_file_path(".gif")

            title = st.text_input(
                "Başlık ekleyin: ", timelapse_title
            )
            RGB = st.selectbox(
                "RGB bant kombinasyonu ekleyin:",
                [
                    "Red/Green/Blue",
                    "NIR/Red/Green",
                    "SWIR2/SWIR1/NIR",
                    "NIR/SWIR1/Red",
                    "SWIR2/NIR/Red",
                    "SWIR2/SWIR1/Red",
                    "SWIR1/NIR/Blue",
                    "NIR/SWIR1/Blue",
                    "SWIR2/NIR/Green",
                    "SWIR1/NIR/Red",
                    "SWIR2/NIR/SWIR1",
                    "SWIR1/NIR/SWIR2",
                ],
                index=0,
            )

            frequency = st.selectbox(
                "Görüntü sıklığı seçin:",
                ["year", "quarter", "month"],
                index=0,
            )

            with st.expander("Timelapsi özelleştirme"):

                speed = st.slider("Dakika başına kare (FPS):", 1, 30, timelapse_speed)
                dimensions = st.slider(
                    "Maksimum Yükseklik (Genişlik*Yükseklik)", 768, 2000, 768
                )
                progress_bar_color = st.color_picker(
                    "Bar rengi:", "#0000ff"
                )
                years = st.slider(
                    "Başlangıç ve Bitiş yılı:",
                    sensor_start_year,
                    date.today().year,
                    (sensor_start_year, date.today().year),
                )
                months = st.slider("Başlangıç ve bitiş ayı:", 1, 12, (1, 12))
                font_size = st.slider("Font büyüklüğü:", 10, 50, 30)
                font_color = st.color_picker("Font rengi:", "#ffffff")
                apply_fmask = st.checkbox(
                    "fmask uygula (Bulut, kar, gölgeleri kaldırır)", True
                )
                font_type = st.selectbox(
                    "Font tipini seçin:",
                    ["arial.ttf", "alibaba.otf"],
                    index=0,
                )
                fading = st.slider(
                    "Fading duration (seconds) for each frame:", 0.0, 3.0, 0.0
                )
                mp4 = st.checkbox("MP4 olarak kaydedilsin mi?", True)

            empty_text = st.empty()
            empty_image = st.empty()
            empty_fire_image = st.empty()
            empty_video = st.container()
            submitted = st.form_submit_button("Submit")
            if submitted:

                if selected_roi == "Uploaded GeoJSON" and data is None:
                    st.warning("Adımlar: Harita üzerinde poligon çizin ➡ GeoJSON olarak export edin ➡ Uygulamaya upload edin ➡ Tarih aralığı seçin.")
                else:

                    empty_text.text("Hesaplanıyor... Lütfen bekleyin...")

                    start_year = years[0]
                    end_year = years[1]
                    start_date = str(months[0]).zfill(2) + "-01"
                    end_date = str(months[1]).zfill(2) + "-30"
                    bands = RGB.split("/")

                    try:
                        if selected_satellite == "landsat-8":
                            out_gif = geemap.landsat_timelapse(
                                roi=roi,
                                out_gif=out_gif,
                                start_year=start_year,
                                end_year=end_year,
                                start_date=start_date,
                                end_date=end_date,
                                bands=bands,
                                apply_fmask=apply_fmask,
                                frames_per_second=speed,
                                dimensions=dimensions,
                                frequency=frequency,
                                date_format=None,
                                title=title,
                                title_xy=("2%", "90%"),
                                add_text=True,
                                text_xy=("2%", "2%"),
                                text_sequence=None,
                                font_type=font_type,
                                font_size=font_size,
                                font_color=font_color,
                                add_progress_bar=True,
                                progress_bar_color=progress_bar_color,
                                progress_bar_height=5,
                                loop=0,
                                mp4=mp4,
                                fading=fading,
                            )
                        elif selected_satellite == "sentinel-2":
                            out_gif = geemap.sentinel2_timelapse(
                                roi=roi,
                                out_gif=out_gif,
                                start_year=start_year,
                                end_year=end_year,
                                start_date=start_date,
                                end_date=end_date,
                                bands=bands,
                                apply_fmask=apply_fmask,
                                frames_per_second=speed,
                                dimensions=dimensions,
                                frequency=frequency,
                                date_format=None,
                                title=title,
                                title_xy=("2%", "90%"),
                                add_text=True,
                                text_xy=("2%", "2%"),
                                text_sequence=None,
                                font_type=font_type,
                                font_size=font_size,
                                font_color=font_color,
                                add_progress_bar=True,
                                progress_bar_color=progress_bar_color,
                                progress_bar_height=5,
                                loop=0,
                                mp4=mp4,
                                fading=fading,
                            )
                    except:
                        empty_text.error(
                            "Hata meydana geldi. Muhtemelen çok fazla data istediniz. Roi'yi veya süreyi kısaltmayı deneyin."
                        )
                        st.stop()

                    if out_gif is not None and os.path.exists(out_gif):

                        empty_text.text(
                            "GIF'e sağ tıklayarak bilgisayara kaydedin👇"
                        )
                        empty_image.image(out_gif)

                        out_mp4 = out_gif.replace(".gif", ".mp4")
                        if mp4 and os.path.exists(out_mp4):
                            with empty_video:
                                st.text(
                                    "MP4'e sağ tıklayarak bilgisayara kaydedin👇"
                                )
                                st.video(out_gif.replace(".gif", ".mp4"))

                    else:
                        empty_text.error(
                            "Hata meydana geldi. Muhtemelen çok fazla data istediniz. Roi'yi veya süreyi kısaltmayı deneyin."
                        )
    with col1:  # left column
        st.info(
            "Adımlar: Harita üzerinde poligon çizin ➡ GeoJSON olarak export edin"
            " ➡ Uygulamaya upload edin"
            " ➡ Tarih aralığı seçin."
        )

        utils.map_search(main_map)

        if st.session_state.get("roi"):
            main_map.center_object(st.session_state["roi"])

        main_map.to_streamlit(height=600)

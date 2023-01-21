"""
Yangın Analizi Sayfası.
"""
import json
import datetime
from datetime import date
import os
import ee
import folium
import geemap.foliumap as geemap
import streamlit as st
from random import randint
from . import rois, satellite_params, utils

SENTINEL = satellite_params.satellite["sentinel-2"]["name"]
SENTINEL_LAUNCH = satellite_params.satellite["sentinel-2"]["launch"]
MAP_HEIGHT = 600
CRS = "epsg:4326"  # Coordinate Reference System
DAY_WINDOW = datetime.timedelta(days=6)
rgb_vis_params = satellite_params.satellite["sentinel-2"]["rgb_vis_params"]
false_color_vis_params = satellite_params.satellite["sentinel-2"][
    "false_color_vis_params"
]


def app():
    """
    The main app that streamlit will render for fire analysis page.
    """
    # Sayfadaki sütunların oluşturılması.
    col1, col2 = st.columns([2, 1])

    if st.session_state.get("zoom_level") is None:
        st.session_state["zoom_level"] = 4

    # Uygulamada gözüken haritanın özellikleri
    main_map = geemap.Map(
        basemap="ROADMAP",
        plugin_Draw=True,
        Draw_export=True,
        locate_control=True,
        plugin_LatLngPopup=False,
    )

    # pre fire ve post fire tarihlerinin hesabı.
    pre_fire = date.today() - 2 * DAY_WINDOW
    post_fire = date.today() - DAY_WINDOW

    with col2:  # Sağ Sütun
        with st.form("submit_form",clear_on_submit=False):

            if 'key' not in st.session_state:
                st.session_state.key = str(randint(1000, 100000000))

            data = st.file_uploader(
                label="ROI olarak kullanmak için şekil dosyası ekleyin.",
                type=["geojson", "kml", "zip", "kmz"],
            )

            selected_roi = st.selectbox(
                "Çalışılacak ROI'yi seçin veya eklenilmiş dosyayı yükleyin.",
                ["Yüklenilen dosyayı seç"] + list(rois.fire_cases.keys()),
                index=0,
            )

            if selected_roi != "Yüklenilen dosyayı seç":  # rois.py dosyasından gelen roiler
                st.session_state["roi"] = rois.fire_cases[selected_roi]["region"]
                pre_fire = date.fromisoformat(
                    rois.fire_cases[selected_roi]["date_range"][0]
                )
                post_fire = date.fromisoformat(
                    rois.fire_cases[selected_roi]["date_range"][1]
                )

            elif data:  # eğer roi kullanıcadan gelecekse
                st.session_state["roi"] = utils.uploaded_file_to_gdf(data)

            # Yangın öncesi ve sonrası görüntülerin tarihinin belirlenmesi
            pre_fire = st.date_input(  # to update dates according to the user selection
                "Yangın Başlangıç Tarihi",
                pre_fire,
                min_value=SENTINEL_LAUNCH,
                max_value=date.today() - 2 * DAY_WINDOW,
            )

            post_fire = st.date_input(
                "Yangın Bitiş Tarihi",
                post_fire,
                min_value=SENTINEL_LAUNCH,
                max_value=date.today() - DAY_WINDOW,
            )

            option = st.selectbox('Yapılacak İşlemi Seçiniz',('NDVI','NBR','Uydu Görüntüsü','False Color'))
            #sistem mainde hata verdiği için geçici olarak devre dışı bırakıldı.
            indirme = False
            #indirme = st.selectbox('Görüntüleri İndirmek İster Misiniz ?',('Hayır','Evet'))
            dates = {
                "prefire_start": str(pre_fire - DAY_WINDOW),
                "prefire_end": str(pre_fire),
                "postfire_start": str(post_fire),
                "postfire_end": str(post_fire + DAY_WINDOW),
            }

            submitted = st.form_submit_button("Submit")

        with st.expander("Grafikleri Görüntüle"):
            empty_total_hectare = st.empty()

            empty_dataframe = st.empty()
            empty_dataframe.info("Lütfen önce ROI ve tarih seçimini yapınız. ")

            empty_plotly_chart = st.empty()
            empty_plotly_chart.text("Grafikler yükleniyor ...")


    with col1:  # Sol Sütun
        st.info(
            "Adımlar: Harita üzerinde poligon çizin ➡ GeoJSON olarak export edin"
            " ➡ Uygulamaya upload edin"
            " ➡ Tarih aralığı seçin."
        )

        utils.map_search(main_map)

        if submitted and st.session_state.get("roi"):

            main_map.center_object(st.session_state["roi"])

            #S2 verilerine erişim
            imagery = ee.ImageCollection(SENTINEL)

            # Yangın öncesi ve sonrası görüntülere tarih ve roi filtresinin uygulanamsı
            prefire = imagery.filterDate(
                dates["prefire_start"], dates["prefire_end"]
            ).filterBounds(st.session_state["roi"])

            postfire = imagery.filterDate(
                dates["postfire_start"], dates["postfire_end"]
            ).filterBounds(st.session_state["roi"])

            # Belirlenmiş olan tarihteki ve roideki  bölgeye ait görüntülerinden ortadaki olanı seçilmesi
            pre_mos = prefire.median().clip(st.session_state["roi"])
            post_mos = postfire.median().clip(st.session_state["roi"])

            # NBR
            pre_nbr = pre_mos.normalizedDifference(["B8", "B12"])
            post_nbr = post_mos.normalizedDifference(["B8", "B12"])

            # NDVI
            pre_ndvi = pre_mos.normalizedDifference(["B8", "B4"])
            post_ndvi = post_mos.normalizedDifference(["B8", "B4"])

            # NBR hesaplanması
            delta_nbr = pre_nbr.subtract(post_nbr).multiply(
                1000
            )

            with open("assets/sld_intervals.xml", "r", encoding="utf-8") as file:
                sld_intervals = file.read()

            # Yapıalcak işlemi seçer line 94 bakınız.
            out_dir = os.path.join(os.path.expanduser('~'),'downloads')
            if option == 'NDVI':
                left_layer=geemap.ee_tile_layer(pre_ndvi,name='Yangın öncesi ndvi',vis_params=utils.ndvi_colors)
                right_layer=geemap.ee_tile_layer(post_ndvi,name='Yangın Sonrası ndvi',vis_params=utils.ndvi_colors)
                main_map.split_map(left_layer,right_layer)
                st.write('Seçilen İlk Tarih =',pre_fire,' Seçilen İkinci Tarih =',post_fire,' Seçilen Bölge =',selected_roi,' Seçilen İşlem:', option)

                #Eğer indirmek isteniyorsa indirme işlemini gerçekleştirir
                if indirme =='Evet':
                    filename = os.path.join(out_dir,'pre_ndvi.tif')
                    filename2 = os.path.join(out_dir,'post_ndvi.tif')
                    prettyImg=pre_ndvi.visualize(bands=['nd'])
                    prettyImg2=post_ndvi.visualize(bands=['nd'])
                    geemap.ee_export_image(
                        prettyImg,filename=filename,scale=10,region=st.session_state['roi'],file_per_band=True)
                    geemap.ee_export_image(
                        prettyImg2,filename=filename2,scale=10,region=st.session_state['roi'],file_per_band=True)
                    st.write('Görüntüler indirildi.(İndirilenler klasörüne bakınız),(Eğer roi fazla büyükse görüntüler inmez)')

            elif option == 'NBR':
                delta_nbr_sld = delta_nbr.sldStyle(sld_intervals)
                left_layer=geemap.ee_tile_layer(post_mos,name='Yangın Oncesi Goruntu ',vis_params=rgb_vis_params)
                right_layer=geemap.ee_tile_layer(delta_nbr_sld,name="dNBR")
                main_map.split_map(left_layer,right_layer)
                st.write('Seçilen İlk Tarih =',pre_fire,' Seçilen İkinci Tarih =',post_fire,' Seçilen Bölge =',selected_roi,' Seçilen İşlem:', option)

                #Eğer indirmek isteniyorsa indirme işlemini gerçekleştirir
                if indirme == 'Evet':
                    filename = os.path.join(out_dir,'nbr.tif')
                    geemap.ee_export_image(
                        delta_nbr_sld,filename=filename,scale=10,region=st.session_state['roi'],file_per_band=True)
                    st.write('Görüntü indirildi.(İndirilenler klasörüne bakınız),(Eğer roi fazla büyükse görüntü inmez)')

            elif option == 'Uydu Görüntüsü':
                left_layer=geemap.ee_tile_layer(pre_mos,name='Yangın Oncesi Goruntu ',vis_params=rgb_vis_params)
                right_layer=geemap.ee_tile_layer(post_mos,name='Yangın Sonrası Goruntu',vis_params=rgb_vis_params)
                main_map.split_map(left_layer,right_layer)
                st.write('Seçilen İlk Tarih =',pre_fire,' Seçilen İkinci Tarih =',post_fire,' Seçilen Bölge =',selected_roi,' Seçilen İşlem:', option)

                #Eğer indirmek isteniyorsa indirme işlemini gerçekleştirir
                if indirme == 'Evet':
                    filename = os.path.join(out_dir,'pre_true_color.tif')
                    filename2 = os.path.join(out_dir,'post_true_color.tif')
                    prettyImg=pre_mos.visualize(bands=['B4','B3','B2'],min=0,max=2000)
                    prettyImg2=post_mos.visualize(bands=['B4','B3','B2'],min=0,max=2000)
                    geemap.ee_export_image(
                        prettyImg,filename=filename,scale=10,region=st.session_state['roi'],file_per_band=True)
                    geemap.ee_export_image(
                        prettyImg2,filename=filename2,scale=10,region=st.session_state['roi'],file_per_band=True)
                    st.write('Görüntüler indirildi.(İndirilenler klasörüne bakınız),(Eğer roi fazla büyükse görüntüler inmez)')

            elif option =='False Color':
                left_layer=geemap.ee_tile_layer(pre_mos,name='Yangın öncesi false color ',vis_params=false_color_vis_params)
                right_layer=geemap.ee_tile_layer(post_mos,name='Yangın Sonrası False Color',vis_params=false_color_vis_params)
                main_map.split_map(left_layer,right_layer)
                st.write('Seçilen İlk Tarih =',pre_fire,' Seçilen İkinci Tarih =',post_fire,' Seçilen Bölge =',selected_roi,' Seçilen İşlem:', option)

                #Eğer indirmek isteniyorsa indirme işlemini gerçekleştirir
                if indirme == 'Evet':
                    filename = os.path.join(out_dir,'pre_false_color.tif')
                    filename2 = os.path.join(out_dir,'post_false_color.tif')
                    prettyImg=pre_mos.visualize(bands=['B8','B3','B2'],min=128,max=2898)
                    prettyImg2=post_mos.visualize(bands=['B8','B3','B2'],min=128,max=2898)
                    geemap.ee_export_image(
                        prettyImg,filename=filename,scale=10,region=st.session_state['roi'],file_per_band=True)
                    geemap.ee_export_image(
                        prettyImg2,filename=filename2,scale=10,region=st.session_state['roi'],file_per_band=True)
                    st.write('Görüntüler indirildi.(İndirilenler klasörüne bakınız),(Eğer roi fazla büyükse görüntüler inmez)')



            # Hektar hesabının yapıldığı alan
            number_of_pixels = utils.get_pixel_counts(
                delta_nbr, st.session_state["roi"]
            )

            total_hectare = sum(number_of_pixels) * 900 / 10000

            empty_total_hectare.info(
                f"Toplam seçilen alan {round(total_hectare, 2)} hektardır."
            )

            dnbr_dataframe = utils.calculate_dnbr_dataframe(number_of_pixels)
            empty_dataframe.write(dnbr_dataframe, unsafe_allow_html=True)

            plotly_charts = utils.get_plotly_charts(number_of_pixels)
            empty_plotly_chart.plotly_chart(plotly_charts, use_container_width=True)


        main_map.to_streamlit(height=600)
        # $coordinates =
        st.info("Seçilen alanın koordinatları")

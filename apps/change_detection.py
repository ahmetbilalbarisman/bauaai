"""
The page for fire analysis page.
"""

import datetime
from datetime import date
from random import randint

import ee
import folium
import geemap.foliumap as geemap
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
# from st_aggrid import AgGrid, GridOptionsBuilder
# from st_aggrid.grid_options_builder import GridOptionsBuilder
import seaborn as sns
import streamlit as st
from PIL import Image

from . import rois, satellite_params, utils

SENTINEL = satellite_params.satellite["sentinel-2"]["name"]
SENTINEL_LAUNCH = satellite_params.satellite["sentinel-2"]["launch"]
MAP_HEIGHT = 600
CRS = "epsg:4326"  # Coordinate Reference System
DAY_WINDOW = datetime.timedelta(days=6)
rgb_vis_params = satellite_params.satellite["sentinel-2"]["rgb_vis_params"]
false_color_vis_params = satellite_params.satellite["sentinel-2"]["false_color_vis_params"]
change_vis_params = {min: 0, max: 1, 'palette': ['black', 'white']}

@st.cache
def uploaded_file_to_gdf(data):
    import os
    import tempfile
    import uuid

    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    return gdf

# Function to mask clouds from the pixel quality band of Sentinel-2 SR data.
def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask).copyProperties(image, ["system:time_start"])

CLD_PRB_THRESH = 40
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

def add_cloud_bands(img):
    """Define a function to add the s2cloudless probability layer and
    derived cloud mask as bands to an S2 SR image input
    """
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    """fine a function to add dark pixels, cloud projection, and
    identified shadows as bands to an S2 SR image input. Note that
    the image input needs to be the result of the above add_cloud_bands function
    """
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    """Define a function to assemble all of the cloud and
    cloud shadow components and produce the final mask.
    """
    # Add cloud component bands.
    img_cloud = add_cloud_bands(img)

    # Add cloud shadow component bands.
    img_cloud_shadow = add_shadow_bands(img_cloud)

    # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))

    # Add the final cloud-shadow mask to the image.
    # return img_cloud_shadow.addBands(is_cld_shdw)
    # Define a function to assemble all of the cloud and
    # cloud shadow components and produce the final mask.
    return img.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    """Define a function to apply the cloud mask to each image in the collection.
    """
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)

def get_s2_col_default(defined_roi, str_start_date, str_end_date, thr_cloud_probability, selected_method_name):
    """get S2 collection wtih default cloud mask
    """
    # img_collection_name = 'COPERNICUS/S2'
    img_collection_name = 'COPERNICUS/S2_SR_HARMONIZED'
    img_collection = ee.ImageCollection(img_collection_name)

    s2_sr_col = (img_collection
        .filterDate(str_start_date, str_end_date)
        .filterBounds(defined_roi) #intersection
        # .filter(ee.Filter.contains('.geo', defined_roi))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', thr_cloud_probability))
        .sort('date')
        .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYY-MM-dd')))
        .map(maskS2clouds)
        .map(selected_method_name)
        )

    return s2_sr_col

def get_s2_col_cld_prob(defined_roi, str_start_date, str_end_date, thr_cloud_probability, selected_method_name):
    """get S2 collection with cloud probability mask
    """
    # img_collection_name = 'COPERNICUS/S2'
    img_collection_name = 'COPERNICUS/S2_SR_HARMONIZED'
    img_collection = ee.ImageCollection(img_collection_name)

    s2_sr_col = (img_collection
        .filterDate(str_start_date, str_end_date)
        .filterBounds(defined_roi) #intersection
        # .filter(ee.Filter.contains('.geo', defined_roi))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', thr_cloud_probability))
        # .sort('date')
        # .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYY-MM-dd')))
        # # .map(maskS2clouds)
        # .map(selected_method_name)
        )

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(defined_roi)
        .filterDate(str_start_date, str_end_date))

    s2_cloudless_img_col = ee.ImageCollection(
        ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
            }))

    s2_img_col = (s2_cloudless_img_col.map(add_cld_shdw_mask)
                                      .map(apply_cld_shdw_mask)
                                      .sort('date')
                                      .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYY-MM-dd')))
                                      .map(selected_method_name)
                                      )

    return s2_img_col

def coll_mosaic_same_day(s2_img_col):
    """mosaic collectiion"""
    s2_img_col_daily = ee.ImageCollection(
        ee.Join.saveAll('images').apply(**{
        'primary': s2_img_col,
        'secondary': s2_img_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'date',
            'rightField': 'date'
            })
        # 'condition': ee.Filter.And(
        #     ee.Filter.equals({
        #     'leftField': 'date',
        #     'rightField': 'date'
        #     }),
        #     ee.Filter.equals({
        #     'leftField': 'SPACECRAFT_NAME',
        #     'rightField': 'SPACECRAFT_NAME'
        #     }),
        #     ee.Filter.equals({
        #     'leftField': 'SENSING_ORBIT_NUMBER',
        #     'rightField': 'SENSING_ORBIT_NUMBER'
        #     })
        # )
        }))

    s2_img_col_mos = (s2_img_col_daily.map(img_mosaic_same_day)
                                      .map(lambda img: img.set('date', ee.Date(img.date()).format('YYYY-MM-dd')))
                                      )

    return s2_img_col_mos

def img_mosaic_same_day(image):
    """mosaic image"""

    return ee.ImageCollection(ee.List(image.get('images'))) \
      .mosaic() \
      .set('system:time_start', ee.Date(image.get('date')).millis()) \

def get_img_from_col(s2_img_col, date_str, roi=None, band_name=None):
    """Get only one image for given parameters
    """
    date_eedate = ee.Date(date_str)
    s2_img_col_fil = s2_img_col.filterDate(date_eedate, date_eedate.advance(1, 'day'))
    ee_image = ee.Image(s2_img_col_fil.first())

    print("Selected Date:", s2_img_col_fil.aggregate_array('date').getInfo())

    if band_name:
        ee_image = ee_image.select(band_name)
    if roi:
        ee_image = ee_image.clip(roi)

    return ee_image

def get_index_diff_stats(pre_s2_index, post_s2_index, method, defined_roi, thr_index, thr_index_diff):
    """get index difference and related statistics
    """
    d_index = pre_s2_index.subtract(post_s2_index).rename('d_index')

    if method == 'MBI':
        # for indexes that have high values for soils like MBI
        pre_s2_index_mask = pre_s2_index.gt(thr_index)
        classified_d_index = d_index.lt(thr_index_diff).reduce('sum').toInt()
    else:
        # for indexes that have high values for forest like NDVI, EVI
        # return 1 for values greater than threshold
        pre_s2_index_mask = pre_s2_index.gt(thr_index)
        classified_d_index = d_index.gt(thr_index_diff).reduce('sum').toInt()

    single_mask =  classified_d_index.updateMask(classified_d_index.eq(1))  # mask with thr
    single_mask =  single_mask.updateMask(pre_s2_index_mask)  # mask with thr

    stats = single_mask.reduceRegion(
        ee.Reducer.count(), # count pixels in a single class
        defined_roi,
        10 # B4, B8 res is 10m
    )
    pix =  ee.Number(stats.get('sum'))
    pix_num = pix.getInfo()
    # S2 pixel = 10m x 10m --> 100 sqm, pix_num*100/10000
    hect = pix.multiply(100).divide(10000).getInfo()

    # get area percent by class and round to 2 decimals
    # perc = pix.divide(allpixels).multiply(10000).round().divide(100).getInfo()

    return d_index, single_mask, pix_num, hect

def get_fixed_histogram(image, defined_roi, min_hist, max_hist, num_bin, band_name):
    """Get fixed histogram in pandas dataframe
    """
    histogram_dict = image.reduceRegion(
        # ee.Reducer.histogram(10),
        ee.Reducer.fixedHistogram(min_hist, max_hist, num_bin),
        # ee.Reducer.fixedHistogram(min_hist, max_hist, 10).unweighted(),
        defined_roi,
        10,
        )

    histogram = histogram_dict.getInfo()
    # y=histogram[band_name]['histogram']
    hist=histogram[band_name]
    x=[]
    y=[]
    # for i in range(len(y)):
    for i in range(len(hist)):
        # x.append(histogram[band_name]['bucketMin']+i*histogram[band_name]['bucketWidth'])
        x.append("{:.2f}".format(hist[i][0]))
        y.append(hist[i][1])
    # y = np.around(y)
    y = list(map(round, y))
    df_data = pd.DataFrame({'x':x,'y':y})

    return df_data

def get_pixel_count(image, defined_roi, band_name):
    """Get pixel count for given parameters"""
    stat_image = image.select(band_name)
    pixstats = stat_image.reduceRegion(
        ee.Reducer.count(), # count pixels in a single class
        defined_roi,
        10 # B4, B8 res is 10m
    )
    mask_pixels = ee.Number(pixstats.get(band_name))
    total_pix_num = mask_pixels.getInfo()

    return total_pix_num

# Function to calculate and add an NDVI band
def add_ndvi(image):
    """NDVI value range −1 to +1
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('index')
    return image.addBands([ndvi])

def add_evi(image):
    """Enhanced Vegetation Index EVI value range = 0-1
    """
    # Compute the EVI using an expression.
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': image.select('B8').divide(10000),
            'RED': image.select('B4').divide(10000),
            'BLUE': image.select('B2').divide(10000)
        }).rename("index")

    return image.addBands(evi)

def add_mbi(image):
    """Modified Bare Soil Index
    The MBI value ranges from −0.5 to +1.5
    Bare soil is emphasized and received positive values until maxima.
    """
    mbi = image.expression(
    # "swir+red-nir+blue/swir+red-nir+blue",
    "((swir-swir2-nir)/(swir+swir2+nir) + 0.5)",
    {
        'red': image.select('B4'),
        'blue':  image.select('B2'),
        'nir':  image.select('B8'),
        'swir':  image.select('B11'),
        'swir2':  image.select('B12')
    # }).rename("index")
    }).reproject(**{'crs': image.select('B4').projection()}).rename("index")

    return image.addBands(mbi)

def app():
    """
    The main app that streamlit will render for fire analysis page.
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

    start_date = date.today() - 2 * DAY_WINDOW
    end_date = date.today() - DAY_WINDOW

    with col2:  # right column
        with st.form("submit_form", clear_on_submit=False):

            if 'key' not in st.session_state:
                st.session_state.key = str(randint(1000, 100000000))

            df_hist = st.file_uploader(
                label="ROI olarak kullanmak için şekil dosyası ekleyin.",
                type=["geojson", "kml", "zip", "kmz"],
            )

            selected_roi = st.selectbox(
                "Çalışılacak ROI'yi seçin veya eklenilmiş dosyayı yükleyin.",
                ["Yüklenilen dosyayı seç"] + list(rois.gold_mines.keys()),
                index=0,
            )

            if selected_roi != "Yüklenilen dosyayı seç":  # rois coming from fire_cases
                st.session_state["roi"] = rois.gold_mines[selected_roi]["region"]
                start_date = date.fromisoformat(
                    rois.gold_mines[selected_roi]["date_range"][0]
                )
                end_date = date.fromisoformat(
                    rois.gold_mines[selected_roi]["date_range"][1]
                )
                # st.session_state["gdf_roi"] = gpd.GeoDataFrame(index=[0], crs=CRS, geometry=[rois.gold_mines[selected_roi]])

            elif df_hist:  # if rois coming from users
                st.session_state["roi"] = utils.uploaded_file_to_gdf(df_hist)
                # st.session_state["gdf_roi"] = uploaded_file_to_gdf(data)

            start_date = st.date_input(  # to update dates according to the user selection
                "Başlangıç tarihi",
                start_date,
                min_value=SENTINEL_LAUNCH,
                max_value=date.today() - 2 * DAY_WINDOW,
            )

            end_date = st.date_input(
                "Bitiş tarihi",
                end_date,
                min_value=SENTINEL_LAUNCH,
                max_value=date.today() - DAY_WINDOW,
            )
            method = st.selectbox('İndeks Seçiniz', ['NDVI', 'EVI', 'MBI'])
            cloud_method = st.selectbox('Bulut Maskesi Yöntemi', ['S2 Varsayılan', 'S2 Bulut Olasılığı'])
            cloud_probability = st.selectbox('Bulutluluk Oranı Seçiniz',
                ['10', '20', '30', '40', '50'],
                index=1,)

            submitted = st.form_submit_button("Analiz Et")

        with st.form("submit_clear"):
            submit_clear = st.form_submit_button("Temizle")

    with col1:  # left column
        st.info(
            "Adımlar: Harita üzerinde poligon çizin ➡ GeoJSON olarak export edin"
            " ➡ Uygulamaya upload edin"
            " ➡ Tarih aralığı seçin."
        )

        utils.map_search(main_map)

        if submit_clear:
            # st.session_state["state"] = "state_0"
            # user_session_keys = ["state", "s2_img_col", "df_area", "total_pix_num", "s2_len", \
            #                      "s2_clear_time_list", "hist_dict", "img_dict", "max_hist_y_value", \
            #                     "max_diff_y_value", "min_diff_y_value", "date_slider", "date_seq_slider"]
            # for key in user_session_keys:
            for key in st.session_state.keys():
                del st.session_state[key]

        if submitted:
            st.session_state["state"] = "state_1"

        info_text = st.empty()
        info_text_date = st.empty()
        if "state" in st.session_state.keys() and st.session_state["state"] == "state_1" and st.session_state.get("roi"):
            info_text.text("Hesaplanıyor... Lütfen Bekleyiniz...")

            defined_roi = st.session_state.get("roi")
            str_start_date = str(start_date)
            str_end_date = str(end_date)

            thr_cloud_probability = int(cloud_probability)
            # thr_cloudless_pix = 10000
            thr_cloudless_perc = 0.95
            # print(defined_roi.toGeoJSONString())
            # {"type": "Polygon", "coordinates": [
            #         [[26.71264458115938, 40.018525971974405], [26.71264458115938, 39.99577881326442],
            #         [26.73538971360567, 39.99577881326442], [26.73538971360567, 40.018525971974405],
            #         [26.71264458115938, 40.018525971974405]]],
            #         "evenOdd": true}

            if method == 'MBI':
                selected_method_name = add_mbi
                visual_params = utils.mbi_colors
                min_hist = -0.2
                max_hist = 0.5
                default_thr_index = 0.20
                default_thr_index_diff = -0.005
                min_thr_index = 0.1
                max_thr_index = 0.4
                step_thr_index = 0.05
                min_thr_index_diff = -0.024
                max_thr_index_diff = 0.0
                step_thr_index_diff = 0.005
                thr_index_diff_values = ['-0.02', '-0.015', '-0.01', '-0.005', '0']
                index_cmap=utils.mbi_colors_imshow
                index_norm=utils.mbi_colors_norm
            elif method == 'EVI':
                # Enhanced Vegetation Index EVI value range = 0-1
                selected_method_name = add_evi
                # visual_params = utils.evi_colors
                visual_params = utils.ndvi_colors2
                min_hist = 0
                max_hist = 1
                default_thr_index = 0.1
                default_thr_index_diff = 0.05
                min_thr_index = 0.1
                max_thr_index = 0.4
                step_thr_index = 0.05
                min_thr_index_diff = 0.0
                max_thr_index_diff = 0.05
                step_thr_index_diff = 0.2
                thr_index_diff_values = ['0', '0.05', '0.1', '0.15', '0.2']
                # index_cmap=utils.evi_colors_imshow
                # index_norm=utils.evi_colors_norm
                index_cmap=utils.ndvi_colors2_imshow
                index_norm=utils.ndvi_colors2_norm
            else: #ndvi
                selected_method_name = add_ndvi
                visual_params = utils.ndvi_colors2
                min_hist = 0
                max_hist = 1
                default_thr_index = 0.2
                default_thr_index_diff = 0.1
                min_thr_index = 0.1
                max_thr_index = 0.7
                step_thr_index = 0.05
                min_thr_index_diff = 0.0
                max_thr_index_diff = 0.05
                step_thr_index_diff = 0.2
                thr_index_diff_values = ['0', '0.05', '0.1', '0.15', '0.2']
                index_cmap=utils.ndvi_colors2_imshow
                index_norm=utils.ndvi_colors2_norm

            if "df_area" in st.session_state.keys():
                s2_img_col = st.session_state["s2_img_col"]
                df_area = st.session_state["df_area"]
                total_pix_num = st.session_state["total_pix_num"]
                thr_cloudless_pix = int(total_pix_num * thr_cloudless_perc)
                s2_len = st.session_state["s2_len"]
                s2_clear_time_list = st.session_state["s2_clear_time_list"]
                hist_dict = st.session_state["hist_dict"]
                img_dict = st.session_state["img_dict"]
                img_url = st.session_state["img_url"]
                max_hist_y_value = st.session_state["max_hist_y_value"]
                max_diff_y_value = st.session_state["max_diff_y_value"]
                min_diff_y_value = st.session_state["min_diff_y_value"]
            else:
                if cloud_method == 'S2 Varsayılan':
                    s2_img_col = get_s2_col_default(defined_roi, str_start_date, str_end_date,
                                                    thr_cloud_probability, selected_method_name)
                else:
                    # 'S2 Bulut Olasılığı'
                    s2_img_col = get_s2_col_cld_prob(defined_roi, str_start_date, str_end_date,
                                                     thr_cloud_probability, selected_method_name)

                s2_time_list = s2_img_col.aggregate_array('date').getInfo()
                s2_len = len(s2_time_list)
                # print(s2_time_list)

                # s2_img_col = coll_mosaic_same_day(s2_img_col)
                # s2_time_list = s2_img_col.aggregate_array('date').getInfo()
                # print(s2_time_list)

                # This is OK for small collections
                s2_img_col_list = s2_img_col.toList(s2_img_col.size())
                collection_size = s2_img_col_list.size().getInfo()

                s2_clear_time_list = []
                s2_clear_time_list_idx = []
                max_hist_y_value = 0
                max_diff_y_value = 0
                min_diff_y_value = 0
                hist_dict = {}
                img_dict = {}
                img_url = {}
                bin_x_values = []

                # calculate total pixels first
                ee_image = ee.Image(s2_img_col.first())
                ee_mask = ee_image.mask().unmask(0)
                total_pix_num = get_pixel_count(ee_mask, defined_roi, 'index')
                print("Total Pixels: ", total_pix_num)

                thr_cloudless_pix = int(total_pix_num * thr_cloudless_perc)
                print("Pixel threshold: ", thr_cloudless_pix)
                # calculate cloudless pixel get clean ones
                for i in range(0, collection_size):
                    img_date = s2_time_list[i]
                    if img_date in s2_clear_time_list:
                        # if already clear image is added with same date
                        continue
                    # we only had ndvi band
                    s2_index_img = ee.Image(s2_img_col_list.get(i)).select('index')

                    # count number of valid pixels in entire layer
                    # mask  valid pixels
                    mask_pix =  s2_index_img.updateMask(s2_index_img)
                    current_pix_num = get_pixel_count(mask_pix, defined_roi, 'index')
                    print("Date: ", img_date, " Valid Pixels: ", current_pix_num)
                    info_txt = "Tarih: " + img_date + " Geçerli Piksel Sayısı: " + str(current_pix_num)
                    info_text_date.text(info_txt)

                    if current_pix_num > thr_cloudless_pix:
                        # s2_index_img = ee.Image(s2_img_col_list.get(i)).select(['B5', 'B4', 'B3'])
                        # s2_index_img = ee.Image(s2_img_col_list.get(i)).select(['B4', 'B3', 'B2']).clip(defined_roi)

                         # # Old version: geemap.ee_to_numpy does not work for large ROIs
                        # index_np = geemap.ee_to_numpy(s2_index_img, region=defined_roi)
                        image_index_url = s2_index_img.clip(defined_roi).getThumbUrl(visual_params)
                        response = requests.get(image_index_url, stream=True)
                        index_np = Image.open(response.raw)
                        if index_np is not None:
                            img_dict[img_date] = index_np

                            url = s2_index_img.clip(defined_roi).getDownloadUrl({
                                'bands': ['index'],
                                # 'region': defined_roi,
                                'scale': 10,
                                'format': 'GEO_TIFF'
                            })
                            img_url[img_date] = url

                            s2_clear_time_list.append(img_date)
                            s2_clear_time_list_idx.append(i)

                            num_bin = 10
                            df_data = get_fixed_histogram(s2_index_img, defined_roi, min_hist, max_hist, num_bin, "index")

                            hist_dict[img_date] = df_data

                            x = df_data.x.values
                            y = df_data.y.values
                            bin_x_values = x
                            max_y_value = max(y)

                            if max_y_value > max_hist_y_value:
                                max_hist_y_value = max_y_value

                if s2_clear_time_list:
                    # create an empty list to store area values in
                    df_area = pd.DataFrame(s2_clear_time_list[:-1], columns=['pre'])
                    df_area[["post","diff_pixels","hectares"]] = None
                    df_area[x] = None

                    clear_img_len = len(s2_clear_time_list)

                    for i in range(1, clear_img_len):
                        post_i = s2_clear_time_list_idx[i]
                        pre_i = s2_clear_time_list_idx[i-1]
                        post_time = s2_time_list[post_i]
                        pre_time = s2_time_list[pre_i]
                        post_s2_index = ee.Image(s2_img_col_list.get(post_i)).select('index')
                        pre_s2_index = ee.Image(s2_img_col_list.get(pre_i)).select('index')
                        # post_time = s2_time_list[collection_size-1]
                        # pre_time = s2_time_list[0]
                        # post_s2_ndvi = ee.Image(s2_img_col_list.get(collection_size-1)).select('index')
                        # pre_s2_ndvi = ee.Image(s2_img_col_list.get(0)).select('index')

                        d_index, single_mask, pix_num, hect = get_index_diff_stats(
                                                                pre_s2_index, post_s2_index,
                                                                method, defined_roi,
                                                                default_thr_index, default_thr_index_diff)

                        # area_list.append({'pre': pre_time, 'post': post_time, 'diff_pixels': pix_num, 'hectares': hect})
                        print ('Pre:', pre_time, '\tPost: ', post_time, '\tDiff Pixels:', pix_num, '\tHectares:', hect)
                        info_txt = 'Öncesi:' + pre_time + '\tSonrası: ' + post_time + '\tFark Piksel Sayısı: ' + str(pix_num)
                        info_text_date.text(info_txt)
                        df_area.loc[df_area['pre'] == pre_time, ['post','diff_pixels', 'hectares']] = [post_time, pix_num, hect]

                        pre_hist_y_values = hist_dict[pre_time]['y']
                        post_hist_y_values = hist_dict[post_time]['y']
                        diff_hist_y_values = list(post_hist_y_values - pre_hist_y_values)
                        hist_dict[pre_time]['diff'] = diff_hist_y_values

                        df_area.loc[df_area['pre'] == pre_time, bin_x_values] = diff_hist_y_values
                        max_y_value = max(diff_hist_y_values)
                        min_y_value = min(diff_hist_y_values)
                        if max_y_value > max_diff_y_value:
                            max_diff_y_value = max_y_value
                        if min_y_value < min_diff_y_value:
                            min_diff_y_value = min_y_value

                    df_area['hectares'] = pd.to_numeric(df_area['hectares'], errors='ignore')
                    df_area['diff_pixels'] = pd.to_numeric(df_area['diff_pixels'], errors='ignore')
                    df_area.index += 1

                    st.session_state["s2_img_col"] = s2_img_col
                    st.session_state["df_area"] = df_area
                    st.session_state["total_pix_num"] = total_pix_num
                    st.session_state["s2_clear_time_list"] = s2_clear_time_list
                    st.session_state["s2_len"] = s2_len
                    st.session_state["hist_dict"] = hist_dict
                    st.session_state["img_dict"] = img_dict
                    st.session_state["img_url"] = img_url
                    st.session_state["max_hist_y_value"] = max_hist_y_value
                    st.session_state["max_diff_y_value"] = max_diff_y_value
                    st.session_state["min_diff_y_value"] = min_diff_y_value
            # print(df_area)

            if s2_clear_time_list:
                # print("Dates of clear S2 products: ", s2_clear_time_list)
                clear_img_len = len(s2_clear_time_list)

                if "date_slider" in st.session_state.keys():
                    slider_start_date, slider_end_date = st.session_state.date_slider
                    # thr_index = st.session_state.thr_index_slider
                    # thr_index_diff = st.session_state.thr_index_diff_slider
                else:
                    max_hect_idx = df_area['hectares'].idxmax()
                    slider_start_date = df_area.loc[max_hect_idx, 'pre']
                    slider_end_date = df_area.loc[max_hect_idx, 'post']
                    # slider_start_date = s2_clear_time_list[-2]
                    # slider_end_date = s2_clear_time_list[-1]
                if "date_seq_slider" in st.session_state.keys():
                    slider_seq_start_date = st.session_state.date_seq_slider
                else:
                    max_hect_idx = df_area['hectares'].idxmax()
                    slider_seq_start_date = df_area.loc[max_hect_idx, 'pre']

                thr_index = default_thr_index
                thr_index_diff = default_thr_index_diff

                 # # Old version: A shorter version is implemented
                # slider_start_date_obj = datetime.datetime.strptime(slider_start_date, '%Y-%m-%d')
                # slider_start_date_obj += datetime.timedelta(days=1)
                # slider_start_date_plus = slider_start_date_obj.strftime('%Y-%m-%d')
                # slider_end_date_obj = datetime.datetime.strptime(slider_end_date, '%Y-%m-%d')
                # slider_end_date_obj += datetime.timedelta(days=1)
                # slider_end_date_plus = slider_end_date_obj.strftime('%Y-%m-%d')

                # s2_img_col_pre = s2_img_col.filter(ee.Filter.date(slider_start_date, slider_start_date_plus))
                # s2_img_col_post = s2_img_col.filter(ee.Filter.date(slider_end_date, slider_end_date_plus))

                # if cloud_method == 'S2 Varsayılan':
                #     s2_img_col_pre = get_s2_col_default(defined_roi, slider_start_date, slider_start_date_plus,
                #                                     thr_cloud_probability, selected_method_name)
                # else:
                #     # 'S2 Bulut Olasılığı'
                #     s2_img_col_pre = get_s2_col_cld_prob(defined_roi, slider_start_date, slider_start_date_plus,
                #                                     thr_cloud_probability, selected_method_name)


                # if cloud_method == 'S2 Varsayılan':
                #     s2_img_col_post = get_s2_col_default(defined_roi, slider_end_date, slider_end_date_plus,
                #                                     thr_cloud_probability, selected_method_name)
                # else:
                #     # 'S2 Bulut Olasılığı'
                #     s2_img_col_post = get_s2_col_cld_prob(defined_roi, slider_end_date, slider_end_date_plus,
                #                                     thr_cloud_probability, selected_method_name)

                # print("Pre Date:", s2_img_col_pre.aggregate_array('date').getInfo())
                # print("Post Date:", s2_img_col_post.aggregate_array('date').getInfo())

                # s2_img_col_pre_list = s2_img_col_pre.toList(s2_img_col_pre.size())
                # s2_img_col_post_list = s2_img_col_post.toList(s2_img_col_post.size())

                # post_s2 = ee.Image(s2_img_col_post_list.get(0))
                # pre_s2 = ee.Image(s2_img_col_pre_list.get(0))

                # post_s2 = post_s2.clip(defined_roi)
                # pre_s2 = pre_s2.clip(defined_roi)
                # post_s2_index = post_s2.select('index')
                # pre_s2_index = pre_s2.select('index')

                pre_s2 = get_img_from_col(s2_img_col, slider_start_date, defined_roi)
                post_s2 = get_img_from_col(s2_img_col, slider_end_date, defined_roi)
                pre_s2_index = pre_s2.select('index')
                post_s2_index = post_s2.select('index')

                d_index, single_mask, pix_num, hect = get_index_diff_stats(
                                                        pre_s2_index, post_s2_index,
                                                        method, defined_roi,
                                                        thr_index, thr_index_diff)

                # if method == 'NDVI':
                left_layer=geemap.ee_tile_layer(pre_s2_index, name="Pre " + method, vis_params=visual_params)
                right_layer=geemap.ee_tile_layer(post_s2_index, name="Post " + method, vis_params=visual_params)
                main_map.split_map(left_layer, right_layer)
                # main_map.add_layer(pre_s2_index, visual_params, name="Pre " + method)
                # main_map.add_layer(post_s2_index, visual_params, name="Post " + method)
                main_map.add_layer(pre_s2, rgb_vis_params, name="Pre RGB", shown=False)
                main_map.add_layer(post_s2, rgb_vis_params, name="Post RGB", shown=False)
                main_map.add_layer(d_index, visual_params, name="Diff " + method, shown=False)
                main_map.add_layer(single_mask, change_vis_params, name="Change Mask", shown=False)
                main_map.add_layer(defined_roi, {'color': '000000', 'fillColor': '00000099'}, name='ROI', shown=False)

                if method in ('NDVI', 'EVI') :
                    main_map.add_legend(title="Renk Paleti", legend_dict=utils.ndvi_colors2_legend)

                st.write(str_start_date + " ve " + str_end_date + " tarihleri arasında analiz gerçekleştirilmiştir. ")
                st.write("Toplam S2 ürün sayısı: ", s2_len)
                # S2 pixel = 10m x 10m --> 100 sqm, pix_num*100/10000 = /100
                st.write("Toplam Alan: ", "{:.2f}".format(total_pix_num/100),  " ha")
                st.write("{:.2f}".format(thr_cloudless_pix/100), " ha dan büyük (ROI'nin %", "{:.0f}".format(thr_cloudless_perc*100), "'i) bulutsuz alan içeren S2 ürün sayısı: ", clear_img_len)

                st.write("İndeks Görüntüsü ve İlgi Alanı İndeks Histogramları:")
                for pre_time, df_hist in hist_dict.items():
                    # Draw Plot
                    fig = plt.figure(constrained_layout=True, figsize=(30, 10))
                    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig,
                                             width_ratios=[1, 2], height_ratios=[2, 1])
                    ax1 = fig.add_subplot(spec[:, 0])
                    ax2 = fig.add_subplot(spec[0, 1])
                    ax3 = fig.add_subplot(spec[1, 1])
                    # fig, ax2 = plt.subplots(figsize=(30, 7), dpi=80)
                    # fig, ((ax1, ax2), (ax4, ax3)) = plt.subplots(2, 2, figsize=(30, 7), dpi=80,
                    #                              gridspec_kw={'width_ratios': [1, 2], 'height_ratios': [2, 1]})
                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 7), dpi=80,
                    #                              gridspec_kw={'width_ratios': [1, 2]})

                    if method in ('NDVI', 'EVI'):
                        sns.barplot(data=df_hist, x='x', y='y', ax=ax2, palette=utils.ndvi_colors2_hist)
                    else:
                        sns.barplot(data=df_hist, x='x', y='y', ax=ax2)

                    if 'diff' in df_hist:
                        sns.barplot(data=df_hist, x='x', y='diff', ax=ax3, color="red")

                    ax1.xaxis.set_ticks([])
                    ax1.yaxis.set_ticks([])

                    ax2.set_xlabel('İndeks Değeri', size=24)
                    ax2.set_ylabel('Toplam Piksel Sayısı', size=24)
                    ax2.xaxis.set_tick_params(labelsize=20)
                    ax2.yaxis.set_tick_params(labelsize=20)
                    ax2.set_ylim([0, max_hist_y_value])
                    ax2.grid(True, color ="gray")
                    ax2.set_axisbelow(True)

                    ax3.xaxis.set_tick_params(labelsize=20)
                    ax3.yaxis.set_tick_params(labelsize=20)
                    ax3.set_xlabel('')
                    ax3.set_ylabel('Değişen Piksel Sayısı', size=24)
                    ax3.set_ylim([min_diff_y_value, max_diff_y_value])
                    ax3.grid(True, color ="gray")
                    ax3.set_axisbelow(True)

                    # st.write("Tarih:", pre_time)
                    if pre_time in img_dict:
                        index_np = img_dict[pre_time]
                        ax1.imshow(index_np, cmap=index_cmap, norm=index_norm)
                        ax1.set_title(pre_time, size=24)

                    st.pyplot(fig)
                    # fig.tight_layout()
                    if pre_time in img_dict:
                        # st.write("geotiff ([indir](", img_url[pre_time], "))")
                        st.markdown("[:arrow_down_small:](%s)" % img_url[pre_time])

                st.write("Yöntem: ", method)
                st.write("İndeks Eşik Değeri: ", thr_index)
                st.write("Fark Eşik Değeri: ", thr_index_diff)

                st.dataframe(df_area)

                # # selected_indices = st.multiselect('Select rows:', df_area.index)

                # options_builder = GridOptionsBuilder.from_dataframe(df_area)
                # options_builder.configure_selection("single”")
                # grid_options = options_builder.build()
                # grid_return = AgGrid(df_area, grid_options)
                # selected_rows = grid_return["selected_rows"]
                # st.write(selected_rows)

                with st.form(key='seq_form'):
                    selected_seq_start_date = st.select_slider(
                        'İndeks farkını  ardışık (3) olarak hesaplamak istediğiniz tarihi seçiniz',
                        options=s2_clear_time_list,
                        key='date_seq_slider',
                        value=slider_seq_start_date
                    )
                    submit_seq_button = st.form_submit_button(label='Hesapla')

                # Show sequential index difference
                s_idx = s2_clear_time_list.index(slider_seq_start_date)
                e_idx = s_idx + 3
                if e_idx >= clear_img_len:
                    e_idx = clear_img_len-1
                s2_seq_list = s2_clear_time_list[s_idx: e_idx+1]

                pre_date = s2_seq_list.pop(0)
                pre_index_np = img_dict[pre_date]
                pre_s2 = get_img_from_col(s2_img_col, pre_date, defined_roi)
                pre_s2_index = pre_s2.select('index')

                mask = np.zeros(pre_index_np.size, dtype=int)
                # mask = np.zeros(pre_index_np.shape, dtype=int)
                # mask_count = np.zeros(pre_index_np.shape, dtype=int)

                fig_cols = len(s2_seq_list) + 1
                fig2 = plt.figure(constrained_layout=True, figsize=(30, 30))
                spec = gridspec.GridSpec(ncols=fig_cols, nrows=3, figure=fig2)
                fig2.suptitle('Ardışık 3 Görüntünün Analizi', fontsize=28)

                color_patches = []
                color_patches2 = []
                mask_colors = ["darkred", "red", "orange"]
                cred_legend = ["+1", "+2", "+3"]
                for i, img_date in enumerate(s2_seq_list):
                    color_patch = mpatches.Patch(color=mask_colors[i], label=img_date)
                    color_patch2 = mpatches.Patch(color=mask_colors[i], label=cred_legend[i])
                    color_patches.append(color_patch)
                    color_patches2.append(color_patch2)

                ax1 = fig2.add_subplot(spec[0, 0])
                ax1.imshow(pre_index_np, cmap=index_cmap, norm=index_norm)
                ax1.xaxis.set_ticks([])
                ax1.yaxis.set_ticks([])
                ax1.set_title(pre_date, size=24)

                ax2 = fig2.add_subplot(spec[1, 0])
                ax2.imshow(mask,cmap=utils.mask_seq_imshow, norm=utils.mask_colors_norm)
                ax2.legend(title="Değişim Maske Lejantı", handles=color_patches[0:fig_cols],
                            loc='center', fontsize=24, title_fontsize=24)
                ax2.set_axis_off()

                ax3 = fig2.add_subplot(spec[2, 0])
                ax3.imshow(mask,cmap=utils.mask_seq_imshow, norm=utils.mask_colors_norm)
                ax3.legend(title="Güvenirlilik Lejantı", handles=color_patches2[0:fig_cols],
                            loc='center', fontsize=24, title_fontsize=24)
                ax3.set_axis_off()

                mask_count_gee = ee.Image(0).select([0], ["sum"]).int()
                mask_new_gee = ee.Image(0).select([0], ["sum"]).int()
                mask_new_gee = mask_new_gee.reproject(**{'crs': pre_s2_index.projection()}) \
                                           .clip(defined_roi)
                for i, post_date in enumerate(s2_seq_list):
                    post_index_np = img_dict[post_date]

                    # # Old version: get all data to np then process
                    # diff_index_np = pre_index_np - post_index_np
                    # if method == 'MBI':
                    #     # for indexes that have high values for soils like MBI
                    #     pre_mask = pre_index_np > thr_index
                    #     diff_mask = diff_index_np < thr_index_diff
                    # else:
                    #     # for indexes that have high values for forest like NDVI, EVI
                    #     pre_mask = pre_index_np > thr_index
                    #     diff_mask = diff_index_np > thr_index_diff

                    # new_mask = mask == 0
                    # mask_idx = pre_mask & diff_mask # single mask

                    # new_mask_idx = mask_idx & new_mask # new change

                    # mask[new_mask_idx] = i+1
                    # mask_count[mask_idx] += 1

                    # num_change_pixel = np.count_nonzero(mask == i+1)
                    # count_vals, num_cred_pixels = np.unique(mask_count, return_counts=True)

                    post_s2 = get_img_from_col(s2_img_col, post_date, defined_roi)
                    post_s2_index = post_s2.select('index')

                    d_index, single_mask, pix_num, hect = get_index_diff_stats(
                                                        pre_s2_index, post_s2_index,
                                                        method, defined_roi,
                                                        thr_index, thr_index_diff)

                    single_mask = single_mask.unmask(0)
                    mask_count_gee = mask_count_gee.select('sum').add(single_mask.select('sum'))
                    mask_new_gee = mask_new_gee.select('sum').where(
                        single_mask.select('sum').gt(0).And(mask_new_gee.select('sum').eq(0)), i+1)

                    df_data_count = get_fixed_histogram(mask_count_gee, defined_roi, 0, 4, 4, "sum")
                    df_data_new = get_fixed_histogram(mask_new_gee, defined_roi, 0, 4, 4, "sum")

                    count_vals = df_data_count.x.values
                    num_cred_pixels = df_data_count.y.values
                    num_change_pixel = df_data_new.y.values[i+1]

                    image_index_url = mask_count_gee.select('sum').getThumbUrl(utils.mask_seq_colors)
                    response = requests.get(image_index_url, stream=True)
                    mask_count_pil = Image.open(response.raw)

                    image_index_url = mask_new_gee.select('sum').getThumbUrl(utils.mask_seq_colors)
                    response = requests.get(image_index_url, stream=True)
                    mask_pil = Image.open(response.raw)

                    ax1 = fig2.add_subplot(spec[0, i + 1])
                    ax1.imshow(post_index_np, cmap=index_cmap, norm=index_norm)
                    ax1.xaxis.set_ticks([])
                    ax1.yaxis.set_ticks([])
                    ax1.set_title(post_date, size=24)

                    ax2 = fig2.add_subplot(spec[1, i + 1])
                    # ax2.imshow(mask, cmap=utils.mask_seq_imshow, norm=utils.mask_colors_norm)
                    ax2.imshow(mask_pil)
                    ax2.xaxis.set_ticks([])
                    ax2.yaxis.set_ticks([])
                    # ax2.set_xlabel('Değişim (Pix): ' + str(num_change_pixel), size=22)
                    # S2 pixel = 10m x 10m --> 100 sqm, pix_num*100/10000  = /100
                    ax2.set_xlabel('Değişim (Ha): ' + "{:.2f}".format(num_change_pixel/100), size=22)

                    ax3 = fig2.add_subplot(spec[2, i + 1])
                    # ax3.imshow(mask_count, cmap=utils.mask_seq_imshow, norm=utils.mask_colors_norm)
                    ax3.imshow(mask_count_pil)
                    ax3.xaxis.set_ticks([])
                    ax3.yaxis.set_ticks([])
                    x_label_str = ""
                    for count_val, cred_pixel in zip(count_vals[1:], num_cred_pixels[1:]):
                        # x_label_str += "+" + str(count_val) + " (Pix): " + str(cred_pixel) + "\n"
                        # S2 pixel = 10m x 10m --> 100 sqm, pix_num*100/10000  = /100
                        x_label_str += "+" + str(int(float(count_val))) + " (Ha): " + "{:.2f}".format(cred_pixel/100) + "\n"
                    ax3.set_xlabel(x_label_str, size=22)
                    # ax3.set_xlabel(str(count_vals[1:]) + str(num_cred_pixels[1:]), size=22)

                st.pyplot(fig2)

                with st.form(key='diff_form'):
                    # thr_index = st.slider('İndeks Eşik Değeri',
                    #     min_value=min_thr_index,
                    #     max_value=max_thr_index,
                    #     step=step_thr_index,
                    #     value=default_thr_index,
                    #     key='thr_index_slider')
                    # thr_index_diff = st.select_slider('İndeks Fark Eşik Değeri',
                    #     min_value=min_thr_index_diff,
                    #     max_value=max_thr_index_diff,
                    #     step=step_thr_index_diff,
                    #     value=default_thr_index_diff,
                    #     key='thr_index_diff_slider')
                    # thr_index_diff = st.select_slider('İndeks Fark Eşik Değeri',
                    #     options=thr_index_diff_values,
                    #     value=('-0.005'),
                    #     key='thr_index_diff_slider')
                    selected_start_date, selected_end_date = st.select_slider(
                        'İndeks farkını görmek istediğiniz tarihleri seçiniz.',
                        options=s2_clear_time_list,
                        key='date_slider',
                        value=(slider_start_date, slider_end_date)
                    )
                    submit_diff_button = st.form_submit_button(label='Hesapla')

                # st.write('Pre Date: ', selected_start_date, 'End Date: ', selected_end_date)
                st.write('Önceki Tarih: ', selected_start_date, 'Sonraki Tarih: ', selected_end_date, 'Fark (hect): ', hect)
                st.write("Seçilen İndeks Eşik Değeri: ", thr_index)
                st.write("Seçilen İndeks Fark Eşik Değeri: ", thr_index_diff)

                st.write("Görüntüler:")
            else:
                st.write("1 ha dan büyük bulutsuz alan içeren S2 ürün sayısı: 0")
            main_map.center_object(defined_roi)
            folium.map.LayerControl("topright", collapsed=True).add_to(main_map)

        main_map.to_streamlit(height=600)
        info_text.text("")
        info_text_date.text("")

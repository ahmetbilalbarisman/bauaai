"""
About page module
"""

import streamlit as st


def app():
    """
    About page app function
    """
    st.markdown(
        "<h1 style='text-align: center; '>HAKKINDA</h1>", unsafe_allow_html=True
    )

    st.subheader("")

    st.markdown(
        """
        BAU Uygulamalı Yapay Zeka Araştırma Merkezi ve TEMA işbirliği ile geliştirilen bu uygulama, uzaktan algılama verilerine dayanarak
        orman tahribatının tespit ve analiz edilmesi amacıyla geliştirilmiştir. Uygulamada [Google Earth Engine](https://earthengine.google.com) kataloğunda bulunan
        [Sentinel-2](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2) ve [Landsat-8](https://landsat.gsfc.nasa.gov/satellites/landsat-8/)
        uydularından elde edilen veriler kullanılmıştır. Elde edilen veriler [geemap](https://geemap.org) ve [leafmap](https://leafmap.org) gibi açık kaynak
        haritalama kütüphaneleri kullanılarak işlenmiş ve [streamlit](https://streamlit.io) kullanılarak web aplikasyonu haline getirilmiştir.
"""
    )

    st.subheader("")

    st.subheader("Sorumluluk Reddi Beyanı")
    st.markdown(
        """
    Bu çalışma ve içerdiği sorgu / analiz sonuçları tahmini değerler içermektedir. Veriler / sonuçlar, herhangi bir arazi gözlemine
    dayanmadan uydu görüntüleri üzerinden görüntü işleme yazılımlarıyla otomatik – yarı otomatik olarak elde edilmiş olduğundan,
    arazi gözlemleri ile uyumsuzluk gösterebilir.
    """
    )

    st.subheader("")

    st.subheader("Geliştirenler")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write(
            """[Osman Faruk BAYRAM](https://github.com/osbm)  \n[Efe Can KIRBIYIK](https://github.com/EFCK)  \n[Ahmet Bilal BARIŞMAN](https://github.com/ahmetbilalbarisman)  \n Aleyna
        KURT  \n[Beyza BAYRAK](https://github.com/beyzabayrakk)
        """
        )
        st.write(
            """
             Aleyna Benan Aydı
        """
        )
    with col2:
        st.write(
            """
        """
        )

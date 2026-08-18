# =============================================================================
# MACROECONOMIC STRESS TESTING DASHBOARD
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, date
from io import BytesIO
from pathlib import Path

from groq import Groq

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Macro Stress Testing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="locked"
)


# =============================================================================
# PATH CONFIG
# =============================================================================

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "vietbank_logo.png"


# =============================================================================
# GLOBAL CSS - DARK THEME
# =============================================================================

st.markdown(
    """
    <style>

    /* MAIN APP */
    .stApp {
        background-color: #0E0E0E;
        color: #FFFFFF;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1500px;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #2A2A2A;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    /* MAIN TITLE */
    .main-title {
        font-size: 36px;
        font-weight: 750;
        color: #FFFFFF;
        line-height: 1.2;
        margin-top: 5px;
        margin-bottom: 7px;
    }

    /* SUBTITLE */
    .subtitle {
        font-size: 15px;
        color: #B3B3B3;
        margin-bottom: 5px;
    }

    /* SECTION TITLE */
    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    /* GENERAL INFO CARD */
    .info-card {
        background-color: #181818;
        padding: 22px 24px;
        border-radius: 12px;
        border: 1px solid #2A2A2A;
        margin-bottom: 15px;
        color: #E5E5E5;
        font-size: 15px;
        line-height: 1.7;
    }

    /* FEATURE CARD */
    .feature-card {
        background-color: #181818;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #2A2A2A;
        min-height: 190px;
    }

    .feature-title {
        font-size: 17px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 10px;
    }

    .feature-text {
        font-size: 14px;
        color: #BFC3C9;
        line-height: 1.65;
    }

    /* DEVELOPER CARD */
    .developer-card {
        background-color: #181818;
        padding: 22px 25px;
        border-radius: 10px;
        border: 1px solid #2A2A2A;
        margin-bottom: 15px;
    }

    .developer-name {
        font-size: 24px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }

    .developer-position {
        font-size: 13px;
        color: #B3B3B3;
    }

    /* CONTACT */
    .contact-card {
        background-color: #181818;
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid #2A2A2A;
        color: #E5E5E5;
        font-size: 14px;
        line-height: 2;
    }

    .contact-label {
        font-weight: 700;
        color: #FFFFFF;
    }

    /* FOOTER */
    .footer-text {
        text-align: center;
        color: #777777;
        font-size: 12px;
        padding-top: 10px;
    }

    /* STREAMLIT TITLES */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    /* NORMAL STREAMLIT TEXT */
    p, span, label {
        color: #E5E5E5;
    }

    /* RADIO BUTTON TEXT */
    div[data-testid="stRadio"] label {
        color: #FFFFFF !important;
    }

    /* DIVIDER */
    hr {
        border-color: #2A2A2A !important;
    }

    /* ALERT / INFO BOX */
    div[data-testid="stAlert"] {
        background-color: #181818;
        color: #FFFFFF;
        border: 1px solid #2A2A2A;
    }

    /* HIDE STREAMLIT DEFAULT ELEMENTS */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def show_logo(width=None):
    """
    Display Vietbank logo if the logo file exists.
    """

    if LOGO_PATH.exists():

        if width is not None:
            st.image(str(LOGO_PATH), width=width)
        else:
            st.image(str(LOGO_PATH), use_container_width=True)

    else:
        st.warning(
            "Logo not found. Please add "
            "`assets/vietbank_logo.png` to the project folder."
        )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    # Logo
    if LOGO_PATH.exists():
        st.image(
            str(LOGO_PATH),
            use_container_width=True
        )
    else:
        st.markdown("### VIETBANK")

    st.markdown("---")

    # Navigation
    menu = st.radio(
        "Navigation",
        [
            "Overview",
            "Macro Data",
            "Data Download",
            "Macro Analysis",
            "Stress Scenario",
            "Stress Test",
            "AI Analyst"
        ],
        index=0
    )

    st.markdown("---")

    st.caption("Macroeconomic Stress Testing Dashboard")
    st.caption("Credit Risk Management")


# =============================================================================
# TAB 0 - OVERVIEW
# =============================================================================

def tab0():

    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------

    col_logo, col_title = st.columns(
        [1.1, 5],
        vertical_alignment="center"
    )

    with col_logo:

        if LOGO_PATH.exists():
            st.image(
                str(LOGO_PATH),
                use_container_width=True
            )
        else:
            st.markdown(
                """
                ## VIETBANK
                """
            )


    with col_title:

        st.markdown(
            """
            <div class="main-title">
                Dashboard Phân tích Vĩ mô & Kiểm tra Sức chịu đựng
            </div>
    
            <div class="subtitle">
                Giám sát kinh tế vĩ mô • Phân tích kịch bản • Stress Test • Phân tích hỗ trợ bởi AI
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # =========================================================================
    # ABOUT DASHBOARD
    # =========================================================================
    
    st.markdown(
        """
        <div class="section-title">
            Giới thiệu Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="info-card">
    
        <b>Dashboard Phân tích Vĩ mô & Kiểm tra Sức chịu đựng</b>
        được xây dựng nhằm hỗ trợ theo dõi, phân tích và đánh giá
        diễn biến kinh tế vĩ mô có liên quan đến hoạt động quản lý rủi ro tín dụng.
    
        <br><br>
    
        Dashboard tích hợp dữ liệu kinh tế vĩ mô, dữ liệu thị trường tài chính,
        phân tích kịch bản, mô hình Stress Test và công nghệ AI trên một
        nền tảng phân tích tập trung.
    
        <br><br>
    
        Hệ thống hỗ trợ nhận diện các yếu tố rủi ro vĩ mô và đánh giá
        tác động tiềm tàng của các kịch bản bất lợi đến danh mục tín dụng.
    
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # PURPOSE
    # =========================================================================

    
    
    st.markdown(
        '<div class="section-title">Mục đích Dashboard</div>',
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("### 📊 Giám sát kinh tế vĩ mô")
    
            st.write(
                """
                Theo dõi các chỉ tiêu kinh tế vĩ mô trọng yếu như
                tăng trưởng GDP, lạm phát, lãi suất, tỷ giá,
                tăng trưởng tín dụng và các biến động của thị trường tài chính.
                """
            )
    
    with col2:
        with st.container(border=True):
            st.markdown("### ⚠️ Kiểm tra sức chịu đựng")
    
            st.write(
                """
                Xây dựng các kịch bản cơ sở, bất lợi và bất lợi nghiêm trọng
                nhằm đánh giá tác động của biến động kinh tế vĩ mô
                đến rủi ro tín dụng và danh mục tín dụng.
                """
            )
    
    with col3:
        with st.container(border=True):
            st.markdown("### 🤖 Phân tích hỗ trợ bởi AI")
    
            st.write(
                """
                Ứng dụng AI để hỗ trợ nhận diện rủi ro vĩ mô,
                phân tích xu hướng, xây dựng kịch bản Stress Test
                và diễn giải kết quả phân tích.
                """
            )


    # =========================================================================
    # CORE FUNCTIONS
    # =========================================================================

    
    st.markdown(
        '<div class="section-title">Chức năng chính</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="info-card">
    
        <b>01. Dữ liệu vĩ mô</b><br>
        Thu thập, tổng hợp và trực quan hóa các chỉ tiêu kinh tế vĩ mô
        và dữ liệu thị trường tài chính.
    
        <br><br>
    
        <b>02. Tải dữ liệu</b><br>
        Cho phép lựa chọn các chỉ tiêu, khoảng thời gian và tần suất dữ liệu
        để tải xuống dưới định dạng CSV hoặc Excel.
    
        <br><br>
    
        <b>03. Phân tích vĩ mô</b><br>
        Phân tích xu hướng lịch sử, mức độ biến động và mối quan hệ
        giữa các biến kinh tế vĩ mô.
    
        <br><br>
    
        <b>04. Kịch bản Stress Test</b><br>
        Xây dựng kịch bản cơ sở, bất lợi, bất lợi nghiêm trọng
        và các kịch bản tùy chỉnh.
    
        <br><br>
    
        <b>05. Stress Test</b><br>
        Đánh giá tác động của các cú sốc kinh tế vĩ mô
        đến các chỉ tiêu rủi ro tín dụng.
    
        <br><br>
    
        <b>06. Trợ lý AI</b><br>
        Hỗ trợ phân tích diễn biến kinh tế vĩ mô, nhận diện rủi ro,
        xây dựng kịch bản và diễn giải kết quả Stress Test.
    
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # DEVELOPER
    # =========================================================================

    
    st.markdown("## 👤 Người xây dựng")
    
    with st.container(border=True):
    
        col1, col2 = st.columns([1, 4])
    
        with col1:
            st.markdown("**Họ và tên**")
            st.markdown("**Chức danh**")
            st.markdown("**Đơn vị**")
    
        with col2:
            st.write("Nguyễn Thị Chinh")
            st.write("Chuyên viên cao cấp")
            st.write("Phòng Quản lý rủi ro tín dụng")


    # =========================================================================
    # CONTACT
    # =========================================================================
    
    st.markdown("## 📩 Thông tin liên hệ")
    
    with st.container(border=True):
    
        col1, col2 = st.columns([1, 3])
    
        with col1:
            st.markdown("**Đơn vị**")
            st.markdown("**Ngân hàng**")
            st.markdown("**Người xây dựng**")
            st.markdown("**Email**")
    
        with col2:
            st.write("Phòng Quản lý rủi ro tín dụng")
            st.write("Vietbank")
            st.write("Chinh, Nguyen Thi")
            st.write("nguyenthichinh@vietbank.com.vn")





    # =========================================================================
    # DISCLAIMER
    # =========================================================================

    st.markdown("## ⚠️ Lưu ý sử dụng")

    with st.container(border=True):

        st.write(
            """
            Dashboard được xây dựng nhằm hỗ trợ hoạt động phân tích,
            giám sát các yếu tố kinh tế vĩ mô và thực hiện Stress Test
            phục vụ công tác quản lý rủi ro.

            Các kết quả phân tích, mô hình và nội dung do AI hỗ trợ
            chỉ mang tính chất tham khảo và hỗ trợ quá trình đánh giá,
            không thay thế cho các quyết định chuyên môn.
            """
        )


    # =========================================================================
    # FOOTER
    # =========================================================================

    
    st.divider()
    
    current_year = datetime.now().year
    
    st.caption(
        f"© {current_year} Vietbank • Dashboard Giám sát Vĩ mô & Stress Test "
        "• Phục vụ mục đích phân tích và quản lý rủi ro nội bộ."
    )


# =============================================================================
# TAB 1 - MACRO DATA
# =============================================================================

def tab1():

    st.title("Dữ liệu vĩ mô")

    st.caption(
        "Theo dõi và phân tích các chỉ tiêu kinh tế vĩ mô "
        "phục vụ xây dựng kịch bản và Stress Test."
    )

    st.divider()


    # =========================================================================
    # DANH MỤC QUỐC GIA
    # =========================================================================

    countries = {
        "Việt Nam": "VNM",
        "Hoa Kỳ": "USA",
        "Trung Quốc": "CHN",
        "Nhật Bản": "JPN",
        "Hàn Quốc": "KOR",
        "Singapore": "SGP",
        "Thái Lan": "THA",
        "Indonesia": "IDN",
        "Malaysia": "MYS",
        "Philippines": "PHL"
    }


    # =========================================================================
    # DANH MỤC CHỈ TIÊU
    # =========================================================================

    indicators = {

        "Tăng trưởng kinh tế": {

            "Tăng trưởng GDP (%)":
                "NY.GDP.MKTP.KD.ZG",

            "GDP bình quân đầu người (USD)":
                "NY.GDP.PCAP.CD",

            "GDP (USD)":
                "NY.GDP.MKTP.CD",

        },


        "Lạm phát & Giá cả": {

            "Lạm phát CPI (%)":
                "FP.CPI.TOTL.ZG",

            "Chỉ số giá tiêu dùng CPI":
                "FP.CPI.TOTL",

            "GDP Deflator (%)":
                "NY.GDP.DEFL.KD.ZG",

        },


        "Lãi suất & Tiền tệ": {

            "Lãi suất cho vay (%)":
                "FR.INR.LEND",

            "Lãi suất tiền gửi (%)":
                "FR.INR.DPST",

            "Lãi suất thực (%)":
                "FR.INR.RINR",

            "Cung tiền rộng (% GDP)":
                "FM.LBL.BMNY.GD.ZS",

        },


        "Tín dụng": {

            "Tín dụng khu vực tư nhân (% GDP)":
                "FS.AST.PRVT.GD.ZS",

            "Tín dụng nội địa khu vực tư nhân (% GDP)":
                "FS.AST.DOMS.GD.ZS",

        },


        "Lao động": {

            "Tỷ lệ thất nghiệp (%)":
                "SL.UEM.TOTL.ZS",

            "Tỷ lệ tham gia lực lượng lao động (%)":
                "SL.TLF.CACT.ZS",

        },


        "Khu vực đối ngoại": {

            "Tài khoản vãng lai (% GDP)":
                "BN.CAB.XOKA.GD.ZS",

            "FDI ròng (% GDP)":
                "BX.KLT.DINV.WD.GD.ZS",

            "Dự trữ ngoại hối (USD)":
                "FI.RES.TOTL.CD",

            "Nợ nước ngoài (% GNI)":
                "DT.DOD.DECT.GN.ZS",

        }
    }


    # =========================================================================
    # BỘ LỌC
    # =========================================================================

    st.subheader("Bộ lọc dữ liệu")

    col1, col2, col3 = st.columns(3)


    # -------------------------------------------------------------------------
    # COUNTRY
    # -------------------------------------------------------------------------

    with col1:

        selected_countries = st.multiselect(
            "Quốc gia",
            options=list(countries.keys()),
            default=["Việt Nam"],
            placeholder="Chọn quốc gia"
        )


    # -------------------------------------------------------------------------
    # GROUP
    # -------------------------------------------------------------------------

    with col2:

        selected_group = st.selectbox(
            "Nhóm chỉ tiêu",
            list(indicators.keys()),
            index=0
        )


    # -------------------------------------------------------------------------
    # INDICATOR
    # -------------------------------------------------------------------------

    with col3:

        selected_indicator = st.selectbox(
            "Chỉ tiêu",
            list(indicators[selected_group].keys())
        )

        indicator_code = indicators[selected_group][selected_indicator]


    # =========================================================================
    # THỜI GIAN
    # =========================================================================

    current_year = datetime.now().year

    col_start, col_end = st.columns(2)

    with col_start:

        start_year = st.number_input(
            "Từ năm",
            min_value=1960,
            max_value=current_year,
            value=2000,
            step=1
        )

    with col_end:

        end_year = st.number_input(
            "Đến năm",
            min_value=1960,
            max_value=current_year,
            value=current_year,
            step=1
        )


    # =========================================================================
    # VALIDATE
    # =========================================================================

    if start_year > end_year:

        st.error(
            "Năm bắt đầu không được lớn hơn năm kết thúc."
        )

        return


    # =========================================================================
    # HÀM DOWNLOAD WORLD BANK
    # =========================================================================

    @st.cache_data(ttl=3600)
    def get_world_bank_data(
        selected_countries,
        indicator_code,
        start_year,
        end_year
    ):
    
        all_data = []
    
        for country_name in selected_countries:
    
            country_code = countries[country_name]
    
            url = (
                f"https://api.worldbank.org/v2/"
                f"country/{country_code}/"
                f"indicator/{indicator_code}"
            )
    
            params = {
                "format": "json",
                "date": f"{start_year}:{end_year}",
                "per_page": 1000
            }
    
            response = requests.get(
                url,
                params=params,
                timeout=20
            )
    
            response.raise_for_status()
    
            data = response.json()
    
            if (
                not isinstance(data, list)
                or len(data) < 2
                or data[1] is None
            ):
                continue
    
            for item in data[1]:
    
                all_data.append(
                    {
                        "Quốc gia": country_name,
                        "Mã quốc gia": country_code,
                        "Năm": item.get("date"),
                        "Giá trị": item.get("value")
                    }
                )
    
        df = pd.DataFrame(all_data)
    
        if df.empty:
            return df
    
        df["Năm"] = pd.to_numeric(
            df["Năm"],
            errors="coerce"
        )
    
        df["Giá trị"] = pd.to_numeric(
            df["Giá trị"],
            errors="coerce"
        )
    
        df = df.dropna(
            subset=["Năm", "Giá trị"]
        )
    
        df = df.sort_values(
            ["Quốc gia", "Năm"]
        ).reset_index(drop=True)
    
        return df


    # =========================================================================
    # LOAD DATA
    # =========================================================================

    if not selected_countries:

        st.warning(
            "Vui lòng chọn ít nhất một quốc gia."
        )
    
        return
    
    
    try:
    
        with st.spinner(
            "Đang tải dữ liệu từ World Bank..."
        ):
    
            df = get_world_bank_data(
                selected_countries,
                indicator_code,
                int(start_year),
                int(end_year)
            )
    
    except requests.exceptions.RequestException as e:
    
        st.error(
            "Không thể kết nối đến World Bank API."
        )
    
        st.caption(str(e))
    
        return
    
    except Exception as e:
    
        st.error(
            "Có lỗi xảy ra trong quá trình xử lý dữ liệu."
        )
    
        st.caption(str(e))
    
        return


    # =========================================================================
    # KIỂM TRA DATA
    # =========================================================================

    if df.empty:

        st.warning(
            f"Không tìm thấy dữ liệu cho chỉ tiêu "
            f"'{selected_indicator}' của {selected_country} "
            f"trong giai đoạn đã chọn."
        )

        return


    # =========================================================================
    # TÍNH TOÁN
    # =========================================================================

    latest_row = df.iloc[-1]

    latest_year = int(
        latest_row["Năm"]
    )

    latest_value = latest_row["Giá trị"]


    # Previous value
    if len(df) >= 2:

        previous_value = df.iloc[-2]["Giá trị"]
        previous_year = int(df.iloc[-2]["Năm"])

        absolute_change = (
            latest_value - previous_value
        )

        if previous_value != 0:

            percentage_change = (
                absolute_change
                / abs(previous_value)
                * 100
            )

        else:

            percentage_change = np.nan

    else:

        previous_value = np.nan
        previous_year = None
        absolute_change = np.nan
        percentage_change = np.nan


    # Historical statistics
    historical_average = df["Giá trị"].mean()

    historical_min = df["Giá trị"].min()

    historical_max = df["Giá trị"].max()

    volatility = df["Giá trị"].std()


    # =========================================================================
    # FORMAT VALUE
    # =========================================================================

    def format_value(value):

        if pd.isna(value):
            return "N/A"

        if abs(value) >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:,.2f} nghìn tỷ"

        elif abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.2f} tỷ"

        elif abs(value) >= 1_000_000:
            return f"{value / 1_000_000:,.2f} triệu"

        else:
            return f"{value:,.2f}"


    # =========================================================================
    # KPI
    # =========================================================================

    st.subheader("Tổng quan chỉ tiêu")


    kpi1, kpi2, kpi3, kpi4 = st.columns(4)


    with kpi1:

        st.metric(
            label=f"Giá trị mới nhất ({latest_year})",
            value=format_value(latest_value)
        )


    with kpi2:

        if pd.notna(absolute_change):

            st.metric(
                label=f"Thay đổi so với {previous_year}",
                value=format_value(absolute_change),
                delta=(
                    f"{percentage_change:+.2f}%"
                    if pd.notna(percentage_change)
                    else None
                )
            )

        else:

            st.metric(
                label="Thay đổi",
                value="N/A"
            )


    with kpi3:

        st.metric(
            label="Trung bình giai đoạn",
            value=format_value(historical_average)
        )


    with kpi4:

        st.metric(
            label="Độ biến động",
            value=format_value(volatility)
        )


    st.caption(
        f"Dữ liệu mới nhất hiện có của chỉ tiêu: {latest_year}. "
        "Năm dữ liệu mới nhất có thể khác năm hiện tại do độ trễ công bố."
    )


    # =========================================================================
    # CHART
    # =========================================================================

    
    st.subheader(
        f"Xu hướng {selected_indicator}"
    )
    
    fig = go.Figure()
    
    for country in selected_countries:
    
        country_df = df[
            df["Quốc gia"] == country
        ].copy()
    
        if country_df.empty:
            continue
    
        fig.add_trace(
            go.Scatter(
                x=country_df["Năm"],
                y=country_df["Giá trị"],
                mode="lines+markers",
                name=country,
                hovertemplate=(
                    f"<b>{country}</b><br>"
                    "Năm: %{x}<br>"
                    "Giá trị: %{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )
    
    
    fig.update_layout(
    
        template="plotly_dark",
    
        height=520,
    
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),
    
        xaxis_title="Năm",
    
        yaxis_title=selected_indicator,
    
        hovermode="x unified",
    
        paper_bgcolor="#0E0E0E",
    
        plot_bgcolor="#0E0E0E",
    
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    
    fig.update_xaxes(
        showgrid=False
    )
    
    fig.update_yaxes(
        gridcolor="#2A2A2A"
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True
    )




    # =========================================================================
    # THỐNG KÊ LỊCH SỬ
    # =========================================================================

    st.subheader("Thống kê lịch sử")


    stat1, stat2, stat3, stat4 = st.columns(4)


    with stat1:

        st.metric(
            "Thấp nhất",
            format_value(historical_min)
        )


    with stat2:

        st.metric(
            "Cao nhất",
            format_value(historical_max)
        )


    with stat3:

        st.metric(
            "Trung bình",
            format_value(historical_average)
        )


    with stat4:

        st.metric(
            "Độ lệch chuẩn",
            format_value(volatility)
        )


    # =========================================================================
    # DATA TABLE
    # =========================================================================

    st.subheader("Dữ liệu chi tiết")


    display_df = df.copy()


    display_df["Năm"] = (
        display_df["Năm"]
        .astype(int)
    )


    display_df["Giá trị"] = (
        display_df["Giá trị"]
        .round(4)
    )


    # YoY change
    display_df["Thay đổi (%)"] = (
        display_df["Giá trị"]
        .pct_change()
        * 100
    )


    display_df["Thay đổi (%)"] = (
        display_df["Thay đổi (%)"]
        .round(2)
    )


    # Mới nhất lên đầu
    display_df = display_df.sort_values(
        "Năm",
        ascending=False
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Năm": st.column_config.NumberColumn(
                "Năm",
                format="%d"
            ),

            "Giá trị": st.column_config.NumberColumn(
                selected_indicator,
                format="%.4f"
            ),

            "Thay đổi (%)": st.column_config.NumberColumn(
                "Thay đổi so với năm trước (%)",
                format="%.2f%%"
            )
        }
    )


    # =========================================================================
    # DOWNLOAD
    # =========================================================================

    
    st.subheader("Tải dữ liệu")
    
    # Tạo dataframe để xuất
    export_df = df.copy()
    
    # Thêm thông tin chỉ tiêu
    export_df["Nhóm chỉ tiêu"] = selected_group
    export_df["Chỉ tiêu"] = selected_indicator
    export_df["Mã chỉ tiêu"] = indicator_code
    export_df["Nguồn"] = "World Bank"
    
    # Sắp xếp cột
    export_df = export_df[
        [
            "Quốc gia",
            "Mã quốc gia",
            "Năm",
            "Nhóm chỉ tiêu",
            "Chỉ tiêu",
            "Mã chỉ tiêu",
            "Giá trị",
            "Nguồn"
        ]
    ]
    
    # Định dạng năm
    export_df["Năm"] = export_df["Năm"].astype(int)
    
    # Sắp xếp
    export_df = export_df.sort_values(
        ["Quốc gia", "Năm"],
        ascending=[True, False]
    ).reset_index(drop=True)
    
    
    # =============================================================================
    # TÊN FILE
    # =============================================================================
    
    # Nếu chỉ chọn 1 quốc gia
    if len(selected_countries) == 1:
    
        file_country = countries[selected_countries[0]]
    
    else:
    
        file_country = "MULTI_COUNTRY"
    
    
    file_name_base = (
        f"{file_country}_"
        f"{indicator_code}_"
        f"{int(start_year)}_{int(end_year)}"
    )
    
    
    # =============================================================================
    # CSV
    # =============================================================================
    
    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8-sig")
    
    
    # =============================================================================
    # EXCEL
    # =============================================================================
    
    excel_buffer = BytesIO()
    
    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:
    
        export_df.to_excel(
            writer,
            index=False,
            sheet_name="Macro Data"
        )
    
    excel_data = excel_buffer.getvalue()
    
    
    # =============================================================================
    # DOWNLOAD BUTTONS
    # =============================================================================
    
    col_csv, col_excel = st.columns(2)
    
    with col_csv:
    
        st.download_button(
            label="⬇️ Tải dữ liệu CSV",
            data=csv_data,
            file_name=f"{file_name_base}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    
    with col_excel:
    
        st.download_button(
            label="⬇️ Tải dữ liệu Excel",
            data=excel_data,
            file_name=f"{file_name_base}.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )
    
    
    # =============================================================================
    # THÔNG TIN FILE
    # =============================================================================
    
    st.caption(
        f"Dữ liệu gồm {len(export_df):,} quan sát • "
        f"{len(selected_countries)} quốc gia • "
        f"Giai đoạn {int(start_year)}–{int(end_year)} • "
        f"Nguồn: World Bank"
    )


# =============================================================================
# TAB 2 - DATA DOWNLOAD
# =============================================================================


def tab2():

    st.title("Tải dữ liệu")

    st.caption(
        "Lựa chọn nhiều quốc gia và nhiều chỉ tiêu kinh tế vĩ mô "
        "để xây dựng bộ dữ liệu phục vụ phân tích và Stress Test."
    )

    st.divider()


    # =========================================================================
    # DANH MỤC QUỐC GIA
    # =========================================================================

    countries = {
        "Việt Nam": "VNM",
        "Hoa Kỳ": "USA",
        "Trung Quốc": "CHN",
        "Nhật Bản": "JPN",
        "Hàn Quốc": "KOR",
        "Singapore": "SGP",
        "Thái Lan": "THA",
        "Indonesia": "IDN",
        "Malaysia": "MYS",
        "Philippines": "PHL"
    }


    # =========================================================================
    # DANH MỤC CHỈ TIÊU
    # =========================================================================

    indicators = {

        "Tăng trưởng GDP (%)":
            "NY.GDP.MKTP.KD.ZG",

        "GDP (USD)":
            "NY.GDP.MKTP.CD",

        "GDP bình quân đầu người (USD)":
            "NY.GDP.PCAP.CD",

        "Lạm phát CPI (%)":
            "FP.CPI.TOTL.ZG",

        "Chỉ số giá tiêu dùng CPI":
            "FP.CPI.TOTL",

        "GDP Deflator (%)":
            "NY.GDP.DEFL.KD.ZG",

        "Lãi suất cho vay (%)":
            "FR.INR.LEND",

        "Lãi suất tiền gửi (%)":
            "FR.INR.DPST",

        "Lãi suất thực (%)":
            "FR.INR.RINR",

        "Cung tiền rộng (% GDP)":
            "FM.LBL.BMNY.GD.ZS",

        "Tín dụng khu vực tư nhân (% GDP)":
            "FS.AST.PRVT.GD.ZS",

        "Tỷ lệ thất nghiệp (%)":
            "SL.UEM.TOTL.ZS",

        "Tỷ lệ tham gia lực lượng lao động (%)":
            "SL.TLF.CACT.ZS",

        "Tài khoản vãng lai (% GDP)":
            "BN.CAB.XOKA.GD.ZS",

        "FDI ròng (% GDP)":
            "BX.KLT.DINV.WD.GD.ZS",

        "Dự trữ ngoại hối (USD)":
            "FI.RES.TOTL.CD",

        "Nợ nước ngoài (% GNI)":
            "DT.DOD.DECT.GN.ZS"
    }


    # =========================================================================
    # BỘ LỌC
    # =========================================================================

    st.subheader("Lựa chọn dữ liệu")


    col1, col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # QUỐC GIA
    # -------------------------------------------------------------------------

    with col1:

        selected_countries = st.multiselect(
            "Quốc gia",
            options=list(countries.keys()),
            default=["Việt Nam"],
            placeholder="Chọn quốc gia"
        )


    # -------------------------------------------------------------------------
    # CHỈ TIÊU
    # -------------------------------------------------------------------------

    with col2:

        selected_indicators = st.multiselect(
            "Chỉ tiêu kinh tế vĩ mô",
            options=list(indicators.keys()),
            default=[
                "Tăng trưởng GDP (%)",
                "Lạm phát CPI (%)"
            ],
            placeholder="Chọn chỉ tiêu"
        )


    # =========================================================================
    # THỜI GIAN
    # =========================================================================

    current_year = datetime.now().year

    col_start, col_end = st.columns(2)


    with col_start:

        start_year = st.number_input(
            "Từ năm",
            min_value=1960,
            max_value=current_year,
            value=2000,
            step=1,
            key="download_start_year"
        )


    with col_end:

        end_year = st.number_input(
            "Đến năm",
            min_value=1960,
            max_value=current_year,
            value=current_year,
            step=1,
            key="download_end_year"
        )


    # =========================================================================
    # VALIDATION
    # =========================================================================

    if start_year > end_year:

        st.error(
            "Năm bắt đầu không được lớn hơn năm kết thúc."
        )

        return


    if not selected_countries:

        st.warning(
            "Vui lòng chọn ít nhất một quốc gia."
        )

        return


    if not selected_indicators:

        st.warning(
            "Vui lòng chọn ít nhất một chỉ tiêu."
        )

        return


    # =========================================================================
    # THÔNG TIN LỰA CHỌN
    # =========================================================================

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric(
            "Số quốc gia",
            len(selected_countries)
        )

    with info2:
        st.metric(
            "Số chỉ tiêu",
            len(selected_indicators)
        )

    with info3:
        st.metric(
            "Số năm",
            int(end_year - start_year + 1)
        )


    st.divider()


    # =========================================================================
    # HÀM LẤY DATA
    # =========================================================================

    @st.cache_data(ttl=3600)
    def download_world_bank_dataset(
        selected_countries,
        selected_indicators,
        start_year,
        end_year
    ):

        records = []


        # ---------------------------------------------------------------------
        # LOOP COUNTRY
        # ---------------------------------------------------------------------

        for country_name in selected_countries:

            country_code = countries[country_name]


            # -----------------------------------------------------------------
            # LOOP INDICATOR
            # -----------------------------------------------------------------

            for indicator_name in selected_indicators:

                indicator_code = indicators[indicator_name]


                url = (
                    f"https://api.worldbank.org/v2/"
                    f"country/{country_code}/"
                    f"indicator/{indicator_code}"
                )


                params = {
                    "format": "json",
                    "date": f"{start_year}:{end_year}",
                    "per_page": 1000
                }


                try:

                    response = requests.get(
                        url,
                        params=params,
                        timeout=20
                    )


                    response.raise_for_status()

                    data = response.json()


                    if (
                        not isinstance(data, list)
                        or len(data) < 2
                        or data[1] is None
                    ):
                        continue


                    for item in data[1]:

                        records.append(
                            {
                                "Quốc gia": country_name,
                                "Mã quốc gia": country_code,
                                "Năm": item.get("date"),
                                "Chỉ tiêu": indicator_name,
                                "Mã chỉ tiêu": indicator_code,
                                "Giá trị": item.get("value"),
                                "Nguồn": "World Bank"
                            }
                        )


                except Exception:
                    continue


        # ---------------------------------------------------------------------
        # DATAFRAME
        # ---------------------------------------------------------------------

        df = pd.DataFrame(records)


        if df.empty:
            return df


        df["Năm"] = pd.to_numeric(
            df["Năm"],
            errors="coerce"
        )


        df["Giá trị"] = pd.to_numeric(
            df["Giá trị"],
            errors="coerce"
        )


        df = df.dropna(
            subset=["Năm", "Giá trị"]
        )


        df["Năm"] = df["Năm"].astype(int)


        df = df.sort_values(
            [
                "Quốc gia",
                "Chỉ tiêu",
                "Năm"
            ]
        ).reset_index(drop=True)


        return df


    # =========================================================================
    # NÚT TẢI DỮ LIỆU
    # =========================================================================

    if st.button(
        "Tải dữ liệu từ World Bank",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Đang tải và tổng hợp dữ liệu..."
            ):

                download_df = download_world_bank_dataset(
                    selected_countries,
                    selected_indicators,
                    int(start_year),
                    int(end_year)
                )


            # -----------------------------------------------------------------
            # STORE SESSION
            # -----------------------------------------------------------------

            st.session_state["download_macro_df"] = download_df


        except Exception as e:

            st.error(
                "Có lỗi xảy ra trong quá trình tải dữ liệu."
            )

            st.caption(str(e))


    # =========================================================================
    # GET DATA FROM SESSION
    # =========================================================================

    download_df = st.session_state.get(
        "download_macro_df",
        pd.DataFrame()
    )


    # =========================================================================
    # PREVIEW
    # =========================================================================

    if not download_df.empty:

        st.success(
            f"Đã tải thành công {len(download_df):,} quan sát."
        )


        st.subheader("Xem trước dữ liệu")


        st.dataframe(
            download_df,
            use_container_width=True,
            hide_index=True,
            column_config={

                "Năm": st.column_config.NumberColumn(
                    "Năm",
                    format="%d"
                ),

                "Giá trị": st.column_config.NumberColumn(
                    "Giá trị",
                    format="%.4f"
                )
            }
        )


        # =====================================================================
        # DẠNG WIDE DATASET
        # =====================================================================

        st.subheader(
            "Dataset phục vụ phân tích"
        )


        wide_df = download_df.pivot_table(
            index=[
                "Quốc gia",
                "Mã quốc gia",
                "Năm"
            ],
            columns="Chỉ tiêu",
            values="Giá trị",
            aggfunc="first"
        ).reset_index()


        # Xóa tên columns index
        wide_df.columns.name = None


        wide_df = wide_df.sort_values(
            [
                "Quốc gia",
                "Năm"
            ],
            ascending=[
                True,
                False
            ]
        )


        st.dataframe(
            wide_df,
            use_container_width=True,
            hide_index=True
        )


        # =====================================================================
        # QUALITY CHECK
        # =====================================================================

        st.subheader(
            "Kiểm tra chất lượng dữ liệu"
        )


        total_cells = (
            len(wide_df)
            * max(
                len(selected_indicators),
                1
            )
        )


        missing_values = wide_df[
            selected_indicators
        ].isna().sum().sum()


        if total_cells > 0:

            missing_rate = (
                missing_values
                / total_cells
                * 100
            )

        else:

            missing_rate = 0


        quality1, quality2, quality3 = st.columns(3)


        with quality1:

            st.metric(
                "Số quan sát",
                f"{len(wide_df):,}"
            )


        with quality2:

            st.metric(
                "Giá trị thiếu",
                f"{missing_values:,}"
            )


        with quality3:

            st.metric(
                "Tỷ lệ thiếu",
                f"{missing_rate:.2f}%"
            )


        # =====================================================================
        # MISSING DATA TABLE
        # =====================================================================

        missing_df = pd.DataFrame({

            "Chỉ tiêu":
                selected_indicators,

            "Số giá trị thiếu":
                [
                    wide_df[x].isna().sum()
                    if x in wide_df.columns
                    else len(wide_df)
                    for x in selected_indicators
                ]
        })


        missing_df["Tỷ lệ thiếu (%)"] = (

            missing_df["Số giá trị thiếu"]
            / len(wide_df)
            * 100

        ).round(2)


        with st.expander(
            "Chi tiết dữ liệu thiếu"
        ):

            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True
            )


        # =====================================================================
        # DOWNLOAD CSV / EXCEL
        # =====================================================================

        st.subheader(
            "Xuất dữ liệu"
        )


        # ---------------------------------------------------------------------
        # CSV
        # ---------------------------------------------------------------------

        csv_data = wide_df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        )


        # ---------------------------------------------------------------------
        # EXCEL
        # ---------------------------------------------------------------------

        excel_buffer = BytesIO()


        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:


            # Wide format
            wide_df.to_excel(
                writer,
                index=False,
                sheet_name="Macro Dataset"
            )


            # Long format
            download_df.to_excel(
                writer,
                index=False,
                sheet_name="Raw Data"
            )


            # Data quality
            missing_df.to_excel(
                writer,
                index=False,
                sheet_name="Data Quality"
            )


        excel_data = excel_buffer.getvalue()


        # ---------------------------------------------------------------------
        # FILE NAME
        # ---------------------------------------------------------------------

        if len(selected_countries) == 1:

            country_name_file = countries[
                selected_countries[0]
            ]

        else:

            country_name_file = "MULTI_COUNTRY"


        file_base = (

            f"MACRO_DATA_"
            f"{country_name_file}_"
            f"{int(start_year)}_"
            f"{int(end_year)}"

        )


        # ---------------------------------------------------------------------
        # BUTTON
        # ---------------------------------------------------------------------

        col_csv, col_excel = st.columns(2)


        with col_csv:

            st.download_button(
                label="Tải CSV",
                data=csv_data,
                file_name=f"{file_base}.csv",
                mime="text/csv",
                use_container_width=True
            )


        with col_excel:

            st.download_button(
                label="Tải Excel",
                data=excel_data,
                file_name=f"{file_base}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True
            )


        # =====================================================================
        # DATA SOURCE
        # =====================================================================

        st.divider()


        st.caption(
            "Nguồn dữ liệu: World Bank Indicators API. "
            "Dữ liệu có thể có độ trễ công bố khác nhau giữa các chỉ tiêu "
            "và giữa các quốc gia."
        )


    else:

        st.info(
            "Chọn quốc gia và chỉ tiêu, sau đó nhấn "
            "'Tải dữ liệu từ World Bank' để tạo dataset."
        )


# =============================================================================
# TAB 3 - MACRO ANALYSIS
# =============================================================================


def tab3():

    st.title("Phân tích vĩ mô")

    st.caption(
        "Phân tích xu hướng, thống kê mô tả và mối quan hệ giữa "
        "các biến kinh tế vĩ mô phục vụ xây dựng kịch bản Stress Test."
    )

    st.divider()


    # =========================================================================
    # DANH MỤC QUỐC GIA
    # =========================================================================

    countries = {
        "Việt Nam": "VNM",
        "Hoa Kỳ": "USA",
        "Trung Quốc": "CHN",
        "Nhật Bản": "JPN",
        "Hàn Quốc": "KOR",
        "Singapore": "SGP",
        "Thái Lan": "THA",
        "Indonesia": "IDN",
        "Malaysia": "MYS",
        "Philippines": "PHL"
    }


    # =========================================================================
    # DANH MỤC CHỈ TIÊU
    # =========================================================================

    indicators = {

        "Tăng trưởng GDP (%)":
            "NY.GDP.MKTP.KD.ZG",

        "Lạm phát CPI (%)":
            "FP.CPI.TOTL.ZG",

        "Lãi suất cho vay (%)":
            "FR.INR.LEND",

        "Lãi suất tiền gửi (%)":
            "FR.INR.DPST",

        "Lãi suất thực (%)":
            "FR.INR.RINR",

        "Cung tiền rộng (% GDP)":
            "FM.LBL.BMNY.GD.ZS",

        "Tín dụng khu vực tư nhân (% GDP)":
            "FS.AST.PRVT.GD.ZS",

        "Tỷ lệ thất nghiệp (%)":
            "SL.UEM.TOTL.ZS",

        "Tài khoản vãng lai (% GDP)":
            "BN.CAB.XOKA.GD.ZS",

        "FDI ròng (% GDP)":
            "BX.KLT.DINV.WD.GD.ZS"
    }


    # =========================================================================
    # BỘ LỌC
    # =========================================================================

    st.subheader("Bộ lọc phân tích")

    col1, col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # QUỐC GIA
    # -------------------------------------------------------------------------

    with col1:

        selected_country = st.selectbox(
            "Quốc gia",
            options=list(countries.keys()),
            index=0,
            key="analysis_country"
        )

        country_code = countries[selected_country]


    # -------------------------------------------------------------------------
    # CHỈ TIÊU
    # -------------------------------------------------------------------------

    with col2:

        selected_indicators = st.multiselect(
            "Chỉ tiêu phân tích",
            options=list(indicators.keys()),
            default=[
                "Tăng trưởng GDP (%)",
                "Lạm phát CPI (%)",
                "Lãi suất thực (%)",
                "Tín dụng khu vực tư nhân (% GDP)"
            ],
            placeholder="Chọn các chỉ tiêu",
            key="analysis_indicators"
        )


    # =========================================================================
    # THỜI GIAN
    # =========================================================================

    current_year = datetime.now().year

    col_start, col_end = st.columns(2)


    with col_start:

        start_year = st.number_input(
            "Từ năm",
            min_value=1960,
            max_value=current_year,
            value=2000,
            step=1,
            key="analysis_start_year"
        )


    with col_end:

        end_year = st.number_input(
            "Đến năm",
            min_value=1960,
            max_value=current_year,
            value=current_year,
            step=1,
            key="analysis_end_year"
        )


    # =========================================================================
    # VALIDATION
    # =========================================================================

    if start_year > end_year:

        st.error(
            "Năm bắt đầu không được lớn hơn năm kết thúc."
        )

        return


    if len(selected_indicators) < 1:

        st.warning(
            "Vui lòng chọn ít nhất một chỉ tiêu để phân tích."
        )

        return


    # =========================================================================
    # HÀM DOWNLOAD DATA
    # =========================================================================

    @st.cache_data(ttl=3600)
    def get_analysis_data(
        country_code,
        selected_indicators,
        start_year,
        end_year
    ):

        all_records = []


        for indicator_name in selected_indicators:

            indicator_code = indicators[indicator_name]


            url = (
                f"https://api.worldbank.org/v2/"
                f"country/{country_code}/"
                f"indicator/{indicator_code}"
            )


            params = {
                "format": "json",
                "date": f"{start_year}:{end_year}",
                "per_page": 1000
            }


            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=20
                )

                response.raise_for_status()

                data = response.json()


                if (
                    not isinstance(data, list)
                    or len(data) < 2
                    or data[1] is None
                ):
                    continue


                for item in data[1]:

                    all_records.append(
                        {
                            "Năm": item.get("date"),
                            "Chỉ tiêu": indicator_name,
                            "Giá trị": item.get("value")
                        }
                    )


            except Exception:
                continue


        df = pd.DataFrame(all_records)


        if df.empty:
            return df


        df["Năm"] = pd.to_numeric(
            df["Năm"],
            errors="coerce"
        )


        df["Giá trị"] = pd.to_numeric(
            df["Giá trị"],
            errors="coerce"
        )


        df = df.dropna(
            subset=[
                "Năm",
                "Giá trị"
            ]
        )


        df["Năm"] = df["Năm"].astype(int)


        return df


    # =========================================================================
    # LOAD DATA
    # =========================================================================

    try:

        with st.spinner(
            "Đang tải dữ liệu phân tích..."
        ):

            analysis_raw = get_analysis_data(
                country_code,
                selected_indicators,
                int(start_year),
                int(end_year)
            )


    except Exception as e:

        st.error(
            "Không thể tải dữ liệu phân tích."
        )

        st.caption(str(e))

        return


    if analysis_raw.empty:

        st.warning(
            "Không tìm thấy dữ liệu phù hợp."
        )

        return


    # =========================================================================
    # PIVOT DATA
    # =========================================================================

    analysis_df = analysis_raw.pivot_table(
        index="Năm",
        columns="Chỉ tiêu",
        values="Giá trị",
        aggfunc="first"
    ).reset_index()


    analysis_df.columns.name = None


    analysis_df = analysis_df.sort_values(
        "Năm"
    ).reset_index(drop=True)


    # Chỉ giữ các indicator thực sự có dữ liệu
    available_indicators = [
        x for x in selected_indicators
        if x in analysis_df.columns
    ]


    if not available_indicators:

        st.warning(
            "Các chỉ tiêu được chọn không có dữ liệu."
        )

        return


    # =========================================================================
    # TỔNG QUAN
    # =========================================================================

    st.subheader(
        f"Tổng quan kinh tế vĩ mô – {selected_country}"
    )


    metric_columns = st.columns(
        min(
            len(available_indicators),
            4
        )
    )


    for i, indicator in enumerate(
        available_indicators[:4]
    ):

        series = analysis_df[
            ["Năm", indicator]
        ].dropna()


        if series.empty:
            continue


        latest = series.iloc[-1]

        latest_year = int(
            latest["Năm"]
        )

        latest_value = latest[indicator]


        # Previous value
        if len(series) >= 2:

            previous_value = (
                series.iloc[-2][indicator]
            )

            delta = (
                latest_value
                - previous_value
            )

        else:

            delta = None


        with metric_columns[i]:

            st.metric(
                label=(
                    f"{indicator} "
                    f"({latest_year})"
                ),

                value=(
                    f"{latest_value:,.2f}"
                ),

                delta=(
                    f"{delta:+.2f}"
                    if delta is not None
                    else None
                )
            )


    # =========================================================================
    # XU HƯỚNG VĨ MÔ
    # =========================================================================

    st.subheader(
        "Xu hướng các chỉ tiêu vĩ mô"
    )


    st.caption(
        "Các chuỗi có đơn vị và thang đo khác nhau. "
        "Có thể sử dụng chế độ chuẩn hóa để so sánh xu hướng."
    )


    normalize = st.toggle(
        "Chuẩn hóa dữ liệu để so sánh xu hướng",
        value=True,
        key="normalize_macro"
    )


    chart_df = analysis_df.copy()


    # -------------------------------------------------------------------------
    # STANDARDIZATION
    # -------------------------------------------------------------------------

    if normalize:

        for indicator in available_indicators:

            mean_value = (
                chart_df[indicator].mean()
            )

            std_value = (
                chart_df[indicator].std()
            )


            if (
                pd.notna(std_value)
                and std_value != 0
            ):

                chart_df[indicator] = (
                    chart_df[indicator]
                    - mean_value
                ) / std_value


    # -------------------------------------------------------------------------
    # CHART
    # -------------------------------------------------------------------------

    fig = go.Figure()


    for indicator in available_indicators:

        indicator_df = chart_df[
            ["Năm", indicator]
        ].dropna()


        fig.add_trace(
            go.Scatter(
                x=indicator_df["Năm"],
                y=indicator_df[indicator],
                mode="lines+markers",
                name=indicator,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Năm: %{x}<br>"
                    "Giá trị: %{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )


    fig.update_layout(

        template="plotly_dark",

        height=550,

        paper_bgcolor="#0E0E0E",

        plot_bgcolor="#0E0E0E",

        hovermode="x unified",

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),

        xaxis_title="Năm",

        yaxis_title=(
            "Z-score"
            if normalize
            else "Giá trị"
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )


    fig.update_xaxes(
        showgrid=False
    )


    fig.update_yaxes(
        gridcolor="#2A2A2A"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =========================================================================
    # THỐNG KÊ MÔ TẢ
    # =========================================================================

    st.subheader(
        "Thống kê mô tả"
    )


    descriptive_data = []


    for indicator in available_indicators:

        series = analysis_df[
            indicator
        ].dropna()


        if series.empty:
            continue


        descriptive_data.append(
            {
                "Chỉ tiêu": indicator,
                "Số quan sát": len(series),
                "Trung bình": series.mean(),
                "Trung vị": series.median(),
                "Thấp nhất": series.min(),
                "Cao nhất": series.max(),
                "Độ lệch chuẩn": series.std()
            }
        )


    descriptive_df = pd.DataFrame(
        descriptive_data
    )


    numeric_columns = [
        "Trung bình",
        "Trung vị",
        "Thấp nhất",
        "Cao nhất",
        "Độ lệch chuẩn"
    ]


    descriptive_df[
        numeric_columns
    ] = descriptive_df[
        numeric_columns
    ].round(2)


    st.dataframe(
        descriptive_df,
        use_container_width=True,
        hide_index=True
    )


    # =========================================================================
    # CORRELATION
    # =========================================================================

    st.subheader(
        "Ma trận tương quan"
    )


    st.caption(
        "Hệ số tương quan giúp đánh giá mức độ đồng biến hoặc nghịch biến "
        "giữa các biến vĩ mô. Tương quan không đồng nghĩa với quan hệ nhân quả."
    )


    if len(available_indicators) >= 2:

        corr_df = analysis_df[
            available_indicators
        ].corr(
            min_periods=3
        )


        # ---------------------------------------------------------------------
        # HEATMAP
        # ---------------------------------------------------------------------

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                zmin=-1,
                zmax=1,
                text=np.round(
                    corr_df.values,
                    2
                ),
                texttemplate="%{text}",
                hovertemplate=(
                    "%{y}<br>"
                    "%{x}<br>"
                    "Tương quan: %{z:.2f}"
                    "<extra></extra>"
                )
            )
        )


        heatmap.update_layout(

            template="plotly_dark",

            height=520,

            paper_bgcolor="#0E0E0E",

            plot_bgcolor="#0E0E0E",

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )


        st.plotly_chart(
            heatmap,
            use_container_width=True
        )


        # =====================================================================
        # CẶP BIẾN
        # =====================================================================

        st.subheader(
            "Phân tích quan hệ giữa hai biến"
        )


        col_x, col_y = st.columns(2)


        with col_x:

            variable_x = st.selectbox(
                "Biến X",
                available_indicators,
                index=0,
                key="macro_x"
            )


        with col_y:

            default_y = (
                1
                if len(available_indicators) > 1
                else 0
            )

            variable_y = st.selectbox(
                "Biến Y",
                available_indicators,
                index=default_y,
                key="macro_y"
            )


        # ---------------------------------------------------------------------
        # CHECK
        # ---------------------------------------------------------------------

        if variable_x == variable_y:

            st.info(
                "Vui lòng chọn hai biến khác nhau để phân tích."
            )

        else:

            pair_df = analysis_df[
                [
                    "Năm",
                    variable_x,
                    variable_y
                ]
            ].dropna()


            if len(pair_df) >= 3:

                correlation = pair_df[
                    variable_x
                ].corr(
                    pair_df[variable_y]
                )


                # =============================================================
                # SIMPLE REGRESSION
                # =============================================================

                x = pair_df[
                    variable_x
                ].values


                y = pair_df[
                    variable_y
                ].values


                slope, intercept = np.polyfit(
                    x,
                    y,
                    1
                )


                predicted_y = (
                    slope * x
                    + intercept
                )


                ss_res = np.sum(
                    (
                        y
                        - predicted_y
                    ) ** 2
                )


                ss_tot = np.sum(
                    (
                        y
                        - np.mean(y)
                    ) ** 2
                )


                if ss_tot != 0:

                    r_squared = (
                        1
                        - ss_res / ss_tot
                    )

                else:

                    r_squared = np.nan


                # =============================================================
                # METRICS
                # =============================================================

                c1, c2, c3 = st.columns(3)


                with c1:

                    st.metric(
                        "Hệ số tương quan",
                        f"{correlation:.3f}"
                    )


                with c2:

                    st.metric(
                        "Hệ số hồi quy",
                        f"{slope:.3f}"
                    )


                with c3:

                    st.metric(
                        "R²",
                        (
                            f"{r_squared:.3f}"
                            if pd.notna(r_squared)
                            else "N/A"
                        )
                    )


                # =============================================================
                # SCATTER
                # =============================================================

                scatter = go.Figure()


                scatter.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name="Quan sát",
                        text=pair_df[
                            "Năm"
                        ],
                        hovertemplate=(
                            "Năm: %{text}<br>"
                            f"{variable_x}: "
                            "%{x:.2f}<br>"
                            f"{variable_y}: "
                            "%{y:.2f}"
                            "<extra></extra>"
                        )
                    )
                )


                # Sort line
                order = np.argsort(x)


                scatter.add_trace(
                    go.Scatter(
                        x=x[order],
                        y=predicted_y[order],
                        mode="lines",
                        name="Đường hồi quy"
                    )
                )


                scatter.update_layout(

                    template="plotly_dark",

                    height=500,

                    paper_bgcolor="#0E0E0E",

                    plot_bgcolor="#0E0E0E",

                    xaxis_title=variable_x,

                    yaxis_title=variable_y,

                    margin=dict(
                        l=20,
                        r=20,
                        t=30,
                        b=20
                    )
                )


                scatter.update_xaxes(
                    gridcolor="#2A2A2A"
                )


                scatter.update_yaxes(
                    gridcolor="#2A2A2A"
                )


                st.plotly_chart(
                    scatter,
                    use_container_width=True
                )


                # =============================================================
                # INTERPRETATION
                # =============================================================

                st.markdown(
                    "#### Diễn giải"
                )


                abs_corr = abs(
                    correlation
                )


                if abs_corr >= 0.7:

                    strength = "mạnh"

                elif abs_corr >= 0.4:

                    strength = "trung bình"

                elif abs_corr >= 0.2:

                    strength = "yếu"

                else:

                    strength = "rất yếu"


                if correlation > 0:

                    direction = "cùng chiều"

                elif correlation < 0:

                    direction = "ngược chiều"

                else:

                    direction = "không có tương quan tuyến tính"


                st.info(
                    f"Trong giai đoạn {int(start_year)}–{int(end_year)}, "
                    f"{variable_x} và {variable_y} có mối tương quan "
                    f"{direction} ở mức {strength} "
                    f"(r = {correlation:.3f}). "
                    "Kết quả này chỉ phản ánh mối liên hệ thống kê và "
                    "không khẳng định quan hệ nhân quả."
                )


            else:

                st.warning(
                    "Không đủ số quan sát chung giữa hai biến để "
                    "thực hiện phân tích."
                )


    else:

        st.info(
            "Chọn ít nhất hai chỉ tiêu để hiển thị "
            "ma trận tương quan và phân tích quan hệ giữa các biến."
        )


    # =========================================================================
    # DỮ LIỆU PHÂN TÍCH
    # =========================================================================

    with st.expander(
        "Xem dữ liệu sử dụng trong phân tích"
    ):

        display_analysis_df = (
            analysis_df
            .sort_values(
                "Năm",
                ascending=False
            )
            .copy()
        )


        st.dataframe(
            display_analysis_df,
            use_container_width=True,
            hide_index=True
        )


    # =========================================================================
    # SOURCE
    # =========================================================================

    st.divider()

    st.caption(
        f"Nguồn dữ liệu: World Bank Indicators API • "
        f"Quốc gia: {selected_country} • "
        f"Giai đoạn: {int(start_year)}–{int(end_year)}"
    )


# =============================================================================
# TAB 4 - STRESS SCENARIO
# =============================================================================


def tab4():

    st.title("Kịch bản Stress Test")

    st.caption(
        "Xây dựng và điều chỉnh các kịch bản kinh tế vĩ mô "
        "phục vụ đánh giá mức độ chịu đựng của danh mục tín dụng."
    )

    st.divider()


    # =========================================================================
    # KỊCH BẢN MẪU
    # =========================================================================

    scenario_templates = {

        "Cơ sở": {
            "GDP Growth (%)": 6.5,
            "Inflation (%)": 3.5,
            "Interest Rate (%)": 5.0,
            "FX Shock (%)": 2.0,
            "Credit Growth (%)": 12.0,
            "Unemployment (%)": 2.3,
            "VN-Index Shock (%)": 0.0
        },

        "Bất lợi": {
            "GDP Growth (%)": 3.0,
            "Inflation (%)": 5.0,
            "Interest Rate (%)": 7.0,
            "FX Shock (%)": 8.0,
            "Credit Growth (%)": 7.0,
            "Unemployment (%)": 3.5,
            "VN-Index Shock (%)": -20.0
        },

        "Bất lợi nghiêm trọng": {
            "GDP Growth (%)": -1.0,
            "Inflation (%)": 7.5,
            "Interest Rate (%)": 9.0,
            "FX Shock (%)": 15.0,
            "Credit Growth (%)": 2.0,
            "Unemployment (%)": 5.0,
            "VN-Index Shock (%)": -35.0
        }
    }


    # =========================================================================
    # CHỌN KỊCH BẢN
    # =========================================================================

    st.subheader("Lựa chọn kịch bản")

    scenario_type = st.selectbox(
        "Loại kịch bản",
        [
            "Cơ sở",
            "Bất lợi",
            "Bất lợi nghiêm trọng",
            "Tùy chỉnh"
        ],
        index=0
    )


    # =========================================================================
    # LOAD GIÁ TRỊ MẶC ĐỊNH
    # =========================================================================

    if scenario_type != "Tùy chỉnh":

        default_values = scenario_templates[
            scenario_type
        ]

    else:

        default_values = {
            "GDP Growth (%)": 5.0,
            "Inflation (%)": 4.0,
            "Interest Rate (%)": 6.0,
            "FX Shock (%)": 5.0,
            "Credit Growth (%)": 10.0,
            "Unemployment (%)": 3.0,
            "VN-Index Shock (%)": -10.0
        }


    # =========================================================================
    # THÔNG TIN KỊCH BẢN
    # =========================================================================

    if scenario_type == "Cơ sở":

        st.info(
            "Kịch bản cơ sở phản ánh điều kiện kinh tế kỳ vọng "
            "trong trường hợp không xảy ra cú sốc lớn."
        )

    elif scenario_type == "Bất lợi":

        st.warning(
            "Kịch bản bất lợi phản ánh môi trường kinh tế suy yếu, "
            "lạm phát và lãi suất tăng, đồng thời thị trường tài chính giảm."
        )

    elif scenario_type == "Bất lợi nghiêm trọng":

        st.error(
            "Kịch bản bất lợi nghiêm trọng mô phỏng suy thoái mạnh, "
            "biến động tỷ giá lớn và suy giảm mạnh trên thị trường tài chính."
        )

    else:

        st.info(
            "Kịch bản tùy chỉnh cho phép người dùng tự thiết lập "
            "toàn bộ các giả định Stress Test."
        )


    # =========================================================================
    # THIẾT LẬP BIẾN VĨ MÔ
    # =========================================================================

    st.subheader("Thiết lập giả định vĩ mô")

    col1, col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # GDP
    # -------------------------------------------------------------------------

    with col1:

        gdp_growth = st.slider(
            "Tăng trưởng GDP (%)",
            min_value=-10.0,
            max_value=15.0,
            value=float(
                default_values["GDP Growth (%)"]
            ),
            step=0.1
        )


    # -------------------------------------------------------------------------
    # INFLATION
    # -------------------------------------------------------------------------

    with col2:

        inflation = st.slider(
            "Lạm phát CPI (%)",
            min_value=-2.0,
            max_value=20.0,
            value=float(
                default_values["Inflation (%)"]
            ),
            step=0.1
        )


    # -------------------------------------------------------------------------
    # INTEREST RATE
    # -------------------------------------------------------------------------

    with col1:

        interest_rate = st.slider(
            "Lãi suất (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(
                default_values["Interest Rate (%)"]
            ),
            step=0.1
        )


    # -------------------------------------------------------------------------
    # FX
    # -------------------------------------------------------------------------

    with col2:

        fx_shock = st.slider(
            "Cú sốc tỷ giá (%)",
            min_value=-20.0,
            max_value=40.0,
            value=float(
                default_values["FX Shock (%)"]
            ),
            step=0.5
        )


    # -------------------------------------------------------------------------
    # CREDIT GROWTH
    # -------------------------------------------------------------------------

    with col1:

        credit_growth = st.slider(
            "Tăng trưởng tín dụng (%)",
            min_value=-10.0,
            max_value=30.0,
            value=float(
                default_values["Credit Growth (%)"]
            ),
            step=0.5
        )


    # -------------------------------------------------------------------------
    # UNEMPLOYMENT
    # -------------------------------------------------------------------------

    with col2:

        unemployment = st.slider(
            "Tỷ lệ thất nghiệp (%)",
            min_value=0.0,
            max_value=15.0,
            value=float(
                default_values["Unemployment (%)"]
            ),
            step=0.1
        )


    # -------------------------------------------------------------------------
    # VNINDEX
    # -------------------------------------------------------------------------

    vnindex_shock = st.slider(
        "Cú sốc VN-Index (%)",
        min_value=-70.0,
        max_value=30.0,
        value=float(
            default_values["VN-Index Shock (%)"]
        ),
        step=1.0
    )


    # =========================================================================
    # TẠO DATAFRAME KỊCH BẢN
    # =========================================================================

    scenario_df = pd.DataFrame(
        {
            "Biến vĩ mô": [
                "Tăng trưởng GDP",
                "Lạm phát CPI",
                "Lãi suất",
                "Tỷ giá",
                "Tăng trưởng tín dụng",
                "Tỷ lệ thất nghiệp",
                "VN-Index"
            ],

            "Giá trị kịch bản": [
                gdp_growth,
                inflation,
                interest_rate,
                fx_shock,
                credit_growth,
                unemployment,
                vnindex_shock
            ],

            "Đơn vị": [
                "%",
                "%",
                "%",
                "%",
                "%",
                "%",
                "%"
            ]
        }
    )


    # =========================================================================
    # HIỂN THỊ KỊCH BẢN
    # =========================================================================

    st.subheader("Tóm tắt kịch bản")

    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Giá trị kịch bản":
                st.column_config.NumberColumn(
                    "Giá trị kịch bản",
                    format="%.2f"
                )
        }
    )


    # =========================================================================
    # KPI TÓM TẮT
    # =========================================================================

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "GDP Growth",
            f"{gdp_growth:.1f}%"
        )

    with k2:

        st.metric(
            "Inflation",
            f"{inflation:.1f}%"
        )

    with k3:

        st.metric(
            "Interest Rate",
            f"{interest_rate:.1f}%"
        )

    with k4:

        st.metric(
            "FX Shock",
            f"{fx_shock:+.1f}%"
        )


    # =========================================================================
    # BIỂU ĐỒ KỊCH BẢN
    # =========================================================================

    st.subheader("Biểu đồ kịch bản")

    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            x=scenario_df["Biến vĩ mô"],
            y=scenario_df["Giá trị kịch bản"],
            text=[
                f"{x:.1f}%"
                for x in scenario_df["Giá trị kịch bản"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Giá trị: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )


    fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        xaxis_title="Biến vĩ mô",
        yaxis_title="Giá trị (%)",
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )


    fig.update_xaxes(
        showgrid=False
    )


    fig.update_yaxes(
        gridcolor="#2A2A2A"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =========================================================================
    # ĐÁNH GIÁ MỨC ĐỘ STRESS
    # =========================================================================

    st.subheader("Đánh giá mức độ căng thẳng")


    stress_score = 0


    # GDP thấp -> stress cao
    if gdp_growth < 0:
        stress_score += 3

    elif gdp_growth < 3:
        stress_score += 2

    elif gdp_growth < 5:
        stress_score += 1


    # Inflation
    if inflation >= 8:
        stress_score += 3

    elif inflation >= 6:
        stress_score += 2

    elif inflation >= 4.5:
        stress_score += 1


    # Interest rate
    if interest_rate >= 10:
        stress_score += 3

    elif interest_rate >= 8:
        stress_score += 2

    elif interest_rate >= 6.5:
        stress_score += 1


    # FX
    if fx_shock >= 15:
        stress_score += 3

    elif fx_shock >= 10:
        stress_score += 2

    elif fx_shock >= 5:
        stress_score += 1


    # Unemployment
    if unemployment >= 6:
        stress_score += 3

    elif unemployment >= 4:
        stress_score += 2

    elif unemployment >= 3:
        stress_score += 1


    # Stock market
    if vnindex_shock <= -40:
        stress_score += 3

    elif vnindex_shock <= -25:
        stress_score += 2

    elif vnindex_shock <= -10:
        stress_score += 1


    # =========================================================================
    # PHÂN LOẠI
    # =========================================================================

    if stress_score >= 12:

        stress_level = "Rất cao"

        st.error(
            f"Mức độ Stress: {stress_level} "
            f"(Stress Score = {stress_score})"
        )

    elif stress_score >= 8:

        stress_level = "Cao"

        st.warning(
            f"Mức độ Stress: {stress_level} "
            f"(Stress Score = {stress_score})"
        )

    elif stress_score >= 4:

        stress_level = "Trung bình"

        st.warning(
            f"Mức độ Stress: {stress_level} "
            f"(Stress Score = {stress_score})"
        )

    else:

        stress_level = "Thấp"

        st.success(
            f"Mức độ Stress: {stress_level} "
            f"(Stress Score = {stress_score})"
        )


    # =========================================================================
    # GỢI Ý DIỄN GIẢI
    # =========================================================================

    st.subheader("Nhận định kịch bản")


    comments = []


    if gdp_growth < 0:

        comments.append(
            "Nền kinh tế rơi vào trạng thái suy giảm, "
            "có thể làm gia tăng rủi ro tín dụng."
        )

    elif gdp_growth < 3:

        comments.append(
            "Tăng trưởng GDP ở mức thấp, "
            "phản ánh môi trường kinh doanh suy yếu."
        )


    if inflation >= 6:

        comments.append(
            "Lạm phát ở mức cao có thể tạo áp lực "
            "lên chi phí hoạt động và sức mua."
        )


    if interest_rate >= 8:

        comments.append(
            "Mặt bằng lãi suất cao có thể làm tăng "
            "nghĩa vụ trả nợ của khách hàng."
        )


    if fx_shock >= 10:

        comments.append(
            "Biến động tỷ giá lớn có thể ảnh hưởng "
            "đến khách hàng có nghĩa vụ ngoại tệ."
        )


    if credit_growth <= 5:

        comments.append(
            "Tăng trưởng tín dụng thấp có thể phản ánh "
            "suy giảm nhu cầu tín dụng hoặc điều kiện tín dụng thắt chặt."
        )


    if unemployment >= 4:

        comments.append(
            "Tỷ lệ thất nghiệp tăng có thể làm suy giảm "
            "khả năng trả nợ của khách hàng cá nhân."
        )


    if vnindex_shock <= -25:

        comments.append(
            "Thị trường chứng khoán giảm mạnh có thể phản ánh "
            "sự suy giảm kỳ vọng và mức độ chấp nhận rủi ro của thị trường."
        )


    if comments:

        for comment in comments:

            st.write(
                f"• {comment}"
            )

    else:

        st.write(
            "Các giả định hiện tại chưa cho thấy mức độ căng thẳng lớn "
            "đối với môi trường kinh tế vĩ mô."
        )


    # =========================================================================
    # LƯU KỊCH BẢN
    # =========================================================================

    st.divider()

    st.subheader("Lưu kịch bản")


    scenario_name = st.text_input(
        "Tên kịch bản",
        value=scenario_type
    )


    if st.button(
        "Lưu kịch bản để chạy Stress Test",
        type="primary",
        use_container_width=True
    ):

        scenario_data = {

            "Tên kịch bản":
                scenario_name,

            "Loại kịch bản":
                scenario_type,

            "GDP Growth":
                gdp_growth,

            "Inflation":
                inflation,

            "Interest Rate":
                interest_rate,

            "FX Shock":
                fx_shock,

            "Credit Growth":
                credit_growth,

            "Unemployment":
                unemployment,

            "VNIndex Shock":
                vnindex_shock,

            "Stress Score":
                stress_score,

            "Stress Level":
                stress_level
        }


        st.session_state[
            "stress_scenario"
        ] = scenario_data


        st.success(
            f"Đã lưu kịch bản '{scenario_name}'. "
            "Bạn có thể chuyển sang tab Stress Test để sử dụng."
        )


    # =========================================================================
    # HIỂN THỊ KỊCH BẢN ĐÃ LƯU
    # =========================================================================

    if "stress_scenario" in st.session_state:

        with st.expander(
            "Xem kịch bản đang được lưu"
        ):

            saved_scenario = (
                st.session_state[
                    "stress_scenario"
                ]
            )

            saved_df = pd.DataFrame(
                saved_scenario.items(),
                columns=[
                    "Thông tin",
                    "Giá trị"
                ]
            )

            st.dataframe(
                saved_df,
                use_container_width=True,
                hide_index=True
            )


# =============================================================================
# TAB 5 - STRESS TEST
# =============================================================================

def tab5():

    st.title("Stress Test")

    st.caption(
        "Đánh giá tác động của các kịch bản kinh tế vĩ mô "
        "đến rủi ro tín dụng và các chỉ tiêu danh mục."
    )

    st.divider()


    # =========================================================================
    # KIỂM TRA KỊCH BẢN ĐÃ LƯU
    # =========================================================================

    if "stress_scenario" not in st.session_state:

        st.warning(
            "Chưa có kịch bản Stress Test được lưu. "
            "Vui lòng chuyển sang tab 'Stress Scenario' "
            "để thiết lập và lưu kịch bản trước."
        )

        return


    scenario = st.session_state["stress_scenario"]


    # =========================================================================
    # THÔNG TIN KỊCH BẢN
    # =========================================================================

    st.subheader("Kịch bản đang sử dụng")


    scenario_name = scenario.get(
        "Tên kịch bản",
        "Không xác định"
    )

    scenario_type = scenario.get(
        "Loại kịch bản",
        "Không xác định"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Tên kịch bản",
            scenario_name
        )


    with col2:

        st.metric(
            "Loại kịch bản",
            scenario_type
        )


    with col3:

        st.metric(
            "Mức độ Stress",
            scenario.get(
                "Stress Level",
                "N/A"
            )
        )


    # =========================================================================
    # HIỂN THỊ CÁC BIẾN VĨ MÔ
    # =========================================================================

    st.subheader("Giả định kinh tế vĩ mô")


    macro_df = pd.DataFrame(
        {
            "Biến vĩ mô": [
                "Tăng trưởng GDP",
                "Lạm phát CPI",
                "Lãi suất",
                "Cú sốc tỷ giá",
                "Tăng trưởng tín dụng",
                "Tỷ lệ thất nghiệp",
                "Cú sốc VN-Index"
            ],

            "Giá trị": [
                scenario.get("GDP Growth", 0),
                scenario.get("Inflation", 0),
                scenario.get("Interest Rate", 0),
                scenario.get("FX Shock", 0),
                scenario.get("Credit Growth", 0),
                scenario.get("Unemployment", 0),
                scenario.get("VNIndex Shock", 0)
            ],

            "Đơn vị": [
                "%",
                "%",
                "%",
                "%",
                "%",
                "%",
                "%"
            ]
        }
    )


    st.dataframe(
        macro_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Giá trị":
                st.column_config.NumberColumn(
                    "Giá trị",
                    format="%.2f"
                )
        }
    )


    st.divider()


    # =========================================================================
    # INPUT DANH MỤC
    # =========================================================================

    st.subheader("Thông số danh mục tín dụng")


    col1, col2 = st.columns(2)


    # -------------------------------------------------------------------------
    # EAD
    # -------------------------------------------------------------------------

    with col1:

        ead = st.number_input(
            "Tổng dư nợ / EAD (tỷ VND)",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )


    # -------------------------------------------------------------------------
    # BASE PD
    # -------------------------------------------------------------------------

    with col2:

        base_pd = st.number_input(
            "PD cơ sở (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1
        )


    # -------------------------------------------------------------------------
    # LGD
    # -------------------------------------------------------------------------

    with col1:

        lgd = st.number_input(
            "LGD (%)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=1.0
        )


    # -------------------------------------------------------------------------
    # NPL
    # -------------------------------------------------------------------------

    with col2:

        base_npl = st.number_input(
            "Tỷ lệ NPL hiện tại (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1
        )


    # -------------------------------------------------------------------------
    # CAR
    # -------------------------------------------------------------------------

    with col1:

        base_car = st.number_input(
            "CAR hiện tại (%)",
            min_value=0.0,
            max_value=100.0,
            value=11.0,
            step=0.1
        )


    # -------------------------------------------------------------------------
    # CAPITAL
    # -------------------------------------------------------------------------

    with col2:

        capital = st.number_input(
            "Vốn tự có (tỷ VND)",
            min_value=0.0,
            value=1500.0,
            step=50.0
        )


    # =========================================================================
    # LẤY BIẾN VĨ MÔ
    # =========================================================================

    gdp = scenario.get(
        "GDP Growth",
        0
    )

    inflation = scenario.get(
        "Inflation",
        0
    )

    interest_rate = scenario.get(
        "Interest Rate",
        0
    )

    fx_shock = scenario.get(
        "FX Shock",
        0
    )

    credit_growth = scenario.get(
        "Credit Growth",
        0
    )

    unemployment = scenario.get(
        "Unemployment",
        0
    )

    vnindex_shock = scenario.get(
        "VNIndex Shock",
        0
    )


    # =========================================================================
    # STRESS MODEL - PROTOTYPE
    # =========================================================================
    #
    # Đây là mô hình minh họa.
    # Sau này có thể thay bằng regression / ML model.
    #
    # PD stress =
    # base PD
    # + GDP effect
    # + inflation effect
    # + rate effect
    # + FX effect
    # + unemployment effect
    # + market effect
    #
    # =========================================================================


    # -------------------------------------------------------------------------
    # GDP EFFECT
    # GDP thấp hơn 6% -> tăng PD
    # -------------------------------------------------------------------------

    gdp_effect = max(
        0,
        6.0 - gdp
    ) * 0.20


    # -------------------------------------------------------------------------
    # INFLATION EFFECT
    # Inflation > 4% -> tăng PD
    # -------------------------------------------------------------------------

    inflation_effect = max(
        0,
        inflation - 4.0
    ) * 0.08


    # -------------------------------------------------------------------------
    # INTEREST RATE EFFECT
    # Rate > 5% -> tăng PD
    # -------------------------------------------------------------------------

    rate_effect = max(
        0,
        interest_rate - 5.0
    ) * 0.12


    # -------------------------------------------------------------------------
    # FX EFFECT
    # -------------------------------------------------------------------------

    fx_effect = max(
        0,
        fx_shock
    ) * 0.03


    # -------------------------------------------------------------------------
    # UNEMPLOYMENT EFFECT
    # -------------------------------------------------------------------------

    unemployment_effect = max(
        0,
        unemployment - 2.5
    ) * 0.25


    # -------------------------------------------------------------------------
    # STOCK MARKET EFFECT
    # Chỉ tính khi VNIndex giảm
    # -------------------------------------------------------------------------

    market_effect = max(
        0,
        -vnindex_shock
    ) * 0.015


    # =========================================================================
    # STRESSED PD
    # =========================================================================

    stressed_pd = (
        base_pd
        + gdp_effect
        + inflation_effect
        + rate_effect
        + fx_effect
        + unemployment_effect
        + market_effect
    )


    stressed_pd = min(
        stressed_pd,
        100
    )


    # =========================================================================
    # EXPECTED LOSS
    # =========================================================================

    base_expected_loss = (
        ead
        * (base_pd / 100)
        * (lgd / 100)
    )


    stressed_expected_loss = (
        ead
        * (stressed_pd / 100)
        * (lgd / 100)
    )


    incremental_loss = (
        stressed_expected_loss
        - base_expected_loss
    )


    # =========================================================================
    # STRESSED NPL
    # =========================================================================

    npl_increase = (
        max(
            0,
            stressed_pd - base_pd
        )
        * 0.65
    )


    stressed_npl = (
        base_npl
        + npl_increase
    )


    stressed_npl = min(
        stressed_npl,
        100
    )


    # =========================================================================
    # STRESSED CAPITAL
    # =========================================================================

    stressed_capital = max(
        0,
        capital - incremental_loss
    )


    # =========================================================================
    # CAR
    # Simplified approximation:
    # CAR stress giảm theo mức tổn thất tăng thêm
    # =========================================================================

    if capital > 0:

        car_reduction = (
            incremental_loss
            / capital
            * base_car
        )

    else:

        car_reduction = 0


    stressed_car = max(
        0,
        base_car - car_reduction
    )


    # =========================================================================
    # KẾT QUẢ STRESS TEST
    # =========================================================================

    st.divider()

    st.subheader("Kết quả Stress Test")


    k1, k2, k3, k4 = st.columns(4)


    # -------------------------------------------------------------------------
    # PD
    # -------------------------------------------------------------------------

    with k1:

        st.metric(
            "PD sau Stress",
            f"{stressed_pd:.2f}%",
            delta=(
                f"{stressed_pd - base_pd:+.2f} điểm %"
            ),
            delta_color="inverse"
        )


    # -------------------------------------------------------------------------
    # NPL
    # -------------------------------------------------------------------------

    with k2:

        st.metric(
            "NPL sau Stress",
            f"{stressed_npl:.2f}%",
            delta=(
                f"{stressed_npl - base_npl:+.2f} điểm %"
            ),
            delta_color="inverse"
        )


    # -------------------------------------------------------------------------
    # EXPECTED LOSS
    # -------------------------------------------------------------------------

    with k3:

        st.metric(
            "Expected Loss",
            f"{stressed_expected_loss:,.2f} tỷ",
            delta=(
                f"{incremental_loss:+,.2f} tỷ"
            ),
            delta_color="inverse"
        )


    # -------------------------------------------------------------------------
    # CAR
    # -------------------------------------------------------------------------

    with k4:

        st.metric(
            "CAR sau Stress",
            f"{stressed_car:.2f}%",
            delta=(
                f"{stressed_car - base_car:+.2f} điểm %"
            )
        )


    # =========================================================================
    # SO SÁNH BASELINE / STRESS
    # =========================================================================

    st.subheader("So sánh trước và sau Stress")


    comparison_df = pd.DataFrame(
        {
            "Chỉ tiêu": [
                "PD (%)",
                "NPL (%)",
                "Expected Loss (tỷ VND)",
                "CAR (%)",
                "Vốn tự có (tỷ VND)"
            ],

            "Trước Stress": [
                base_pd,
                base_npl,
                base_expected_loss,
                base_car,
                capital
            ],

            "Sau Stress": [
                stressed_pd,
                stressed_npl,
                stressed_expected_loss,
                stressed_car,
                stressed_capital
            ]
        }
    )


    comparison_df["Thay đổi"] = (
        comparison_df["Sau Stress"]
        - comparison_df["Trước Stress"]
    )


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Trước Stress":
                st.column_config.NumberColumn(
                    "Trước Stress",
                    format="%.2f"
                ),

            "Sau Stress":
                st.column_config.NumberColumn(
                    "Sau Stress",
                    format="%.2f"
                ),

            "Thay đổi":
                st.column_config.NumberColumn(
                    "Thay đổi",
                    format="%+.2f"
                )
        }
    )


    # =========================================================================
    # BIỂU ĐỒ SO SÁNH
    # =========================================================================

    st.subheader("Tác động của Stress Test")


    chart_df = pd.DataFrame(
        {
            "Chỉ tiêu": [
                "PD",
                "NPL",
                "CAR"
            ],

            "Trước Stress": [
                base_pd,
                base_npl,
                base_car
            ],

            "Sau Stress": [
                stressed_pd,
                stressed_npl,
                stressed_car
            ]
        }
    )


    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            name="Trước Stress",
            x=chart_df["Chỉ tiêu"],
            y=chart_df["Trước Stress"],
            text=[
                f"{x:.2f}%"
                for x in chart_df["Trước Stress"]
            ],
            textposition="outside"
        )
    )


    fig.add_trace(
        go.Bar(
            name="Sau Stress",
            x=chart_df["Chỉ tiêu"],
            y=chart_df["Sau Stress"],
            text=[
                f"{x:.2f}%"
                for x in chart_df["Sau Stress"]
            ],
            textposition="outside"
        )
    )


    fig.update_layout(
        template="plotly_dark",
        height=500,
        barmode="group",
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        xaxis_title="Chỉ tiêu",
        yaxis_title="%",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        )
    )


    fig.update_xaxes(
        showgrid=False
    )


    fig.update_yaxes(
        gridcolor="#2A2A2A"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =========================================================================
    # EXPECTED LOSS CHART
    # =========================================================================

    st.subheader("Tác động đến Expected Loss")


    loss_df = pd.DataFrame(
        {
            "Trạng thái": [
                "Trước Stress",
                "Sau Stress"
            ],

            "Expected Loss": [
                base_expected_loss,
                stressed_expected_loss
            ]
        }
    )


    loss_fig = go.Figure()


    loss_fig.add_trace(
        go.Bar(
            x=loss_df["Trạng thái"],
            y=loss_df["Expected Loss"],
            text=[
                f"{x:,.2f}"
                for x in loss_df["Expected Loss"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Expected Loss: %{y:,.2f} tỷ VND"
                "<extra></extra>"
            )
        )
    )


    loss_fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        yaxis_title="Tỷ VND",
        xaxis_title="",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )


    loss_fig.update_xaxes(
        showgrid=False
    )


    loss_fig.update_yaxes(
        gridcolor="#2A2A2A"
    )


    st.plotly_chart(
        loss_fig,
        use_container_width=True
    )


    # =========================================================================
    # PHÂN RÃ TÁC ĐỘNG LÊN PD
    # =========================================================================

    st.subheader("Phân rã tác động lên PD")


    pd_driver_df = pd.DataFrame(
        {
            "Yếu tố": [
                "GDP",
                "Lạm phát",
                "Lãi suất",
                "Tỷ giá",
                "Thất nghiệp",
                "VN-Index"
            ],

            "Tác động lên PD": [
                gdp_effect,
                inflation_effect,
                rate_effect,
                fx_effect,
                unemployment_effect,
                market_effect
            ]
        }
    )


    pd_driver_df = pd_driver_df.sort_values(
        "Tác động lên PD",
        ascending=False
    )


    driver_fig = go.Figure()


    driver_fig.add_trace(
        go.Bar(
            x=pd_driver_df["Tác động lên PD"],
            y=pd_driver_df["Yếu tố"],
            orientation="h",
            text=[
                f"{x:.2f}"
                for x in pd_driver_df["Tác động lên PD"]
            ],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Tác động PD: %{x:.2f} điểm %"
                "<extra></extra>"
            )
        )
    )


    driver_fig.update_layout(
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        xaxis_title="Tác động lên PD (điểm %)",
        yaxis_title="",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )


    driver_fig.update_xaxes(
        gridcolor="#2A2A2A"
    )


    driver_fig.update_yaxes(
        showgrid=False
    )


    st.plotly_chart(
        driver_fig,
        use_container_width=True
    )


    # =========================================================================
    # ĐÁNH GIÁ RỦI RO
    # =========================================================================

    st.subheader("Đánh giá kết quả")


    if stressed_pd >= 10:

        st.error(
            "PD tăng lên mức rất cao dưới kịch bản Stress. "
            "Danh mục có dấu hiệu chịu tác động đáng kể từ các yếu tố vĩ mô."
        )

    elif stressed_pd >= 5:

        st.warning(
            "PD tăng đáng kể dưới kịch bản Stress. "
            "Cần theo dõi các phân khúc hoặc khách hàng nhạy cảm "
            "với biến động kinh tế vĩ mô."
        )

    else:

        st.success(
            "PD sau Stress vẫn ở mức tương đối kiểm soát "
            "theo các giả định của mô hình hiện tại."
        )


    if stressed_car < 8:

        st.error(
            "CAR sau Stress giảm xuống mức thấp. "
            "Cần đánh giá thêm khả năng hấp thụ tổn thất và nhu cầu vốn."
        )

    elif stressed_car < base_car:

        st.warning(
            "CAR suy giảm dưới tác động của tổn thất tín dụng tăng thêm."
        )


    # =========================================================================
    # MODEL DETAILS
    # =========================================================================

    with st.expander(
        "Xem chi tiết mô hình Stress Test"
    ):

        st.markdown(
            """
            **Mô hình hiện tại là phiên bản prototype.**

            PD sau Stress được ước tính từ PD cơ sở và tác động của:

            - Tăng trưởng GDP
            - Lạm phát
            - Lãi suất
            - Tỷ giá
            - Tỷ lệ thất nghiệp
            - Thị trường chứng khoán

            Expected Loss được tính theo:

            **Expected Loss = EAD × PD × LGD**

            NPL và CAR hiện được ước tính bằng các quan hệ đơn giản để
            minh họa luồng Stress Test.

            Khi triển khai mô hình chính thức, các hệ số tác động cần được
            ước lượng và hiệu chỉnh bằng dữ liệu lịch sử thực tế.
            """
        )


    # =========================================================================
    # LƯU KẾT QUẢ
    # =========================================================================

    stress_result = {

        "Scenario":
            scenario_name,

        "EAD":
            ead,

        "Base PD":
            base_pd,

        "Stressed PD":
            stressed_pd,

        "LGD":
            lgd,

        "Base NPL":
            base_npl,

        "Stressed NPL":
            stressed_npl,

        "Base EL":
            base_expected_loss,

        "Stressed EL":
            stressed_expected_loss,

        "Incremental Loss":
            incremental_loss,

        "Base CAR":
            base_car,

        "Stressed CAR":
            stressed_car,

        "Base Capital":
            capital,

        "Stressed Capital":
            stressed_capital
    }


    st.session_state[
        "stress_test_result"
    ] = stress_result


    # =========================================================================
    # DOWNLOAD RESULT
    # =========================================================================

    st.divider()

    st.subheader("Tải kết quả Stress Test")


    result_df = pd.DataFrame(
        stress_result.items(),
        columns=[
            "Chỉ tiêu",
            "Giá trị"
        ]
    )


    csv_data = result_df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


    st.download_button(
        label="Tải kết quả CSV",
        data=csv_data,
        file_name="stress_test_result.csv",
        mime="text/csv",
        use_container_width=True
    )


    # =========================================================================
    # DISCLAIMER
    # =========================================================================

    st.caption(
        "Lưu ý: Kết quả Stress Test hiện tại được tính từ mô hình prototype "
        "và chỉ phục vụ mục đích xây dựng, thử nghiệm và minh họa dashboard."
    )





# =============================================================================
# TAB 6 - AI ANALYST
# =============================================================================

def tab6():

    st.title("AI Analysis")

    st.caption(
        "AI hỗ trợ diễn giải kịch bản và kết quả Stress Test, "
        "nhận diện các yếu tố rủi ro trọng yếu và tổng hợp nhận định quản trị."
    )

    st.divider()


    # =========================================================================
    # IMPORT GROQ
    # =========================================================================

    try:
        from groq import Groq

    except ImportError:

        st.error(
            "Chưa cài đặt thư viện Groq. "
            "Vui lòng chạy: python -m pip install groq"
        )

        return


    # =========================================================================
    # API KEY
    # =========================================================================

    try:

        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:

        st.warning(
            "Chưa tìm thấy GROQ_API_KEY trong file "
            ".streamlit/secrets.toml"
        )

        return


    # =========================================================================
    # GROQ CLIENT
    # =========================================================================

    client = Groq(
        api_key=api_key
    )


    # =========================================================================
    # LOAD DATA FROM PREVIOUS TABS
    # =========================================================================

    scenario = st.session_state.get(
        "stress_scenario"
    )

    stress_result = st.session_state.get(
        "stress_test_result"
    )


    # =========================================================================
    # VALIDATE DATA
    # =========================================================================

    if scenario is None:

        st.warning(
            "Chưa có kịch bản Stress Test. "
            "Vui lòng thiết lập và lưu kịch bản tại tab "
            "'Kịch bản Stress Test' trước."
        )

        return


    if stress_result is None:

        st.warning(
            "Chưa có kết quả Stress Test. "
            "Vui lòng chạy Stress Test trước khi sử dụng AI Analysis."
        )

        return


    # =========================================================================
    # CURRENT ANALYSIS CONTEXT
    # =========================================================================

    st.subheader("Dữ liệu đầu vào cho AI")


    scenario_name = scenario.get(
        "Tên kịch bản",
        "Không xác định"
    )

    scenario_type = scenario.get(
        "Loại kịch bản",
        "Không xác định"
    )

    stress_level = scenario.get(
        "Stress Level",
        "Không xác định"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Kịch bản",
            scenario_name
        )


    with c2:

        st.metric(
            "Loại kịch bản",
            scenario_type
        )


    with c3:

        st.metric(
            "Mức độ Stress",
            stress_level
        )


    st.caption(
        "AI sử dụng trực tiếp kịch bản và kết quả Stress Test "
        "đã được tạo tại các tab trước. AI không tính toán lại mô hình."
    )


    # =========================================================================
    # OPTIONAL DATA VIEW
    # =========================================================================

    with st.expander(
        "Xem dữ liệu AI đang sử dụng"
    ):

        st.markdown("#### Kịch bản vĩ mô")

        scenario_df = pd.DataFrame(
            scenario.items(),
            columns=[
                "Thông tin",
                "Giá trị"
            ]
        )

        st.dataframe(
            scenario_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown("#### Kết quả Stress Test")

        result_df = pd.DataFrame(
            stress_result.items(),
            columns=[
                "Chỉ tiêu",
                "Giá trị"
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    # =========================================================================
    # BUILD CONTEXT FOR AI
    # =========================================================================

    scenario_context = ""

    for key, value in scenario.items():

        scenario_context += (
            f"- {key}: {value}\n"
        )


    stress_context = ""

    for key, value in stress_result.items():

        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating
            )
        ):

            stress_context += (
                f"- {key}: {float(value):.4f}\n"
            )

        else:

            stress_context += (
                f"- {key}: {value}\n"
            )


    data_context = f"""
DỮ LIỆU ĐẦU VÀO

1. KỊCH BẢN STRESS TEST

{scenario_context}


2. KẾT QUẢ STRESS TEST

{stress_context}
"""


    # =========================================================================
    # SYSTEM PROMPT
    # =========================================================================

    system_prompt = """
Bạn là chuyên gia phân tích rủi ro tín dụng và Stress Testing
trong lĩnh vực ngân hàng tại Việt Nam.

Bạn đang hỗ trợ diễn giải kết quả Stress Test đã được tính toán
bởi một mô hình định lượng trong dashboard.

NGUYÊN TẮC:

1. Luôn trả lời bằng tiếng Việt.
2. Sử dụng văn phong chuyên nghiệp, rõ ràng và phù hợp báo cáo quản trị.
3. Không tính toán lại mô hình Stress Test.
4. Không thay đổi hoặc tự tạo số liệu.
5. Chỉ sử dụng dữ liệu được cung cấp.
6. Phân biệt rõ:
   - giả định kịch bản;
   - kết quả mô hình;
   - nhận định phân tích của AI.
7. Không khẳng định quan hệ nhân quả nếu dữ liệu không đủ để chứng minh.
8. Không đưa ra quyết định tín dụng thay cho người có thẩm quyền.
9. Nếu mô hình là prototype, phải xem kết quả là kết quả mô phỏng.
10. Ưu tiên phân tích:
    - PD;
    - NPL;
    - Expected Loss;
    - Capital;
    - CAR;
    - các yếu tố kinh tế vĩ mô có liên quan.

Mục tiêu của bạn là biến kết quả định lượng thành nhận định
có ý nghĩa đối với công tác quản trị rủi ro.
"""


    # =========================================================================
    # FUNCTION CALL AI
    # =========================================================================

    def run_ai_analysis(
        user_prompt
    ):

        try:

            response = (
            client
            .chat
            .completions
            .create(
                model="openai/gpt-oss-20b",
        
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": (
                            data_context
                            + "\n\n"
                            + user_prompt
                        )
                    }
                ],
        
                temperature=0.2,
                max_tokens=1800
            )
        )


            return (
                response
                .choices[0]
                .message
                .content
            )


        except Exception as e:

            st.error(
                "Không thể kết nối đến AI."
            )

            st.caption(
                str(e)
            )

            return None


    # =========================================================================
    # AI ANALYSIS TYPE
    # =========================================================================

    st.subheader("Phân tích tự động")


    analysis_type = st.selectbox(

        "Loại phân tích",

        [
            "Tổng quan điều hành",
            "Yếu tố rủi ro trọng yếu",
            "Phân tích tác động",
            "Đánh giá khả năng chịu đựng",
            "Chỉ tiêu cần giám sát",
            "Báo cáo tổng hợp"
        ],

        key="ai_analysis_type"
    )


    # =========================================================================
    # PROMPTS
    # =========================================================================

    prompts = {


        # ---------------------------------------------------------------------
        # EXECUTIVE SUMMARY
        # ---------------------------------------------------------------------

        "Tổng quan điều hành":
        """
Hãy viết phần tổng quan điều hành về kết quả Stress Test.

Cấu trúc:

1. Mức độ nghiêm trọng của kịch bản.
2. Các yếu tố vĩ mô đáng chú ý.
3. Những thay đổi quan trọng trong kết quả Stress Test.
4. Rủi ro trọng yếu đối với danh mục.
5. Kết luận ngắn gọn.

Yêu cầu:
- Viết ngắn gọn.
- Phù hợp để trình bày cho cấp quản lý.
- Không tạo thêm số liệu.
""",


        # ---------------------------------------------------------------------
        # RISK DRIVERS
        # ---------------------------------------------------------------------

        "Yếu tố rủi ro trọng yếu":
        """
Hãy xác định các yếu tố rủi ro vĩ mô trọng yếu trong kịch bản.

Phân tích các yếu tố sau nếu có dữ liệu:

- tăng trưởng GDP;
- lạm phát;
- lãi suất;
- tỷ giá;
- tăng trưởng tín dụng;
- thất nghiệp;
- VN-Index.

Với mỗi yếu tố:

1. Mô tả giả định trong kịch bản.
2. Giải thích chiều tác động tiềm năng đến rủi ro tín dụng.
3. Cho biết chỉ tiêu Stress Test nào có thể chịu ảnh hưởng.
4. Xác định yếu tố nào đáng chú ý nhất.

Không tạo số liệu mới.
""",


        # ---------------------------------------------------------------------
        # IMPACT
        # ---------------------------------------------------------------------

        "Phân tích tác động":
        """
Hãy diễn giải kết quả Stress Test.

Phân tích lần lượt:

1. PD trước và sau Stress.
2. NPL trước và sau Stress.
3. Expected Loss trước và sau Stress.
4. Mức tổn thất tăng thêm.
5. Vốn trước và sau Stress.
6. CAR trước và sau Stress.

Không tính lại các chỉ tiêu.

Giải thích ý nghĩa quản trị rủi ro của những thay đổi đã được mô hình tính toán.
""",


        # ---------------------------------------------------------------------
        # RESILIENCE
        # ---------------------------------------------------------------------

        "Đánh giá khả năng chịu đựng":
        """
Dựa trên kết quả Stress Test đã cung cấp,
hãy đánh giá khả năng chịu đựng của danh mục dưới kịch bản hiện tại.

Tập trung vào:

1. Mức tăng PD và NPL.
2. Mức tăng Expected Loss.
3. Mức suy giảm vốn.
4. Mức thay đổi CAR.
5. Các dấu hiệu cho thấy khả năng hấp thụ tổn thất suy giảm.

Không tự đặt ngưỡng pháp lý hoặc ngưỡng nội bộ nếu dữ liệu không cung cấp.

Kết luận theo góc độ:
- tương đối ổn định;
- cần theo dõi;
- chịu áp lực đáng kể;
- chịu áp lực rất cao.

Giải thích cơ sở của nhận định.
""",


        # ---------------------------------------------------------------------
        # MONITORING
        # ---------------------------------------------------------------------

        "Chỉ tiêu cần giám sát":
        """
Dựa trên kịch bản và kết quả Stress Test,
hãy đề xuất các nhóm chỉ tiêu cần được theo dõi.

Chia thành:

1. Chỉ tiêu kinh tế vĩ mô.
2. Chỉ tiêu chất lượng tín dụng.
3. Chỉ tiêu tổn thất tín dụng.
4. Chỉ tiêu vốn.
5. Chỉ báo cảnh báo sớm.

Đối với từng nhóm, giải thích tại sao cần theo dõi.

Không đưa ra quyết định tín dụng.
""",


        # ---------------------------------------------------------------------
        # FULL REPORT
        # ---------------------------------------------------------------------

        "Báo cáo tổng hợp":
        """
Hãy tạo một báo cáo phân tích Stress Test ngắn gọn với cấu trúc:

1. Tóm tắt điều hành
2. Mô tả kịch bản
3. Yếu tố rủi ro vĩ mô trọng yếu
4. Tác động đến PD và NPL
5. Tác động đến Expected Loss
6. Tác động đến vốn và CAR
7. Đánh giá khả năng chịu đựng
8. Chỉ tiêu cần theo dõi
9. Kết luận

Yêu cầu:
- Văn phong phù hợp báo cáo quản trị rủi ro ngân hàng.
- Không tạo thêm số liệu.
- Không tính toán lại mô hình.
"""
    }


    # =========================================================================
    # RUN AI
    # =========================================================================

    if st.button(
        "Phân tích bằng AI",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "AI đang phân tích kết quả Stress Test..."
        ):

            ai_result = run_ai_analysis(
                prompts[
                    analysis_type
                ]
            )


        if ai_result:

            st.session_state[
                "ai_analysis_result"
            ] = ai_result

            st.session_state[
                "ai_analysis_result_type"
            ] = analysis_type


    # =========================================================================
    # DISPLAY RESULT
    # =========================================================================

    if (
        "ai_analysis_result"
        in st.session_state
    ):

        st.divider()

        st.subheader(
            st.session_state.get(
                "ai_analysis_result_type",
                "Kết quả phân tích"
            )
        )


        with st.container(
            border=True
        ):

            st.markdown(
                st.session_state[
                    "ai_analysis_result"
                ]
            )


    # =========================================================================
    # CUSTOM ANALYSIS
    # =========================================================================

    st.divider()

    st.subheader(
        "Phân tích theo yêu cầu"
    )

    st.caption(
        "Đặt câu hỏi cụ thể dựa trên kịch bản và "
        "kết quả Stress Test hiện tại."
    )


    custom_question = st.text_area(

        "Nội dung cần phân tích",

        placeholder=(
            "Ví dụ: Hãy phân tích yếu tố nào có khả năng "
            "tạo áp lực lớn nhất lên chất lượng tín dụng."
        ),

        height=120,

        key="ai_custom_question"
    )


    if st.button(
        "Gửi yêu cầu phân tích",
        use_container_width=True
    ):

        if not custom_question.strip():

            st.warning(
                "Vui lòng nhập nội dung cần phân tích."
            )

        else:

            with st.spinner(
                "AI đang xử lý yêu cầu..."
            ):

                custom_result = run_ai_analysis(
                    custom_question
                )


            if custom_result:

                st.session_state[
                    "ai_custom_result"
                ] = custom_result


    # =========================================================================
    # CUSTOM RESULT
    # =========================================================================

    if (
        "ai_custom_result"
        in st.session_state
    ):

        st.markdown(
            "### Kết quả phân tích theo yêu cầu"
        )


        with st.container(
            border=True
        ):

            st.markdown(
                st.session_state[
                    "ai_custom_result"
                ]
            )


    # =========================================================================
    # RESET AI RESULT
    # =========================================================================

    st.divider()


    if st.button(
        "Xóa kết quả AI"
    ):

        st.session_state.pop(
            "ai_analysis_result",
            None
        )

        st.session_state.pop(
            "ai_analysis_result_type",
            None
        )

        st.session_state.pop(
            "ai_custom_result",
            None
        )

        st.rerun()


    # =========================================================================
    # DISCLAIMER
    # =========================================================================

    st.caption(
        "AI Analysis chỉ hỗ trợ diễn giải kết quả mô hình. "
        "Nội dung do AI tạo ra cần được rà soát trước khi sử dụng "
        "trong báo cáo hoặc quá trình ra quyết định."
    )



# =============================================================================
# MAIN APPLICATION
# =============================================================================

if menu == "Overview":
    tab0()

elif menu == "Macro Data":
    tab1()

elif menu == "Data Download":
    tab2()

elif menu == "Macro Analysis":
    tab3()

elif menu == "Stress Scenario":
    tab4()

elif menu == "Stress Test":
    tab5()

elif menu == "AI Analyst":
    tab6()
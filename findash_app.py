import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from groq import Groq


# ==============================================================================
# Summary
# ==============================================================================


@st.cache_data
def getsummary(ticker):

    stock = yf.Ticker(ticker)
    info = stock.info

    summary = pd.DataFrame({
        "attribute": [
            "Current Price",
            "Previous Close",
            "Open",
            "Day High",
            "Day Low",
            "52 Week High",
            "52 Week Low",
            "Volume",
            "Market Cap",
            "PE Ratio"
        ],
        "value": [
            info.get("currentPrice"),
            info.get("previousClose"),
            info.get("open"),
            info.get("dayHigh"),
            info.get("dayLow"),
            info.get("fiftyTwoWeekHigh"),
            info.get("fiftyTwoWeekLow"),
            info.get("volume"),
            info.get("marketCap"),
            info.get("trailingPE")
        ]
    })

    return summary

### Get data yfinance
@st.cache_data
def getstockdata(ticker):

    df = yf.download(
        ticker,
        period="max",
        progress=False,
        auto_adjust=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df



# =============================================================================
# TAB 0 - OVERVIEW
# =============================================================================

def tab0():

    # -------------------------------------------------------------------------
    # CSS
    # -------------------------------------------------------------------------

    st.markdown(
        """
        <style>

        .overview-container {
            padding: 10px 15px 30px 15px;
        }

        .university-name {
            text-align: center;
            font-size: 23px;
            font-weight: 700;
            color: #00529C;
            margin-top: 5px;
        }

        .faculty-name {
            text-align: center;
            font-size: 17px;
            color: #555555;
            margin-bottom: 25px;
        }

        .project-type {
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            color: #E67E22;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 20px;
        }

        .project-title {
            text-align: center;
            font-size: 38px;
            font-weight: 800;
            color: #1F2937;
            line-height: 1.3;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .project-subtitle {
            text-align: center;
            font-size: 18px;
            color: #64748B;
            margin-bottom: 30px;
        }

        .lecturer-box {
            background: linear-gradient(
                90deg,
                #EAF4FF,
                #F7FBFF
            );
            border-left: 5px solid #00529C;
            padding: 18px 25px;
            border-radius: 10px;
            margin: 20px 0px;
            font-size: 18px;
        }

        .section-title {
            font-size: 25px;
            font-weight: 700;
            color: #00529C;
            margin-top: 30px;
            margin-bottom: 20px;
        }

        .member-card {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0px 2px 7px rgba(0,0,0,0.05);
        }

        .member-number {
            font-size: 14px;
            color: #E67E22;
            font-weight: 600;
        }

        .member-name {
            font-size: 18px;
            font-weight: 650;
            color: #1F2937;
        }

        .thankyou-box {
            background: linear-gradient(
                135deg,
                #F8FBFF,
                #EEF6FF
            );
            padding: 25px 30px;
            border-radius: 15px;
            margin-top: 25px;
            border: 1px solid #DCEAF7;
            text-align: justify;
            line-height: 1.8;
            font-size: 16px;
        }

        .footer-overview {
            text-align: center;
            color: #64748B;
            margin-top: 35px;
            font-size: 14px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # UEH LOGO
    # =========================================================================

    try:

        col_logo1, col_logo2, col_logo3 = st.columns(
            [2, 1, 2]
        )

        with col_logo2:

            st.image(
                "ueh_logo.png",
                use_container_width=True
            )

    except:

        st.markdown(
            "<h2 style='text-align:center; color:#00529C;'>UEH</h2>",
            unsafe_allow_html=True
        )


    # =========================================================================
    # UNIVERSITY
    # =========================================================================

    st.markdown(
        """
        <div class="university-name">
            UNIVERSITY OF ECONOMICS HO CHI MINH CITY
        </div>

        <div class="faculty-name">
            Đại học Kinh tế Thành phố Hồ Chí Minh - UEH
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # PROJECT TITLE
    # =========================================================================

    st.markdown(
        """
        <div class="project-type">
            ĐỒ ÁN GIỮA KỲ
        </div>

        <div class="project-title">
            FINANCIAL & INVESTMENT<br>
            ANALYTICS DASHBOARD
        </div>

        <div class="project-subtitle">
            Xây dựng Dashboard thông tin đầu tư có hỗ trợ Chatbot
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # COURSE
    # =========================================================================

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:19px;
            margin-top:10px;
        ">
            <b>Môn học:</b>
            Phân tích Đầu tư Nâng cao
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # LECTURER
    # =========================================================================
    
    st.markdown(
        """
    <div class="lecturer-box">
    👨‍🏫 <b style="color:#000000;">Giảng viên hướng dẫn:</b>
    <span style="color:#000000; font-weight:600;">TS. Đỗ Như Tài</span>
    </div>
    """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # PROJECT DESCRIPTION
    # =========================================================================

    st.markdown(
        '<div class="section-title">📊 Giới thiệu đồ án</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Đồ án **Financial & Investment Analytics Dashboard** được xây dựng
        nhằm hỗ trợ người dùng theo dõi, trực quan hóa và phân tích thông tin
        tài chính của các tài sản đầu tư.

        Hệ thống tích hợp dữ liệu thị trường, các chỉ số tài chính,
        phân tích giá, mô phỏng Monte Carlo, theo dõi danh mục đầu tư
        và trợ lý Chatbot nhằm hỗ trợ quá trình tìm hiểu và phân tích
        quyết định đầu tư.
        """
    )


    # =========================================================================
    # MAIN FUNCTIONS
    # =========================================================================

    st.markdown(
        '<div class="section-title">🚀 Chức năng chính</div>',
        unsafe_allow_html=True
    )

    function_col1, function_col2, function_col3 = st.columns(3)

    with function_col1:

        st.info(
            """
            **📈 Stock Analytics**

            • Summary

            • Price Chart

            • Statistics

            • Financial Statements
            """
        )

    with function_col2:

        st.info(
            """
            **📊 Investment Analysis**

            • Analyst Analysis

            • Portfolio Trend

            • Risk Analysis

            • Monte Carlo Simulation
            """
        )

    with function_col3:

        st.info(
            """
            **🤖 AI Assistant**

            • Financial Chatbot

            • Stock Explanation

            • Risk Interpretation

            • Investment Concepts
            """
        )


    # =========================================================================
    # TEAM MEMBERS
    # =========================================================================
    
    st.markdown(
        '<div class="section-title">👥 Team Members</div>',
        unsafe_allow_html=True
    )
    
    members = [
        "Nguyễn Thị Chinh",
        "Nguyễn Thị Thu Thảo",
        "Châu Phương Uyên",
        "Bùi Thị Mạnh Quỳnh",
        "Đào Duy Bảo",
        "Mạnh Hồ Kiên"
    ]
    
    # ROW 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f'<div class="member-card">'
            f'<div class="member-number">MEMBER 01</div>'
            f'<div class="member-name">👤 {members[0]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="member-card">'
            f'<div class="member-number">MEMBER 02</div>'
            f'<div class="member-name">👤 {members[1]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    
    # ROW 2
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(
            f'<div class="member-card">'
            f'<div class="member-number">MEMBER 03</div>'
            f'<div class="member-name">👤 {members[2]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f'<div class="member-card">'
            f'<div class="member-number">MEMBER 04</div>'
            f'<div class="member-name">👤 {members[3]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    
    # ROW 3
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown(
            f'<div class="member-card">'
            f'<div class="member-number">MEMBER 05</div>'
            f'<div class="member-name">👤 {members[4]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col6:
        st.markdown(
            f'<div class="member-card">'
            f'<div class="member-number">MEMBER 06</div>'
            f'<div class="member-name">👤 {members[5]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


    # =========================================================================
    # ACKNOWLEDGEMENT
    # =========================================================================
    
    st.markdown(
        '<div style="color:white; font-size:25px; font-weight:700; margin-top:30px;">💙 Lời cảm ơn</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
    <div style="color:white; font-size:16px; line-height:1.8; text-align:justify;">
    Nhóm chúng em xin gửi lời cảm ơn chân thành đến
    <b style="color:white;">TS. Đỗ Như Tài</b>
    đã tận tình hướng dẫn, chia sẻ kiến thức và hỗ trợ nhóm
    trong quá trình học tập cũng như thực hiện đồ án giữa kỳ môn
    <b style="color:white;">Phân tích Đầu tư Nâng cao</b>.
    </div>
    """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
    <div style="color:white; font-size:16px; line-height:1.8; text-align:justify;">
    Thông qua quá trình xây dựng
    <b style="color:white;">Financial & Investment Analytics Dashboard</b>,
    nhóm có cơ hội vận dụng các kiến thức về phân tích tài chính,
    quản trị danh mục đầu tư, đo lường rủi ro, mô phỏng tài chính
    và ứng dụng công nghệ vào phân tích đầu tư.
    </div>
    """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
    <div style="color:white; font-size:16px; line-height:1.8; text-align:justify;">
    Do giới hạn về thời gian và kinh nghiệm thực tế,
    đồ án khó tránh khỏi những thiếu sót.
    Nhóm rất mong nhận được những nhận xét và góp ý từ giảng viên
    để có thể tiếp tục hoàn thiện sản phẩm trong tương lai.
    </div>
    """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
    <div style="color:white; font-size:16px; font-weight:700; margin-top:20px;">
    Nhóm xin chân thành cảm ơn!
    </div>
    """,
        unsafe_allow_html=True
    )


    # =========================================================================
    # FOOTER
    # =========================================================================

    st.markdown(
        """
        <div class="footer-overview">
            Financial & Investment Analytics Dashboard
            <br>
            Midterm Project — Advanced Investment Analysis
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# TAB 1 - SUMMARY
# ==============================================================================

def tab1():

    st.title("📈 Stock Summary Dashboard")

    # =========================
    # Use ticker from Sidebar
    # =========================

    if ticker == "-":
        st.info("👈 Please select a ticker from the sidebar")
        return

    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except:
        info = {}

    company_name = info.get("longName", ticker)

    # =========================
    # Company Name
    # =========================

    st.subheader(f"🏢 {company_name}")
    st.caption(f"Ticker: {ticker}")

    # =========================
    # Report Date
    # =========================
    
    st.write("### 📅 Report Date")
    
    report_date = st.date_input(
        "Select report date",
        value=datetime.today().date(),
        max_value=datetime.today().date(),
        label_visibility="collapsed"
    )
    
    st.caption(
        f"Selected date: {report_date.strftime('%d/%m/%Y')}"
    )
    
    # =========================
    # Historical Data by Report Date
    # =========================
    
    price_data = yf.download(
        ticker,
        start=report_date - timedelta(days=10),
        end=report_date + timedelta(days=1),
        auto_adjust=False,
        progress=False
    )
    
    # Fix MultiIndex columns from yfinance
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.get_level_values(0)
    
    # Only keep trading data up to selected report date
    price_data = price_data[
        price_data.index.date <= report_date
    ]
    
    if not price_data.empty:
    
        # Latest available trading session on or before report_date
        current = float(price_data["Close"].iloc[-1])
        volume = int(price_data["Volume"].iloc[-1])
    
        # Previous trading session
        if len(price_data) >= 2:
            previous = float(price_data["Close"].iloc[-2])
        else:
            previous = current
    
        # Daily change
        change = current - previous
    
        if previous != 0:
            change_pct = change / previous * 100
        else:
            change_pct = 0
    
    else:
        st.warning("No trading data available for the selected date.")
        return
    
    
    # =========================
    # Current Fundamental Data
    # =========================
    
    market_cap = info.get("marketCap", 0) or 0
    week_high = info.get("fiftyTwoWeekHigh", 0) or 0
    week_low = info.get("fiftyTwoWeekLow", 0) or 0
    pe_ratio = info.get("trailingPE", 0) or 0
    eps = info.get("trailingEps", 0) or 0
    
    
    # =========================
    # KPI Cards
    # =========================
    
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric(
        "Close Price",
        f"${current:,.2f}",
        help="Closing price for the selected report date (USD)"
    )
    
    c2.metric(
        "Daily Change",
        f"${change:+,.2f}",
        f"{change_pct:+.2f}%",
        help="Change from the previous trading session"
    )
    
    c3.metric(
        "Market Cap",
        f"${market_cap / 1_000_000_000:.2f}B",
        help="Current Market Capitalization in Billion USD"
    )
    
    c4.metric(
        "Trading Volume",
        f"{volume:,.0f}",
        help="Number of shares traded on the selected report date"
    )
    
    
    c5, c6, c7, c8 = st.columns(4)
    
    c5.metric(
        "52-Week High",
        f"${week_high:,.2f}",
        help="Current 52-week highest price (USD)"
    )
    
    c6.metric(
        "52-Week Low",
        f"${week_low:,.2f}",
        help="Current 52-week lowest price (USD)"
    )
    
    c7.metric(
        "P/E Ratio",
        f"{pe_ratio:,.2f}x",
        help="Current Price-to-Earnings Ratio"
    )
    
    c8.metric(
        "Earnings Per Share",
        f"${eps:,.2f}",
        help="Current EPS (USD per share)"
    )
    

    # =========================
    # Company Profile
    # =========================

    st.subheader("🏢 Company Profile")

    profile = pd.DataFrame({
        "Information": [
            "Ticker",
            "Company",
            "Sector",
            "Industry",
            "Country",
            "Website"
        ],
        "Value": [
            ticker,
            info.get("longName", "N/A"),
            info.get("sector", "N/A"),
            info.get("industry", "N/A"),
            info.get("country", "N/A"),
            info.get("website", "N/A")
        ]
    })

    st.table(profile)

    # =========================
    # Price Chart
    # =========================

    df = yf.download(
        ticker,
        period="5y",
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not df.empty:

        fig = px.area(
            df,
            x=df.index,
            y="Close",
            title=f"{company_name} ({ticker}) - Historical Closing Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    else:
        st.warning("No historical price data available.")        
        
    
#==============================================================================
# Tab 2 Chart
#==============================================================================


#The code below divides the streamlit page into 5 columns. The first two columns
#have a date picker option to select start and end dates and the the other three
#have dropdown selection boxes for duration, interval, and type of plot.

def tab2():

    st.title("📊 Stock Price Chart")

    # =========================
    # Check ticker
    # =========================

    if ticker == "-":
        st.info("👈 Please select a ticker from the sidebar")
        return

    # Company name
    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName", ticker)
    except:
        company_name = ticker

    st.subheader(f"🏢 {company_name}")
    st.caption(f"Ticker: {ticker}")

    st.divider()

    # =========================
    # FILTERS
    # =========================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        start_date = st.date_input(
            "Start Date",
            datetime.today().date() - timedelta(days=365)
        )

    with c2:
        end_date = st.date_input(
            "End Date",
            datetime.today().date()
        )

    with c3:
        duration = st.selectbox(
            "Duration",
            ["Custom", "1mo", "3mo", "6mo", "1y", "3y", "5y", "max"],
            index=4
        )

    with c4:
        interval = st.selectbox(
            "Interval",
            ["1d", "1wk", "1mo"]
        )

    with c5:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Candlestick"]
        )

    # =========================
    # MOVING AVERAGE OPTIONS
    # =========================

    st.write("#### 📈 Technical Overlays")

    ma1, ma2, ma3 = st.columns(3)

    with ma1:
        show_sma20 = st.checkbox("SMA 20", value=True)

    with ma2:
        show_sma50 = st.checkbox("SMA 50", value=True)

    with ma3:
        show_sma200 = st.checkbox("SMA 200", value=False)

    # =========================
    # DOWNLOAD DATA
    # =========================

    try:

        if duration == "Custom":

            df = yf.download(
                ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                progress=False
            )

        else:

            df = yf.download(
                ticker,
                period=duration,
                interval=interval,
                auto_adjust=False,
                progress=False
            )

        # Fix MultiIndex from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.warning("No stock data available for the selected period.")
            return

        # =========================
        # MOVING AVERAGES
        # =========================

        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()

        # =========================
        # PRICE CHART
        # =========================

        st.subheader("📈 Price Movement")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.75, 0.25]
        )

        # -------------------------
        # Line / Candlestick
        # -------------------------

        if chart_type == "Line":

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name="Close Price"
                ),
                row=1,
                col=1
            )

        else:

            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name=ticker
                ),
                row=1,
                col=1
            )

        # =========================
        # SMA
        # =========================

        if show_sma20:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA20"],
                    mode="lines",
                    name="SMA 20"
                ),
                row=1,
                col=1
            )

        if show_sma50:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA50"],
                    mode="lines",
                    name="SMA 50"
                ),
                row=1,
                col=1
            )

        if show_sma200:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA200"],
                    mode="lines",
                    name="SMA 200"
                ),
                row=1,
                col=1
            )

        # =========================
        # VOLUME
        # =========================

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume"
            ),
            row=2,
            col=1
        )

        # =========================
        # CHART SETTINGS
        # =========================

        fig.update_layout(
            height=700,
            title=f"{company_name} ({ticker})",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.02,
                x=0
            )
        )

        fig.update_yaxes(
            title_text="Price (USD)",
            row=1,
            col=1
        )

        fig.update_yaxes(
            title_text="Volume",
            row=2,
            col=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================
        # PRICE STATISTICS
        # =========================

        st.subheader("📊 Period Statistics")

        latest_close = float(df["Close"].iloc[-1])

        period_high = float(df["High"].max())
        period_low = float(df["Low"].min())

        avg_volume = float(df["Volume"].mean())

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Latest Close",
            f"${latest_close:,.2f}"
        )

        s2.metric(
            "Period High",
            f"${period_high:,.2f}"
        )

        s3.metric(
            "Period Low",
            f"${period_low:,.2f}"
        )

        s4.metric(
            "Avg Volume",
            f"{avg_volume:,.0f}"
        )

    except Exception as e:

        st.error(
            f"Unable to load chart data: {e}"
        )
           
             


# =============================================================================
# TAB 3 - STATISTICS
# =============================================================================

# -----------------------------------------------------------------------------
# FORMAT FUNCTIONS
# -----------------------------------------------------------------------------

def format_money_currency(value, symbol):
    if value is None:
        return "N/A"

    try:
        value = float(value)

        if abs(value) >= 1_000_000_000_000:
            return f"{symbol}{value / 1_000_000_000_000:.2f} T"

        elif abs(value) >= 1_000_000_000:
            return f"{symbol}{value / 1_000_000_000:.2f} B"

        elif abs(value) >= 1_000_000:
            return f"{symbol}{value / 1_000_000:.2f} M"

        elif abs(value) >= 1_000:
            return f"{symbol}{value / 1_000:.2f} K"

        else:
            return f"{symbol}{value:,.2f}"

    except (TypeError, ValueError):
        return "N/A"


def format_percent(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"

    except (TypeError, ValueError):
        return "N/A"


def format_number(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):,.2f}"

    except (TypeError, ValueError):
        return "N/A"


def format_price(value, symbol="$"):
    if value is None:
        return "N/A"

    try:
        return f"{symbol}{float(value):,.2f}"

    except (TypeError, ValueError):
        return "N/A"


def format_volume(value):
    if value is None:
        return "N/A"

    try:
        return f"{int(value):,} shares"

    except (TypeError, ValueError):
        return "N/A"


# -----------------------------------------------------------------------------
# TAB 3
# -----------------------------------------------------------------------------

def tab3():

    st.title("Statistics")
    st.write("Ticker:", ticker)

    if ticker == "-":
        st.info("Please select a ticker from the sidebar.")
        return

    # -------------------------------------------------------------------------
    # GET DATA
    # -------------------------------------------------------------------------

    @st.cache_data(ttl=3600)
    def getstats(ticker):

        stock = yf.Ticker(ticker)

        # get_info() ổn định hơn cách gọi stock.info trực tiếp
        info = stock.get_info()

        if not info:
            raise ValueError(
                f"No statistics data returned for {ticker}"
            )

        # ---------------------------------------------------------------------
        # CURRENCY
        # ---------------------------------------------------------------------

        currency = info.get("currency", "USD")

        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥",
            "KRW": "₩",
            "VND": "₫"
        }

        symbol = currency_symbols.get(
            currency,
            currency + " "
        )

        # =====================================================================
        # 1. VALUATION MEASURES
        # =====================================================================

        valuation = pd.DataFrame({
            "Attribute": [
                "Market Cap",
                "Enterprise Value",
                "Trailing P/E",
                "Forward P/E",
                "Price to Book",
                "Price to Sales",
                "Enterprise Value / Revenue",
                "Enterprise Value / EBITDA"
            ],

            "Value": [
                format_money_currency(
                    info.get("marketCap"),
                    symbol
                ),

                format_money_currency(
                    info.get("enterpriseValue"),
                    symbol
                ),

                format_number(
                    info.get("trailingPE")
                ),

                format_number(
                    info.get("forwardPE")
                ),

                format_number(
                    info.get("priceToBook")
                ),

                format_number(
                    info.get("priceToSalesTrailing12Months")
                ),

                format_number(
                    info.get("enterpriseToRevenue")
                ),

                format_number(
                    info.get("enterpriseToEbitda")
                )
            ]
        })

        # =====================================================================
        # 2. PROFITABILITY
        # =====================================================================

        profitability = pd.DataFrame({
            "Attribute": [
                "Profit Margin",
                "Operating Margin",
                "Return on Assets (ROA)",
                "Return on Equity (ROE)"
            ],

            "Value": [
                format_percent(
                    info.get("profitMargins")
                ),

                format_percent(
                    info.get("operatingMargins")
                ),

                format_percent(
                    info.get("returnOnAssets")
                ),

                format_percent(
                    info.get("returnOnEquity")
                )
            ]
        })

        # =====================================================================
        # 3. TRADING INFORMATION
        # =====================================================================

        trading = pd.DataFrame({
            "Attribute": [
                "Current Price",
                "Previous Close",
                "Open",
                "Day Low",
                "Day High",
                "52 Week Low",
                "52 Week High",
                "Volume",
                "Average Volume",
                "Beta"
            ],

            "Value": [
                format_price(
                    info.get("currentPrice"),
                    symbol
                ),

                format_price(
                    info.get("previousClose"),
                    symbol
                ),

                format_price(
                    info.get("open"),
                    symbol
                ),

                format_price(
                    info.get("dayLow"),
                    symbol
                ),

                format_price(
                    info.get("dayHigh"),
                    symbol
                ),

                format_price(
                    info.get("fiftyTwoWeekLow"),
                    symbol
                ),

                format_price(
                    info.get("fiftyTwoWeekHigh"),
                    symbol
                ),

                format_volume(
                    info.get("volume")
                ),

                format_volume(
                    info.get("averageVolume")
                ),

                format_number(
                    info.get("beta")
                )
            ]
        })

        # =====================================================================
        # 4. FINANCIAL HIGHLIGHTS
        # =====================================================================

        financial = pd.DataFrame({
            "Attribute": [
                "Total Revenue",
                "Revenue Per Share",
                "EBITDA",
                "Net Income",
                "Total Cash",
                "Total Debt",
                "Debt to Equity",
                "Free Cash Flow"
            ],

            "Value": [
                format_money_currency(
                    info.get("totalRevenue"),
                    symbol
                ),

                format_price(
                    info.get("revenuePerShare"),
                    symbol
                ),

                format_money_currency(
                    info.get("ebitda"),
                    symbol
                ),

                format_money_currency(
                    info.get("netIncomeToCommon"),
                    symbol
                ),

                format_money_currency(
                    info.get("totalCash"),
                    symbol
                ),

                format_money_currency(
                    info.get("totalDebt"),
                    symbol
                ),

                format_number(
                    info.get("debtToEquity")
                ),

                format_money_currency(
                    info.get("freeCashflow"),
                    symbol
                )
            ]
        })

        # =====================================================================
        # 5. SHARE STATISTICS
        # =====================================================================

        shares = pd.DataFrame({
            "Attribute": [
                "Shares Outstanding",
                "Float Shares",
                "Shares Short",
                "Short Ratio",
                "Held by Insiders",
                "Held by Institutions"
            ],

            "Value": [
                format_volume(
                    info.get("sharesOutstanding")
                ),

                format_volume(
                    info.get("floatShares")
                ),

                format_volume(
                    info.get("sharesShort")
                ),

                format_number(
                    info.get("shortRatio")
                ),

                format_percent(
                    info.get("heldPercentInsiders")
                ),

                format_percent(
                    info.get("heldPercentInstitutions")
                )
            ]
        })

        # =====================================================================
        # 6. DIVIDENDS
        # =====================================================================

        five_year_dividend = info.get(
            "fiveYearAvgDividendYield"
        )

        if five_year_dividend is not None:
            try:
                five_year_dividend = (
                    f"{float(five_year_dividend):.2f}%"
                )
            except (TypeError, ValueError):
                five_year_dividend = "N/A"
        else:
            five_year_dividend = "N/A"

        dividends = pd.DataFrame({
            "Attribute": [
                "Dividend Rate",
                "Dividend Yield",
                "Payout Ratio",
                "5 Year Average Dividend Yield"
            ],

            "Value": [
                format_price(
                    info.get("dividendRate"),
                    symbol
                ),

                format_percent(
                    info.get("dividendYield")
                ),

                format_percent(
                    info.get("payoutRatio")
                ),

                five_year_dividend
            ]
        })

        return (
            valuation,
            profitability,
            trading,
            financial,
            shares,
            dividends,
            currency
        )

    # -------------------------------------------------------------------------
    # LOAD AND DISPLAY DATA
    # -------------------------------------------------------------------------

    try:

        (
            valuation,
            profitability,
            trading,
            financial,
            shares,
            dividends,
            currency
        ) = getstats(ticker)

        # ---------------------------------------------------------------------
        # CURRENCY LABEL
        # ---------------------------------------------------------------------

        st.caption(
            f"Financial data currency: {currency}"
        )

        # =====================================================================
        # DASHBOARD LAYOUT
        # =====================================================================

        col1, col2 = st.columns(2)

        # =====================================================================
        # LEFT COLUMN
        # =====================================================================

        with col1:

            st.header("Valuation Measures")

            st.dataframe(
                valuation.set_index("Attribute"),
                use_container_width=True
            )

            st.header("Profitability")

            st.dataframe(
                profitability.set_index("Attribute"),
                use_container_width=True
            )

            st.header("Financial Highlights")

            st.dataframe(
                financial.set_index("Attribute"),
                use_container_width=True
            )

        # =====================================================================
        # RIGHT COLUMN
        # =====================================================================

        with col2:

            st.header("Trading Information")

            st.dataframe(
                trading.set_index("Attribute"),
                use_container_width=True
            )

            st.header("Share Statistics")

            st.dataframe(
                shares.set_index("Attribute"),
                use_container_width=True
            )

            st.header("Dividends & Splits")

            st.dataframe(
                dividends.set_index("Attribute"),
                use_container_width=True
            )

    # -------------------------------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------------------------------

    except Exception as e:

        st.error(
            "Unable to load Statistics data from Yahoo Finance."
        )

        st.write(
            "Error type:",
            type(e).__name__
        )

        st.write(
            "Error details:"
        )

        st.code(
            str(e)
        )
         
         
         

# =============================================================================
# TAB 4 - FINANCIALS
# =============================================================================

def tab4():

    st.title("Financials")

    st.write("Ticker:", ticker)

    if ticker == "-":
        st.info("Please select a ticker from the sidebar.")
        return


    # -------------------------------------------------------------------------
    # USER OPTIONS
    # -------------------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        statement = st.selectbox(
            "Show",
            [
                "Income Statement",
                "Balance Sheet",
                "Cash Flow"
            ]
        )

    with col2:
        period = st.selectbox(
            "Period",
            [
                "Yearly",
                "Quarterly"
            ]
        )


    # -------------------------------------------------------------------------
    # GET FINANCIAL DATA
    # -------------------------------------------------------------------------

    @st.cache_data(ttl=3600)
    def get_financial_statement(ticker, statement, period):

        stock = yf.Ticker(ticker)

        # ================================================================
        # INCOME STATEMENT
        # ================================================================

        if statement == "Income Statement":

            if period == "Yearly":
                data = stock.income_stmt

            else:
                data = stock.quarterly_income_stmt


        # ================================================================
        # BALANCE SHEET
        # ================================================================

        elif statement == "Balance Sheet":

            if period == "Yearly":
                data = stock.balance_sheet

            else:
                data = stock.quarterly_balance_sheet


        # ================================================================
        # CASH FLOW
        # ================================================================

        elif statement == "Cash Flow":

            if period == "Yearly":
                data = stock.cashflow

            else:
                data = stock.quarterly_cashflow


        else:
            data = pd.DataFrame()

        return data


    # -------------------------------------------------------------------------
    # FORMAT FINANCIAL VALUES
    # -------------------------------------------------------------------------

    def format_financial_value(value):

        if pd.isna(value):
            return "N/A"

        try:

            value = float(value)

            if abs(value) >= 1_000_000_000_000:

                return f"{value / 1_000_000_000_000:,.2f} T"

            elif abs(value) >= 1_000_000_000:

                return f"{value / 1_000_000_000:,.2f} B"

            elif abs(value) >= 1_000_000:

                return f"{value / 1_000_000:,.2f} M"

            elif abs(value) >= 1_000:

                return f"{value / 1_000:,.2f} K"

            else:

                return f"{value:,.2f}"

        except:
            return str(value)


    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------

    try:

        data = get_financial_statement(
            ticker,
            statement,
            period
        )


        # ---------------------------------------------------------------------
        # CHECK EMPTY DATA
        # ---------------------------------------------------------------------

        if data is None or data.empty:

            st.warning(
                f"No {period.lower()} {statement.lower()} data available "
                f"for {ticker}."
            )

            return


        # ---------------------------------------------------------------------
        # GET CURRENCY
        # ---------------------------------------------------------------------

        stock = yf.Ticker(ticker)

        try:
            info = stock.info
            currency = info.get(
                "financialCurrency",
                info.get("currency", "USD")
            )

        except:
            currency = "USD"


        st.caption(
            f"Financial statement currency: {currency}"
        )


        # ---------------------------------------------------------------------
        # FORMAT DATE COLUMNS
        # ---------------------------------------------------------------------

        display_data = data.copy()

        new_columns = []

        for col in display_data.columns:

            try:
                new_columns.append(
                    pd.to_datetime(col).strftime("%Y-%m-%d")
                )

            except:
                new_columns.append(str(col))

        display_data.columns = new_columns


        # ---------------------------------------------------------------------
        # FORMAT VALUES
        # ---------------------------------------------------------------------

        display_data = display_data.map(
            format_financial_value
        )


        # ---------------------------------------------------------------------
        # RENAME INDEX
        # ---------------------------------------------------------------------

        display_data.index.name = "Financial Item"


        # ---------------------------------------------------------------------
        # DISPLAY TITLE
        # ---------------------------------------------------------------------

        st.subheader(
            f"{period} {statement}"
        )


        # ---------------------------------------------------------------------
        # DISPLAY TABLE
        # ---------------------------------------------------------------------

        st.dataframe(
            display_data,
            use_container_width=True,
            height=650
        )


    # -------------------------------------------------------------------------
    # ERROR HANDLING
    # -------------------------------------------------------------------------

    except Exception as e:

        st.error(
            "Unable to load financial statement data."
        )

        st.write("Error details:")

        st.code(str(e))               
        
      
        
      
# =============================================================================
# TAB 5 - ANALYSIS
# =============================================================================

def tab5():

    st.title("Analysis")
    st.write("Ticker:", ticker)

    # -------------------------------------------------------------------------
    # CHECK TICKER
    # -------------------------------------------------------------------------

    if ticker == "-":
        st.info("Please select a ticker from the sidebar.")
        return


    # =========================================================================
    # GET ANALYST DATA
    # =========================================================================

    @st.cache_data(ttl=3600)
    def get_analysis_data(ticker):

        stock = yf.Ticker(ticker)

        data = {}

        # ---------------------------------------------------------------------
        # Analyst Recommendations
        # ---------------------------------------------------------------------

        try:
            data["recommendations"] = stock.recommendations
        except Exception:
            data["recommendations"] = pd.DataFrame()


        # ---------------------------------------------------------------------
        # Earnings Estimate
        # ---------------------------------------------------------------------

        try:
            data["earnings_estimate"] = stock.earnings_estimate
        except Exception:
            data["earnings_estimate"] = pd.DataFrame()


        # ---------------------------------------------------------------------
        # Revenue Estimate
        # ---------------------------------------------------------------------

        try:
            data["revenue_estimate"] = stock.revenue_estimate
        except Exception:
            data["revenue_estimate"] = pd.DataFrame()


        # ---------------------------------------------------------------------
        # EPS Trend
        # ---------------------------------------------------------------------

        try:
            data["eps_trend"] = stock.eps_trend
        except Exception:
            data["eps_trend"] = pd.DataFrame()


        # ---------------------------------------------------------------------
        # EPS Revisions
        # ---------------------------------------------------------------------

        try:
            data["eps_revisions"] = stock.eps_revisions
        except Exception:
            data["eps_revisions"] = pd.DataFrame()


        # ---------------------------------------------------------------------
        # Growth Estimates
        # ---------------------------------------------------------------------

        try:
            data["growth_estimates"] = stock.growth_estimates
        except Exception:
            data["growth_estimates"] = pd.DataFrame()


        # ---------------------------------------------------------------------
        # Analyst Price Targets
        # ---------------------------------------------------------------------

        try:
            price_targets = stock.analyst_price_targets

            if isinstance(price_targets, dict):
                data["price_targets"] = price_targets
            else:
                data["price_targets"] = {}

        except Exception:
            data["price_targets"] = {}


        # ---------------------------------------------------------------------
        # Upgrades / Downgrades
        # ---------------------------------------------------------------------

        try:
            data["upgrades"] = stock.upgrades_downgrades
        except Exception:
            data["upgrades"] = pd.DataFrame()


        return data


    # =========================================================================
    # FORMAT ANALYSIS VALUE
    # =========================================================================

    def format_analysis_value(value):

        # ---------------------------------------------------------------------
        # Handle None
        # ---------------------------------------------------------------------

        if value is None:
            return "N/A"

        # ---------------------------------------------------------------------
        # Handle NaN
        # ---------------------------------------------------------------------

        try:
            if pd.isna(value):
                return "N/A"
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # Format Numeric Value
        # ---------------------------------------------------------------------

        try:

            value = float(value)

            if abs(value) >= 1_000_000_000_000:
                return (
                    f"{value / 1_000_000_000_000:,.2f} T"
                )

            elif abs(value) >= 1_000_000_000:
                return (
                    f"{value / 1_000_000_000:,.2f} B"
                )

            elif abs(value) >= 1_000_000:
                return (
                    f"{value / 1_000_000:,.2f} M"
                )

            elif abs(value) >= 1_000:
                return (
                    f"{value / 1_000:,.2f} K"
                )

            else:
                return f"{value:,.2f}"

        except (TypeError, ValueError):
            return str(value)


    # =========================================================================
    # FORMAT TARGET PRICE
    # =========================================================================

    def format_target_price(value, currency):

        if value is None:
            return "N/A"

        try:

            value = float(value)

            return f"{value:,.2f} {currency}"

        except (TypeError, ValueError):
            return "N/A"


    # =========================================================================
    # LOAD DATA
    # =========================================================================

    try:

        analysis = get_analysis_data(ticker)

        stock = yf.Ticker(ticker)

        # ---------------------------------------------------------------------
        # GET CURRENCY
        # ---------------------------------------------------------------------

        try:

            info = stock.get_info()

            if info:
                currency = info.get("currency", "USD")
            else:
                currency = "USD"

        except Exception:
            currency = "USD"


        st.caption(
            f"Analyst data currency: {currency}"
        )


        # =====================================================================
        # 1. ANALYST PRICE TARGET
        # =====================================================================

        st.header("Analyst Price Target")

        price_target = analysis.get(
            "price_targets",
            {}
        )


        # ---------------------------------------------------------------------
        # IMPORTANT:
        # Initialize variables first to prevent:
        # "local variable 'current' referenced before assignment"
        # ---------------------------------------------------------------------

        current = None
        low = None
        mean = None
        median = None
        high = None


        if price_target:

            current = price_target.get(
                "current"
            )

            low = price_target.get(
                "low"
            )

            mean = price_target.get(
                "mean"
            )

            median = price_target.get(
                "median"
            )

            high = price_target.get(
                "high"
            )


            # =================================================================
            # ROW 1
            # =================================================================

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Current Price",
                    format_target_price(
                        current,
                        currency
                    )
                )


            with col2:

                st.metric(
                    "Target Low",
                    format_target_price(
                        low,
                        currency
                    )
                )


            with col3:

                st.metric(
                    "Target Mean",
                    format_target_price(
                        mean,
                        currency
                    )
                )


            # =================================================================
            # ROW 2
            # =================================================================

            col4, col5, col6 = st.columns(3)


            with col4:

                st.metric(
                    "Target Median",
                    format_target_price(
                        median,
                        currency
                    )
                )


            with col5:

                st.metric(
                    "Target High",
                    format_target_price(
                        high,
                        currency
                    )
                )


            # -----------------------------------------------------------------
            # Potential Upside / Downside
            # -----------------------------------------------------------------

            with col6:

                if (
                    current is not None
                    and mean is not None
                    and current != 0
                ):

                    try:

                        current_float = float(
                            current
                        )

                        mean_float = float(
                            mean
                        )

                        upside = (
                            (
                                mean_float
                                - current_float
                            )
                            / current_float
                        ) * 100


                        st.metric(
                            "Potential Upside / Downside",
                            f"{upside:.2f}%"
                        )

                    except (
                        TypeError,
                        ValueError,
                        ZeroDivisionError
                    ):

                        st.metric(
                            "Potential Upside / Downside",
                            "N/A"
                        )

                else:

                    st.metric(
                        "Potential Upside / Downside",
                        "N/A"
                    )


        else:

            st.info(
                "No analyst price target data available."
            )


        st.divider()


        # =====================================================================
        # 2. ANALYST RECOMMENDATIONS
        # =====================================================================

        st.header(
            "Analyst Recommendations"
        )


        recommendations = analysis.get(
            "recommendations",
            pd.DataFrame()
        )


        if (
            recommendations is not None
            and isinstance(
                recommendations,
                pd.DataFrame
            )
            and not recommendations.empty
        ):

            recommendations_display = (
                recommendations.copy()
            )


            st.dataframe(
                recommendations_display,
                use_container_width=True
            )


            # -----------------------------------------------------------------
            # RECOMMENDATION CHART
            # -----------------------------------------------------------------

            try:

                latest = (
                    recommendations_display.iloc[0]
                )


                categories = [
                    "Strong Buy",
                    "Buy",
                    "Hold",
                    "Sell",
                    "Strong Sell"
                ]


                values = [
                    latest.get(
                        "strongBuy",
                        0
                    ),

                    latest.get(
                        "buy",
                        0
                    ),

                    latest.get(
                        "hold",
                        0
                    ),

                    latest.get(
                        "sell",
                        0
                    ),

                    latest.get(
                        "strongSell",
                        0
                    )
                ]


                recommendation_chart = (
                    pd.DataFrame(
                        {
                            "Recommendation":
                                categories,

                            "Analysts":
                                values
                        }
                    )
                )


                fig = px.bar(
                    recommendation_chart,
                    x="Recommendation",
                    y="Analysts",
                    title=(
                        "Latest Analyst "
                        "Recommendations"
                    ),
                    text="Analysts"
                )


                fig.update_layout(
                    height=450
                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


            except Exception:
                pass


        else:

            st.info(
                "No analyst recommendation "
                "data available."
            )


        st.divider()


        # =====================================================================
        # 3. EARNINGS ESTIMATE
        # =====================================================================

        st.header(
            "Earnings Estimate"
        )


        earnings = analysis.get(
            "earnings_estimate",
            pd.DataFrame()
        )


        if (
            earnings is not None
            and isinstance(
                earnings,
                pd.DataFrame
            )
            and not earnings.empty
        ):

            earnings_display = (
                earnings.copy()
            )


            earnings_display = (
                earnings_display.map(
                    format_analysis_value
                )
            )


            st.dataframe(
                earnings_display,
                use_container_width=True
            )


        else:

            st.info(
                "No earnings estimate "
                "data available."
            )


        st.divider()


        # =====================================================================
        # 4. REVENUE ESTIMATE
        # =====================================================================

        st.header(
            "Revenue Estimate"
        )


        revenue = analysis.get(
            "revenue_estimate",
            pd.DataFrame()
        )


        if (
            revenue is not None
            and isinstance(
                revenue,
                pd.DataFrame
            )
            and not revenue.empty
        ):

            revenue_display = (
                revenue.copy()
            )


            revenue_display = (
                revenue_display.map(
                    format_analysis_value
                )
            )


            st.dataframe(
                revenue_display,
                use_container_width=True
            )


        else:

            st.info(
                "No revenue estimate "
                "data available."
            )


        st.divider()


        # =====================================================================
        # 5. EPS TREND
        # =====================================================================

        st.header(
            "EPS Trend"
        )


        eps_trend = analysis.get(
            "eps_trend",
            pd.DataFrame()
        )


        if (
            eps_trend is not None
            and isinstance(
                eps_trend,
                pd.DataFrame
            )
            and not eps_trend.empty
        ):

            eps_trend_display = (
                eps_trend.copy()
            )


            eps_trend_display = (
                eps_trend_display.map(
                    format_analysis_value
                )
            )


            st.dataframe(
                eps_trend_display,
                use_container_width=True
            )


        else:

            st.info(
                "No EPS trend data available."
            )


        st.divider()


        # =====================================================================
        # 6. EPS REVISIONS
        # =====================================================================

        st.header(
            "EPS Revisions"
        )


        eps_revisions = analysis.get(
            "eps_revisions",
            pd.DataFrame()
        )


        if (
            eps_revisions is not None
            and isinstance(
                eps_revisions,
                pd.DataFrame
            )
            and not eps_revisions.empty
        ):

            eps_revisions_display = (
                eps_revisions.copy()
            )


            eps_revisions_display = (
                eps_revisions_display.map(
                    format_analysis_value
                )
            )


            st.dataframe(
                eps_revisions_display,
                use_container_width=True
            )


        else:

            st.info(
                "No EPS revision data available."
            )


        st.divider()


        # =====================================================================
        # 7. GROWTH ESTIMATES
        # =====================================================================

        st.header(
            "Growth Estimates"
        )


        growth = analysis.get(
            "growth_estimates",
            pd.DataFrame()
        )


        if (
            growth is not None
            and isinstance(
                growth,
                pd.DataFrame
            )
            and not growth.empty
        ):

            growth_display = (
                growth.copy()
            )


            # -----------------------------------------------------------------
            # Convert decimal growth ratios to percentage
            # Example: 0.15 -> 15.00%
            # -----------------------------------------------------------------

            for col in growth_display.columns:

                growth_display[col] = (
                    growth_display[col].apply(
                        lambda x:
                        (
                            f"{float(x) * 100:.2f}%"
                            if (
                                isinstance(
                                    x,
                                    (int, float, np.number)
                                )
                                and not pd.isna(x)
                            )
                            else x
                        )
                    )
                )


            st.dataframe(
                growth_display,
                use_container_width=True
            )


        else:

            st.info(
                "No growth estimate "
                "data available."
            )


        st.divider()


        # =====================================================================
        # 8. UPGRADES & DOWNGRADES
        # =====================================================================

        st.header(
            "Upgrades & Downgrades"
        )


        upgrades = analysis.get(
            "upgrades",
            pd.DataFrame()
        )


        if (
            upgrades is not None
            and isinstance(
                upgrades,
                pd.DataFrame
            )
            and not upgrades.empty
        ):

            upgrades_display = (
                upgrades.copy()
            )


            # -----------------------------------------------------------------
            # Sort newest first
            # -----------------------------------------------------------------

            try:

                upgrades_display = (
                    upgrades_display.sort_index(
                        ascending=False
                    )
                )

            except Exception:
                pass


            # -----------------------------------------------------------------
            # Show latest 20
            # -----------------------------------------------------------------

            upgrades_display = (
                upgrades_display.head(20)
            )


            st.dataframe(
                upgrades_display,
                use_container_width=True
            )


        else:

            st.info(
                "No upgrades or downgrades "
                "data available."
            )


    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    except Exception as e:

        st.error(
            "Unable to load analyst data "
            "from Yahoo Finance."
        )

        st.write(
            "Error type:",
            type(e).__name__
        )

        st.write(
            "Error details:"
        )

        st.code(
            str(e)
        )
            
           
# =============================================================================
# TAB 6 - MONTE CARLO SIMULATION
# =============================================================================

def tab6():

    st.title("Monte Carlo Simulation")
    st.write("Ticker:", ticker)

    if ticker == "-":
        st.info("Please select a ticker from the sidebar.")
        return

    # =========================================================================
    # USER INPUT
    # =========================================================================

    col1, col2 = st.columns(2)

    with col1:
        simulations = st.selectbox(
            "Number of Simulations (n)",
            [200, 500, 1000],
            index=1
        )

    with col2:
        time_horizon = st.selectbox(
            "Time Horizon (Trading Days)",
            [30, 60, 90, 180, 252],
            index=2
        )

    # =========================================================================
    # GET HISTORICAL DATA
    # =========================================================================

    @st.cache_data(ttl=3600)
    def get_mc_data(ticker):

        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        # Fix MultiIndex returned by newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    # =========================================================================
    # MONTE CARLO FUNCTION
    # =========================================================================

    def monte_carlo_simulation(
        close_price,
        time_horizon,
        simulations
    ):

        # ---------------------------------------------------------------------
        # DAILY RETURNS
        # ---------------------------------------------------------------------

        daily_returns = (
            close_price
            .pct_change()
            .dropna()
        )

        # Historical average daily return
        mean_return = daily_returns.mean()

        # Historical daily volatility
        daily_volatility = daily_returns.std()

        # Current stock price
        current_price = float(close_price.iloc[-1])

        # ---------------------------------------------------------------------
        # CREATE SIMULATION
        # ---------------------------------------------------------------------

        simulation_df = pd.DataFrame(
            index=range(1, time_horizon + 1)
        )

        for i in range(simulations):

            prices = []

            last_price = current_price

            for day in range(time_horizon):

                # Generate random daily return
                random_return = np.random.normal(
                    mean_return,
                    daily_volatility
                )

                # Calculate future stock price
                future_price = (
                    last_price *
                    (1 + random_return)
                )

                prices.append(future_price)

                last_price = future_price

            simulation_df[i] = prices

        return (
            simulation_df,
            current_price,
            mean_return,
            daily_volatility
        )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    try:

        stock_data = get_mc_data(ticker)

        if stock_data.empty:

            st.warning(
                "No historical stock data available."
            )

            return

        close_price = stock_data["Close"].dropna()

        if len(close_price) < 30:

            st.warning(
                "Not enough historical data to run Monte Carlo Simulation."
            )

            return

        # =====================================================================
        # GET CURRENCY
        # =====================================================================

        try:

            stock = yf.Ticker(ticker)

            info = stock.info

            currency = info.get(
                "currency",
                "USD"
            )

        except:

            currency = "USD"

        # =====================================================================
        # RUN MONTE CARLO
        # =====================================================================

        (
            mc,
            current_price,
            mean_return,
            daily_volatility
        ) = monte_carlo_simulation(
            close_price,
            time_horizon,
            simulations
        )

        # =====================================================================
        # SUMMARY
        # =====================================================================

        st.subheader("Simulation Parameters")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Current Price (USD)",
                f"{current_price:,.2f}"
            )

        with c2:

            st.metric(
                "Daily Return",
                f"{mean_return * 100:.2f}%"
            )

        with c3:

            st.metric(
                "Daily Volatility",
                f"{daily_volatility * 100:.2f}%"
            )

        with c4:

            st.metric(
                "Simulations",
                f"{simulations:,}"
            )

        st.caption(
            f"Forecast horizon: {time_horizon} trading days"
        )

        st.divider()

        # =====================================================================
        # MONTE CARLO CHART
        # =====================================================================

        st.subheader("Simulated Stock Price Paths")

        fig = go.Figure()

        # To keep the chart readable, display maximum 200 paths
        number_paths_display = min(
            simulations,
            200
        )

        for i in range(number_paths_display):

            fig.add_trace(

                go.Scatter(

                    x=mc.index,

                    y=mc[i],

                    mode="lines",

                    line=dict(
                        width=1
                    ),

                    opacity=0.25,

                    showlegend=False
                )
            )

        # Current price reference line
        fig.add_hline(

            y=current_price,

            line_dash="dash",

            annotation_text=(
                f"Current Price: "
                f"{current_price:,.2f} {currency}"
            )
        )

        fig.update_layout(

            title=(
                f"Monte Carlo Simulation for {ticker} "
                f"- Next {time_horizon} Trading Days"
            ),

            xaxis_title="Trading Day",

            yaxis_title=f"Stock Price ({currency})",

            height=600,

            hovermode="x"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            f"Displaying {number_paths_display} of "
            f"{simulations} simulated paths."
        )

        # =====================================================================
        # ENDING PRICE DISTRIBUTION
        # =====================================================================

        st.divider()

        st.subheader("Distribution of Ending Prices")

        ending_prices = mc.iloc[-1].values

        # ---------------------------------------------------------------------
        # STATISTICS
        # ---------------------------------------------------------------------

        mean_ending_price = np.mean(
            ending_prices
        )

        median_ending_price = np.median(
            ending_prices
        )

        percentile_5 = np.percentile(
            ending_prices,
            5
        )

        percentile_95 = np.percentile(
            ending_prices,
            95
        )

        # =====================================================================
        # HISTOGRAM
        # =====================================================================

        fig2 = go.Figure()

        fig2.add_trace(

            go.Histogram(

                x=ending_prices,

                nbinsx=50,

                name="Ending Price"
            )
        )

        # 5th percentile
        fig2.add_vline(

            x=percentile_5,

            line_dash="dash",

            annotation_text=(
                f"5th Percentile: "
                f"{percentile_5:,.2f}"
            )
        )

        # Mean
        fig2.add_vline(

            x=mean_ending_price,

            line_dash="dot",

            annotation_text=(
                f"Mean: "
                f"{mean_ending_price:,.2f}"
            )
        )

        fig2.update_layout(

            title=(
                f"Distribution of Stock Price "
                f"After {time_horizon} Trading Days"
            ),

            xaxis_title=f"Ending Price ({currency})",

            yaxis_title="Frequency",

            height=500
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =====================================================================
        # FORECAST STATISTICS
        # =====================================================================

        st.subheader("Forecast Statistics")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Mean Ending Price",
                f"{mean_ending_price:,.2f} {currency}"
            )

            st.metric(
                "5th Percentile",
                f"{percentile_5:,.2f} {currency}"
            )

        with col2:

            st.metric(
                "Median Ending Price",
                f"{median_ending_price:,.2f} {currency}"
            )

            st.metric(
                "95th Percentile",
                f"{percentile_95:,.2f} {currency}"
            )

        # =====================================================================
        # VALUE AT RISK
        # =====================================================================

        st.divider()

        st.subheader("Value at Risk (VaR)")

        # VaR at 95% confidence level
        VaR_95 = (
            current_price -
            percentile_5
        )

        # VaR percentage
        VaR_percent = (
            VaR_95 /
            current_price
        ) * 100

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "VaR 95%",
                f"{VaR_95:,.2f} {currency}"
            )

        with col2:

            st.metric(
                "VaR 95% (%)",
                f"{VaR_percent:.2f}%"
            )

        st.caption(
            "VaR 95% represents the potential loss threshold "
            "based on the 5th percentile of simulated ending prices."
        )

        # =====================================================================
        # PROBABILITY OF PROFIT / LOSS
        # =====================================================================

        st.divider()

        st.subheader("Probability Analysis")

        probability_profit = (
            np.mean(
                ending_prices >
                current_price
            ) * 100
        )

        probability_loss = (
            np.mean(
                ending_prices <
                current_price
            ) * 100
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Probability of Profit",
                f"{probability_profit:.2f}%"
            )

        with col2:

            st.metric(
                "Probability of Loss",
                f"{probability_loss:.2f}%"
            )

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    except Exception as e:

        st.error(
            "Unable to run Monte Carlo Simulation."
        )

        st.write(
            "Error details:"
        )

        st.code(
            str(e)
        )         
     
  
# =============================================================================
# TAB 7 - YOUR PORTFOLIO'S TREND
# =============================================================================

def tab7():

    st.title("Your Portfolio's Trend")

    alltickers = [
        'AAPL',
        'MSFT',
        'GOOG',
        'META',
        'AMZN',
        'NVDA',
        'TSLA',
        'NFLX',
        'AMD',
        'BTC-USD',
        'ETH-USD',
        '^GSPC',
        '^IXIC'
    ]

    selected_tickers = st.multiselect(
        "Select tickers in your portfolio",
        options=alltickers,
        default=['AAPL']
    )

    if len(selected_tickers) == 0:
        st.info("Please select at least one ticker.")
        return

    df = pd.DataFrame()

    for stock_ticker in selected_tickers:

        data = yf.download(
            stock_ticker,
            period='5y',
            progress=False,
            auto_adjust=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if not data.empty and 'Close' in data.columns:
            df[stock_ticker] = data['Close']

    if df.empty:
        st.warning("Unable to load portfolio data.")
        return

    st.subheader("Portfolio Price Trend")

    fig = px.line(
        df,
        x=df.index,
        y=df.columns,
        labels={
            "value": "Price",
            "Date": "Date",
            "variable": "Ticker"
        }
    )

    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend_title="Ticker"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    
# =============================================================================
# TAB 8 - FINANCIAL CHATBOT - GROQ
# =============================================================================

def tab8():

    st.title("Financial Chatbot")

    st.write(
        "Ask questions about stocks, financial ratios, risk, "
        "portfolio management and investment analysis."
    )

    # =========================================================================
    # GROQ API KEY
    # =========================================================================

    try:
        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:
        st.error(
            "Groq API Key was not found. "
            "Please add GROQ_API_KEY to .streamlit/secrets.toml"
        )
        return

    # Create Groq client
    client = Groq(api_key=api_key)

    # =========================================================================
    # CURRENT STOCK DATA
    # =========================================================================

    stock_context = ""

    if ticker != "-":

        try:

            stock = yf.Ticker(ticker)
            info = stock.info

            currency = info.get("currency", "USD")

            # =================================================================
            # FORMAT NUMBER
            # =================================================================

            def format_number(value):

                if value is None:
                    return "N/A"

                try:
                    return f"{float(value):,.2f}"

                except (TypeError, ValueError):
                    return str(value)

            # =================================================================
            # FORMAT PERCENTAGE
            # =================================================================

            def format_percent(value):

                if value is None:
                    return "N/A"

                try:
                    return f"{float(value) * 100:.2f}%"

                except (TypeError, ValueError):
                    return "N/A"

            # =================================================================
            # FORMAT MARKET CAP
            # =================================================================

            def format_market_cap(value):

                if value is None:
                    return "N/A"

                try:

                    value = float(value)

                    if abs(value) >= 1_000_000_000_000:
                        return (
                            f"{value / 1_000_000_000_000:.2f} T"
                        )

                    elif abs(value) >= 1_000_000_000:
                        return (
                            f"{value / 1_000_000_000:.2f} B"
                        )

                    elif abs(value) >= 1_000_000:
                        return (
                            f"{value / 1_000_000:.2f} M"
                        )

                    else:
                        return f"{value:,.2f}"

                except (TypeError, ValueError):
                    return "N/A"

            # =================================================================
            # STOCK CONTEXT
            # =================================================================

            stock_context = f"""
CURRENT STOCK DATA

Ticker:
{ticker}

Company:
{info.get("longName", "N/A")}

Sector:
{info.get("sector", "N/A")}

Industry:
{info.get("industry", "N/A")}

Current Price:
{format_number(info.get("currentPrice"))} {currency}

Previous Close:
{format_number(info.get("previousClose"))} {currency}

Market Capitalization:
{format_market_cap(info.get("marketCap"))} {currency}

Trailing P/E:
{format_number(info.get("trailingPE"))}

Forward P/E:
{format_number(info.get("forwardPE"))}

EPS:
{format_number(info.get("trailingEps"))} {currency}

Price to Book:
{format_number(info.get("priceToBook"))}

ROE:
{format_percent(info.get("returnOnEquity"))}

ROA:
{format_percent(info.get("returnOnAssets"))}

Profit Margin:
{format_percent(info.get("profitMargins"))}

Beta:
{format_number(info.get("beta"))}

52 Week High:
{format_number(info.get("fiftyTwoWeekHigh"))} {currency}

52 Week Low:
{format_number(info.get("fiftyTwoWeekLow"))} {currency}

Dividend Yield:
{format_percent(info.get("dividendYield"))}
"""

            # =================================================================
            # SHOW CURRENT STOCK
            # =================================================================

            st.info(
                f"Currently analyzing: "
                f"{ticker} - "
                f"{info.get('longName', '')}"
            )

        except Exception:

            stock_context = f"""
Selected ticker: {ticker}

Detailed financial information is currently unavailable.
"""

    else:

        stock_context = """
No ticker is currently selected.
"""

        st.info(
            "Select a ticker from the sidebar "
            "for stock-specific analysis."
        )

    # =========================================================================
    # CHAT HISTORY
    # =========================================================================

    if "groq_chat_history" not in st.session_state:
        st.session_state.groq_chat_history = []

    # =========================================================================
    # DISPLAY CHAT HISTORY
    # =========================================================================

    for message in st.session_state.groq_chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # =========================================================================
    # USER INPUT
    # =========================================================================

    question = st.chat_input(
        "Ask a financial question..."
    )

    if question:

        # Display question
        with st.chat_message("user"):
            st.markdown(question)

        # =====================================================================
        # SYSTEM PROMPT
        # =====================================================================

        system_prompt = f"""
You are a Financial Analysis Assistant integrated into a
Financial Investment Dashboard.

Your purpose is to help users understand financial and
investment information.

You can explain:

- Stock prices
- Financial statements
- Revenue and profit
- P/E Ratio
- EPS
- ROE
- ROA
- Profit Margin
- Market Capitalization
- Beta
- Volatility
- Risk and Return
- Portfolio Diversification
- CAPM
- APT
- Sharpe Ratio
- Value at Risk (VaR)
- Monte Carlo Simulation
- Investment concepts


==================================================
CURRENT DASHBOARD DATA
==================================================

{stock_context}


==================================================
INSTRUCTIONS
==================================================

1. Answer in Vietnamese unless the user asks for English.

2. If the user says "cổ phiếu này", "mã này",
   or "this stock", they mean the ticker supplied
   in CURRENT DASHBOARD DATA.

3. Use the financial data supplied by the dashboard.

4. Do not invent financial numbers.

5. Round numerical values to 2 decimal places.

6. Include currency units when discussing prices.

7. Explain financial terminology clearly and simply.

8. When analyzing a stock, discuss both positive factors
   and potential risks.

9. Never guarantee future investment returns.

10. Never claim that a stock will definitely increase
    or decrease.

11. If there is insufficient information, clearly say so.

12. Keep responses concise and suitable for a
    financial dashboard.
"""

        # =====================================================================
        # BUILD MESSAGES
        # =====================================================================

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Add previous chat history
        messages.extend(
            st.session_state.groq_chat_history[-8:]
        )

        # Add current question
        messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # Save user question
        st.session_state.groq_chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        # =====================================================================
        # CALL GROQ
        # =====================================================================

        try:

            with st.chat_message("assistant"):

                with st.spinner("Analyzing..."):

                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.3
                    )

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                st.markdown(answer)

            # Save AI answer
            st.session_state.groq_chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(
                "Unable to connect to Groq."
            )

            st.write("Error details:")

            st.code(str(e))

    # =========================================================================
    # CLEAR CHAT
    # =========================================================================

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.groq_chat_history = []

        st.rerun() 


# =============================================================================
# MAIN BODY
# =============================================================================

def run():

    # Danh sách ticker
    ticker_list = [
        '-',
        'AAPL',
        'MSFT',
        'GOOG',
        'META',
        'AMZN',
        'NVDA',
        'TSLA',
        'NFLX',
        'AMD',
        'BTC-USD',
        'ETH-USD',
        '^GSPC',
        '^IXIC'
    ]

    # Ticker được sử dụng cho các tab
    global ticker

    ticker = st.sidebar.selectbox(
        "Select a ticker",
        ticker_list
    )

    # Menu
    select_tab = st.sidebar.radio(
        "Select tab",
        [
            'Overview',
            'Summary',
            'Chart',
            'Statistics',
            'Financials',
            'Analysis',
            'Monte Carlo Simulation',
            "Your Portfolio's Trend",
            "Financial Chatbot" 
        ]
    )

    # Chọn tab
    if select_tab == 'Overview':
        tab0()
        
    if select_tab == 'Summary':
        tab1()

    elif select_tab == 'Chart':
        tab2()

    elif select_tab == 'Statistics':
        tab3()

    elif select_tab == 'Financials':
        tab4()

    elif select_tab == 'Analysis':
        tab5()

    elif select_tab == 'Monte Carlo Simulation':
        tab6()

    elif select_tab == "Your Portfolio's Trend":
        tab7()
        
    elif select_tab == "Financial Chatbot":
        tab8() 


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    run()
    


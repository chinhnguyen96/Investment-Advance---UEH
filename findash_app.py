import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap

from groq import Groq

# ==============================================================================
# AI / MACHINE LEARNING IMPORTS
# ==============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    GRU,
    Attention,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

from scipy.optimize import minimize


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


# ==============================================================================
# ABLATION STUDY - GRU TRAINING
# ==============================================================================

@st.cache_resource(show_spinner=False)
def train_gru_ablation(
    ticker,
    sequence_length,
    use_technical,
    use_attention
):

    # --------------------------------------------------------------------------
    # Prepare AI data
    # --------------------------------------------------------------------------

    (
        raw_df,
        data,
        feature_data,
        latest_features,
        current_price
    ) = prepare_ai_data(ticker)


    # --------------------------------------------------------------------------
    # Feature groups
    # --------------------------------------------------------------------------

    basic_features = [
        "Return",
        "Lag1",
        "Lag5",
        "Volume_Change"
    ]

    technical_features = [
        "SMA20_Ratio",
        "SMA50_Ratio",
        "MACD_Ratio",
        "RSI14",
        "Momentum10",
        "Volatility20"
    ]


    features = basic_features.copy()

    if use_technical:
        features += technical_features


    # --------------------------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------------------------

    missing_columns = [
        col for col in features
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )


    if "Target" not in data.columns:
        raise ValueError(
            "Target column was not found."
        )


    # --------------------------------------------------------------------------
    # X / y
    # --------------------------------------------------------------------------

    X_df = data[features].copy()

    y = data["Target"].values


    # --------------------------------------------------------------------------
    # Train / Validation / Test split
    # --------------------------------------------------------------------------

    train_end, valid_end = get_split_points(data)


    # --------------------------------------------------------------------------
    # Standardization
    # --------------------------------------------------------------------------

    scaler = StandardScaler()

    # Fit scaler ONLY on Training Set
    scaler.fit(
        X_df.iloc[:train_end]
    )

    X_scaled = scaler.transform(
        X_df
    )


    # --------------------------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------------------------

    X_train, y_train, train_pos = create_sequences(
        X_scaled,
        y,
        sequence_length,
        sequence_length - 1,
        train_end
    )


    X_valid, y_valid, valid_pos = create_sequences(
        X_scaled,
        y,
        sequence_length,
        train_end,
        valid_end
    )


    X_test, y_test, test_pos = create_sequences(
        X_scaled,
        y,
        sequence_length,
        valid_end,
        len(data)
    )


    if (
        len(X_train) == 0
        or len(X_valid) == 0
        or len(X_test) == 0
    ):
        raise ValueError(
            "Not enough sequence data for Ablation Study."
        )


    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------

    tf.keras.backend.clear_session()

    np.random.seed(42)
    tf.random.set_seed(42)


    # ==========================================================================
    # GRU WITHOUT ATTENTION
    # ==========================================================================

    if not use_attention:

        model = Sequential([
            Input(
                shape=(
                    sequence_length,
                    len(features)
                )
            ),

            GRU(
                64
            ),

            Dropout(
                0.20
            ),

            Dense(
                32,
                activation="relu"
            ),

            Dense(
                1
            )
        ])


    # ==========================================================================
    # ATTENTION-GRU
    # ==========================================================================

    else:

        inputs = Input(
            shape=(
                sequence_length,
                len(features)
            )
        )

        x = GRU(
            64,
            return_sequences=True
        )(inputs)

        attention_output = Attention()(
            [x, x]
        )

        x = GlobalAveragePooling1D()(
            attention_output
        )

        x = Dropout(
            0.20
        )(x)

        x = Dense(
            32,
            activation="relu"
        )(x)

        outputs = Dense(
            1
        )(x)

        model = Model(
            inputs=inputs,
            outputs=outputs
        )


    # --------------------------------------------------------------------------
    # Compile
    # --------------------------------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="mse"
    )


    # --------------------------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------------------------

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    )


    # --------------------------------------------------------------------------
    # Train
    # --------------------------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        validation_data=(
            X_valid,
            y_valid
        ),
        epochs=50,
        batch_size=32,
        shuffle=False,
        verbose=0,
        callbacks=[
            early_stop
        ]
    )


    # --------------------------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------------------------

    valid_pred = (
        model.predict(
            X_valid,
            verbose=0
        )
        .reshape(-1)
    )


    # --------------------------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------------------------

    test_pred = (
        model.predict(
            X_test,
            verbose=0
        )
        .reshape(-1)
    )


    # --------------------------------------------------------------------------
    # Validation DataFrame
    # --------------------------------------------------------------------------

    valid_df = pd.DataFrame(
        {
            "Actual": y_valid,
            "Predicted": valid_pred
        },
        index=data.index[
            valid_pos
        ]
    )


    # --------------------------------------------------------------------------
    # Test DataFrame
    # --------------------------------------------------------------------------

    test_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": test_pred
        },
        index=data.index[
            test_pos
        ]
    )


    return valid_df, test_df

# =============================================================================
# TAB 0 - OVERVIEW
# =============================================================================

def tab0():

    # =========================================================================
    # CSS
    # =========================================================================

    st.markdown("""
<style>

/* --------------------------------------------------------------------------
   GENERAL
-------------------------------------------------------------------------- */

.overview-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.3;
    margin-top: 12px;
    margin-bottom: 8px;
}

.overview-subtitle {
    text-align: center;
    font-size: 18px;
    color: #CBD5E1;
    line-height: 1.6;
    margin-bottom: 25px;
}

.project-type {
    text-align: center;
    font-size: 17px;
    font-weight: 700;
    color: #F59E0B;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 15px;
}

.university-name {
    text-align: center;
    font-size: 22px;
    font-weight: 800;
    color: #4EA8FF;
    margin-top: 5px;
}

.faculty-name {
    text-align: center;
    font-size: 16px;
    color: #CBD5E1;
    margin-top: 5px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #FFFFFF;
    margin-top: 38px;
    margin-bottom: 18px;
}


/* --------------------------------------------------------------------------
   LECTURER
-------------------------------------------------------------------------- */

.lecturer-card {
    border: 1px solid #334155;
    border-left: 5px solid #3B82F6;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 25px 0;
    background: rgba(30, 41, 59, 0.55);
    color: #FFFFFF;
    font-size: 17px;
    line-height: 1.7;
}

.lecturer-name {
    font-weight: 700;
    color: #FFFFFF;
}


/* --------------------------------------------------------------------------
   GRID
-------------------------------------------------------------------------- */

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin-top: 10px;
    margin-bottom: 15px;
}


/* --------------------------------------------------------------------------
   INFORMATION CARD
-------------------------------------------------------------------------- */

.info-card {
    border: 1px solid #334155;
    background: rgba(30, 41, 59, 0.55);
    border-radius: 14px;
    padding: 18px;
    min-height: 150px;
    overflow-wrap: anywhere;
}

.card-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.card-title {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 10px;
    line-height: 1.4;
}

.card-text {
    color: #CBD5E1;
    font-size: 14px;
    line-height: 1.8;
}


/* --------------------------------------------------------------------------
   AI PIPELINE
-------------------------------------------------------------------------- */

.pipeline-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: stretch;
    gap: 8px;
    margin-top: 15px;
    margin-bottom: 20px;
}

.pipeline-step {
    flex: 1 1 140px;
    max-width: 180px;
    min-width: 130px;
    border: 1px solid #334155;
    background: rgba(30, 41, 59, 0.65);
    border-radius: 14px;
    padding: 16px 10px;
    text-align: center;
}

.pipeline-number {
    width: 30px;
    height: 30px;
    margin: 0 auto 9px auto;
    border-radius: 50%;
    background: #2563EB;
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}

.pipeline-title {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 750;
    line-height: 1.5;
}

.pipeline-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #60A5FA;
    font-size: 22px;
    font-weight: 800;
}


/* --------------------------------------------------------------------------
   MODEL CARD
-------------------------------------------------------------------------- */

.model-card {
    border: 1px solid #334155;
    background: rgba(30, 41, 59, 0.60);
    border-radius: 14px;
    padding: 20px 14px;
    text-align: center;
    min-height: 135px;
}

.model-type {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 600;
    line-height: 1.5;
    margin-bottom: 12px;
}

.model-name {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 800;
    line-height: 1.5;
    overflow-wrap: anywhere;
}


/* --------------------------------------------------------------------------
   REQUIREMENTS
-------------------------------------------------------------------------- */

.requirement-card {
    border: 1px solid #334155;
    background: rgba(30, 41, 59, 0.60);
    border-radius: 14px;
    padding: 20px 12px;
    text-align: center;
    min-height: 145px;
}

.requirement-label {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
    line-height: 1.5;
}

.requirement-value {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 800;
    line-height: 1.55;
    overflow-wrap: anywhere;
}

.requirement-highlight {
    color: #34D399;
}


/* --------------------------------------------------------------------------
   TEAM MEMBERS
-------------------------------------------------------------------------- */

.member-card {
    border: 1px solid #334155;
    background: rgba(30, 41, 59, 0.55);
    border-radius: 12px;
    padding: 17px 18px;
    margin-bottom: 12px;
    min-height: 85px;
}

.member-number {
    font-size: 12px;
    color: #F59E0B;
    font-weight: 750;
    letter-spacing: 0.7px;
    margin-bottom: 6px;
}

.member-name {
    font-size: 17px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.45;
    overflow-wrap: anywhere;
}


/* --------------------------------------------------------------------------
   ACKNOWLEDGEMENT
-------------------------------------------------------------------------- */

.ack-card {
    border: 1px solid #334155;
    border-radius: 14px;
    background: rgba(30, 41, 59, 0.45);
    padding: 22px 25px;
    color: #FFFFFF;
    font-size: 15.5px;
    line-height: 1.85;
    text-align: justify;
}


/* --------------------------------------------------------------------------
   FOOTER
-------------------------------------------------------------------------- */

.footer-overview {
    text-align: center;
    color: #94A3B8;
    margin-top: 40px;
    padding-top: 18px;
    border-top: 1px solid #334155;
    font-size: 14px;
    line-height: 1.8;
}


/* --------------------------------------------------------------------------
   MOBILE
-------------------------------------------------------------------------- */

@media (max-width: 700px) {

    .overview-title {
        font-size: 27px;
    }

    .overview-subtitle {
        font-size: 15px;
    }

    .university-name {
        font-size: 18px;
    }

    .section-title {
        font-size: 21px;
    }

    .pipeline-container {
        flex-direction: column;
    }

    .pipeline-step {
        max-width: 100%;
        width: 100%;
    }

    .pipeline-arrow {
        transform: rotate(90deg);
        height: 20px;
    }

}

</style>
""", unsafe_allow_html=True)


    # =========================================================================
    # UEH LOGO
    # =========================================================================

    try:

        logo_col1, logo_col2, logo_col3 = st.columns([2, 1, 2])

        with logo_col2:
            st.image(
                "ueh_logo.png",
                use_container_width=True
            )

    except Exception:

        st.markdown(
            "<h2 style='text-align:center;color:#4EA8FF;'>UEH</h2>",
            unsafe_allow_html=True
        )


    # =========================================================================
    # UNIVERSITY
    # =========================================================================

    st.markdown("""
<div class="university-name">
ĐẠI HỌC KINH TẾ THÀNH PHỐ HỒ CHÍ MINH
</div>

<div class="faculty-name">
University of Economics Ho Chi Minh City - UEH
</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # PROJECT TITLE
    # =========================================================================

    st.markdown("""
<div class="project-type">
ĐỒ ÁN CUỐI KỲ
</div>

<div class="overview-title">
ỨNG DỤNG AI TRONG<br>
PHÂN TÍCH ĐẦU TƯ
</div>

<div class="overview-subtitle">
Dự báo lợi suất cổ phiếu và tối ưu hóa danh mục đầu tư<br>
bằng Trí tuệ nhân tạo
</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # COURSE
    # =========================================================================

    st.markdown("""
<div style="
text-align:center;
font-size:18px;
color:#FFFFFF;
line-height:1.7;
margin-top:10px;
">
<strong>Môn học:</strong> Phân tích Đầu tư Nâng cao
</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # LECTURER
    # =========================================================================

    st.markdown("""
<div class="lecturer-card">
👨‍🏫 <strong>Giảng viên hướng dẫn:</strong>
<span class="lecturer-name">TS. Đỗ Như Tài</span>
</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # PROJECT OVERVIEW
    # =========================================================================

    st.markdown(
        '<div class="section-title">📊 Giới thiệu đồ án</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        **Ứng dụng AI trong Phân tích Đầu tư** được xây dựng nhằm ứng dụng
        **Trí tuệ nhân tạo, Học máy và Học sâu** vào quá trình phân tích và ra quyết định đầu tư.

        Hệ thống sử dụng dữ liệu thị trường để **dự báo lợi suất cổ phiếu**,
        tạo tín hiệu **Mua / Nắm giữ / Bán (Buy / Hold / Sell)** và đánh giá
        hiệu quả chiến lược thông qua **Backtesting** trên dữ liệu ngoài mẫu.

        Bên cạnh đó, hệ thống hỗ trợ **tối ưu hóa danh mục đầu tư** dựa trên
        kết quả dự báo của mô hình AI, kết hợp với các công cụ phân tích tài chính,
        phân tích kỹ thuật, mô phỏng Monte Carlo và Financial Chatbot.
        """
    )


    # =========================================================================
    # AI PIPELINE
    # =========================================================================

    st.markdown(
        '<div class="section-title">🧠 Quy trình ứng dụng AI trong đầu tư</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="pipeline-container">

<div class="pipeline-step">
<div class="pipeline-number">1</div>
<div class="pipeline-title">Thu thập<br>dữ liệu</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">2</div>
<div class="pipeline-title">Tiền xử lý<br>dữ liệu</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">3</div>
<div class="pipeline-title">Xây dựng<br>đặc trưng</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">4</div>
<div class="pipeline-title">Train<br>Validation<br>Test</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">5</div>
<div class="pipeline-title">Huấn luyện<br>mô hình AI</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">6</div>
<div class="pipeline-title">Dự báo<br>lợi suất</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">7</div>
<div class="pipeline-title">Mua<br>Nắm giữ<br>Bán</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">8</div>
<div class="pipeline-title">Đánh giá<br>Backtesting</div>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-step">
<div class="pipeline-number">9</div>
<div class="pipeline-title">Tối ưu hóa<br>danh mục</div>
</div>

</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # MAIN FUNCTIONS
    # =========================================================================

    st.markdown(
        '<div class="section-title">🚀 Chức năng chính</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="card-grid">

<div class="info-card">
<div class="card-icon">📈</div>
<div class="card-title">Phân tích tài chính</div>
<div class="card-text">
Tổng quan cổ phiếu<br>
Biểu đồ giá<br>
Thống kê tài chính<br>
Báo cáo tài chính<br>
Phân tích từ chuyên gia
</div>
</div>

<div class="info-card">
<div class="card-icon">📊</div>
<div class="card-title">Phân tích đầu tư</div>
<div class="card-text">
Chỉ báo kỹ thuật<br>
Xu hướng danh mục<br>
Mô phỏng Monte Carlo<br>
Phân tích rủi ro<br>
Value at Risk (VaR)
</div>
</div>

<div class="info-card">
<div class="card-icon">🤖</div>
<div class="card-title">Ứng dụng AI</div>
<div class="card-text">
Dự báo bằng AI<br>
Tín hiệu Mua / Nắm giữ / Bán<br>
Backtesting<br>
Ablation Study<br>
Tối ưu hóa danh mục<br>
Financial Chatbot
</div>
</div>

</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # MODELS
    # =========================================================================

    st.markdown(
        '<div class="section-title">🤖 Các mô hình sử dụng</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="card-grid">

<div class="model-card">
<div class="model-type">
Mô hình cơ sở<br>
(Baseline)
</div>
<div class="model-name">
Linear<br>Regression
</div>
</div>

<div class="model-card">
<div class="model-type">
Học máy<br>
(Machine Learning)
</div>
<div class="model-name">
XGBoost
</div>
</div>

<div class="model-card">
<div class="model-type">
Học sâu<br>
(Deep Learning)
</div>
<div class="model-name">
LSTM
</div>
</div>

<div class="model-card">
<div class="model-type">
Mô hình đề xuất
</div>
<div class="model-name">
Attention-GRU
</div>
</div>

</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # EVALUATION
    # =========================================================================

    st.markdown(
        '<div class="section-title">📏 Chỉ tiêu đánh giá mô hình</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="card-grid">

<div class="info-card">
<div class="card-title">
🎯 Khả năng dự báo
</div>

<div class="card-text">
Sai số tuyệt đối trung bình (MAE)<br>
Sai số bình phương trung bình (RMSE)<br>
Độ chính xác xu hướng (Directional Accuracy)
</div>
</div>


<div class="info-card">

<div class="card-title">
💰 Hiệu quả đầu tư
</div>

<div class="card-text">
Lợi suất tích lũy (Cumulative Return)<br>
Tăng trưởng kép (CAGR)<br>
Biến động (Volatility)<br>
Sharpe Ratio<br>
Mức sụt giảm tối đa (Maximum Drawdown)<br>
Tỷ lệ giao dịch thắng (Win Rate)
</div>

</div>

</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # FINAL PROJECT REQUIREMENTS
    # =========================================================================

    st.markdown(
        '<div class="section-title">🎯 Yêu cầu đồ án cuối kỳ</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="card-grid">

<div class="requirement-card">
<div class="requirement-label">
CHIA TẬP DỮ LIỆU
</div>

<div class="requirement-value">
Train<br>
Validation<br>
Test
</div>
</div>


<div class="requirement-card">
<div class="requirement-label">
HUẤN LUYỆN MÔ HÌNH
</div>

<div class="requirement-value">
Baseline<br>
ML / DL<br>
Mô hình đề xuất
</div>
</div>


<div class="requirement-card">
<div class="requirement-label">
KIỂM SOÁT DỮ LIỆU
</div>

<div class="requirement-value">
Tránh<br>
Data Leakage
</div>
</div>


<div class="requirement-card">
<div class="requirement-label">
MỤC TIÊU ĐÁNH GIÁ
</div>

<div class="requirement-value requirement-highlight">
Sharpe Ratio<br>
≥ 1.80
</div>
</div>


<div class="requirement-card">
<div class="requirement-label">
THỰC NGHIỆM
</div>

<div class="requirement-value">
Ablation<br>
Study
</div>
</div>

</div>
""", unsafe_allow_html=True)


    st.info(
        """
        **Nguyên tắc đánh giá:** Tập **Train** và **Validation** được sử dụng
        để lựa chọn mô hình và điều chỉnh siêu tham số. Tập **Test** chỉ được
        sử dụng để đánh giá cuối cùng nhằm hạn chế **Data Leakage**.
        """
    )


    # =========================================================================
    # DASHBOARD STRUCTURE
    # =========================================================================

    st.markdown(
        '<div class="section-title">🖥️ Cấu trúc Dashboard</div>',
        unsafe_allow_html=True
    )

    dashboard_structure = pd.DataFrame({

        "Tab": [
            "Overview",
            "Summary",
            "Chart",
            "Statistics",
            "Financials",
            "Analysis",
            "Monte Carlo Simulation",
            "Your Portfolio's Trend",
            "AI Prediction",
            "AI Backtesting",
            "AI Portfolio Optimization",
            "Financial Chatbot"
        ],

        "Chức năng": [
            "Giới thiệu đồ án và phương pháp nghiên cứu",
            "Tổng quan cổ phiếu và các chỉ số chính",
            "Biểu đồ giá và phân tích kỹ thuật",
            "Các chỉ số thống kê tài chính",
            "Báo cáo tài chính doanh nghiệp",
            "Thông tin và dự báo của chuyên gia",
            "Mô phỏng rủi ro và Value at Risk",
            "Theo dõi xu hướng danh mục đầu tư",
            "Dự báo lợi suất và tín hiệu giao dịch bằng AI",
            "Đánh giá chiến lược trên dữ liệu ngoài mẫu",
            "Tối ưu hóa tỷ trọng danh mục bằng AI",
            "Trợ lý AI hỗ trợ phân tích tài chính"
        ]

    })

    st.dataframe(
        dashboard_structure,
        use_container_width=True,
        hide_index=True
    )


    # =========================================================================
    # TEAM MEMBERS
    # =========================================================================

    st.markdown(
        '<div class="section-title">👥 Thành viên nhóm</div>',
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

    for i in range(0, len(members), 2):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
<div class="member-card">
<div class="member-number">
THÀNH VIÊN {i + 1:02d}
</div>

<div class="member-name">
👤 {members[i]}
</div>
</div>
""",
                unsafe_allow_html=True
            )


        if i + 1 < len(members):

            with col2:

                st.markdown(
                    f"""
<div class="member-card">
<div class="member-number">
THÀNH VIÊN {i + 2:02d}
</div>

<div class="member-name">
👤 {members[i + 1]}
</div>
</div>
""",
                    unsafe_allow_html=True
                )


    # =========================================================================
    # ACKNOWLEDGEMENT
    # =========================================================================

    st.markdown(
        '<div class="section-title">💙 Lời cảm ơn</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="ack-card">

Nhóm chúng em xin gửi lời cảm ơn chân thành đến
<strong>TS. Đỗ Như Tài</strong>
đã tận tình hướng dẫn, chia sẻ kiến thức và hỗ trợ nhóm
trong quá trình học tập cũng như thực hiện đồ án cuối kỳ môn
<strong>Phân tích Đầu tư Nâng cao</strong>.

<br><br>

Thông qua quá trình xây dựng
<strong>Ứng dụng AI trong Phân tích Đầu tư</strong>,
nhóm có cơ hội vận dụng các kiến thức về phân tích tài chính,
quản trị danh mục đầu tư, Machine Learning, Deep Learning,
đo lường rủi ro và ứng dụng trí tuệ nhân tạo trong
phân tích và ra quyết định đầu tư.

<br><br>

Do giới hạn về thời gian, dữ liệu và kinh nghiệm thực tế,
đồ án khó tránh khỏi những thiếu sót. Nhóm rất mong nhận được
những nhận xét và góp ý từ giảng viên để tiếp tục hoàn thiện
mô hình và ứng dụng trong tương lai.

<br><br>

<strong>Nhóm xin chân thành cảm ơn!</strong>

</div>
""", unsafe_allow_html=True)


    # =========================================================================
    # FOOTER
    # =========================================================================

    st.markdown("""
<div class="footer-overview">

<strong>
Ứng dụng AI trong Phân tích Đầu tư
</strong>

<br>

Đồ án cuối kỳ — Phân tích Đầu tư Nâng cao

</div>
""", unsafe_allow_html=True)


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


# =============================================================================
# YAHOO FINANCE - CACHED INFO
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_info(ticker):
    # Lấy thông tin doanh nghiệp từ Yahoo Finance
    # Cache dữ liệu trong 1 giờ để hạn chế Too Many Requests

    try:
        stock = yf.Ticker(ticker)
        info = stock.get_info()

        if info:
            return info

    except Exception:
        pass

    return {}


# =============================================================================
# TAB 3 - STATISTICS
# =============================================================================

def tab3():

    st.title("📊 Statistics")

    # =========================================================================
    # CHECK TICKER
    # =========================================================================

    if ticker == "-":
        st.info("👈 Please select a ticker from the sidebar")
        return


    # =========================================================================
    # GET CACHED DATA
    # =========================================================================

    info = get_ticker_info(ticker)

    if not info:
        st.warning(
            "Yahoo Finance is temporarily unavailable or rate-limiting "
            "requests. Please try again later."
        )
        return


    # =========================================================================
    # COMPANY INFORMATION
    # =========================================================================

    company_name = info.get("longName", ticker)

    st.subheader(f"🏢 {company_name}")
    st.caption(f"Ticker: {ticker}")

    st.divider()


    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def format_number(value):
        # Định dạng số lớn thành K / M / B / T

        if value is None:
            return "N/A"

        try:
            value = float(value)

            if abs(value) >= 1_000_000_000_000:
                return f"{value / 1_000_000_000_000:.2f}T"

            elif abs(value) >= 1_000_000_000:
                return f"{value / 1_000_000_000:.2f}B"

            elif abs(value) >= 1_000_000:
                return f"{value / 1_000_000:.2f}M"

            elif abs(value) >= 1_000:
                return f"{value / 1_000:.2f}K"

            else:
                return f"{value:,.2f}"

        except (TypeError, ValueError):
            return str(value)


    def format_percent(value):
        # Chuyển dữ liệu dạng 0.25 thành 25.00%

        if value is None:
            return "N/A"

        try:
            return f"{float(value) * 100:.2f}%"

        except (TypeError, ValueError):
            return "N/A"


    def create_table(data):
        # Tạo DataFrame thống nhất cho các bảng

        df = pd.DataFrame(
            data,
            columns=["Attribute", "Value"]
        )

        return df.set_index("Attribute")


    # =========================================================================
    # MAIN LAYOUT
    # =========================================================================

    c1, c2 = st.columns(2)


    # =========================================================================
    # LEFT COLUMN
    # =========================================================================

    with c1:

        # =====================================================================
        # VALUATION MEASURES
        # =====================================================================

        st.header("💰 Valuation Measures")

        valuation = create_table([

            (
                "Market Cap",
                format_number(
                    info.get("marketCap")
                )
            ),

            (
                "Enterprise Value",
                format_number(
                    info.get("enterpriseValue")
                )
            ),

            (
                "Trailing P/E",
                format_number(
                    info.get("trailingPE")
                )
            ),

            (
                "Forward P/E",
                format_number(
                    info.get("forwardPE")
                )
            ),

            (
                "PEG Ratio",
                format_number(
                    info.get("pegRatio")
                )
            ),

            (
                "Price / Sales",
                format_number(
                    info.get("priceToSalesTrailing12Months")
                )
            ),

            (
                "Price / Book",
                format_number(
                    info.get("priceToBook")
                )
            ),

            (
                "Enterprise Value / Revenue",
                format_number(
                    info.get("enterpriseToRevenue")
                )
            ),

            (
                "Enterprise Value / EBITDA",
                format_number(
                    info.get("enterpriseToEbitda")
                )
            )

        ])

        st.table(valuation)


        # =====================================================================
        # FINANCIAL HIGHLIGHTS
        # =====================================================================

        st.header("📈 Financial Highlights")


        # ---------------------------------------------------------------------
        # PROFITABILITY
        # ---------------------------------------------------------------------

        st.subheader("Profitability")

        profitability = create_table([

            (
                "Profit Margin",
                format_percent(
                    info.get("profitMargins")
                )
            ),

            (
                "Operating Margin",
                format_percent(
                    info.get("operatingMargins")
                )
            ),

            (
                "Gross Margin",
                format_percent(
                    info.get("grossMargins")
                )
            ),

            (
                "EBITDA Margin",
                format_percent(
                    info.get("ebitdaMargins")
                )
            )

        ])

        st.table(profitability)


        # ---------------------------------------------------------------------
        # MANAGEMENT EFFECTIVENESS
        # ---------------------------------------------------------------------

        st.subheader("Management Effectiveness")

        management = create_table([

            (
                "Return on Assets (ROA)",
                format_percent(
                    info.get("returnOnAssets")
                )
            ),

            (
                "Return on Equity (ROE)",
                format_percent(
                    info.get("returnOnEquity")
                )
            )

        ])

        st.table(management)


        # ---------------------------------------------------------------------
        # INCOME STATEMENT
        # ---------------------------------------------------------------------

        st.subheader("Income Statement")

        income = create_table([

            (
                "Total Revenue",
                format_number(
                    info.get("totalRevenue")
                )
            ),

            (
                "Revenue Per Share",
                format_number(
                    info.get("revenuePerShare")
                )
            ),

            (
                "Gross Profit",
                format_number(
                    info.get("grossProfits")
                )
            ),

            (
                "EBITDA",
                format_number(
                    info.get("ebitda")
                )
            ),

            (
                "Net Income",
                format_number(
                    info.get("netIncomeToCommon")
                )
            ),

            (
                "Diluted EPS",
                format_number(
                    info.get("trailingEps")
                )
            )

        ])

        st.table(income)


        # ---------------------------------------------------------------------
        # BALANCE SHEET
        # ---------------------------------------------------------------------

        st.subheader("Balance Sheet")

        balance = create_table([

            (
                "Total Cash",
                format_number(
                    info.get("totalCash")
                )
            ),

            (
                "Cash Per Share",
                format_number(
                    info.get("totalCashPerShare")
                )
            ),

            (
                "Total Debt",
                format_number(
                    info.get("totalDebt")
                )
            ),

            (
                "Debt / Equity",
                format_number(
                    info.get("debtToEquity")
                )
            ),

            (
                "Current Ratio",
                format_number(
                    info.get("currentRatio")
                )
            ),

            (
                "Book Value Per Share",
                format_number(
                    info.get("bookValue")
                )
            )

        ])

        st.table(balance)


        # ---------------------------------------------------------------------
        # CASH FLOW
        # ---------------------------------------------------------------------

        st.subheader("Cash Flow")

        cashflow = create_table([

            (
                "Operating Cash Flow",
                format_number(
                    info.get("operatingCashflow")
                )
            ),

            (
                "Free Cash Flow",
                format_number(
                    info.get("freeCashflow")
                )
            )

        ])

        st.table(cashflow)


    # =========================================================================
    # RIGHT COLUMN
    # =========================================================================

    with c2:

        st.header("📊 Trading Information")


        # =====================================================================
        # STOCK PRICE HISTORY
        # =====================================================================

        st.subheader("Stock Price History")

        price_history = create_table([

            (
                "Current Price",
                format_number(
                    info.get(
                        "currentPrice",
                        info.get("regularMarketPrice")
                    )
                )
            ),

            (
                "Previous Close",
                format_number(
                    info.get("previousClose")
                )
            ),

            (
                "Open",
                format_number(
                    info.get("open")
                )
            ),

            (
                "Day High",
                format_number(
                    info.get("dayHigh")
                )
            ),

            (
                "Day Low",
                format_number(
                    info.get("dayLow")
                )
            ),

            (
                "52 Week High",
                format_number(
                    info.get("fiftyTwoWeekHigh")
                )
            ),

            (
                "52 Week Low",
                format_number(
                    info.get("fiftyTwoWeekLow")
                )
            ),

            (
                "50 Day Average",
                format_number(
                    info.get("fiftyDayAverage")
                )
            ),

            (
                "200 Day Average",
                format_number(
                    info.get("twoHundredDayAverage")
                )
            )

        ])

        st.table(price_history)


        # =====================================================================
        # SHARE STATISTICS
        # =====================================================================

        st.subheader("Share Statistics")

        shares = create_table([

            (
                "Volume",
                format_number(
                    info.get("volume")
                )
            ),

            (
                "Average Volume",
                format_number(
                    info.get("averageVolume")
                )
            ),

            (
                "Shares Outstanding",
                format_number(
                    info.get("sharesOutstanding")
                )
            ),

            (
                "Float Shares",
                format_number(
                    info.get("floatShares")
                )
            ),

            (
                "% Held by Insiders",
                format_percent(
                    info.get("heldPercentInsiders")
                )
            ),

            (
                "% Held by Institutions",
                format_percent(
                    info.get("heldPercentInstitutions")
                )
            ),

            (
                "Short Ratio",
                format_number(
                    info.get("shortRatio")
                )
            ),

            (
                "Beta",
                format_number(
                    info.get("beta")
                )
            )

        ])

        st.table(shares)


        # =====================================================================
        # DIVIDENDS & SPLITS
        # =====================================================================

        st.subheader("Dividends & Splits")

        dividends = create_table([

            (
                "Dividend Rate",
                format_number(
                    info.get("dividendRate")
                )
            ),

            (
                "Dividend Yield",
                format_percent(
                    info.get("dividendYield")
                )
            ),

            (
                "Payout Ratio",
                format_percent(
                    info.get("payoutRatio")
                )
            ),

            (
                "5 Year Average Dividend Yield",
                format_number(
                    info.get("fiveYearAvgDividendYield")
                )
            ),

            (
                "Last Dividend Value",
                format_number(
                    info.get("lastDividendValue")
                )
            )

        ])

        st.table(dividends)


    # =========================================================================
    # DATA SOURCE
    # =========================================================================

    st.divider()

    st.caption(
        "📌 Source: Yahoo Finance via yfinance | "
        "Company information is cached for 1 hour to reduce API requests."
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
# ANALYSIS DATA - CACHE
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_analysis_data(ticker):
    """
    Get analyst data from Yahoo Finance.
    Cache for 1 hour to reduce repeated requests.
    """

    stock = yf.Ticker(ticker)
    data = {}

    # Analyst Price Targets
    try:
        data["price_targets"] = stock.analyst_price_targets
    except Exception:
        data["price_targets"] = None

    # Recommendations
    try:
        data["recommendations"] = stock.recommendations
    except Exception:
        data["recommendations"] = None

    # Recommendation Summary
    try:
        data["recommendations_summary"] = stock.recommendations_summary
    except Exception:
        data["recommendations_summary"] = None

    # Earnings Estimate
    try:
        data["earnings_estimate"] = stock.earnings_estimate
    except Exception:
        data["earnings_estimate"] = None

    # Revenue Estimate
    try:
        data["revenue_estimate"] = stock.revenue_estimate
    except Exception:
        data["revenue_estimate"] = None

    # EPS Trend
    try:
        data["eps_trend"] = stock.eps_trend
    except Exception:
        data["eps_trend"] = None

    # EPS Revisions
    try:
        data["eps_revisions"] = stock.eps_revisions
    except Exception:
        data["eps_revisions"] = None

    # Growth Estimates
    try:
        data["growth_estimates"] = stock.growth_estimates
    except Exception:
        data["growth_estimates"] = None

    # Upgrades / Downgrades
    try:
        data["upgrades"] = stock.upgrades_downgrades
    except Exception:
        data["upgrades"] = None

    return data


# =============================================================================
# TAB 5 - ANALYSIS
# =============================================================================

def tab5():

    st.title("📈 Analyst Analysis")

    # =========================================================================
    # CHECK TICKER
    # =========================================================================

    if ticker == "-":
        st.info("👈 Please select a ticker from the sidebar")
        return


    # =========================================================================
    # COMPANY INFORMATION
    # =========================================================================

    # Dùng cache chung đã tạo cho Tab 3
    info = get_ticker_info(ticker)

    company_name = info.get("longName", ticker) if info else ticker

    st.subheader(f"🏢 {company_name}")
    st.caption(f"Ticker: {ticker} | Currency: USD")

    st.divider()


    # =========================================================================
    # GET ANALYSIS DATA
    # =========================================================================

    with st.spinner("Loading analyst data..."):
        analysis = get_analysis_data(ticker)


    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def valid_dataframe(df):
        """
        Kiểm tra DataFrame có dữ liệu thực sự hay không.
        """

        if df is None:
            return False

        if not isinstance(df, pd.DataFrame):
            return False

        if df.empty:
            return False

        # Loại bỏ hàng/cột hoàn toàn NaN
        temp = df.dropna(how="all").dropna(axis=1, how="all")

        return not temp.empty


    def clean_dataframe(df):
        """
        Làm sạch DataFrame trước khi hiển thị.
        """

        if not valid_dataframe(df):
            return None

        df = df.copy()

        # Xóa hàng toàn NaN
        df = df.dropna(how="all")

        # Xóa cột toàn NaN
        df = df.dropna(axis=1, how="all")

        # Thay NaN còn lại bằng "-"
        df = df.fillna("-")

        return df


    def format_price(value):
        """
        Format USD.
        """

        try:

            if value is None:
                return "N/A"

            value = float(value)

            if np.isnan(value):
                return "N/A"

            return f"${value:,.2f}"

        except (TypeError, ValueError):
            return "N/A"


    # =========================================================================
    # ANALYST PRICE TARGET
    # =========================================================================

    price_targets = analysis.get("price_targets")

    if isinstance(price_targets, dict) and price_targets:

        st.subheader("🎯 Analyst Price Target")

        current = price_targets.get("current")
        low = price_targets.get("low")
        mean = price_targets.get("mean")
        median = price_targets.get("median")
        high = price_targets.get("high")

        # Chỉ hiển thị metric nào có dữ liệu
        metrics = []

        if current is not None:
            metrics.append(("Current Price", current))

        if low is not None:
            metrics.append(("Target Low", low))

        if mean is not None:
            metrics.append(("Target Mean", mean))

        if median is not None:
            metrics.append(("Target Median", median))

        if high is not None:
            metrics.append(("Target High", high))


        if metrics:

            cols = st.columns(len(metrics))

            for col, (label, value) in zip(cols, metrics):

                col.metric(
                    label,
                    format_price(value)
                )


            # -----------------------------------------------------------------
            # Upside / Downside Potential
            # -----------------------------------------------------------------

            if current and mean:

                try:

                    upside = (
                        (float(mean) - float(current))
                        / float(current)
                    ) * 100

                    if upside > 0:

                        st.success(
                            f"📈 Analyst mean target implies "
                            f"approximately **{upside:.2f}% upside potential** "
                            f"from the current price."
                        )

                    elif upside < 0:

                        st.warning(
                            f"📉 Analyst mean target implies "
                            f"approximately **{abs(upside):.2f}% downside potential** "
                            f"from the current price."
                        )

                    else:

                        st.info(
                            "Analyst mean target is approximately "
                            "equal to the current price."
                        )

                except Exception:
                    pass


        st.divider()


    # =========================================================================
    # RECOMMENDATION SUMMARY
    # =========================================================================

    recommendations_summary = analysis.get(
        "recommendations_summary"
    )

    if valid_dataframe(recommendations_summary):

        st.subheader("⭐ Analyst Recommendations")

        rec = clean_dataframe(
            recommendations_summary
        )

        # -------------------------------------------------------------
        # Show latest recommendation period as KPI
        # -------------------------------------------------------------

        try:

            latest = rec.iloc[0]

            recommendation_columns = [
                "strongBuy",
                "buy",
                "hold",
                "sell",
                "strongSell"
            ]

            available = [
                col
                for col in recommendation_columns
                if col in rec.columns
            ]

            if available:

                cols = st.columns(len(available))

                labels = {
                    "strongBuy": "Strong Buy",
                    "buy": "Buy",
                    "hold": "Hold",
                    "sell": "Sell",
                    "strongSell": "Strong Sell"
                }

                for col_ui, column in zip(cols, available):

                    value = latest[column]

                    col_ui.metric(
                        labels[column],
                        value
                    )

        except Exception:
            pass


        # -------------------------------------------------------------
        # Full recommendation table
        # -------------------------------------------------------------

        with st.expander(
            "📋 View Recommendation History",
            expanded=False
        ):

            st.dataframe(
                rec,
                use_container_width=True
            )


        st.divider()


    # =========================================================================
    # EARNINGS + REVENUE
    # =========================================================================

    earnings = clean_dataframe(
        analysis.get("earnings_estimate")
    )

    revenue = clean_dataframe(
        analysis.get("revenue_estimate")
    )


    if earnings is not None or revenue is not None:

        st.subheader("💰 Earnings & Revenue Estimates")

        c1, c2 = st.columns(2)


        # ---------------------------------------------------------------------
        # Earnings
        # ---------------------------------------------------------------------

        with c1:

            if earnings is not None:

                st.markdown("#### Earnings Estimate")

                st.dataframe(
                    earnings,
                    use_container_width=True
                )

            else:

                st.caption(
                    "Earnings estimates are currently unavailable."
                )


        # ---------------------------------------------------------------------
        # Revenue
        # ---------------------------------------------------------------------

        with c2:

            if revenue is not None:
        
                st.markdown("#### Revenue Estimate")
        
                # Copy để không làm thay đổi dữ liệu gốc
                revenue_display = revenue.copy()
        
                # Đổi tên period cho dễ hiểu
                period_names = {
                    "0q": "Current Quarter",
                    "+1q": "Next Quarter",
                    "0y": "Current Year",
                    "+1y": "Next Year"
                }
        
                # Nếu period đang là index
                revenue_display = revenue_display.reset_index()
        
                # Đổi tên cột đầu thành Period
                revenue_display.rename(
                    columns={
                        revenue_display.columns[0]: "Period",
                        "avg": "Average",
                        "low": "Low",
                        "high": "High",
                        "yearAgoRevenue": "Year Ago",
                        "growth": "Growth"
                    },
                    inplace=True
                )
        
                # Đổi 0q, +1q... thành tên dễ đọc
                revenue_display["Period"] = (
                    revenue_display["Period"]
                    .astype(str)
                    .replace(period_names)
                )
        
                # Các cột tiền → Billion USD
                money_columns = [
                    "Average",
                    "Low",
                    "High",
                    "Year Ago"
                ]
        
                for col in money_columns:
        
                    if col in revenue_display.columns:
        
                        revenue_display[col] = pd.to_numeric(
                            revenue_display[col],
                            errors="coerce"
                        ).apply(
                            lambda x:
                            f"${x / 1_000_000_000:.2f}B"
                            if pd.notna(x)
                            else "-"
                        )
        
                # Growth → %
                if "Growth" in revenue_display.columns:
        
                    revenue_display["Growth"] = pd.to_numeric(
                        revenue_display["Growth"],
                        errors="coerce"
                    ).apply(
                        lambda x:
                        f"{x * 100:.2f}%"
                        if pd.notna(x)
                        else "-"
                    )
        
                st.dataframe(
                    revenue_display,
                    use_container_width=True,
                    hide_index=True
                )


    # =========================================================================
    # EPS TREND
    # =========================================================================

    eps_trend = clean_dataframe(
        analysis.get("eps_trend")
    )

    if eps_trend is not None:

        st.subheader("📊 EPS Trend")

        st.dataframe(
            eps_trend,
            use_container_width=True
        )

        st.divider()


    # =========================================================================
    # EPS REVISIONS
    # =========================================================================

    eps_revisions = clean_dataframe(
        analysis.get("eps_revisions")
    )

    if eps_revisions is not None:

        st.subheader("🔄 EPS Revisions")

        st.dataframe(
            eps_revisions,
            use_container_width=True
        )

        st.divider()


    # =========================================================================
    # GROWTH ESTIMATES
    # =========================================================================

    growth = clean_dataframe(
        analysis.get("growth_estimates")
    )

    if growth is not None:

        st.subheader("🚀 Growth Estimates")

        st.dataframe(
            growth,
            use_container_width=True
        )

        st.divider()


    # =========================================================================
    # UPGRADES / DOWNGRADES
    # =========================================================================

    upgrades = clean_dataframe(
        analysis.get("upgrades")
    )

    if upgrades is not None:

        st.subheader("📢 Analyst Upgrades & Downgrades")

        # Chỉ lấy dữ liệu gần đây để tránh bảng quá dài
        try:
            upgrades_display = upgrades.tail(20)
        except Exception:
            upgrades_display = upgrades


        st.dataframe(
            upgrades_display,
            use_container_width=True
        )

        st.divider()


    # =========================================================================
    # CHECK IF EVERYTHING IS EMPTY
    # =========================================================================

    available_sections = [

        isinstance(
            price_targets,
            dict
        ) and bool(price_targets),

        valid_dataframe(
            recommendations_summary
        ),

        earnings is not None,

        revenue is not None,

        eps_trend is not None,

        eps_revisions is not None,

        growth is not None,

        upgrades is not None

    ]


    if not any(available_sections):

        st.warning(
            "⚠️ Analyst data is currently unavailable for this ticker. "
            "Yahoo Finance may not provide analyst coverage for this asset, "
            "or requests may be temporarily rate-limited."
        )


    # =========================================================================
    # NOTE
    # =========================================================================

    st.caption(
        "📌 Source: Yahoo Finance via yfinance. "
        "Analyst estimates and recommendations may not be available "
        "for all stocks, indices or cryptocurrencies."
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


# ==============================================================================
# AI HELPERS FOR TAB8,9
# ==============================================================================

AI_FEATURES = [
    "Return",
    "Lag1",
    "Lag5",
    "SMA20_Ratio",
    "SMA50_Ratio",
    "MACD_Ratio",
    "RSI14",
    "Momentum10",
    "Volatility20",
    "Volume_Change"
]


# ------------------------------------------------------------------------------
# Flatten yfinance columns
# ------------------------------------------------------------------------------

def flatten_yf_columns(df):

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# ------------------------------------------------------------------------------
# RSI
# ------------------------------------------------------------------------------

def calculate_rsi(close, period=14):

    delta = close.diff()                              # Thay đổi giá

    gain = delta.clip(lower=0)                       # Chỉ giữ phần tăng
    loss = -delta.clip(upper=0)                      # Chỉ giữ phần giảm

    avg_gain = gain.rolling(period).mean()           # Trung bình mức tăng
    avg_loss = loss.rolling(period).mean()           # Trung bình mức giảm

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ------------------------------------------------------------------------------
# Download + Feature Engineering
# ------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def prepare_ai_data(ticker):

    df = yf.download(
        ticker,
        period="10y",                                 # Dùng dữ liệu dài hơn cho AI
        auto_adjust=False,
        progress=False
    )

    df = flatten_yf_columns(df)

    if df.empty:
        raise ValueError(f"No historical data available for {ticker}.")

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required_columns:

        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.copy()

    # ======================================================================
    # BASIC RETURNS
    # ======================================================================

    df["Return"] = df["Close"].pct_change()            # Daily return

    df["Lag1"] = df["Return"].shift(1)                 # Return t-1
    df["Lag5"] = df["Return"].shift(5)                 # Return t-5


    # ======================================================================
    # MOVING AVERAGES
    # ======================================================================

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # Dùng ratio thay vì raw price để model ổn định hơn
    df["SMA20_Ratio"] = df["Close"] / df["SMA20"] - 1
    df["SMA50_Ratio"] = df["Close"] / df["SMA50"] - 1


    # ======================================================================
    # MACD
    # ======================================================================

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Ratio"] = (
        df["MACD"] /
        df["Close"].replace(0, np.nan)
    )


    # ======================================================================
    # RSI
    # ======================================================================

    df["RSI14"] = calculate_rsi(
        df["Close"],
        period=14
    )


    # ======================================================================
    # MOMENTUM
    # ======================================================================

    df["Momentum10"] = (
        df["Close"].pct_change(10)
    )


    # ======================================================================
    # VOLATILITY
    # ======================================================================

    df["Volatility20"] = (
        df["Return"]
        .rolling(20)
        .std()
    )


    # ======================================================================
    # VOLUME
    # ======================================================================

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )


    # ======================================================================
    # TARGET
    # ======================================================================

    # Dùng dữ liệu ngày t dự báo return từ t → t+1
    df["Target"] = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1
    )


    # ======================================================================
    # FEATURE DATA
    # ======================================================================

    feature_data = df[
        AI_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_data = feature_data.dropna()


    # Model dataset phải có Target
    model_data = df[
        AI_FEATURES + ["Target"]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    model_data = model_data.dropna()


    if len(model_data) < 300:

        raise ValueError(
            "Not enough observations for AI training."
        )


    # Latest feature row dùng để dự báo ngày tiếp theo
    latest_features = feature_data.iloc[-1:].copy()

    latest_date = latest_features.index[-1]

    current_price = float(
        df.loc[latest_date, "Close"]
    )


    return (
        df,
        model_data,
        feature_data,
        latest_features,
        current_price
    )


# ------------------------------------------------------------------------------
# Train / Validation / Test Split
# ------------------------------------------------------------------------------

def get_split_points(data):

    n = len(data)

    train_end = int(n * 0.60)                         # 60% train
    valid_end = int(n * 0.80)                         # 20% validation

    return train_end, valid_end


# ------------------------------------------------------------------------------
# Create sequence for LSTM / GRU
# ------------------------------------------------------------------------------

def create_sequences(
    X,
    y,
    sequence_length,
    start_target,
    end_target
):

    sequences = []
    targets = []
    positions = []

    for target_pos in range(
        start_target,
        end_target
    ):

        start_pos = (
            target_pos
            - sequence_length
            + 1
        )

        if start_pos < 0:
            continue

        sequences.append(
            X[
                start_pos:
                target_pos + 1
            ]
        )

        targets.append(
            y[target_pos]
        )

        positions.append(
            target_pos
        )

    return (
        np.asarray(sequences),
        np.asarray(targets),
        np.asarray(positions)
    )


# ------------------------------------------------------------------------------
# Build Deep Learning Model
# ------------------------------------------------------------------------------

def build_deep_model(
    model_name,
    sequence_length,
    n_features
):

    tf.keras.backend.clear_session()

    tf.random.set_seed(42)
    np.random.seed(42)


    # ======================================================================
    # LSTM
    # ======================================================================

    if model_name == "LSTM":

        model = Sequential([
            Input(
                shape=(
                    sequence_length,
                    n_features
                )
            ),

            LSTM(
                64
            ),

            Dropout(
                0.20
            ),

            Dense(
                32,
                activation="relu"
            ),

            Dense(
                1
            )
        ])


    # ======================================================================
    # GRU
    # ======================================================================

    elif model_name == "GRU":

        model = Sequential([
            Input(
                shape=(
                    sequence_length,
                    n_features
                )
            ),

            GRU(
                64
            ),

            Dropout(
                0.20
            ),

            Dense(
                32,
                activation="relu"
            ),

            Dense(
                1
            )
        ])


    # ======================================================================
    # ATTENTION-GRU
    # ======================================================================

    elif model_name == "Attention-GRU":

        inputs = Input(
            shape=(
                sequence_length,
                n_features
            )
        )

        x = GRU(
            64,
            return_sequences=True
        )(inputs)

        attention_output = Attention()(
            [x, x]
        )

        x = GlobalAveragePooling1D()(
            attention_output
        )

        x = Dropout(
            0.20
        )(x)

        x = Dense(
            32,
            activation="relu"
        )(x)

        outputs = Dense(
            1
        )(x)

        model = Model(
            inputs,
            outputs
        )


    else:

        raise ValueError(
            f"Unknown deep model: {model_name}"
        )


    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# ------------------------------------------------------------------------------
# Train AI Model
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def train_ai_model(
    ticker,
    model_name,
    sequence_length=30
):

    (
        raw_df,
        data,
        feature_data,
        latest_features,
        current_price
    ) = prepare_ai_data(ticker)


    X_df = data[AI_FEATURES]
    y_series = data["Target"]

    train_end, valid_end = get_split_points(
        data
    )


    # ======================================================================
    # SCALER
    # ======================================================================

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit ONLY Train data → tránh data leakage
    scaler.fit(
        X_df.iloc[:train_end]
    )

    X_scaled = scaler.transform(
        X_df
    )

    y = y_series.values


    # ======================================================================
    # LINEAR REGRESSION / XGBOOST
    # ======================================================================

    if model_name in [
        "Linear Regression",
        "XGBoost"
    ]:

        X_train = X_scaled[:train_end]
        y_train = y[:train_end]

        X_valid = X_scaled[
            train_end:
            valid_end
        ]

        y_valid = y[
            train_end:
            valid_end
        ]

        X_test = X_scaled[
            valid_end:
        ]

        y_test = y[
            valid_end:
        ]


        # ------------------------------------------------------------------
        # Linear Regression
        # ------------------------------------------------------------------

        if model_name == "Linear Regression":

            model = LinearRegression()

            model.fit(
                X_train,
                y_train
            )


        # ------------------------------------------------------------------
        # XGBoost
        # ------------------------------------------------------------------

        else:

            model = XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.80,
                colsample_bytree=0.80,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train,
                y_train
            )


        valid_pred = model.predict(
            X_valid
        )

        test_pred = model.predict(
            X_test
        )


        valid_index = data.index[
            train_end:
            valid_end
        ]

        test_index = data.index[
            valid_end:
        ]


        # Latest prediction
        latest_scaled = scaler.transform(
            latest_features[
                AI_FEATURES
            ]
        )

        latest_prediction = float(
            model.predict(
                latest_scaled
            )[0]
        )


        # Feature importance
        if model_name == "XGBoost":

            feature_importance = pd.Series(
                model.feature_importances_,
                index=AI_FEATURES
            ).sort_values(
                ascending=False
            )

        else:

            feature_importance = pd.Series(
                np.abs(model.coef_),
                index=AI_FEATURES
            ).sort_values(
                ascending=False
            )


    # ======================================================================
    # LSTM / GRU / ATTENTION-GRU
    # ======================================================================

    else:

        # Train sequences
        X_train, y_train, train_pos = create_sequences(
            X_scaled,
            y,
            sequence_length,
            sequence_length - 1,
            train_end
        )


        # Validation may use past Train rows as historical context
        X_valid, y_valid, valid_pos = create_sequences(
            X_scaled,
            y,
            sequence_length,
            train_end,
            valid_end
        )


        # Test may use previous Validation rows as context
        X_test, y_test, test_pos = create_sequences(
            X_scaled,
            y,
            sequence_length,
            valid_end,
            len(data)
        )


        if (
            len(X_train) == 0
            or len(X_valid) == 0
            or len(X_test) == 0
        ):

            raise ValueError(
                "Not enough sequence data."
            )


        model = build_deep_model(
            model_name,
            sequence_length,
            len(AI_FEATURES)
        )


        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )


        model.fit(
            X_train,
            y_train,
            validation_data=(
                X_valid,
                y_valid
            ),
            epochs=50,
            batch_size=32,
            verbose=0,
            shuffle=False,
            callbacks=[
                early_stop
            ]
        )


        valid_pred = (
            model.predict(
                X_valid,
                verbose=0
            )
            .reshape(-1)
        )

        test_pred = (
            model.predict(
                X_test,
                verbose=0
            )
            .reshape(-1)
        )


        valid_index = data.index[
            valid_pos
        ]

        test_index = data.index[
            test_pos
        ]


        # ------------------------------------------------------------------
        # Latest sequence
        # ------------------------------------------------------------------

        latest_full_scaled = scaler.transform(
            feature_data[
                AI_FEATURES
            ]
        )

        latest_sequence = (
            latest_full_scaled[
                -sequence_length:
            ]
        )

        latest_sequence = np.expand_dims(
            latest_sequence,
            axis=0
        )

        latest_prediction = float(
            model.predict(
                latest_sequence,
                verbose=0
            )[0][0]
        )


        feature_importance = None


    # ======================================================================
    # VALIDATION DATAFRAME
    # ======================================================================

    valid_df = pd.DataFrame(
        {
            "Actual": y_valid,
            "Predicted": valid_pred
        },
        index=valid_index
    )


    # ======================================================================
    # TEST DATAFRAME
    # ======================================================================

    test_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": test_pred
        },
        index=test_index
    )


    # ======================================================================
    # MODEL METRICS
    # ======================================================================

    mae = mean_absolute_error(
        test_df["Actual"],
        test_df["Predicted"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            test_df["Actual"],
            test_df["Predicted"]
        )
    )

    directional_accuracy = (
        np.sign(
            test_df["Actual"]
        )
        ==
        np.sign(
            test_df["Predicted"]
        )
    ).mean()


    return {
        "model": model,
        "scaler": scaler,

        "data": data,
        "raw": raw_df,

        "valid": valid_df,
        "test": test_df,

        "latest_prediction":
            latest_prediction,

        "current_price":
            current_price,

        "mae":
            mae,

        "rmse":
            rmse,

        "directional_accuracy":
            directional_accuracy,

        "feature_importance":
            feature_importance,

        "train_end_date":
            data.index[train_end - 1],

        "validation_end_date":
            data.index[valid_end - 1],

        "test_start_date":
            data.index[valid_end]
    }


# ------------------------------------------------------------------------------
# Generate Trading Signal
# ------------------------------------------------------------------------------

def create_signal(
    prediction,
    threshold
):

    signal = np.where(
        prediction > threshold,
        1,
        np.where(
            prediction < -threshold,
            -1,
            0
        )
    )

    return signal


# ------------------------------------------------------------------------------
# Strategy Returns
# ------------------------------------------------------------------------------

def calculate_strategy_returns(
    actual_returns,
    predictions,
    threshold,
    transaction_cost=0.001
):

    actual_returns = np.asarray(
        actual_returns,
        dtype=float
    )

    predictions = np.asarray(
        predictions,
        dtype=float
    )


    signal = create_signal(
        predictions,
        threshold
    )


    # Signal tại t được dùng cho Target return t → t+1
    gross_return = (
        signal
        *
        actual_returns
    )


    # Turnover
    turnover = np.abs(
        np.diff(
            signal,
            prepend=0
        )
    )


    cost = (
        turnover
        *
        transaction_cost
    )


    net_return = (
        gross_return
        -
        cost
    )


    return (
        signal,
        net_return
    )


# ------------------------------------------------------------------------------
# Backtest Metrics
# ------------------------------------------------------------------------------

def investment_metrics(
    returns
):

    returns = pd.Series(
        returns
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()


    if returns.empty:

        return {
            "Cumulative Return": np.nan,
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "Win Rate": np.nan
        }


    equity = (
        1 + returns
    ).cumprod()


    cumulative_return = (
        equity.iloc[-1]
        - 1
    )


    n = len(returns)

    years = (
        n / 252
    )


    if (
        years > 0
        and equity.iloc[-1] > 0
    ):

        cagr = (
            equity.iloc[-1]
            ** (1 / years)
            - 1
        )

    else:

        cagr = np.nan


    volatility = (
        returns.std()
        *
        np.sqrt(252)
    )


    if returns.std() > 0:

        sharpe = (
            returns.mean()
            /
            returns.std()
            *
            np.sqrt(252)
        )

    else:

        sharpe = np.nan


    running_max = equity.cummax()

    drawdown = (
        equity
        /
        running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )


    win_rate = (
        returns > 0
    ).mean()


    return {
        "Cumulative Return":
            cumulative_return,

        "CAGR":
            cagr,

        "Volatility":
            volatility,

        "Sharpe Ratio":
            sharpe,

        "Max Drawdown":
            max_drawdown,

        "Win Rate":
            win_rate
    }


# ------------------------------------------------------------------------------
# Choose threshold using VALIDATION ONLY
# ------------------------------------------------------------------------------

def choose_threshold(
    valid_df,
    transaction_cost=0.001
):

    thresholds = np.linspace(
        0,
        0.01,
        21
    )


    best_threshold = 0
    best_sharpe = -np.inf


    for threshold in thresholds:

        _, strategy_return = (
            calculate_strategy_returns(
                valid_df["Actual"],
                valid_df["Predicted"],
                threshold,
                transaction_cost
            )
        )

        metrics = investment_metrics(
            strategy_return
        )

        sharpe = metrics[
            "Sharpe Ratio"
        ]


        if (
            pd.notna(sharpe)
            and sharpe > best_sharpe
        ):

            best_sharpe = sharpe
            best_threshold = threshold


    return (
        best_threshold,
        best_sharpe
    )


# ------------------------------------------------------------------------------
# AI Backtest
# ------------------------------------------------------------------------------

def backtest_ai_model(
    result,
    transaction_cost=0.001
):

    # Threshold chỉ được chọn trên Validation
    threshold, valid_sharpe = choose_threshold(
        result["valid"],
        transaction_cost
    )


    test_df = result["test"].copy()


    signal, strategy_return = (
        calculate_strategy_returns(
            test_df["Actual"],
            test_df["Predicted"],
            threshold,
            transaction_cost
        )
    )


    test_df["Signal"] = signal

    test_df[
        "Strategy_Return"
    ] = strategy_return


    test_df[
        "Equity"
    ] = (
        1
        +
        test_df["Strategy_Return"]
    ).cumprod()


    metrics = investment_metrics(
        test_df["Strategy_Return"]
    )


    return (
        test_df,
        metrics,
        threshold,
        valid_sharpe
    )


# ------------------------------------------------------------------------------
# SMA Baseline
# ------------------------------------------------------------------------------

def backtest_sma_strategy(
    ticker,
    test_start_date,
    transaction_cost=0.001
):

    df = yf.download(
        ticker,
        period="10y",
        auto_adjust=False,
        progress=False
    )

    df = flatten_yf_columns(
        df
    )

    df["Return"] = (
        df["Close"]
        .pct_change()
    )

    df["SMA20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )


    # Long / Flat rule
    df["Signal"] = np.where(
        df["SMA20"] > df["SMA50"],
        1,
        0
    )


    # Signal ở t sử dụng cho return t → t+1
    df["Forward_Return"] = (
        df["Close"].shift(-1)
        /
        df["Close"]
        - 1
    )


    df = df[
        df.index >= test_start_date
    ].dropna()


    turnover = (
        df["Signal"]
        .diff()
        .abs()
        .fillna(
            df["Signal"].abs()
        )
    )


    df["Strategy_Return"] = (
        df["Signal"]
        *
        df["Forward_Return"]
        -
        turnover
        *
        transaction_cost
    )


    df["Equity"] = (
        1
        +
        df["Strategy_Return"]
    ).cumprod()


    metrics = investment_metrics(
        df["Strategy_Return"]
    )


    return (
        df,
        metrics
    )


# ------------------------------------------------------------------------------
# Buy & Hold
# ------------------------------------------------------------------------------

def backtest_buy_hold(
    test_returns
):

    returns = pd.Series(
        test_returns
    ).dropna()

    equity = (
        1 + returns
    ).cumprod()


    metrics = investment_metrics(
        returns
    )


    return (
        equity,
        metrics
    )



# ==============================================================================
# TAB 8 - AI PREDICTION
# ==============================================================================

def tab8():

    st.title("🤖 AI Stock Prediction")


    # =========================================================================
    # CHECK TICKER
    # =========================================================================

    if ticker == "-":

        st.info(
            "👈 Please select a ticker from the sidebar."
        )

        return


    # =========================================================================
    # COMPANY INFORMATION
    # =========================================================================

    try:

        info = yf.Ticker(
            ticker
        ).get_info()

        company_name = info.get(
            "longName",
            ticker
        )

    except Exception:

        company_name = ticker


    st.subheader(
        f"🏢 {company_name}"
    )

    st.caption(
        f"Ticker: {ticker}"
    )

    st.divider()


    # =========================================================================
    # MODEL SETTINGS
    # =========================================================================

    st.subheader(
        "⚙️ AI Model Settings"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        model_name = st.selectbox(
            "Select Model",
            [
                "Linear Regression",
                "XGBoost",
                "LSTM",
                "Attention-GRU"
            ],
            index=3
        )


    with c2:

        sequence_length = st.selectbox(
            "Sequence Length",
            [
                10,
                20,
                30,
                60
            ],
            index=2
        )


    with c3:

        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            [
                "1 Trading Day"
            ]
        )


    # =========================================================================
    # TRAIN / LOAD MODEL
    # =========================================================================

    try:

        with st.spinner(
            f"Running {model_name}..."
        ):

            result = train_ai_model(
                ticker,
                model_name,
                sequence_length
            )


    except Exception as e:

        st.error(
            "Unable to train or run the AI model."
        )

        st.code(
            str(e)
        )

        return


    # =========================================================================
    # DATA SPLIT
    # =========================================================================

    st.subheader(
        "🗂️ Time-Series Data Split"
    )


    s1, s2, s3 = st.columns(3)


    s1.metric(
        "Training End",
        result[
            "train_end_date"
        ].strftime(
            "%Y-%m-%d"
        )
    )


    s2.metric(
        "Validation End",
        result[
            "validation_end_date"
        ].strftime(
            "%Y-%m-%d"
        )
    )


    s3.metric(
        "Test Start",
        result[
            "test_start_date"
        ].strftime(
            "%Y-%m-%d"
        )
    )


    st.caption(
        "Data are split chronologically. "
        "The scaler is fitted only on the Training Set."
    )


    # =========================================================================
    # LATEST FEATURES
    # =========================================================================

    st.divider()

    st.subheader(
        "📊 Latest Input Features"
    )


    latest_row = (
        prepare_ai_data(
            ticker
        )[2]
        .iloc[-1]
    )


    f1, f2, f3, f4 = st.columns(4)


    f1.metric(
        "Daily Return",
        f"{latest_row['Return'] * 100:.2f}%"
    )


    f2.metric(
        "RSI 14",
        f"{latest_row['RSI14']:.2f}"
    )


    f3.metric(
        "Momentum 10",
        f"{latest_row['Momentum10'] * 100:.2f}%"
    )


    f4.metric(
        "Volatility 20",
        f"{latest_row['Volatility20'] * 100:.2f}%"
    )


    # =========================================================================
    # PREDICTION
    # =========================================================================

    st.divider()

    st.subheader(
        "🎯 AI Prediction"
    )


    predicted_return = result[
        "latest_prediction"
    ]


    current_price = result[
        "current_price"
    ]


    predicted_price = (
        current_price
        *
        (
            1
            +
            predicted_return
        )
    )


    # Threshold được chọn trên Validation Set
    best_threshold, valid_sharpe = (
        choose_threshold(
            result["valid"],
            transaction_cost=0.001
        )
    )


    if predicted_return > best_threshold:

        signal = "🟢 BUY"

    elif predicted_return < -best_threshold:

        signal = "🔴 SELL"

    else:

        signal = "🟡 HOLD"


    p1, p2, p3, p4 = st.columns(4)


    p1.metric(
        "Current Price",
        f"${current_price:,.2f}"
    )


    p2.metric(
        "Predicted Return",
        f"{predicted_return * 100:+.3f}%"
    )


    p3.metric(
        "Estimated Next Price",
        f"${predicted_price:,.2f}"
    )


    p4.metric(
        "AI Signal",
        signal
    )


    st.caption(
        f"Trading threshold selected on Validation Set: "
        f"{best_threshold * 100:.2f}%"
    )


    # =========================================================================
    # MODEL PERFORMANCE
    # =========================================================================

    st.divider()

    st.subheader(
        "📏 Test Set Prediction Metrics"
    )


    m1, m2, m3 = st.columns(3)


    m1.metric(
        "MAE",
        f"{result['mae']:.6f}"
    )


    m2.metric(
        "RMSE",
        f"{result['rmse']:.6f}"
    )


    m3.metric(
        "Directional Accuracy",
        f"{result['directional_accuracy'] * 100:.2f}%"
    )


    # =========================================================================
    # ACTUAL VS PREDICTED
    # =========================================================================

    st.subheader(
        "📈 Actual vs Predicted Return"
    )


    plot_data = (
        result["test"]
        .tail(150)
        .reset_index()
    )


    date_column = (
        plot_data.columns[0]
    )


    plot_long = plot_data.melt(
        id_vars=[
            date_column
        ],
        value_vars=[
            "Actual",
            "Predicted"
        ],
        var_name="Series",
        value_name="Return"
    )


    fig = px.line(
        plot_long,
        x=date_column,
        y="Return",
        color="Series",
        title=(
            f"{model_name} - "
            "Actual vs Predicted Return"
        )
    )


    fig.update_layout(
        hovermode="x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    if result[
        "feature_importance"
    ] is not None:

        st.subheader(
            "🔎 Feature Importance"
        )


        importance = (
            result[
                "feature_importance"
            ]
            .reset_index()
        )


        importance.columns = [
            "Feature",
            "Importance"
        ]


        fig_imp = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                f"{model_name} Feature Importance"
            )
        )


        st.plotly_chart(
            fig_imp,
            use_container_width=True
        )


# =============================================================================
# ABLATION STUDY for tab 9
# =============================================================================

def run_ablation_study(
    ticker,
    sequence_length,
    transaction_cost
):

    experiments = [
        {
            "name": "A",
            "description": "GRU + Basic Features",
            "technical": False,
            "attention": False
        },
        {
            "name": "A + B",
            "description": "GRU + Technical Indicators",
            "technical": True,
            "attention": False
        },
        {
            "name": "A + B + C",
            "description": "Attention-GRU",
            "technical": True,
            "attention": True
        }
    ]

    results = []

    for exp in experiments:

        # ---------------------------------------------------------------------
        # Train model for each Ablation configuration
        # ---------------------------------------------------------------------

        valid_df, test_df = train_gru_ablation(
            ticker=ticker,
            sequence_length=sequence_length,
            use_technical=exp["technical"],
            use_attention=exp["attention"]
        )

        # ---------------------------------------------------------------------
        # Check data
        # ---------------------------------------------------------------------

        if valid_df is None or test_df is None:
            continue

        if valid_df.empty or test_df.empty:
            continue

        if (
            "Actual" not in valid_df.columns
            or "Predicted" not in valid_df.columns
            or "Actual" not in test_df.columns
            or "Predicted" not in test_df.columns
        ):
            continue

        # ---------------------------------------------------------------------
        # Select threshold using VALIDATION SET only
        # ---------------------------------------------------------------------

        threshold, validation_sharpe = choose_threshold(
            valid_df,
            transaction_cost
        )

        # ---------------------------------------------------------------------
        # Backtest TEST SET
        # ---------------------------------------------------------------------

        signal, strategy_return = calculate_strategy_returns(
            test_df["Actual"],
            test_df["Predicted"],
            threshold,
            transaction_cost
        )

        # ---------------------------------------------------------------------
        # Investment metrics
        # ---------------------------------------------------------------------

        metrics = investment_metrics(
            strategy_return
        )

        # ---------------------------------------------------------------------
        # Prediction metrics
        # ---------------------------------------------------------------------

        actual = np.asarray(
            test_df["Actual"],
            dtype=float
        )

        predicted = np.asarray(
            test_df["Predicted"],
            dtype=float
        )

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        directional_accuracy = np.mean(
            np.sign(actual)
            ==
            np.sign(predicted)
        )

        # ---------------------------------------------------------------------
        # Save result
        # ---------------------------------------------------------------------

        results.append({
            "Experiment": exp["name"],
            "Model": exp["description"],
            "MAE": mae,
            "RMSE": rmse,
            "Directional Accuracy": directional_accuracy,
            "Threshold": threshold,
            "Validation Sharpe": validation_sharpe,
            "Cumulative Return": metrics.get(
                "Cumulative Return",
                np.nan
            ),
            "Sharpe Ratio": metrics.get(
                "Sharpe Ratio",
                np.nan
            ),
            "Max Drawdown": metrics.get(
                "Max Drawdown",
                np.nan
            )
        })

    return pd.DataFrame(results)



# ==============================================================================
# TAB 9 - AI BACKTESTING
# ==============================================================================

def tab9():

    st.title("📈 AI Strategy Backtesting")


    # =========================================================================
    # CHECK TICKER
    # =========================================================================

    if ticker == "-":

        st.info(
            "👈 Please select a ticker from the sidebar."
        )

        return


    # =========================================================================
    # SETTINGS
    # =========================================================================

    st.subheader("⚙️ Backtesting Settings")

    c1, c2, c3 = st.columns(3)


    # -------------------------------------------------------------------------
    # Strategy selection
    # -------------------------------------------------------------------------

    with c1:

        selected_models = st.multiselect(
            "Select Strategies",
            [
                "Buy & Hold",
                "SMA Strategy",
                "Linear Regression",
                "XGBoost",
                "LSTM",
                "Attention-GRU"
            ],
            default=[
                "Buy & Hold",
                "XGBoost",
                "Attention-GRU"
            ]
        )


    # -------------------------------------------------------------------------
    # Transaction cost
    # -------------------------------------------------------------------------

    with c2:

        transaction_cost_pct = st.number_input(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.10,
            step=0.05
        )


    # -------------------------------------------------------------------------
    # Sequence length
    # -------------------------------------------------------------------------

    with c3:

        sequence_length = st.selectbox(
            "Sequence Length",
            [
                10,
                20,
                30,
                60
            ],
            index=2,
            key="backtest_sequence"
        )


    # -------------------------------------------------------------------------
    # Check strategy selection
    # -------------------------------------------------------------------------

    if not selected_models:

        st.warning(
            "Please select at least one strategy."
        )

        return


    # Convert % → decimal

    transaction_cost = (
        transaction_cost_pct
        /
        100
    )


    # =========================================================================
    # EXPLANATION
    # =========================================================================

    st.info(
        """
        **Backtesting principle**

        • Train / Validation được sử dụng để huấn luyện và lựa chọn mô hình  
        • Trading threshold được lựa chọn trên Validation Set  
        • Test Set chỉ được sử dụng để đánh giá cuối cùng  
        • Transaction Cost được tính vào Strategy Return  
        • Không điều chỉnh mô hình dựa trên kết quả Test Set
        """
    )


    # =========================================================================
    # RESULT CONTAINERS
    # =========================================================================

    performance_rows = []

    equity_curves = {}

    ai_results = {}


    # =========================================================================
    # REFERENCE MODEL
    # =========================================================================
    #
    # Dùng Linear Regression để xác định cùng Test Period
    # cho tất cả các chiến lược
    # =========================================================================

    try:

        with st.spinner(
            "Preparing Test Set..."
        ):

            reference = train_ai_model(
                ticker,
                "Linear Regression",
                sequence_length
            )


    except Exception as e:

        st.error(
            "Unable to prepare Test Set."
        )

        st.code(
            str(e)
        )

        return


    test_start_date = reference[
        "test_start_date"
    ]


    # =========================================================================
    # TEST PERIOD
    # =========================================================================

    st.subheader(
        "🗂️ Out-of-Sample Test Period"
    )

    t1, t2, t3 = st.columns(3)


    t1.metric(
        "Training End",
        reference[
            "train_end_date"
        ].strftime(
            "%Y-%m-%d"
        )
    )


    t2.metric(
        "Validation End",
        reference[
            "validation_end_date"
        ].strftime(
            "%Y-%m-%d"
        )
    )


    t3.metric(
        "Test Start",
        reference[
            "test_start_date"
        ].strftime(
            "%Y-%m-%d"
        )
    )


    st.caption(
        "Dữ liệu được chia theo thời gian để hạn chế Data Leakage."
    )


    # =========================================================================
    # BUY & HOLD
    # =========================================================================

    if "Buy & Hold" in selected_models:

        try:

            test_returns = (
                reference[
                    "data"
                ]
                .loc[
                    reference[
                        "test"
                    ].index,
                    "Target"
                ]
            )


            (
                equity,
                metrics
            ) = backtest_buy_hold(
                test_returns
            )


            equity.index = (
                test_returns.index
            )


            equity_curves[
                "Buy & Hold"
            ] = equity


            performance_rows.append(
                {
                    "Strategy":
                        "Buy & Hold",

                    **metrics
                }
            )


        except Exception as e:

            st.warning(
                f"Buy & Hold failed: {e}"
            )


    # =========================================================================
    # SMA STRATEGY
    # =========================================================================

    if "SMA Strategy" in selected_models:

        try:

            (
                sma_df,
                sma_metrics
            ) = backtest_sma_strategy(
                ticker,
                test_start_date,
                transaction_cost
            )


            equity_curves[
                "SMA Strategy"
            ] = sma_df[
                "Equity"
            ]


            performance_rows.append(
                {
                    "Strategy":
                        "SMA Strategy",

                    **sma_metrics
                }
            )


        except Exception as e:

            st.warning(
                f"SMA Strategy failed: {e}"
            )


    # =========================================================================
    # AI MODELS
    # =========================================================================

    ai_model_names = [
        "Linear Regression",
        "XGBoost",
        "LSTM",
        "Attention-GRU"
    ]


    for model_name in ai_model_names:

        if model_name not in selected_models:
            continue


        try:

            with st.spinner(
                f"Backtesting {model_name}..."
            ):

                # -------------------------------------------------------------
                # Train model
                # -------------------------------------------------------------

                result = train_ai_model(
                    ticker,
                    model_name,
                    sequence_length
                )


                # -------------------------------------------------------------
                # Backtest
                # -------------------------------------------------------------

                (
                    backtest_df,
                    metrics,
                    threshold,
                    validation_sharpe
                ) = backtest_ai_model(
                    result,
                    transaction_cost
                )


            # -------------------------------------------------------------
            # Save result
            # -------------------------------------------------------------

            ai_results[
                model_name
            ] = {

                "result":
                    result,

                "backtest":
                    backtest_df,

                "threshold":
                    threshold,

                "validation_sharpe":
                    validation_sharpe
            }


            # -------------------------------------------------------------
            # Equity
            # -------------------------------------------------------------

            equity_curves[
                model_name
            ] = backtest_df[
                "Equity"
            ]


            # -------------------------------------------------------------
            # Performance row
            # -------------------------------------------------------------

            performance_rows.append(
                {

                    "Strategy":
                        model_name,

                    **metrics,

                    "Threshold":
                        threshold
                }
            )


        except Exception as e:

            st.warning(
                f"{model_name} failed: {e}"
            )


    # =========================================================================
    # CHECK RESULT
    # =========================================================================

    if not performance_rows:

        st.warning(
            "No backtesting results available."
        )

        return


    # =========================================================================
    # PERFORMANCE TABLE
    # =========================================================================

    results_df = pd.DataFrame(
        performance_rows
    )


    st.divider()

    st.subheader(
        "🏆 Test Set Performance"
    )


    display_results = (
        results_df.copy()
    )


    # -------------------------------------------------------------------------
    # Format percentage columns
    # -------------------------------------------------------------------------

    percentage_columns = [
        "Cumulative Return",
        "CAGR",
        "Volatility",
        "Max Drawdown",
        "Win Rate"
    ]


    for col in percentage_columns:

        if col in display_results.columns:

            display_results[
                col
            ] = (
                display_results[
                    col
                ]
                .apply(
                    lambda x:
                    f"{x * 100:.2f}%"
                    if pd.notna(x)
                    else "-"
                )
            )


    # -------------------------------------------------------------------------
    # Sharpe
    # -------------------------------------------------------------------------

    if "Sharpe Ratio" in display_results.columns:

        display_results[
            "Sharpe Ratio"
        ] = (
            display_results[
                "Sharpe Ratio"
            ]
            .apply(
                lambda x:
                f"{x:.2f}"
                if pd.notna(x)
                else "-"
            )
        )


    # -------------------------------------------------------------------------
    # Threshold
    # -------------------------------------------------------------------------

    if "Threshold" in display_results.columns:

        display_results[
            "Threshold"
        ] = (
            display_results[
                "Threshold"
            ]
            .apply(
                lambda x:
                f"{x * 100:.2f}%"
                if pd.notna(x)
                else "-"
            )
        )


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    # =========================================================================
    # BEST STRATEGY
    # =========================================================================

    valid_sharpe_df = (
        results_df
        .dropna(
            subset=[
                "Sharpe Ratio"
            ]
        )
    )


    if not valid_sharpe_df.empty:

        best_row = (
            valid_sharpe_df
            .sort_values(
                "Sharpe Ratio",
                ascending=False
            )
            .iloc[0]
        )


        st.subheader(
            "🥇 Best Strategy"
        )


        b1, b2, b3 = st.columns(3)


        b1.metric(
            "Strategy",
            best_row[
                "Strategy"
            ]
        )


        b2.metric(
            "Sharpe Ratio",
            f"{best_row['Sharpe Ratio']:.2f}"
        )


        b3.metric(
            "Target",
            "≥ 1.80"
        )


        if best_row[
            "Sharpe Ratio"
        ] >= 1.8:

            st.success(
                "✅ Sharpe Ratio requirement ≥ 1.8 "
                "was achieved on the Test Set."
            )

        else:

            st.warning(
                "⚠️ Sharpe Ratio on the Test Set "
                "is below the target of 1.8."
            )


    # =========================================================================
    # EQUITY CURVE
    # =========================================================================

    st.divider()

    st.subheader(
        "📈 Equity Curve"
    )


    if equity_curves:

        equity_df = pd.DataFrame(
            equity_curves
        )


        equity_df = (
            equity_df
            .sort_index()
            .ffill()
        )


        fig = px.line(
            equity_df,
            x=equity_df.index,
            y=equity_df.columns,
            title=(
                "Out-of-Sample Test Set "
                "Equity Curve"
            )
        )


        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            legend_title="Strategy"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    else:

        st.info(
            "No equity curves available."
        )


    # =========================================================================
    # AI PREDICTION METRICS
    # =========================================================================

    if ai_results:

        st.divider()

        st.subheader(
            "🤖 AI Prediction Metrics"
        )


        prediction_rows = []


        for (
            model_name,
            item
        ) in ai_results.items():

            model_result = item[
                "result"
            ]


            prediction_rows.append(
                {

                    "Model":
                        model_name,

                    "MAE":
                        model_result[
                            "mae"
                        ],

                    "RMSE":
                        model_result[
                            "rmse"
                        ],

                    "Directional Accuracy":
                        model_result[
                            "directional_accuracy"
                        ],

                    "Validation Sharpe":
                        item[
                            "validation_sharpe"
                        ],

                    "Trading Threshold":
                        item[
                            "threshold"
                        ]
                }
            )


        pred_metrics = pd.DataFrame(
            prediction_rows
        )


        # ---------------------------------------------------------------------
        # Format
        # ---------------------------------------------------------------------

        pred_metrics[
            "MAE"
        ] = (
            pred_metrics[
                "MAE"
            ]
            .map(
                lambda x:
                f"{x:.6f}"
            )
        )


        pred_metrics[
            "RMSE"
        ] = (
            pred_metrics[
                "RMSE"
            ]
            .map(
                lambda x:
                f"{x:.6f}"
            )
        )


        pred_metrics[
            "Directional Accuracy"
        ] = (
            pred_metrics[
                "Directional Accuracy"
            ]
            .map(
                lambda x:
                f"{x * 100:.2f}%"
            )
        )


        pred_metrics[
            "Validation Sharpe"
        ] = (
            pred_metrics[
                "Validation Sharpe"
            ]
            .map(
                lambda x:
                f"{x:.2f}"
                if pd.notna(x)
                else "-"
            )
        )


        pred_metrics[
            "Trading Threshold"
        ] = (
            pred_metrics[
                "Trading Threshold"
            ]
            .map(
                lambda x:
                f"{x * 100:.2f}%"
            )
        )


        st.dataframe(
            pred_metrics,
            use_container_width=True,
            hide_index=True
        )


    # =========================================================================
    # ABLATION STUDY
    # =========================================================================

    st.divider()

    st.subheader(
        "🧪 Ablation Study"
    )


    st.caption(
        "Đánh giá đóng góp của từng nhóm đặc trưng "
        "và Attention đối với hiệu quả dự báo và đầu tư."
    )


    # -------------------------------------------------------------------------
    # Experiment description
    # -------------------------------------------------------------------------

    a1, a2, a3 = st.columns(3)


    with a1:

        st.info(
            """
            **A — Basic Features**

            • Daily Return  
            • Lag Return  
            • Volume Change
            """
        )


    with a2:

        st.info(
            """
            **A + B — Technical Indicators**

            • SMA  
            • MACD  
            • RSI  
            • Momentum  
            • Volatility
            """
        )


    with a3:

        st.info(
            """
            **A + B + C — Proposed Model**

            • Basic Features  
            • Technical Indicators  
            • Attention Mechanism
            """
        )


    # =========================================================================
    # RUN ABLATION
    # =========================================================================

     

    try:

        with st.spinner("Running Ablation Study..."):

            ablation_results = run_ablation_study(
                ticker,
                sequence_length,
                transaction_cost
            )

        # ---------------------------------------------------------------------
        # CHECK RESULT
        # ---------------------------------------------------------------------

        if ablation_results is None or ablation_results.empty:

            st.info(
                "Không đủ dữ liệu để thực hiện Ablation Study."
            )

        else:

            # =================================================================
            # DISPLAY TABLE
            # =================================================================

            display_ablation = ablation_results.copy()

            # MAE
            if "MAE" in display_ablation.columns:
                display_ablation["MAE"] = (
                    display_ablation["MAE"]
                    .map(
                        lambda x: f"{x:.6f}"
                        if pd.notna(x)
                        else "-"
                    )
                )

            # RMSE
            if "RMSE" in display_ablation.columns:
                display_ablation["RMSE"] = (
                    display_ablation["RMSE"]
                    .map(
                        lambda x: f"{x:.6f}"
                        if pd.notna(x)
                        else "-"
                    )
                )

            # Directional Accuracy
            if "Directional Accuracy" in display_ablation.columns:
                display_ablation["Directional Accuracy"] = (
                    display_ablation["Directional Accuracy"]
                    .map(
                        lambda x: f"{x * 100:.2f}%"
                        if pd.notna(x)
                        else "-"
                    )
                )

            # Threshold
            if "Threshold" in display_ablation.columns:
                display_ablation["Threshold"] = (
                    display_ablation["Threshold"]
                    .map(
                        lambda x: f"{x * 100:.2f}%"
                        if pd.notna(x)
                        else "-"
                    )
                )

            # Validation Sharpe
            if "Validation Sharpe" in display_ablation.columns:
                display_ablation["Validation Sharpe"] = (
                    display_ablation["Validation Sharpe"]
                    .map(
                        lambda x: f"{x:.2f}"
                        if pd.notna(x)
                        else "-"
                    )
                )

            # Cumulative Return
            if "Cumulative Return" in display_ablation.columns:
                display_ablation["Cumulative Return"] = (
                    display_ablation["Cumulative Return"]
                    .map(
                        lambda x: f"{x * 100:.2f}%"
                        if pd.notna(x)
                        else "-"
                    )
                )

            # Sharpe Ratio
            if "Sharpe Ratio" in display_ablation.columns:
                display_ablation["Sharpe Ratio"] = (
                    display_ablation["Sharpe Ratio"]
                    .map(
                        lambda x: f"{x:.2f}"
                        if pd.notna(x)
                        else "-"
                    )
                )

            # Max Drawdown
            if "Max Drawdown" in display_ablation.columns:
                display_ablation["Max Drawdown"] = (
                    display_ablation["Max Drawdown"]
                    .map(
                        lambda x: f"{x * 100:.2f}%"
                        if pd.notna(x)
                        else "-"
                    )
                )

            st.dataframe(
                display_ablation,
                use_container_width=True,
                hide_index=True
            )

            # =================================================================
            # SHARPE RATIO CHART
            # =================================================================

            if "Sharpe Ratio" in ablation_results.columns:

                st.subheader("📊 So sánh Sharpe Ratio")

                fig_ablation = px.bar(
                    ablation_results,
                    x="Experiment",
                    y="Sharpe Ratio",
                    text="Sharpe Ratio",
                    title="Ablation Study – Test Set Sharpe Ratio"
                )

                fig_ablation.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside"
                )

                fig_ablation.add_hline(
                    y=1.8,
                    line_dash="dash",
                    annotation_text="Target Sharpe = 1.8"
                )

                fig_ablation.update_layout(
                    xaxis_title="Experiment",
                    yaxis_title="Sharpe Ratio"
                )

                st.plotly_chart(
                    fig_ablation,
                    use_container_width=True
                )

            # =================================================================
            # BEST ABLATION RESULT
            # =================================================================

            valid_results = ablation_results.dropna(
                subset=["Sharpe Ratio"]
            )

            if not valid_results.empty:

                best = (
                    valid_results
                    .sort_values(
                        "Sharpe Ratio",
                        ascending=False
                    )
                    .iloc[0]
                )

                best_experiment = str(best["Experiment"])
                best_model = str(best["Model"])
                best_sharpe = float(best["Sharpe Ratio"])

                st.subheader("🏅 Kết quả Ablation tốt nhất")

                # -------------------------------------------------------------
                # Dùng Streamlit native thay vì HTML
                # → tránh lỗi <div> hiển thị thành text
                # → tránh lỗi indentation của HTML
                # -------------------------------------------------------------

                r1, r2, r3 = st.columns([1, 2, 1])

                with r1:
                    st.markdown("##### Thí nghiệm tốt nhất")
                    st.success(best_experiment)

                with r2:
                    st.markdown("##### Mô hình tốt nhất")
                    st.success(best_model)

                with r3:
                    st.markdown("##### Test Sharpe Ratio")
                    st.success(f"{best_sharpe:.2f}")

                # -------------------------------------------------------------
                # INTERPRETATION
                # -------------------------------------------------------------

                if best_experiment == "A + B + C":

                    st.success(
                        "✅ Mô hình đầy đủ A + B + C đạt Sharpe Ratio "
                        "cao nhất. Kết quả cho thấy Technical Indicators "
                        "và Attention có đóng góp tích cực."
                    )

                else:

                    st.info(
                        f"📌 Thí nghiệm **{best_experiment}** đạt kết quả "
                        f"tốt nhất với Sharpe Ratio = **{best_sharpe:.2f}**. "
                        "Mô hình đầy đủ A + B + C chưa phải cấu hình tốt nhất, "
                        "do đó cần biện luận kết quả dựa trên thực nghiệm."
                    )

    # =========================================================================
    # QUAN TRỌNG: EXCEPT PHẢI NGANG HÀNG VỚI TRY
    # =========================================================================

    except Exception as e:

        st.warning(
            f"Ablation Study unavailable: {e}"
        )


    # =========================================================================
    # FOOTNOTE
    # =========================================================================

    st.divider()

    st.caption(
        "📌 Backtesting results are calculated on the out-of-sample Test Set. "
        "Trading thresholds are selected using Validation data only."
    )



# ==============================================================================
# TAB 10 - AI PORTFOLIO OPTIMIZATION
# ==============================================================================

def tab10():

    st.title(
        "🧠 AI Portfolio Optimization"
    )


    st.write(
        "Portfolio weights are optimized using "
        "AI-predicted expected returns and "
        "historical covariance."
    )


    # =========================================================================
    # SETTINGS
    # =========================================================================

    available_assets = [
        "AAPL",
        "MSFT",
        "GOOG",
        "META",
        "AMZN",
        "NVDA",
        "TSLA",
        "NFLX",
        "AMD"
    ]


    selected_assets = st.multiselect(
        "Select Assets",
        available_assets,
        default=[
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOG"
        ]
    )


    if len(selected_assets) < 2:

        st.warning(
            "Please select at least two assets."
        )

        return


    c1, c2, c3 = st.columns(3)


    with c1:

        model_name = st.selectbox(
            "AI Model",
            [
                "Linear Regression",
                "XGBoost",
                "LSTM",
                "Attention-GRU"
            ],
            index=3,
            key="portfolio_ai_model"
        )


    with c2:

        sequence_length = (
            st.selectbox(
                "Sequence Length",
                [
                    10,
                    20,
                    30,
                    60
                ],
                index=2,
                key="portfolio_sequence"
            )
        )


    with c3:

        risk_free_rate_pct = (
            st.number_input(
                "Annual Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.25
            )
        )


    risk_free_rate = (
        risk_free_rate_pct
        /
        100
    )


    # =========================================================================
    # HISTORICAL PRICE DATA
    # =========================================================================

    try:

        prices = yf.download(
            selected_assets,
            period="5y",
            auto_adjust=False,
            progress=False
        )


        if isinstance(
            prices.columns,
            pd.MultiIndex
        ):

            close = prices[
                "Close"
            ].copy()

        else:

            close = prices[
                ["Close"]
            ].copy()

            close.columns = (
                selected_assets
            )


        close = close.dropna(
            how="all"
        )


        if close.empty:

            st.warning(
                "No portfolio data available."
            )

            return


    except Exception as e:

        st.error(
            f"Unable to load portfolio data: {e}"
        )

        return


    # =========================================================================
    # NORMALIZED PERFORMANCE
    # =========================================================================

    st.subheader(
        "📈 Historical Asset Performance"
    )


    normalized = (
        close
        /
        close.iloc[0]
        *
        100
    )


    fig_history = px.line(
        normalized,
        x=normalized.index,
        y=normalized.columns,
        title="Normalized Performance (Base = 100)"
    )


    fig_history.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        hovermode="x unified",
        legend_title="Ticker"
    )


    st.plotly_chart(
        fig_history,
        use_container_width=True
    )


    # =========================================================================
    # AI EXPECTED RETURNS
    # =========================================================================

    st.divider()

    st.subheader(
        "🤖 AI Expected Returns"
    )


    predicted_returns = {}

    failed_assets = []


    progress = st.progress(
        0
    )


    for i, asset in enumerate(
        selected_assets
    ):

        try:

            result = train_ai_model(
                asset,
                model_name,
                sequence_length
            )


            predicted_returns[
                asset
            ] = result[
                "latest_prediction"
            ]


        except Exception as e:

            failed_assets.append(
                (
                    asset,
                    str(e)
                )
            )


        progress.progress(
            (
                i + 1
            )
            /
            len(
                selected_assets
            )
        )


    progress.empty()


    if len(
        predicted_returns
    ) < 2:

        st.error(
            "Not enough successful AI predictions "
            "to optimize a portfolio."
        )

        return


    valid_assets = list(
        predicted_returns.keys()
    )


    ai_mu = pd.Series(
        predicted_returns
    ).reindex(
        valid_assets
    )


    ai_return_table = pd.DataFrame({
        "Ticker":
            valid_assets,

        "Predicted Daily Return":
            ai_mu.values,

        "Predicted Return (%)":
            ai_mu.values
            *
            100
    })


    display_ai_return = (
        ai_return_table[
            [
                "Ticker",
                "Predicted Return (%)"
            ]
        ]
        .copy()
    )


    display_ai_return[
        "Predicted Return (%)"
    ] = (
        display_ai_return[
            "Predicted Return (%)"
        ]
        .map(
            lambda x:
            f"{x:+.3f}%"
        )
    )


    st.dataframe(
        display_ai_return,
        use_container_width=True,
        hide_index=True
    )


    if failed_assets:

        with st.expander(
            "Assets with unavailable predictions"
        ):

            for asset, error in failed_assets:

                st.write(
                    f"{asset}: {error}"
                )


    # =========================================================================
    # COVARIANCE MATRIX
    # =========================================================================

    portfolio_close = (
        close[
            valid_assets
        ]
        .dropna()
    )


    returns = (
        portfolio_close
        .pct_change()
        .dropna()
    )


    # Gần nhất 252 trading days
    covariance = (
        returns
        .tail(252)
        .cov()
    )


    # =========================================================================
    # PORTFOLIO OPTIMIZATION
    # =========================================================================

    n_assets = len(
        valid_assets
    )


    initial_weights = (
        np.ones(
            n_assets
        )
        /
        n_assets
    )


    risk_free_daily = (
        (
            1
            +
            risk_free_rate
        )
        ** (
            1 / 252
        )
        - 1
    )


    def negative_sharpe(
        weights
    ):

        portfolio_return = (
            np.dot(
                weights,
                ai_mu.values
            )
        )


        portfolio_volatility = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    covariance.values,
                    weights
                )
            )
        )


        if portfolio_volatility == 0:

            return 1e6


        sharpe = (
            portfolio_return
            -
            risk_free_daily
        ) / portfolio_volatility


        return -sharpe


    constraints = (
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w)
                - 1
        },
    )


    bounds = tuple(
        (
            0.0,
            1.0
        )
        for _ in range(
            n_assets
        )
    )


    optimization = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )


    if not optimization.success:

        st.error(
            "Portfolio optimization failed."
        )

        st.code(
            optimization.message
        )

        return


    optimal_weights = (
        optimization.x
    )


    # =========================================================================
    # PORTFOLIO METRICS
    # =========================================================================

    expected_daily_return = (
        np.dot(
            optimal_weights,
            ai_mu.values
        )
    )


    daily_volatility = np.sqrt(
        np.dot(
            optimal_weights.T,
            np.dot(
                covariance.values,
                optimal_weights
            )
        )
    )


    annual_return = (
        expected_daily_return
        *
        252
    )


    annual_volatility = (
        daily_volatility
        *
        np.sqrt(252)
    )


    if annual_volatility > 0:

        portfolio_sharpe = (
            annual_return
            -
            risk_free_rate
        ) / annual_volatility

    else:

        portfolio_sharpe = np.nan


    # =========================================================================
    # OPTIMAL WEIGHTS
    # =========================================================================

    st.divider()

    st.subheader(
        "⚖️ AI-Optimized Portfolio"
    )


    weights_df = pd.DataFrame({
        "Ticker":
            valid_assets,

        "Weight":
            optimal_weights
    })


    weights_df = (
        weights_df
        .sort_values(
            "Weight",
            ascending=False
        )
    )


    display_weights = (
        weights_df.copy()
    )


    display_weights[
        "Weight"
    ] = (
        display_weights[
            "Weight"
        ]
        .map(
            lambda x:
            f"{x * 100:.2f}%"
        )
    )


    st.dataframe(
        display_weights,
        use_container_width=True,
        hide_index=True
    )


    # =========================================================================
    # WEIGHT CHART
    # =========================================================================

    fig_weights = px.pie(
        weights_df,
        names="Ticker",
        values="Weight",
        title="Optimal Portfolio Allocation"
    )


    st.plotly_chart(
        fig_weights,
        use_container_width=True
    )


    # =========================================================================
    # KPI
    # =========================================================================

    k1, k2, k3 = st.columns(3)


    k1.metric(
        "Expected Annual Return",
        f"{annual_return * 100:.2f}%"
    )


    k2.metric(
        "Annual Volatility",
        f"{annual_volatility * 100:.2f}%"
    )


    k3.metric(
        "Expected Sharpe Ratio",
        (
            f"{portfolio_sharpe:.2f}"
            if pd.notna(
                portfolio_sharpe
            )
            else "N/A"
        )
    )


    # =========================================================================
    # EQUAL WEIGHT COMPARISON
    # =========================================================================

    st.divider()

    st.subheader(
        "📊 AI Portfolio vs Equal Weight"
    )


    equal_weights = (
        np.ones(
            n_assets
        )
        /
        n_assets
    )


    equal_daily_return = (
        np.dot(
            equal_weights,
            ai_mu.values
        )
    )


    equal_daily_vol = np.sqrt(
        np.dot(
            equal_weights.T,
            np.dot(
                covariance.values,
                equal_weights
            )
        )
    )


    equal_annual_return = (
        equal_daily_return
        *
        252
    )


    equal_annual_vol = (
        equal_daily_vol
        *
        np.sqrt(252)
    )


    equal_sharpe = (
        (
            equal_annual_return
            -
            risk_free_rate
        )
        /
        equal_annual_vol
        if equal_annual_vol > 0
        else np.nan
    )


    comparison = pd.DataFrame({
        "Portfolio": [
            "AI Optimized",
            "Equal Weight"
        ],

        "Expected Return": [
            annual_return,
            equal_annual_return
        ],

        "Volatility": [
            annual_volatility,
            equal_annual_vol
        ],

        "Sharpe Ratio": [
            portfolio_sharpe,
            equal_sharpe
        ]
    })


    display_comparison = (
        comparison.copy()
    )


    display_comparison[
        "Expected Return"
    ] = (
        display_comparison[
            "Expected Return"
        ]
        .map(
            lambda x:
            f"{x * 100:.2f}%"
        )
    )


    display_comparison[
        "Volatility"
    ] = (
        display_comparison[
            "Volatility"
        ]
        .map(
            lambda x:
            f"{x * 100:.2f}%"
        )
    )


    display_comparison[
        "Sharpe Ratio"
    ] = (
        display_comparison[
            "Sharpe Ratio"
        ]
        .map(
            lambda x:
            f"{x:.2f}"
            if pd.notna(x)
            else "N/A"
        )
    )


    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "AI Expected Returns are generated by the selected prediction model. "
        "Portfolio risk is estimated from historical covariance."
    )

    
# =============================================================================
# TAB 11 - FINANCIAL CHATBOT - GROQ
# =============================================================================

def tab11():

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
                        model="llama-3.3-70b-versatile",
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


# ==============================================================================
# MAIN BODY
# ==============================================================================

def run():

    ticker_list = [
        "-",
        "AAPL",
        "MSFT",
        "GOOG",
        "META",
        "AMZN",
        "NVDA",
        "TSLA",
        "NFLX",
        "AMD",
        "BTC-USD",
        "ETH-USD",
        "^GSPC",
        "^IXIC"
    ]

    global ticker

    ticker = st.sidebar.selectbox(
        "Select a ticker",
        ticker_list
    )

    tabs = {
        "Overview": tab0,
        "Summary": tab1,
        "Chart": tab2,
        "Statistics": tab3,
        "Financials": tab4,
        "Analysis": tab5,
        "Monte Carlo Simulation": tab6,
        "Your Portfolio's Trend": tab7,
        "AI Prediction": tab8,
        "AI Backtesting": tab9,
        "AI Portfolio Optimization": tab10,
        "Financial Chatbot": tab11
    }

    select_tab = st.sidebar.radio(
        "Select tab",
        list(tabs.keys())
    )

    tabs[select_tab]()


if __name__ == "__main__":
    run()
    


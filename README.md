# Investment-Advance - UEH

* Team members:
1. Nguyễn Thị Thu Thảo
2. Nguyễn Thị Chinh: Build Dashboard
3. Đào Duy Bảo
4. Châu Phương Uyên
5. Mạnh Hồ Kiên
6. Bùi Thị Mạnh Quỳnh

Link dashboard:
https://ai-investment-advance-ntc.streamlit.app

### Artificial Intelligence Applications in Investment Analysis

### 1. Problem Definition

Clearly define the investment problem, including:

* **Input:** Identify the input variables/features used in the model.
* **Output:** Define the expected output or prediction target.
* **Evaluation Metrics:** Specify appropriate metrics for evaluating both predictive performance and investment performance.

### 2. Data Collection, Analysis, and Preprocessing

* Collect relevant financial and market data.
* Perform Exploratory Data Analysis (EDA).
* Clean and preprocess the dataset.
* Handle missing values and other data-quality issues.
* Perform feature engineering where appropriate.
* Prepare the data for model training.

### 3. Model Training

#### 3.1. Train / Validation / Test Split

Divide the dataset into:

* **Training Set**
* **Validation Set**
* **Test Set**

The **Training and Validation Sets** should be used for:

* Model selection.
* Hyperparameter tuning.
* Model development.

The **Test Set** should only be used for the final evaluation.

Special attention must be paid to preventing **data leakage**.

The investment strategy/model should aim to achieve:

> **Sharpe Ratio ≥ 1.8 on the Test Set**

#### 3.2. Baseline Models

Develop baseline models for comparison.

**Traditional models may include:**

* Linear Regression
* CAPM
* APT
* Rule-based strategies
* Other conventional investment models

**Common Machine Learning / Deep Learning models may include:**

* MLP
* CNN
* RNN
* LSTM
* GRU
* Transformer

The selected models should provide meaningful benchmarks for evaluating the proposed model.

### 4. Proposed Model

Develop a proposed AI/ML/DL model that improves upon the baseline approaches.

The proposed model should clearly describe:

* Model architecture.
* Input features.
* Training process.
* Hyperparameters.
* Prediction output.
* How the model is applied to the investment problem.

### 5. Ablation Study

Conduct **Ablation Studies** to evaluate the contribution of different components or features of the proposed model.

For example:

| Experiment | Components        | Performance |
| ---------- | ----------------- | ----------: |
| Model 1    | Feature A         |          x% |
| Model 2    | Feature A + B     |      x + 2% |
| Model 3    | Feature A + B + C |      x + 5% |

The purpose of the Ablation Study is to determine whether each additional component actually improves model performance.

### 6. Model Evaluation

Evaluate and compare the performance of:

* Traditional baseline models.
* Machine Learning / Deep Learning models.
* Proposed model.

The evaluation should include appropriate **prediction metrics** and **investment performance metrics**.

Particular attention should be given to:

* Model performance on the **Test Set**.
* Sharpe Ratio.
* Risk-adjusted return.
* Comparison with baseline strategies.
* Robustness of the proposed model.

### 7. Discussion

Discuss the model results in the context of **investment analysis**.

The discussion should address:

* Why the proposed model performs better or worse than the baseline models.
* The financial interpretation of the results.
* The effectiveness of AI in supporting investment decisions.
* Risk and return implications.
* Model limitations.
* Potential overfitting and data leakage issues.
* Practical applicability of the proposed approach.

### 8. Streamlit Application

Develop a **Streamlit application** to demonstrate the proposed investment analysis system.

The application should visually present relevant outputs such as:

* Market and stock data.
* Model predictions.
* Investment/trading signals.
* Model performance.
* Backtesting results.
* Sharpe Ratio and risk metrics.
* Portfolio analysis or optimization, where applicable.

### 9. GitHub Repository

Upload the complete project to **GitHub**, including:

* Source code.
* Streamlit application.
* Model implementation.
* Data-processing code.
* Required dependencies.
* `requirements.txt`.
* Project documentation / README.
* Instructions for running the application.

### Overall Project Workflow

**Problem Definition → Data Collection → Data Preprocessing → Train/Validation/Test Split → Baseline Models → AI/Deep Learning Models → Proposed Model → Ablation Study → Model Evaluation → Investment Discussion → Streamlit Application → GitHub**


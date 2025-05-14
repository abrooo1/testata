import streamlit as st
import numpy as np
from math import sqrt, erf

# Page Config
st.set_page_config(page_title="Correlation App", layout="centered")
st.title("📊 Correlation Analysis with Hypothesis Testing")
st.markdown("Select or enter two variables to compute their **Pearson correlation coefficient** and perform a **hypothesis test**.")

# Predefined Datasets
DATASETS = {
    "Example 1 - Academic Performance": {
        "x": "3.2, 4.5, 6.7, 2.8, 5.1, 7.3, 3.9, 4.6, 6.2, 5.5",
        "y": "28, 35, 45, 22, 38, 50, 30, 36, 44, 40"
    },
    "Example 2 - Sales vs Advertising": {
        "x": "10, 20, 30, 40, 50, 60, 70, 80, 90, 100",
        "y": "100, 150, 200, 220, 280, 300, 350, 400, 450, 500"
    },
    "Example 3 - Temperature vs Ice Cream Sales": {
        "x": "20, 22, 25, 23, 27, 28, 30, 29, 26, 24",
        "y": "150, 160, 200, 180, 220, 240, 250, 230, 210, 190"
    }
}

# Dataset Selection Dropdown
selected_dataset = st.selectbox("Choose a dataset:", options=list(DATASETS.keys()))
default_x = DATASETS[selected_dataset]["x"]
default_y = DATASETS[selected_dataset]["y"]

# Input Fields
col1, col2 = st.columns(2)
with col1:
    x_input = st.text_area("Variable X (comma-separated)", value=default_x, height=150)
with col2:
    y_input = st.text_area("Variable Y (comma-separated)", value=default_y, height=150)

# Calculate Button
if st.button("🔍 Calculate Correlation"):
    try:
        # Convert inputs
        x = list(map(float, x_input.strip().split(',')))
        y = list(map(float, y_input.strip().split(',')))

        if len(x) != len(y):
            st.error("❌ Both datasets must have the same number of values.")
        else:
            n = len(x)
            mean_x = sum(x) / n
            mean_y = sum(y) / n

            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denom_x = sqrt(sum((xi - mean_x)**2 for xi in x))
            denom_y = sqrt(sum((yi - mean_y)**2 for yi in y))

            r = numerator / (denom_x * denom_y)

            # Hypothesis Test
            t_stat = r * sqrt((n - 2) / (1 - r**2))
            p_value = 2 * (1 - t_cdf(abs(t_stat), n - 2))

            # Display Results
            st.subheader("📈 Results")
            st.metric(label="Sample Size", value=str(n))
            st.metric(label="Pearson's r", value=f"{r:.3f}")
            st.metric(label="p-value", value=f"{p_value:.4f}")

            # Interpretation
            st.markdown("### 🔍 Interpretation:")
            alpha = 0.05
            if p_value < alpha:
                st.success("✅ Reject null hypothesis: Significant correlation (p < 0.05)")
            else:
                st.warning("⚠️ Fail to reject null hypothesis: No significant correlation (p ≥ 0.05)")

            # Additional explanation
            st.markdown("""
            - **Pearson's r** ranges from -1 to +1:
              - Close to +1 → strong positive linear relationship
              - Close to -1 → strong negative linear relationship
              - Near 0 → no linear relationship
            - **p-value** tells us whether the correlation is statistically significant.
            """)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# Helper Functions (T-Distribution CDF Approximation)
def t_cdf(x, dof):
    if dof <= 0:
        return 0.5
    if dof > 30:
        return normal_cdf(x)
    
    a = dof / 2
    b = 0.5
    x_beta = dof / (dof + x * x)

    beta_val = np.exp(lbeta(a, b))
    ibeta_val = ibeta(x_beta, a, b)
    prob = ibeta_val

    return prob / 2 + (0.5 if x > 0 else 0)


def normal_cdf(x):
    return (1 + erf(x / sqrt(2))) / 2


def lbeta(a, b):
    return np.log(gamma(a)) + np.log(gamma(b)) - np.log(gamma(a + b))


def gamma(x):
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507341404690403,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    if x < 0.5:
        return np.pi / (np.sin(np.pi * x) * gamma(1 - x))
    x -= 1
    tmp = p[0]
    for i in range(1, len(p)):
        tmp += p[i] / (x + i)
    t = x + len(p) - 1.5
    return sqrt(2 * np.pi) * pow(t, x + 0.5) * np.exp(-t) * tmp


def ibeta(x, a, b):
    if x == 0:
        return 0
    if x == 1:
        return beta(a, b)
    bt = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta(a, b))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(x, a, b) / a
    else:
        return 1 - bt * betacf(1 - x, b, a) / b


def beta(a, b):
    return np.exp(lbeta(a, b))


def betacf(x, a, b):
    MAXIT = 100
    EPS = 3e-7
    FPMIN = 1e-30
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1 / d
        h *= d * c
        if abs(h - 1) < EPS:
            break
    return h
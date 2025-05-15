import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from math import sqrt, erf

# --- Helper Functions ---

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

# --- Main App Starts Here ---

st.set_page_config(page_title="Correlation App with Cascading Filters", layout="centered")
st.title("📊 Correlation Analysis with Cascading Filtering")
st.markdown("Filter by 'Region contains', and see cascading updates in other fields.")

try:
    # Load Excel file
    df = pd.read_excel("data.xlsx")

    if df.empty:
        st.warning("❌ The Excel file is empty.")
    else:
        st.success("✅ Data loaded successfully!")
        st.dataframe(df.head())

        # --- FILTERING OPTIONS ---
        st.subheader("🔍 Apply Filters")

        # Region filter
        region_col = "Region"
        if region_col not in df.columns:
            st.error(f"❌ Column '{region_col}' not found in data.")
        else:
            region_filter = st.text_input("Keep rows where 'Region' contains:")
            filtered_df = df.copy()

            if region_filter:
                filtered_df = filtered_df[filtered_df[region_col].str.contains(region_filter, case=False, na=False)]

            st.info(f"✅ {len(filtered_df)} rows remain after region filter.")

            # Cascading: Select another column to show filtered options
            other_cols = [col for col in filtered_df.columns if col != region_col]
            cascade_col = st.selectbox("Select a column to view filtered values:", options=other_cols)

            if cascade_col:
                unique_vals = filtered_df[cascade_col].dropna().unique()
                st.markdown(f"### 🔄 Unique values in '{cascade_col}':")
                st.write(unique_vals)

        # Select Variables for Correlation
        st.subheader("📈 Select Variables for Correlation Analysis")
        columns = filtered_df.columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
            var_x = st.selectbox("Select Variable X", options=columns)
        with col2:
            var_y = st.selectbox("Select Variable Y", options=columns)

        if var_x == var_y:
            st.error("❌ Please select two different columns.")
        else:
            x = filtered_df[var_x].dropna().values
            y = filtered_df[var_y].dropna().values

            if len(x) != len(y):
                st.warning("⚠️ Length mismatch: Trimming to shortest length.")
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]

            if len(x) < 2:
                st.error("❌ At least 2 data points are required for correlation.")
            else:
                # Calculate Pearson Correlation
                n = len(x)
                mean_x = np.mean(x)
                mean_y = np.mean(y)

                numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
                denom_x = sqrt(sum((xi - mean_x)**2 for xi in x))
                denom_y = sqrt(sum((yi - mean_y)**2 for yi in y))

                r = numerator / (denom_x * denom_y)

                # Hypothesis Test
                t_stat = r * sqrt((n - 2) / (1 - r**2))
                p_value = 2 * (1 - t_cdf(abs(t_stat), n - 2))

                # Display Results
                st.subheader("📊 Results")
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

                # Scatter Plot
                st.subheader("📉 Scatter Plot with Regression Line")
                fig, ax = plt.subplots()
                ax.scatter(x, y, color='blue', label='Data')
                slope = r * (np.std(y) / np.std(x))
                intercept = mean_y - slope * mean_x
                ax.plot(x, slope * x + intercept, color='red', label='Regression Line')
                ax.set_xlabel(var_x)
                ax.set_ylabel(var_y)
                ax.legend()
                st.pyplot(fig)

                # Export Options
                st.subheader("📤 Export Results")
                result_df = pd.DataFrame({
                    'X': x,
                    'Y': y,
                    'Predicted Y': slope * x + intercept,
                    'Residuals': y - (slope * x + intercept)
                })

                csv = result_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='correlation_results.csv',
                    mime='text/csv'
                )

                # Generate PDF
                def generate_pdf():
                    pdf_path = "correlation_report.pdf"
                    c = canvas.Canvas(pdf_path, pagesize=letter)
                    c.drawString(50, 750, f"Correlation Report: {var_x} vs {var_y}")
                    c.drawString(50, 730, f"Pearson's r: {r:.3f}")
                    c.drawString(50, 710, f"p-value: {p_value:.4f}")
                    c.drawString(50, 690, f"Interpretation: {'Significant' if p_value < alpha else 'Not Significant'}")
                    c.save()
                    return pdf_path

                if st.button("📄 Generate PDF Report"):
                    pdf_file = generate_pdf()
                    with open(pdf_file, "rb") as f:
                        st.download_button("📥 Download PDF Report", f, file_name="correlation_report.pdf")

except FileNotFoundError:
    st.error("❌ Excel file not found. Make sure 'data.xlsx' exists in the same directory.")
except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")

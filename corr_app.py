import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Page config
st.set_page_config(page_title="PCA Calculator", layout="wide")

# Title and description
st.markdown("""
<div style='text-align: center; padding: 14px;'>
    <h2 style='margin: 0; font-size: 24px;'>Principal Component Analysis (PCA) Calculator</h2>
    <p style='font-size: 16px; color: #555;'>Upload your dataset and perform PCA interactively.</p>
</div>
""", unsafe_allow_html=True)

# Divider
st.markdown("<hr>", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if df.empty:
            st.warning("❌ The uploaded file is empty.")
        else:
            st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

            # Display first few rows
            with st.expander("📊 View Raw Data"):
                st.dataframe(df.head())

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("⚠️ At least two numeric columns are required for PCA.")
            else:
                # Select columns for PCA
                selected_cols = st.multiselect("🔢 Select columns for PCA", options=numeric_cols, default=numeric_cols[:5])
                if len(selected_cols) < 2:
                    st.warning("⚠️ Please select at least two numeric columns for PCA.")
                else:
                    X = df[selected_cols].dropna()
                    if len(X) < 2:
                        st.error("❌ Not enough complete rows after dropping missing values.")
                    else:
                        # Scaling
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        # Perform PCA
                        pca = PCA()
                        pca.fit(X_scaled)
                        components = pca.transform(X_scaled)

                        explained_variance_ratio = pca.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance_ratio)

                        st.markdown("### 📈 Scree Plot – Explained Variance per Principal Component")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.bar(range(len(explained_variance_ratio)), explained_variance_ratio, label="Individual")
                        ax.plot(range(len(cumulative_variance)), cumulative_variance, 'r-o', label="Cumulative")
                        ax.set_xlabel("Principal Components")
                        ax.set_ylabel("Variance Explained (%)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        st.markdown("### 🧮 Transformed Principal Components")
                        pc_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(len(selected_cols))])
                        st.dataframe(pc_df.head())

                        st.markdown("### 🔀 Biplot of First Two Principal Components")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(pc_df["PC1"], pc_df["PC2"], alpha=0.7)

                        # Add variable loadings as vectors
                        loadings = pca.components_.T
                        for i, (name, vector) in enumerate(zip(selected_cols, loadings)):
                            ax.arrow(0, 0, vector[0] * 3, vector[1] * 3, head_width=0.1, length_includes_head=True, color='r')
                            ax.text(vector[0] * 3.2, vector[1] * 3.2, name, color='g', ha='center', va='center')

                        ax.set_xlabel("PC1")
                        ax.set_ylabel("PC2")
                        ax.axhline(0, color='gray', linestyle='--')
                        ax.axvline(0, color='gray', linestyle='--')
                        ax.set_title("PCA Biplot (Loadings + Scores)")
                        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
else:
    st.info("📥 Please upload a CSV or Excel file to begin analysis.")

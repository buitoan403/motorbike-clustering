# app_clustering_motobikes.py
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ======================================================
# HÀM TIỀN XỬ LÝ DỮ LIỆU
# ======================================================

def parse_price_to_million(s: str):
    """Chuẩn hóa chuỗi giá về đơn vị triệu đồng."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower()

    # loại bỏ ký tự không cần thiết
    s = s.replace("\u00a0", " ")
    s = s.replace("vnđ", "").replace("vnd", "").replace("đ", "")
    s = s.replace(",", ".").strip()

    m = re.search(r"(\d+\.?\d*)", s)
    if not m:
        return np.nan

    num = float(m.group(1))

    if "triệu" in s or " tr" in s:
        return num
    if "nghìn" in s or "ngàn" in s or "k" in s:
        return num / 1000
    # nếu chỉ ghi dạng 20.000.000
    if num > 1000:
        return num / 1_000_000
    return num


@st.cache_data
def load_and_prepare_data(data_path: str):
    """Đọc & xử lý dữ liệu, trả về df, danh sách cột, preprocess, ma trận X."""
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df_raw = pd.read_csv(data_path)
    else:
        df_raw = pd.read_excel(data_path)

    df = df_raw.copy()

    # Tự tìm cột khoảng giá min / max
    min_col_txt = [c for c in df.columns if "min" in c.lower()][0]
    max_col_txt = [c for c in df.columns if "max" in c.lower()][0]

    df["Khoảng giá min (triệu)"] = df[min_col_txt].apply(parse_price_to_million)
    df["Khoảng giá max (triệu)"] = df[max_col_txt].apply(parse_price_to_million)

    # Cột Giá chính
    if "Giá" in df.columns:
        df["Giá"] = pd.to_numeric(df["Giá"], errors="coerce")
        mask = df["Giá"].isna()
        df.loc[mask, "Giá"] = df.loc[
            mask, ["Khoảng giá min (triệu)", "Khoảng giá max (triệu)"]
        ].mean(axis=1)
    else:
        df["Giá"] = df[["Khoảng giá min (triệu)", "Khoảng giá max (triệu)"]].mean(axis=1)

    # Tuổi xe
    df["Năm đăng ký"] = pd.to_numeric(df["Năm đăng ký"], errors="coerce")
    df["Tuổi xe"] = 2025 - df["Năm đăng ký"]

    # Km
    if "Số Km đã đi" in df.columns:
        df["Số Km đã đi"] = pd.to_numeric(df["Số Km đã đi"], errors="coerce")
    else:
        df["Số Km đã đi"] = np.nan

    # Các cột dùng phân cụm
    numeric_cols = ["Giá", "Tuổi xe", "Số Km đã đi"]
    categorical_cols = [
        "Thương hiệu",
        "Dòng xe",
        "Loại xe",
        "Dung tích xe",
        "Xuất xứ",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    X = df[numeric_cols + categorical_cols].copy()

    pre_num = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre_cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", pre_num, numeric_cols),
            ("cat", pre_cat, categorical_cols),
        ]
    )

    X_prep = preprocess.fit_transform(X)
    X_dense = X_prep.toarray()

    return df, numeric_cols, categorical_cols, preprocess, X_dense


# ======================================================
# KMEANS & TIỆN ÍCH
# ======================================================

def run_kmeans(df, numeric_cols, X_dense, K: int):
    """Chạy KMeans, trả về model + kết quả."""
    model = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = model.fit_predict(X_dense)

    sil = silhouette_score(X_dense, labels)

    dfc = df.copy()
    dfc["cluster"] = labels

    summary = (
        dfc.groupby("cluster")[numeric_cols]
        .agg(["count", "mean", "min", "max"])
        .round(2)
    )

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)

    return {
        "model": model,
        "dfc": dfc,
        "summary": summary,
        "silhouette": sil,
        "X_pca": X_pca,
        "K": K,
    }


def seg_label(c: int) -> str:
    return f"Phân khúc {c + 1}"


# ======================================================
# ẢNH HEADER: xe.png
# ======================================================

def get_xe_image_path():
    """Trả về đường dẫn xe.png nếu tồn tại, không báo lỗi nếu không có."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "xe.png")
    if os.path.exists(path):
        return path
    return None


def render_header():
    """Tiêu đề + ảnh xe ở góc phải."""
    img_path = get_xe_image_path()

    if img_path:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Phân khúc xe máy đã qua sử dụng – Streamlit GUI")
        with col2:
            st.image(img_path, use_column_width=True)
    else:
        st.title("Phân khúc xe máy đã qua sử dụng – Streamlit GUI")


# ======================================================
# CÁC TRANG GIAO DIỆN
# ======================================================

def page_business_problem():
    st.header("Vấn đề kinh doanh – Phân khúc xe máy cũ")
    st.write(
        """
Ứng dụng này nhằm:
- Phân nhóm các xe máy đã qua sử dụng thành các **phân khúc thị trường khác nhau**;
- Hỗ trợ bên bán xây dựng chiến lược giá & marketing;
- Hỗ trợ người mua nhận diện phân khúc xe phù hợp với nhu cầu & ngân sách.

Dữ liệu bao gồm các thông tin: **Giá, Năm đăng ký, Số Km đã đi, Thương hiệu, Dòng xe, Loại xe, Dung tích, Xuất xứ**.
"""
    )


def page_evaluation(result):
    st.header("Đánh giá & Báo cáo")

    st.subheader("1️⃣ Thông tin mô hình")
    st.write(f"- Số phân khúc (K): **{result['K']}**")
    st.write(f"- Giá trị Silhouette: **{result['silhouette']:.4f}**")
    st.markdown(
        """
- Silhouette càng lớn (gần 1) → các phân khúc càng tách biệt, chất lượng phân cụm càng tốt.
"""
    )

    st.subheader("2️⃣ Thống kê theo từng phân khúc")
    summary = result["summary"].copy()
    summary.index = [seg_label(i) for i in summary.index]
    st.dataframe(summary, use_container_width=True)


def page_cluster_and_predict(df, numeric_cols, categorical_cols, preprocess, result):
    st.header("Khám phá & Dự đoán phân khúc")

    dfc = result["dfc"]
    X_pca = result["X_pca"]
    model = result["model"]

    # ----- PCA plot
    st.subheader("🌈 Trực quan PCA 2D theo phân khúc")
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = sorted(dfc["cluster"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))

    for cl, color in zip(clusters, colors):
        mask = dfc["cluster"] == cl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=10,
            color=color,
            label=seg_label(cl),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)

    # ----- Bảng chi tiết từng phân khúc
    st.subheader("📄 Danh sách xe theo phân khúc")
    choice = st.selectbox(
        "Chọn phân khúc muốn xem:",
        clusters,
        format_func=seg_label,
    )
    st.dataframe(
        dfc[dfc["cluster"] == choice].reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")

    # ----- Form dự đoán phân khúc cho xe người dùng
    st.subheader("🛵 Dự đoán phân khúc cho xe của bạn")

    defaults = {c: float(df[c].median()) for c in numeric_cols}
    cats = {c: sorted(df[c].dropna().unique()) for c in categorical_cols}

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        thuong_hieu = col1.selectbox("Thương hiệu", cats.get("Thương hiệu", [""]))
        dong_xe = col2.selectbox("Dòng xe", cats.get("Dòng xe", [""]))
        loai_xe = col3.selectbox("Loại xe", cats.get("Loại xe", [""]))

        col4, col5, col6 = st.columns(3)
        dung_tich = col4.selectbox("Dung tích xe", cats.get("Dung tích xe", [""]))
        xuat_xu = col5.selectbox("Xuất xứ", cats.get("Xuất xứ", [""]))
        gia = col6.number_input(
            "Giá (triệu đồng)", value=defaults.get("Giá", 20.0), min_value=0.0
        )

        col7, col8 = st.columns(2)
        nam_dk = col7.number_input(
            "Năm đăng ký",
            min_value=1990,
            max_value=2025,
            value=int(df["Năm đăng ký"].median()),
        )
        so_km = col8.number_input(
            "Số Km đã đi",
            value=defaults.get("Số Km đã đi", 30000.0),
            min_value=0.0,
            step=1000.0,
        )

        submit = st.form_submit_button("🔍 Dự đoán phân khúc")

    if submit:
        tuoi_xe = 2025 - nam_dk

        row = {
            "Giá": gia,
            "Tuổi xe": tuoi_xe,
            "Số Km đã đi": so_km,
            "Thương hiệu": thuong_hieu,
            "Dòng xe": dong_xe,
            "Loại xe": loai_xe,
            "Dung tích xe": dung_tich,
            "Xuất xứ": xuat_xu,
        }

        X_user = preprocess.transform(pd.DataFrame([row])).toarray()
        pred = int(model.predict(X_user)[0])

        st.success(f"✅ Xe của bạn được xếp vào **{seg_label(pred)}**.")


def page_team():
    st.header("Thông tin nhóm thực hiện")
    st.write(
        """
**Nhóm học viên:**
1. Mai Bảo Ngọc  
2. Bùi Ngọc Toản  
3. Nguyễn Vũ Duy  
"""
    )


# ======================================================
# MAIN APP
# ======================================================

def main():
    st.set_page_config(
        page_title="Phân khúc xe máy – Streamlit",
        layout="wide",
    )

    # Header có ảnh xe.png
    render_header()

    # Sidebar: chọn trang & cấu hình
    page = st.sidebar.radio(
        "",  # ẩn label "Chọn trang:"
        [
            "Vấn đề kinh doanh",
            "Đánh giá & Báo cáo",
            "Khám phá & Dự đoán phân khúc",
            "Thông tin nhóm",
        ],
    )

    raw_path = st.sidebar.text_input(
        "",  # ẩn label
        value="data_motobikes_clean.xlsx",
        label_visibility="collapsed",
    )

    K = st.sidebar.slider("Số phân khúc (K)", min_value=2, max_value=8, value=3)

    # Đường dẫn dữ liệu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, raw_path.strip().strip('"').strip("'"))

    if not os.path.exists(data_path):
        st.error("❌ Không tìm thấy file dữ liệu, hãy kiểm tra lại tên file.")
        return

    # Load dữ liệu & phân cụm
    df, numeric_cols, categorical_cols, preprocess, X_dense = load_and_prepare_data(
        data_path
    )
    result = run_kmeans(df, numeric_cols, X_dense, K)

    # Điều hướng trang
    if page == "Vấn đề kinh doanh":
        page_business_problem()
    elif page == "Đánh giá & Báo cáo":
        page_evaluation(result)
    elif page == "Khám phá & Dự đoán phân khúc":
        page_cluster_and_predict(df, numeric_cols, categorical_cols, preprocess, result)
    else:
        page_team()


if __name__ == "__main__":
    main()

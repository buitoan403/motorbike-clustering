# app_clustering_motobikes.py
# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


# =====================
# HÀM XỬ LÝ DỮ LIỆU
# =====================

def parse_price_to_million(s):
    """Chuẩn hóa chuỗi giá về đơn vị triệu đồng."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower().strip()
    s = s.replace("\u00a0", " ").replace("vnđ", "").replace("vnd", "").strip()

    m = re.search(r"(\d+[.,]?\d*)", s)
    if not m:
        return np.nan
    num = float(m.group(1).replace(".", "").replace(",", "."))

    if re.search(r"triệu|tr\b", s):
        return num
    if re.search(r"nghìn|ngàn|k\b", s):
        return num / 1000
    return num / 1_000_000


@st.cache_data
def load_and_prepare_data(data_path: str):
    """
    Đọc & xử lý dữ liệu, trả về:
    - df: DataFrame đã chuẩn hóa
    - numeric_cols: danh sách biến số
    - categorical_cols: danh sách biến phân loại
    - preprocess: ColumnTransformer đã fit
    - X_dense: ma trận đặc trưng đã chuẩn hóa (numpy array)
    """

    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df_raw = pd.read_csv(data_path)
    else:
        df_raw = pd.read_excel(data_path)

    df = df_raw.copy()

    # Tìm cột text chứa khoảng giá
    min_col_txt = [c for c in df.columns if "Khoảng giá min" in c][0]
    max_col_txt = [c for c in df.columns if "Khoảng giá max" in c][0]

    df["Khoảng giá min (triệu)"] = df[min_col_txt].apply(parse_price_to_million)
    df["Khoảng giá max (triệu)"] = df[max_col_txt].apply(parse_price_to_million)

    # Điền thiếu theo Dòng xe, Thương hiệu
    if "Dòng xe" in df.columns:
        for c in ["Khoảng giá min (triệu)", "Khoảng giá max (triệu)"]:
            df[c] = df.groupby("Dòng xe")[c].transform(lambda x: x.fillna(x.mean()))
    if "Thương hiệu" in df.columns:
        for c in ["Khoảng giá min (triệu)", "Khoảng giá max (triệu)"]:
            df[c] = df.groupby("Thương hiệu")[c].transform(lambda x: x.fillna(x.mean()))

    # Giá (triệu)
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

    # Số Km
    if "Số Km đã đi" in df.columns:
        df["Số Km đã đi"] = pd.to_numeric(df["Số Km đã đi"], errors="coerce")

    # Các cột dùng cho mô hình
    numeric_cols = ["Giá", "Tuổi xe", "Số Km đã đi"]
    categorical_cols = [
        "Thương hiệu",
        "Dòng xe",
        "Tình trạng",
        "Loại xe",
        "Dung tích xe",
        "Xuất xứ",
        "Chính sách bảo hành",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    X = df[numeric_cols + categorical_cols].copy()

    # Tiền xử lý
    numeric_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ]
    )

    X_prep = preprocess.fit_transform(X)
    X_dense = X_prep.toarray()

    return df, numeric_cols, categorical_cols, preprocess, X_dense


# =====================
# PHÂN CỤM KMEANS
# =====================

def run_kmeans(df, numeric_cols, X_dense, K: int, random_state: int = 42):
    """
    Chạy KMeans với K cụm, trả về:
    - K, silhouette, df_clustered, summary, X_pca, model
    """
    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_dense)

    sil = silhouette_score(X_dense, labels)

    df_clu = df.copy()
    df_clu["cluster"] = labels  # 0,1,2,... => hiển thị là Phân khúc 1,2,3,...

    summary = (
        df_clu.groupby("cluster")[numeric_cols]
        .agg(["count", "mean", "min", "max"])
        .round(2)
    )

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_dense)

    return {
        "K": K,
        "silhouette": sil,
        "df_clustered": df_clu,
        "summary": summary,
        "X_pca": X_pca,
        "model": km,
    }


def seg_label(cluster_id: int) -> str:
    """Đổi số cụm 0,1,2 thành 'Phân khúc 1, 2, 3'."""
    return f"Phân khúc {cluster_id + 1}"


# =========
# TRANG GIAO DIỆN
# =========

def page_business_problem():
    st.header("Business Problem – Phân cụm xe máy đã qua sử dụng")
    st.write(
        """
Mục tiêu: phân loại các xe máy cũ thành **các phân khúc thị trường (Phân khúc 1, 2, 3,...)** 
dựa trên các đặc điểm như **giá**, **tuổi xe**, **số km đã đi**, **thương hiệu**, **dòng xe**, v.v.

Ứng dụng cho phép:
- Chọn **K (số phân khúc)** linh hoạt;
- Xem thống kê mô tả theo từng phân khúc;
- Trực quan hoá các phân khúc trên mặt phẳng PCA;
- Nhập thông tin 1 chiếc xe bất kỳ để xem **nó thuộc phân khúc nào**.
"""
    )


def page_evaluation(result):
    st.header("Evaluation & Report – Đánh giá mô hình KMeans")

    K = result["K"]
    sil = result["silhouette"]

    st.subheader("1️⃣ Thông tin mô hình")
    st.write(f"- Thuật toán: **KMeans**")
    st.write(f"- Số phân khúc (K): **{K}**")
    st.write(f"- Giá trị Silhouette: **{sil:.4f}**")

    st.markdown(
        """
Silhouette càng lớn (gần 1) → các phân khúc càng tách biệt, chất lượng phân cụm càng tốt.  
Giá trị quanh **0.3–0.6** thường là chấp nhận được với dữ liệu thực tế.
"""
    )

    st.subheader("2️⃣ Thống kê mô tả theo từng phân khúc")
    summary = result["summary"].copy()
    summary.index = [seg_label(i) for i in summary.index]
    st.dataframe(summary, use_container_width=True)


def page_cluster_and_predict(df, numeric_cols, categorical_cols, preprocess, result):
    st.header("Khám phá & Dự đoán phân khúc")

    dfc = result["df_clustered"]
    X_pca = result["X_pca"]

    # ====== PHẦN 1: TRỰC QUAN & BẢNG CHI TIẾT ======
    st.subheader("🌈 Trực quan PCA 2D theo phân khúc")

    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(dfc["cluster"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))

    for cl, color in zip(clusters, colors):
        mask = dfc["cluster"] == cl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=8,
            color=color,
            label=seg_label(cl),
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📄 Chi tiết theo phân khúc")
    selected = st.selectbox(
        "Chọn phân khúc để xem chi tiết:",
        clusters,
        format_func=seg_label,
    )
    st.dataframe(
        dfc[dfc["cluster"] == selected].reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")

    # ====== PHẦN 2: FORM NHẬP THÔNG TIN XE NGƯỜI DÙNG ======
    st.subheader("🛵 Nhập thông tin xe của bạn để dự đoán **phân khúc**")

    model = result["model"]

    # Giá trị mặc định numeric (median)
    defaults = {c: float(df[c].median()) for c in numeric_cols}

    # Giá trị categorical từ dữ liệu thực tế
    cat_options = {c: sorted(df[c].dropna().unique()) for c in categorical_cols}

    with st.form("form_predict"):
        # Hàng 1: Brand, Year, Engine Capacity
        row1 = st.columns(3)
        thuong_hieu = (
            row1[0].selectbox("Thương hiệu (Brand)", cat_options.get("Thương hiệu", []))
            if "Thương hiệu" in categorical_cols
            else row1[0].text_input("Thương hiệu (Brand)", "")
        )

        nam_dk = row1[1].number_input(
            "Năm đăng ký (Year of Registration)",
            value=int(df["Năm đăng ký"].median()) if "Năm đăng ký" in df.columns else 2018,
            step=1,
        )
        # Tuổi xe sẽ tính lại từ Năm đăng ký
        tuoi_tinh = 2025 - nam_dk

        dung_tich = (
            row1[2].selectbox("Dung tích xe (Engine Capacity)", cat_options.get("Dung tích xe", []))
            if "Dung tích xe" in categorical_cols
            else row1[2].text_input("Dung tích xe (Engine Capacity)", "")
        )

        # Hàng 2: Type, Km, Origin
        row2 = st.columns(3)
        loai_xe = (
            row2[0].selectbox("Loại xe (Type)", cat_options.get("Loại xe", []))
            if "Loại xe" in categorical_cols
            else row2[0].text_input("Loại xe (Type)", "")
        )

        km = row2[1].number_input(
            "Số Km đã đi (Kilometers Travelled)",
            value=float(defaults.get("Số Km đã đi", 30000.0)),
            min_value=0.0,
            step=1000.0,
        )

        xuat_xu = (
            row2[2].selectbox("Xuất xứ (Origin)", cat_options.get("Xuất xứ", []))
            if "Xuất xứ" in categorical_cols
            else row2[2].text_input("Xuất xứ (Origin)", "")
        )

        # Hàng 3: Tình trạng, Giá, Bảo hành
        row3 = st.columns(3)
        tinh_trang = (
            row3[0].selectbox("Tình trạng (Condition)", cat_options.get("Tình trạng", []))
            if "Tình trạng" in categorical_cols
            else row3[0].text_input("Tình trạng (Condition)", "")
        )

        gia = row3[1].number_input(
            "Giá (triệu VND) – Price (million VND)",
            value=float(defaults.get("Giá", 20.0)),
            min_value=0.0,
            step=1.0,
        )

        if "Chính sách bảo hành" in categorical_cols:
            bao_hanh = row3[2].selectbox(
                "Chính sách bảo hành (Warranty)",
                cat_options.get("Chính sách bảo hành", []),
            )
        else:
            bao_hanh = row3[2].text_input("Chính sách bảo hành (Warranty)", "")

        submitted = st.form_submit_button("🔍 Dự đoán phân khúc")

    if submitted:
        # Xây dựng DataFrame 1 dòng cho xe người dùng
        data_dict = {}

        for c in numeric_cols:
            if c == "Giá":
                data_dict[c] = gia
            elif c == "Tuổi xe":
                data_dict[c] = tuoi_tinh
            elif c == "Số Km đã đi":
                data_dict[c] = km

        for c in categorical_cols:
            if c == "Thương hiệu":
                data_dict[c] = thuong_hieu
            elif c == "Dòng xe":
                data_dict[c] = loai_xe  # nếu dữ liệu có Dòng xe riêng, có thể sửa lại
            elif c == "Tình trạng":
                data_dict[c] = tinh_trang
            elif c == "Loại xe":
                data_dict[c] = loai_xe
            elif c == "Dung tích xe":
                data_dict[c] = dung_tich
            elif c == "Xuất xứ":
                data_dict[c] = xuat_xu
            elif c == "Chính sách bảo hành":
                data_dict[c] = bao_hanh

        user_df = pd.DataFrame([data_dict])

        # Tiền xử lý & dự đoán cụm
        X_user = preprocess.transform(user_df)
        X_user_dense = X_user.toarray()
        cluster_user = int(model.predict(X_user_dense)[0])

        phan_khuc = seg_label(cluster_user)
        st.success(f"✅ Xe của bạn được xếp vào **{phan_khuc}**.")

        # Hiển thị nhanh thống kê phân khúc đó
        st.markdown("#### Đặc điểm thống kê của phân khúc này (theo dữ liệu thị trường):")
        summary = result["summary"]
        if cluster_user in summary.index:
            info = summary.loc[cluster_user].to_frame(name="Giá trị").round(2)
            st.dataframe(info, use_container_width=True)
        else:
            st.write("Không tìm thấy thống kê cho phân khúc này.")


def page_team():
    st.header("Thông tin nhóm thực hiện")
    st.markdown(
        """
**Nhóm học viên thực hiện:**

1. **Mai Bảo Ngọc**  
2. **Bùi Ngọc Toản**  
3. **Nguyễn Vũ Duy**
"""
    )


# =========
# MAIN APP
# =========

def main():
    st.set_page_config(page_title="Motorbike Clustering & Recommendation", layout="wide")

    st.title("Motorbike Clustering & Recommendation")

    st.sidebar.header("Menu")
    page = st.sidebar.radio(
        "Chọn trang:",
        [
            "Business Problem",
            "Evaluation & Report",
            "Khám phá & Dự đoán phân khúc",
            "Thông tin nhóm",
        ],
    )

    st.sidebar.header("Cấu hình dữ liệu")
    raw_path = st.sidebar.text_input(
        "Tên file dữ liệu (.xlsx / .csv):",
        value="data_motobikes_clean.xlsx",
    )

    K = st.sidebar.slider("Số phân khúc K (KMeans)", min_value=2, max_value=8, value=3, step=1)

    # Đường dẫn tuyệt đối tới file dữ liệu (cùng thư mục với app)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, raw_path.strip().strip('"').strip("'"))

    if not os.path.exists(data_path):
        st.error("❌ Không tìm thấy file dữ liệu. Vui lòng để file cùng thư mục với app.")
        return

    # Đọc dữ liệu & phân cụm
    with st.spinner("Đang tải dữ liệu và chạy mô hình phân cụm..."):
        df, numeric_cols, categorical_cols, preprocess, X_dense = load_and_prepare_data(
            data_path
        )
        result = run_kmeans(df, numeric_cols, X_dense, K)

    # Hiển thị từng trang
    if page == "Business Problem":
        page_business_problem()
    elif page == "Evaluation & Report":
        page_evaluation(result)
    elif page == "Khám phá & Dự đoán phân khúc":
        page_cluster_and_predict(df, numeric_cols, categorical_cols, preprocess, result)
    else:
        page_team()


if __name__ == "__main__":
    main()

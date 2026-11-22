"""Trang 'Mô hình' của app TNGT: thao tác với các mô hình cuối của Bước 3 (xây dựng mô hình).

Đọc các mô hình đã tinh chỉnh (bai_tap_lon/3_mo_hinh/mo_hinh_cuoi_*.joblib, do tinh_chinh.py tạo) và các bảng
trong schema mo_hinh của PostgreSQL. Không học lại mô hình nào.
"""
from __future__ import annotations

import sys
from itertools import cycle
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

GOC = Path(__file__).resolve().parents[1] / "bai_tap_lon"
MO_HINH = GOC / "3_mo_hinh"
sys.path[:0] = [str(GOC), str(MO_HINH), str(GOC / "2_tien_xu_ly")]
import csdl  # noqa: E402
import huan_luyen as hl  # noqa: E402  (cách tạo cột gốc + bộ mã hoá dùng chung với các script mô hình)

KHOA = ["nguon", "dong_goc"]
NHAN = "co_nguoi_chet"
TEN_HUONG = {"du_bao": "Dự báo (có tỉnh, huyện, đường)", "khong_dia_ly": "Không địa lý"}
THU = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]   # thu = 1..7, 1 = thứ 2
KHONG_RO_DUONG = "(không rõ / không ghi)"
TEN_COT = {"cap_duong": "Cấp đường", "khu_vuc": "Khu vực", "khung_gio": "Khung giờ", "vung": "Vùng", "mua": "Mùa",
           "dip_tet": "Dịp Tết", "giai_doan_covid": "Giai đoạn COVID", "giao_lo": "Tại giao lộ", "cau": "Trên cầu",
           "deo_doc": "Đèo dốc", "ly_trinh": "Có ghi lý trình (km)", "so_nha": "Có ghi số nhà",
           "noi_dong_nguoi": "Nơi đông người", "cuoi_tuan": "Cuối tuần", "gio_thieu": "Không rõ giờ"}


# ---------------------------------------------------------------- DỮ LIỆU VÀ MÔ HÌNH (có cache)

@st.cache_resource(show_spinner="Đang nạp mô hình...")
def nap(ten: str):
    duong_dan = MO_HINH / ten
    return joblib.load(duong_dan) if duong_dan.exists() else None


@st.cache_data(show_spinner=False)
def doc_bang(schema: str, ten: str) -> pd.DataFrame | None:
    try:
        return csdl.doc(schema, ten)
    except Exception:
        return None


@st.cache_data(show_spinner="Đang chuẩn bị dữ liệu...")
def khung_tat_ca() -> pd.DataFrame:
    """Cột gốc (như lúc học) của toàn bộ 33.062 vụ, kèm tập train/test."""
    khung = hl.kn.tao_khung(csdl.doc("tien_xu_ly", "tai_nan_loi_sach")).astype({"dong_goc": "int64"})
    chia = csdl.doc("mo_hinh", "chia_phan_loai")[KHOA + ["tap"]].astype({"dong_goc": "int64"})
    return khung.merge(chia, on=KHOA, validate="1:1").sort_values(KHOA).reset_index(drop=True)


@st.cache_data(show_spinner="Đang chấm tập test...")
def xac_suat_test(huong: str) -> tuple[np.ndarray, np.ndarray]:
    goi = nap(f"mo_hinh_cuoi_phan_loai_{huong}.joblib")
    d = khung_tat_ca()
    d = d[d["tap"] == "test"]
    return d[NHAN].to_numpy(int), goi["mo_hinh"].predict_proba(d[goi["cot"]])[:, 1]


@st.cache_data(show_spinner="Đang dự báo 2022...")
def du_bao_2022() -> pd.DataFrame:
    goi = nap("mo_hinh_cuoi_hoi_quy.joblib")
    o = hl.du_lieu_hoi_quy()
    te = o[o["tap"] == "test"].copy()
    te["du_bao"] = np.clip(goi["mo_hinh"].predict(te), 0, None)
    te["ky"] = te["ky"].astype(str)
    return te


def goc_ve_gia_tri(sin: float, cos: float, chu_ky: int) -> float:
    """Đổi cặp sin/cos về lại giá trị gốc (giờ, tháng-1, thứ-1)."""
    return (np.arctan2(sin, cos) % (2 * np.pi)) * chu_ky / (2 * np.pi)


def nhi_phan(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and set(s.dropna().unique()) <= {0, 1}


def giong_nhau(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    return bool(np.isclose(a, b)) if isinstance(a, (float, np.floating)) else a == b


def dien_mau(dong: pd.Series, cot: list[str], huong: str, d: pd.DataFrame) -> None:
    """Đưa giá trị của một vụ có thật vào các ô nhập (session_state)."""
    ss = st.session_state
    for c in cot:
        if c.endswith("_cos") or c not in dong.index:
            continue
        if c == "gio_so_sin":
            ss[f"{huong}_gio"] = int(round(goc_ve_gia_tri(dong["gio_so_sin"], dong["gio_so_cos"], 24))) % 24
        elif c == "thang_sin":
            ss[f"{huong}_thang"] = int(round(goc_ve_gia_tri(dong["thang_sin"], dong["thang_cos"], 12))) % 12 + 1
        elif c == "thu_sin":
            ss[f"{huong}_thu"] = THU[int(round(goc_ve_gia_tri(dong["thu_sin"], dong["thu_cos"], 7))) % 7]
        elif c == "khoa_huyen":
            ss[f"{huong}_huyen"] = str(dong[c]).split("|", 1)[-1]
        elif c == "khoa_duong":
            ss[f"{huong}_duong"] = str(dong[c]).split("|", 1)[-1] if pd.notna(dong[c]) else KHONG_RO_DUONG
        elif nhi_phan(d[c]):
            ss[f"{huong}_{c}"] = bool(dong[c])
        elif pd.api.types.is_numeric_dtype(d[c]):
            ss[f"{huong}_{c}"] = float(dong[c])
        else:
            ss[f"{huong}_{c}"] = str(dong[c])


# ---------------------------------------------------------------- TRANG

st.title(":material/model_training: Mô hình")
st.caption("Các mô hình cuối của Bước 3, đã tinh chỉnh ở bước 6 (tinh_chinh.py). Trang này chỉ dùng mô hình đã lưu, "
           "không học lại.")

thieu = [t for t in ("mo_hinh_cuoi_phan_loai_du_bao.joblib", "mo_hinh_cuoi_phan_loai_khong_dia_ly.joblib",
                     "mo_hinh_cuoi_hoi_quy.joblib", "mo_hinh_cuoi_phan_cum.joblib") if not (MO_HINH / t).exists()]
if thieu:
    st.error("Chưa có mô hình cuối: " + ", ".join(thieu) + ". Chạy lần lượt các script trong "
             "`bai_tap_lon/3_mo_hinh/` (đến `tinh_chinh.py`) rồi tải lại trang.", icon=":material/error:")
    st.stop()

tab_doan, tab_nguong, tab_hq, tab_cum, tab_luat = st.tabs(
    ["Đoán thử một vụ", "Kết quả phân loại & ngưỡng", "Dự báo số vụ 2022", "Phân cụm tỉnh", "Luật kết hợp"])

# ---- 1. ĐOÁN THỬ MỘT VỤ
with tab_doan:
    huong = st.segmented_control("Mô hình", list(TEN_HUONG), format_func=TEN_HUONG.get, default="du_bao",
                                 key="huong_doan") or "du_bao"
    goi = nap(f"mo_hinh_cuoi_phan_loai_{huong}.joblib")
    cot = goi["cot"]
    d = khung_tat_ca()
    tham_so = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in goi["tham_so"].items())
    st.caption(f"{goi['ten']} ({tham_so}) · {len(cot)} cột gốc")

    test = d[d["tap"] == "test"].reset_index(drop=True)
    if st.button("Lấy một vụ có thật trong tập test", icon=":material/shuffle:"):
        i = int(np.random.default_rng().integers(len(test)))
        dien_mau(test.loc[i], cot, huong, d)
        st.session_state[f"{huong}_mau"] = i

    ss = st.session_state
    vao: dict = {}
    with st.container(border=True):
        o_nhap = cycle(st.columns(3))

        def o_tiep():
            return next(o_nhap)

        if "tinh" in cot:
            tinh_ds = sorted(d["tinh"].dropna().unique())
            tinh = o_tiep().selectbox("Tỉnh / thành phố", tinh_ds, key=f"{huong}_tinh")
            vao["tinh"] = tinh
            if "khoa_huyen" in cot:
                huyen_ds = sorted(d.loc[d["tinh"] == tinh, "khoa_huyen"].dropna().str.split("|", n=1).str[-1].unique())
                if ss.get(f"{huong}_huyen") not in huyen_ds:
                    ss.pop(f"{huong}_huyen", None)
                huyen = o_tiep().selectbox("Quận / huyện", huyen_ds, key=f"{huong}_huyen")
                vao["khoa_huyen"] = f"{tinh}|{huyen}"
            if "khoa_duong" in cot:
                duong_ds = [KHONG_RO_DUONG] + sorted(
                    d.loc[d["tinh"] == tinh, "khoa_duong"].dropna().str.split("|", n=1).str[-1].unique())
                if ss.get(f"{huong}_duong") not in duong_ds:
                    ss.pop(f"{huong}_duong", None)
                duong = o_tiep().selectbox("Tên đường", duong_ds, key=f"{huong}_duong")
                vao["khoa_duong"] = np.nan if duong == KHONG_RO_DUONG else f"{tinh}|{duong}"
        if "gio_so_sin" in cot:
            gio = o_tiep().slider("Giờ xảy ra", 0, 23, 20, key=f"{huong}_gio")
            a = 2 * np.pi * gio / 24
            vao["gio_so_sin"], vao["gio_so_cos"] = np.sin(a), np.cos(a)
        if "thang_sin" in cot:
            thang = o_tiep().slider("Tháng", 1, 12, 6, key=f"{huong}_thang")
            a = 2 * np.pi * (thang - 1) / 12
            vao["thang_sin"], vao["thang_cos"] = np.sin(a), np.cos(a)
        if "thu_sin" in cot:
            thu = o_tiep().selectbox("Thứ", THU, key=f"{huong}_thu")
            a = 2 * np.pi * THU.index(thu) / 7
            vao["thu_sin"], vao["thu_cos"] = np.sin(a), np.cos(a)
        for c in cot:
            if c in vao or c.endswith(("_sin", "_cos")) or c in ("tinh", "khoa_huyen", "khoa_duong"):
                continue
            if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
                ds = sorted(d[c].dropna().astype(str).unique())
                vao[c] = o_tiep().selectbox(TEN_COT.get(c, c), ds, key=f"{huong}_{c}")
            elif nhi_phan(d[c]):
                vao[c] = int(o_tiep().checkbox(TEN_COT.get(c, c), key=f"{huong}_{c}"))
            else:
                vao[c] = o_tiep().number_input(TEN_COT.get(c, c), value=float(d[c].median()), key=f"{huong}_{c}")

    x = pd.DataFrame([vao])[cot].astype(d[cot].dtypes.to_dict())   # cùng kiểu cột như lúc học (NaN của tên đường)
    xs = float(goi["mo_hinh"].predict_proba(x)[0, 1])
    nen = d.loc[d["tap"] == "train", NHAN].mean()
    with st.container(horizontal=True):
        st.metric("Xác suất có người chết", f"{xs:.0%}", f"{(xs - nen) * 100:+.0f} điểm % so với trung bình",
                  delta_color="inverse", border=True)
        st.metric("Mô hình đoán", "Có người chết" if xs >= 0.5 else "Không có người chết", border=True)
        if f"{huong}_mau" in ss:
            thuc = test.loc[ss[f"{huong}_mau"]]
            if all(giong_nhau(vao[c], thuc[c]) for c in cot):
                st.metric("Thực tế (vụ lấy từ test)", "Có người chết" if thuc[NHAN] else "Không có người chết",
                          border=True)
    st.caption(f"Trung bình của train: {nen:.1%} vụ có người chết. Ngưỡng đoán: 0,5. Tên tỉnh, huyện, đường lấy từ dữ "
               "liệu; nơi có ít vụ thì mô hình kéo về tỉ lệ chung. Kết quả là mức độ liên quan trong dữ liệu báo cáo, "
               "không phải nguyên nhân.")

# ---- 2. KẾT QUẢ PHÂN LOẠI & NGƯỠNG
with tab_nguong:
    kq = doc_bang("mo_hinh", "ket_qua_test")
    if kq is not None:
        bang = (kq[kq["bai_toan"] == "Phân loại"]
                .pivot_table(index=["huong", "mo_hinh"], columns="chi_so", values="gia_tri").reset_index())
        chon = kq[(kq["bai_toan"] == "Phân loại") & kq["duoc_chon"].astype(bool)][["huong", "mo_hinh"]].drop_duplicates()
        bang["Được chọn"] = bang.set_index(["huong", "mo_hinh"]).index.isin(chon.set_index(["huong", "mo_hinh"]).index)
        bang["huong"] = bang["huong"].map(TEN_HUONG)
        st.subheader("Điểm trên tập test (6.613 vụ)")
        st.dataframe(bang[["huong", "mo_hinh", "Được chọn", "Accuracy", "Precision", "Recall", "F1", "AUC"]],
                     hide_index=True, column_config={"huong": "Hướng", "mo_hinh": "Mô hình",
                                                     **{c: st.column_config.NumberColumn(c, format="%.3f")
                                                        for c in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}})
        st.caption("Mô hình được chọn theo điểm kiểm định chéo, trước khi xem tập test; các dòng khác chỉ để tham khảo.")

    st.subheader("Đổi ngưỡng đoán")
    h2 = st.segmented_control("Mô hình", list(TEN_HUONG), format_func=TEN_HUONG.get, default="du_bao",
                              key="huong_nguong") or "du_bao"
    y, p = xac_suat_test(h2)
    nguong = st.slider("Đoán “có người chết” khi xác suất ≥", 0.20, 0.80, 0.50, 0.01)
    du = (p >= nguong).astype(int)
    tp, fp = int(((du == 1) & (y == 1)).sum()), int(((du == 1) & (y == 0)).sum())
    fn, tn = int(((du == 0) & (y == 1)).sum()), int(((du == 0) & (y == 0)).sum())
    with st.container(horizontal=True):
        st.metric("Accuracy", f"{(tp + tn) / len(y):.3f}", border=True)
        st.metric("Precision", f"{tp / (tp + fp):.3f}" if tp + fp else "—", border=True)
        st.metric("Recall (bắt được vụ có chết)", f"{tp / (tp + fn):.3f}", border=True)
    st.dataframe(pd.DataFrame({"Đoán: không chết": [tn, fn], "Đoán: có chết": [fp, tp]},
                              index=["Thật: không chết", "Thật: có chết"]))
    st.caption(f"Hạ ngưỡng thì bắt được nhiều vụ có người chết hơn (Recall tăng) nhưng báo nhầm nhiều hơn (Precision "
               f"giảm). Ở ngưỡng {nguong:.2f}: bỏ sót {fn:,d} / {tp + fn:,d} vụ có người chết, báo nhầm {fp:,d} vụ."
               .replace(",", "."))

# ---- 3. DỰ BÁO SỐ VỤ 2022
with tab_hq:
    te = du_bao_2022()
    goi_hq = nap("mo_hinh_cuoi_hoi_quy.joblib")
    vung = st.pills("Vùng", sorted(te["vung"].unique()), selection_mode="multi", key="vung_hq")
    x = te[te["vung"].isin(vung)] if vung else te
    thang = x.groupby("ky")[["so_vu", "du_bao"]].sum().reset_index()
    dai = thang.melt("ky", var_name="loai", value_name="so").replace({"so_vu": "Thực tế", "du_bao": "Dự báo"})
    bieu_do = alt.Chart(dai).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("ky:N", title="Tháng"), y=alt.Y("so:Q", title="Số vụ", scale=alt.Scale(zero=True)),
        color=alt.Color("loai:N", title=None, scale=alt.Scale(domain=["Thực tế", "Dự báo"],
                                                               range=["#1f6feb", "#d9480f"])),
        strokeDash=alt.StrokeDash("loai:N", legend=None, scale=alt.Scale(domain=["Thực tế", "Dự báo"],
                                                                          range=[[1, 0], [6, 3]])),
        tooltip=[alt.Tooltip("ky:N", title="Tháng"), alt.Tooltip("loai:N", title=""),
                 alt.Tooltip("so:Q", title="Số vụ", format=",.0f")])
    st.altair_chart(bieu_do, alt="Số vụ theo tháng năm 2022: thực tế và dự báo")
    sai = te["so_vu"] - te["du_bao"]
    with st.container(horizontal=True):
        st.metric("RMSE (vụ / ô)", f"{np.sqrt((sai ** 2).mean()):.2f}".replace(".", ","), border=True)
        st.metric("MAE (vụ / ô)", f"{sai.abs().mean():.2f}".replace(".", ","), border=True)
        st.metric("R²", f"{1 - (sai ** 2).sum() / ((te['so_vu'] - te['so_vu'].mean()) ** 2).sum():.3f}"
                  .replace(".", ","), border=True)
    st.caption(f"Ridge (alpha = {goi_hq['alpha']:g}) học trên 01/2020–12/2021, dự báo 01–07/2022. Mỗi ô là tháng × vùng "
               "× khung giờ × thứ; chỉ số trên tính cho cả nước. Mô hình không lường được cú sốc như giãn cách 2021.")

# ---- 4. PHÂN CỤM TỈNH
with tab_cum:
    goi_cum = nap("mo_hinh_cuoi_phan_cum.joblib")
    cum = doc_bang("mo_hinh", "cum_tinh_cuoi")
    tam = pd.DataFrame(goi_cum["mo_hinh"].cluster_centers_, index=range(1, goi_cum["k"] + 1),
                       columns=[c.removesuffix("_z") for c in goi_cum["cot"]]).round(2)
    tam.insert(0, "Số tỉnh", cum["cum"].value_counts().sort_index())
    st.subheader(f"{goi_cum['k']} cụm, tâm cụm theo độ lệch chuẩn")
    st.dataframe(tam, column_config={c: st.column_config.NumberColumn(c, format="%+.2f") for c in tam.columns[1:]})
    st.caption("Giá trị là số độ lệch chuẩn so với trung bình 63 tỉnh (+ cao hơn, − thấp hơn). Silhouette 0,18 và "
               "độ ổn định ARI 0,45: các tỉnh không chia thành nhóm tách bạch, nên đọc cụm như xu hướng, không như ranh giới.")
    c = st.selectbox("Xem tỉnh trong cụm", sorted(cum["cum"].unique()), format_func=lambda k: f"Cụm {k}")
    st.write(", ".join(sorted(cum.loc[cum["cum"] == c, "tinh"])))

# ---- 5. LUẬT KẾT HỢP
with tab_luat:
    luat = doc_bang("mo_hinh", "luat_cuoi")
    phai = st.segmented_control("Vế phải", ["co_nguoi_chet=1", "co_nguoi_chet=0"],
                                format_func={"co_nguoi_chet=1": "→ có người chết",
                                             "co_nguoi_chet=0": "→ không có người chết"}.get,
                                default="co_nguoi_chet=1", key="ve_phai")
    tim = st.text_input("Tìm trong vế trái", placeholder="ví dụ: Quốc lộ, Mùa khô, Hồ Chí Minh")
    x = luat[luat["ve_phai"] == phai] if phai else luat
    if tim:
        x = x[x["ve_trai"].str.contains(tim, case=False, regex=False)]
    x = x.assign(support=x["support"] * 100, confidence=x["confidence"] * 100)
    st.dataframe(x[["ve_trai", "ve_phai", "support", "confidence", "lift"]].sort_values("lift", ascending=False),
                 hide_index=True, column_config={
                     "ve_trai": st.column_config.TextColumn("Vế trái (điều kiện)", width="large"),
                     "ve_phai": "Vế phải",
                     "support": st.column_config.NumberColumn("Support", format="%.1f%%"),
                     "confidence": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
                     "lift": st.column_config.NumberColumn("Lift", format="%.2f")})
    st.caption("Ngưỡng: support ≥ 5%, confidence ≥ 0,55, tối đa 3 item; bỏ luật thừa. Lift > 1 nghĩa là điều kiện đi "
               "cùng kết quả nhiều hơn mức trung bình. Luật mô tả dữ liệu báo cáo, không phải quan hệ nhân quả.")

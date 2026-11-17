"""ỨNG DỤNG STREAMLIT — TIỀN XỬ LÝ DỮ LIỆU TAI NẠN GIAO THÔNG ĐƯỜNG BỘ VIỆT NAM (Nhóm 6)

Trình bày lại toàn bộ phần Chuẩn bị dữ liệu (Data Preparation) đã chạy bằng các script:
    1_thu_thap_du_lieu/thu_thap.py        -> tab "1. Thu thập"
    2_tien_xu_ly/tien_xu_ly.py            -> tab "2a. Làm sạch"
    2_tien_xu_ly/bien_doi.py, ky_nghe_dac_trung.py, pipeline.py
                                          -> tab "3. Thử pipeline", "4. Kỹ nghệ đặc trưng", "5. Đầu ra"
Ứng dụng CHỈ ĐỌC các file kết quả, không ghi gì vào dự án, không có mô hình dự báo.

Chạy (từ D:/PDM, sau khi đã chạy đủ các script trên):
       .venv/Scripts/python -m streamlit run bai_tap_lon/app_tien_xu_ly.py
"""
from __future__ import annotations

import sys
import unicodedata
from datetime import date, time
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.utils import estimator_html_repr

GOC = Path(__file__).resolve().parent
B1, B2 = GOC / "1_thu_thap_du_lieu", GOC / "2_tien_xu_ly"
for thu_muc in (B2, B1):
    sys.path.insert(0, str(thu_muc))

from ky_nghe_dac_trung import CO_MOI, KHONG_RO, NHAN, TaoDacTrungDong, tao_khung  # noqa: E402
from thu_thap import rut_dac_trung_chung  # noqa: E402
from tien_xu_ly import sua_vi_tri_dau  # noqa: E402

KHOA = ["nguon", "dong_goc"]
HUONG = {"du_bao": "Dự báo (có mã hoá tỉnh/huyện/đường)", "khong_dia_ly": "Không địa lý (để giải thích)"}
MAU_CHINH = "#2a78d6"

GIAI_THICH = {
    "khung_gio": "khung giờ (nhóm A, bước 1)", "vung": "vùng của tỉnh (nhóm A)", "mua": "mùa mưa/khô theo vùng + tháng",
    "dip_tet": "vị trí so với đợt nghỉ Tết", "giai_doan_covid": "giai đoạn giãn cách", "cap_duong": "cấp đường",
    "khu_vuc": "nội thị / nông thôn", "gio_so_sin": "sin(2π·giờ/24)", "gio_so_cos": "cos(2π·giờ/24)",
    "thang_sin": "sin(2π·(tháng−1)/12)", "thang_cos": "cos(2π·(tháng−1)/12)", "thu_sin": "sin(2π·(thứ−1)/7)",
    "thu_cos": "cos(2π·(thứ−1)/7)", "gio_thieu": "1 nếu không có giờ", "cuoi_tuan": "1 nếu Thứ Bảy/Chủ nhật",
    "giao_lo": "địa điểm có ngã ba/tư, giao lộ, vòng xoay…", "cau": "địa điểm có chữ 'cầu'",
    "deo_doc": "địa điểm có 'đèo'/'dốc'", "ly_trinh": "có lý trình km (vd 82+500)",
    "so_nha": "địa chỉ bắt đầu bằng số nhà", "noi_dong_nguoi": "chợ, trường, bến xe, KCN, bệnh viện…",
    "dip_le": "trong ±2 ngày quanh 1/1, 30/4, 1/5, 2/9", "dem": "giờ ≥ 18 hoặc < 6",
    "dem_x_quoc_lo": "đêm VÀ quốc lộ/cao tốc", "dem_x_cuoi_tuan": "đêm VÀ cuối tuần",
    "tinh": "tỉnh", "khoa_huyen": "tỉnh|huyện", "khoa_duong": "tỉnh|tên đường đã chuẩn hoá",
}

st.set_page_config(page_title="Tiền xử lý dữ liệu TNGT – Nhóm 6", layout="wide")


# ---------------------------------------------------------------------------
# Đọc dữ liệu (có cache)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang đọc dữ liệu…")
def doc(thu_muc: str, ten: str) -> pd.DataFrame | None:
    duong_dan = Path(thu_muc) / f"{ten}.parquet"
    return pd.read_parquet(duong_dan) if duong_dan.exists() else None


@st.cache_data
def doc_txt(duong_dan: str) -> str | None:
    p = Path(duong_dan)
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_resource
def nap_joblib(duong_dan: str):
    p = Path(duong_dan)
    return joblib.load(p) if p.exists() else None


@st.cache_data(show_spinner="Đang tạo đặc trưng cho toàn bộ dữ liệu…")
def khung_day_du(_sach: pd.DataFrame, so_dong: int) -> pd.DataFrame:
    return tao_khung(_sach)


def can(df, ten: str, script: str) -> bool:
    if df is None:
        st.warning(f"Chưa có **{ten}** — hãy chạy `{script}` trước.")
        return False
    return True


def bieu_do_cot(df: pd.DataFrame, nhan: str, gia_tri: str, tieu_de_truc: str, dinh_dang: str = ",.0f",
                ngang: bool = True, sap_xep: bool = True, cao: int | None = None):
    """Biểu đồ cột một màu, có tooltip; dùng cho mọi so sánh độ lớn trong ứng dụng."""
    thu_tu = "-x" if ngang else None
    if ngang:
        enc = dict(y=alt.Y(f"{nhan}:N", sort=thu_tu if sap_xep else None, title=None),
                   x=alt.X(f"{gia_tri}:Q", title=tieu_de_truc))
    else:
        enc = dict(x=alt.X(f"{nhan}:N", sort=None, title=None), y=alt.Y(f"{gia_tri}:Q", title=tieu_de_truc))
    c = (alt.Chart(df).mark_bar(color=MAU_CHINH, cornerRadiusEnd=4)
         .encode(**enc, tooltip=[alt.Tooltip(f"{nhan}:N", title="Nhóm"),
                                 alt.Tooltip(f"{gia_tri}:Q", title=tieu_de_truc, format=dinh_dang)]))
    st.altair_chart(c.properties(height=cao or max(160, 26 * len(df)) if ngang else 280), width="stretch")


def nhung_txt(tieu_de: str, duong_dan: Path) -> None:
    noi_dung = doc_txt(str(duong_dan))
    with st.expander(tieu_de):
        st.code(noi_dung, language=None) if noi_dung else st.caption(f"Chưa có {duong_dan.name}")


# ---------------------------------------------------------------------------
loi = doc(str(B1), "bang_loi")
sach = doc(str(B2), "tai_nan_loi_sach")

st.title("Tiền xử lý dữ liệu tai nạn giao thông đường bộ Việt Nam")
st.caption("Nhóm 6 · Khai thác dữ liệu · 15/12/2019 – 14/08/2022 · Bảng lõi (11 thuộc tính chung + 12 đặc trưng nhóm A)")
with st.sidebar:
    st.header("Quy trình")
    st.markdown(
        "1. **Thu thập** — 3 file Excel của Cục CSGT → bảng lõi\n"
        "2. **Tiền xử lý**\n"
        "   - a. Làm sạch — trùng lặp, nhiễu, ngoại lai, thiếu\n"
        "   - b. Biến đổi — sin/cos, One-Hot, chia train/test\n"
        "   - c. Kỹ nghệ đặc trưng — đặc trưng mới, Target Encoding, chọn lọc\n"
        "   - d. Pipeline — gói b–c vào một đối tượng scikit-learn")
    st.info("Ứng dụng chỉ trình bày phần **tiền xử lý**. Chưa có mô hình dự báo.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Thu thập", "2a. Làm sạch", "2b. Thử pipeline", "2c. Kỹ nghệ đặc trưng", "2d. Đầu ra"])

# ---------------------------------------------------------------------------
with tab1:
    if can(loi, "bang_loi.parquet", "thu_thap.py"):
        ngay = pd.to_datetime(loi["ngay"], errors="coerce")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số vụ (bảng lõi)", f"{len(loi):,d}")
        c2.metric("Số cột", loi.shape[1])
        c3.metric("Số nguồn", loi["nguon"].nunique())
        c4.metric("Số tỉnh", loi["tinh"].nunique())
        st.markdown(f"Khoảng thời gian: **{ngay.min():%d/%m/%Y} – {ngay.max():%d/%m/%Y}**")
        st.subheader("Số vụ theo nguồn")
        bieu_do_cot(loi["nguon"].astype(str).value_counts().rename_axis("nguon").reset_index(name="so_vu"),
                    "nguon", "so_vu", "Số vụ", sap_xep=False)
        st.subheader("Số vụ theo tháng")
        theo_thang = ngay.dt.to_period("M").value_counts().sort_index()
        df_thang = pd.DataFrame({"thang": theo_thang.index.to_timestamp(), "so_vu": theo_thang.to_numpy()})
        st.altair_chart(
            alt.Chart(df_thang).mark_line(color=MAU_CHINH, strokeWidth=2, point=True)
            .encode(x=alt.X("thang:T", title=None, axis=alt.Axis(format="%m/%Y")),
                    y=alt.Y("so_vu:Q", title="Số vụ"),
                    tooltip=[alt.Tooltip("thang:T", title="Tháng", format="%m/%Y"),
                             alt.Tooltip("so_vu:Q", title="Số vụ", format=",d")])
            .properties(height=280), width="stretch")
        st.caption("Tháng đầu (12/2019) và tháng cuối (08/2022) không trọn tháng — phần làm sạch gắn cờ thang_khong_tron.")
        nhung_txt("Báo cáo kiểm tra bước thu thập (kiem_tra.txt)", B1 / "kiem_tra.txt")

# ---------------------------------------------------------------------------
with tab2:
    if can(loi, "bang_loi.parquet", "thu_thap.py") and can(sach, "tai_nan_loi_sach.parquet", "tien_xu_ly.py"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trước làm sạch", f"{len(loi):,d} vụ")
        c2.metric("Sau làm sạch", f"{len(sach):,d} vụ", delta=f"{len(sach) - len(loi):,d} trùng hoàn toàn")
        c3.metric("Giờ sai định dạng đã sửa", f"{int(pd.to_numeric(sach['gio_da_sua']).sum()):,d}")
        c4.metric("Giờ mặc định 00:00:00 (gắn cờ)", f"{int(pd.to_numeric(sach['co_gio_mac_dinh']).sum()):,d}")

        st.subheader("Xem trước → sau theo từng cột")
        st.caption("So từng ô của cùng một vụ (khoá nguon + dong_goc) giữa bảng lõi thô và bảng sạch.")
        cot = st.selectbox("Cột", ["gio", "tinh", "huyen", "xa", "ten_duong", "km", "dia_diem", "cap_duong", "khu_vuc"])
        gop = loi[KHOA + [cot]].merge(sach[KHOA + [cot]], on=KHOA, suffixes=("_truoc", "_sau"))
        truoc = gop[f"{cot}_truoc"].astype("string").fillna("(trống)")
        sau = gop[f"{cot}_sau"].astype("string").fillna("(trống)")
        khac = truoc != sau
        st.markdown(f"**{int(khac.sum()):,d}** ô của cột `{cot}` bị thay đổi.")
        if khac.any():
            bang = (pd.DataFrame({"Trước": truoc[khac], "Sau": sau[khac]}).value_counts()
                    .rename("Số ô").reset_index().head(100))
            st.dataframe(bang, width="stretch", hide_index=True)

        st.subheader("Bản ghi bị xoá vì trùng hoàn toàn")
        k_sach = set(sach["nguon"].astype(str) + "#" + sach["dong_goc"].astype(str))
        bi_xoa = loi[~(loi["nguon"].astype(str) + "#" + loi["dong_goc"].astype(str)).isin(k_sach)]
        st.dataframe(bi_xoa[KHOA + ["ngay", "gio", "tinh", "huyen", "loai_duong", "so_chet", "so_bi_thuong"]],
                     width="stretch", hide_index=True)

        st.subheader("Ô trống trước → sau")
        dong = []
        for c in ["gio_so", "khung_gio", "vung", "mua", "cap_duong", "khu_vuc", "ten_duong", "km"]:
            if c in loi and c in sach:
                dong.append({"Cột": c, "Trống trước": loi[c].isna().mean(), "Trống sau": sach[c].isna().mean(),
                             f"'{KHONG_RO}' sau": (sach[c].astype("string") == KHONG_RO).mean()})
        st.dataframe(pd.DataFrame(dong).style.format({c: "{:.1%}" for c in ["Trống trước", "Trống sau",
                                                                              f"'{KHONG_RO}' sau"]}),
                     width="stretch", hide_index=True)
        st.caption("cap_duong / khu_vuc trống được điền 'Không rõ' (giữ thành một nhóm riêng, không xoá dòng). "
                   "ten_duong / km để trống vì là văn bản tự do.")
        nhung_txt("Báo cáo kiểm tra bước làm sạch (kiem_tra_tien_xu_ly.txt)", B2 / "kiem_tra_tien_xu_ly.txt")

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Đưa một vụ đi qua pipeline tiền xử lý")
    huong = st.radio("Pipeline", list(HUONG), format_func=HUONG.get, horizontal=True)
    pl = nap_joblib(str(B2 / f"pipeline_tien_xu_ly_{huong}.joblib"))
    if can(sach, "tai_nan_loi_sach.parquet", "tien_xu_ly.py") and can(pl, f"pipeline_tien_xu_ly_{huong}.joblib",
                                                                      "pipeline.py"):
        nguon_vao = st.radio("Bản ghi", ["Lấy một vụ có sẵn", "Tự nhập một vụ mới"], horizontal=True)
        if nguon_vao == "Lấy một vụ có sẵn":
            c1, c2 = st.columns([1, 3])
            if c1.button("Chọn ngẫu nhiên"):
                st.session_state["vi_tri"] = int(np.random.default_rng().integers(len(sach)))
            vi_tri = c2.number_input("Vị trí dòng trong bảng sạch", 0, len(sach) - 1,
                                     st.session_state.get("vi_tri", 0))
            ban_ghi = sach.iloc[[int(vi_tri)]].reset_index(drop=True)
            st.caption(f"Nhãn thực tế co_nguoi_chet = {ban_ghi[NHAN].iloc[0]} — pipeline KHÔNG dùng nhãn khi biến đổi.")
        else:
            c1, c2, c3 = st.columns(3)
            ngay_nhap = c1.date_input("Ngày", date(2022, 9, 2), min_value=date(2019, 12, 15),
                                      max_value=date(2022, 12, 31), format="DD/MM/YYYY")
            gio_nhap = c2.time_input("Giờ", time(21, 30))
            ds_tinh = sorted(sach["tinh"].dropna().unique())
            tinh = c3.selectbox("Tỉnh", ds_tinh, index=ds_tinh.index("Hà Nội") if "Hà Nội" in ds_tinh else 0)
            ds_huyen = sorted(sach.loc[sach["tinh"] == tinh, "huyen"].dropna().unique())
            huyen = c1.selectbox("Huyện", ds_huyen)
            loai_duong = c2.selectbox("Loại đường", sorted(sach["loai_duong"].dropna().unique()))
            ten_duong = c3.text_input("Tên đường", "Quốc lộ 3")
            km = c1.text_input("Km / số nhà", "Km 25+300")
            vi_tri_cu_the = c2.text_input("Vị trí cụ thể", "Ngã tư Phù Lỗ")
            xa = c3.text_input("Xã / phường", "")
            chuan = lambda s: sua_vi_tri_dau(unicodedata.normalize("NFC", s.strip())) if s and s.strip() else pd.NA
            dia_diem = ", ".join(x for x in [vi_tri_cu_the, xa, huyen, tinh] if x and x.strip())
            tho = pd.DataFrame({
                "nguon": ["moi"], "dong_goc": [0], "ngay": [pd.Timestamp(ngay_nhap)],
                "gio": pd.array([gio_nhap.strftime("%H:%M:00")], dtype="string"),
                "tinh": pd.array([tinh], dtype="string"), "huyen": pd.array([huyen], dtype="string"),
                "xa": pd.array([chuan(xa)], dtype="string"), "loai_duong": pd.array([loai_duong], dtype="string"),
                "ten_duong": pd.array([chuan(ten_duong)], dtype="string"), "km": pd.array([chuan(km)], dtype="string"),
                "dia_diem": pd.array([chuan(dia_diem)], dtype="string"), "so_chet": [np.nan]})
            ban_ghi, _ = rut_dac_trung_chung(tho)
            for c in ("cap_duong", "khu_vuc"):
                ban_ghi[c] = ban_ghi[c].fillna(KHONG_RO)
            st.caption("Bản ghi mới đi qua đúng các hàm của bước 1 (rut_dac_trung_chung — đặc trưng nhóm A) "
                       "và phần làm sạch (chuẩn hoá Unicode, sửa vị trí dấu, điền 'Không rõ') trước khi vào pipeline.")

        a, b, c = st.columns([1, 1.2, 1.2])
        with a:
            st.markdown("**① Đầu vào — dòng bảng lõi sạch**")
            hien = ["ngay", "gio", "tinh", "huyen", "loai_duong", "ten_duong", "km", "dia_diem", "gio_so",
                    "khung_gio", "thu", "cuoi_tuan", "thang", "vung", "mua", "dip_tet", "giai_doan_covid",
                    "cap_duong", "khu_vuc"]
            v = ban_ghi[[x for x in hien if x in ban_ghi]].iloc[0].astype(str)
            st.dataframe(v.rename("giá trị").to_frame(), width="stretch", height=700)
        with b:
            st.markdown("**② Sau `dac_trung_dong` (TaoDacTrungDong) — không học**")
            dong = TaoDacTrungDong().transform(ban_ghi).drop(columns=KHOA).iloc[0]
            st.dataframe(pd.DataFrame({"giá trị": dong.map(lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)),
                                       "ý nghĩa": [GIAI_THICH.get(k, "") for k in dong.index]}),
                         width="stretch", height=700)
        with c:
            st.markdown("**③ Sau `chon_ma_hoa` (ColumnTransformer đã học trên train)**")
            ten = pl.named_steps["chon_ma_hoa"].get_feature_names_out()
            X = pd.Series(pl.transform(ban_ghi)[0], index=ten)
            st.dataframe(pd.DataFrame({"giá trị": X.round(4)}).style.apply(
                lambda s: ["font-weight: 700" if v != 0 else "color: #8a8a8a" for v in s], subset=["giá trị"]),
                width="stretch", height=700)
            st.caption(f"{len(X)} cột, {int((X != 0).sum())} cột khác 0 (in đậm). "
                       + ("Cột tinh / khoa_huyen / khoa_duong = tỉ lệ có người chết (đã làm mượt) học từ train; "
                          "tan_suat_* = log(1 + số vụ cùng huyện/đường trong train)." if huong == "du_bao" else
                          "Hướng này không dùng bất kỳ thông tin tỉnh/huyện/đường nào."))

# ---------------------------------------------------------------------------
with tab4:
    cau_hinh = nap_joblib(str(B2 / "bo_dac_trung.joblib"))
    if can(sach, "tai_nan_loi_sach.parquet", "tien_xu_ly.py"):
        st.subheader("10 đặc trưng mới theo dòng — tỉ lệ có người chết khi có / không có")
        x = khung_day_du(sach, len(sach))
        dong = []
        for c in CO_MOI:
            co = x[c] == 1
            dong.append({"Đặc trưng": c, "Ý nghĩa": GIAI_THICH[c], "Số vụ": int(co.sum()), "Tỉ lệ vụ": co.mean(),
                         "Có chết khi CÓ": x.loc[co, NHAN].mean() if co.any() else np.nan,
                         "Có chết khi KHÔNG": x.loc[~co, NHAN].mean()})
        st.dataframe(pd.DataFrame(dong).style.format({"Số vụ": "{:,d}", "Tỉ lệ vụ": "{:.1%}",
                                                      "Có chết khi CÓ": "{:.1%}", "Có chết khi KHÔNG": "{:.1%}"}),
                     width="stretch", hide_index=True)
    if can(cau_hinh, "bo_dac_trung.joblib", "ky_nghe_dac_trung.py"):
        st.subheader("So sánh 4 bộ đặc trưng — kiểm định chéo 5 lần trên train")
        dong = []
        for bo, kq in cau_hinh["ket_qua_cv"].items():
            dong.append({"Bộ đặc trưng": bo, "AUC Logistic": kq["logistic"]["auc"], "F1 Logistic": kq["logistic"]["f1"],
                         "AUC HistGB": kq["histgb"]["auc"], "F1 HistGB": kq["histgb"]["f1"]})
        bang = pd.DataFrame(dong)
        st.dataframe(bang.style.format({c: "{:.3f}" for c in bang.columns[1:]}), width="stretch",
                     hide_index=True)
        bieu_do_cot(bang, "Bộ đặc trưng", "AUC HistGB", "AUC (HistGB)", ".3f", sap_xep=False)
        st.caption("AUC = 0,5 là đoán ngẫu nhiên. Chênh lệch giữa 'ma_hoa_dia_ly' và 'khong_dia_ly' là phần "
                   "dự báo đến từ thông tin tỉnh/huyện/đường — có khả năng cao phản ánh thói quen báo cáo "
                   "của địa phương hơn là mức nguy hiểm thật.")

        st.subheader("Chọn lọc đặc trưng — AUC giảm bao nhiêu khi tráo ngẫu nhiên một nhóm cột")
        h = st.radio("Hướng", list(HUONG), format_func=HUONG.get, horizontal=True, key="huong_tq")
        ch = cau_hinh["huong"][h]
        qt = ch.get("tam_quan_trong")
        if qt is None:
            st.info("bo_dac_trung.joblib chưa lưu mức quan trọng — chạy lại `ky_nghe_dac_trung.py`.")
        else:
            qt = qt.assign(trang_thai=np.where(qt["nhom"].isin(ch["nhom"]), "Giữ", "Bỏ"))
            bieu_do_cot(qt, "nhom", "auc_giam", "AUC giảm khi tráo", ".4f")
            st.dataframe(qt.rename(columns={"nhom": "Nhóm", "auc_giam": "AUC giảm", "sd": "Độ lệch chuẩn",
                                            "trang_thai": "Kết quả"}).drop(columns="bo", errors="ignore")
                         .style.format({"AUC giảm": "{:+.4f}", "Độ lệch chuẩn": "{:.4f}"}),
                         width="stretch", hide_index=True)
            st.caption("Bỏ nhóm nếu AUC giảm không vượt quá max(2 × độ lệch chuẩn, 0,0001), rồi kiểm định chéo "
                       "lại để chắc AUC không giảm quá 0,002.")
        nhung_txt("Báo cáo kỹ nghệ đặc trưng (kiem_tra_ky_nghe_dac_trung.txt)", B2 / "kiem_tra_ky_nghe_dac_trung.txt")

# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Các bảng đầu ra của phần tiền xử lý")
    DS = [("train", "Biến đổi – train (one-hot + sin/cos)"), ("test", "Biến đổi – test"),
          ("train_fe", "Kỹ nghệ đặc trưng – train, hướng dự báo"), ("test_fe", "Kỹ nghệ đặc trưng – test, hướng dự báo"),
          ("train_fe_khong_dia_ly", "Kỹ nghệ đặc trưng – train, không địa lý"),
          ("test_fe_khong_dia_ly", "Kỹ nghệ đặc trưng – test, không địa lý"),
          ("giao_dich", "Dạng giao dịch cho luật kết hợp"), ("ho_so_tinh", "Hồ sơ 63 tỉnh cho phân cụm")]
    bang = {ten: doc(str(B2), ten) for ten, _ in DS}
    st.dataframe(pd.DataFrame([{"Bảng": ten, "Dùng cho": mo_ta,
                                "Số dòng": None if bang[ten] is None else len(bang[ten]),
                                "Số cột": None if bang[ten] is None else bang[ten].shape[1]} for ten, mo_ta in DS]),
                 width="stretch", hide_index=True)
    chon = st.selectbox("Xem bảng", [ten for ten, _ in DS if bang[ten] is not None])
    if chon:
        st.dataframe(bang[chon].head(200), width="stretch")

    hs = bang["ho_so_tinh"]
    if hs is not None:
        st.subheader("Hồ sơ tỉnh")
        chi_so = st.selectbox("Chỉ số", [c for c in hs.columns if c not in ("tinh", "so_vu") and not c.endswith("_z")])
        thang_do = st.radio("Thang đo", ["Gốc", "Chuẩn hoá z-score"], horizontal=True)
        cot_ve = chi_so if thang_do == "Gốc" else f"{chi_so}_z"
        bieu_do_cot(hs[["tinh", cot_ve]].sort_values(cot_ve, ascending=False), "tinh", cot_ve, cot_ve, ".3f")

    gd = bang["giao_dich"]
    if gd is not None:
        st.subheader("Item phổ biến nhất trong dạng giao dịch (độ hỗ trợ)")
        ho_tro = gd.drop(columns=KHOA).mean().sort_values(ascending=False).head(20)
        bieu_do_cot(ho_tro.rename_axis("item").reset_index(name="ho_tro"), "item", "ho_tro", "Độ hỗ trợ", ".1%")

    st.subheader("Cấu trúc pipeline tiền xử lý")
    for h, mo_ta in HUONG.items():
        pl = nap_joblib(str(B2 / f"pipeline_tien_xu_ly_{h}.joblib"))
        if pl is not None:
            st.markdown(f"**{mo_ta}**")
            st.iframe(estimator_html_repr(pl), height=260)
    nhung_txt("Báo cáo pipeline (kiem_tra_pipeline.txt)", B2 / "kiem_tra_pipeline.txt")

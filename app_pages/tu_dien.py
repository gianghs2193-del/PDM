import streamlit as st

from du_lieu import NGUON, csv_du_lieu, doc_du_lieu, doc_file, doc_tu_dien

td = doc_tu_dien()
df = doc_du_lieu()

st.title(":material/menu_book: Từ điển dữ liệu & tải về")

with st.container(horizontal=True):
    st.download_button("Dữ liệu tích hợp (.csv)", csv_du_lieu(), "du_lieu_tich_hop.csv", mime="text/csv",
                       icon=":material/download:", type="primary")
    st.download_button("Từ điển dữ liệu (.csv)", doc_file("tu_dien_du_lieu.csv"), "tu_dien_du_lieu.csv",
                       mime="text/csv", icon=":material/download:")
    st.download_button("Báo cáo kiểm tra (.txt)", doc_file("kiem_tra.txt"), "kiem_tra.txt",
                       mime="text/plain", icon=":material/download:")
st.caption("Dữ liệu được đọc từ bảng `thu_thap.du_lieu_tich_hop` trong PostgreSQL; bản CSV (UTF-8 có BOM để mở "
           "bằng Excel) cũng nằm trong thư mục `bai_tap_lon/1_thu_thap_du_lieu/`.")

st.subheader("Từ điển dữ liệu")
tim = st.text_input("Tìm thuộc tính", placeholder="ví dụ: đường, PT1, thời tiết", live=True)
xem = td
if tim:
    m = td.apply(lambda r: r.astype(str).str.contains(tim, case=False, regex=False).any(), axis=1)
    xem = td[m]
st.dataframe(
    xem,
    hide_index=True,
    column_config={
        "ten_cot_goc": st.column_config.TextColumn("Tên cột gốc", width="medium"),
        "ten_cot": "Tên mới",
        "nhom": "Nhóm",
        "kieu": "Kiểu",
        "mo_ta": st.column_config.TextColumn("Mô tả", width="large"),
        "co_o_nguon": "Có ở nguồn",
        "ty_le_thieu_%": st.column_config.ProgressColumn("% thiếu", min_value=0, max_value=100, format="%.1f%%"),
    },
    alt="Từ điển dữ liệu 55 thuộc tính",
)

st.subheader("Xem trước dữ liệu")
c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
with c1:
    nguon = st.segmented_control("Nguồn", NGUON, default="2021", key="nguon_xem")
with c2:
    chi_co = st.toggle("Chỉ bản ghi bị gắn cờ nghi vấn")
x = df[df["nguon"] == nguon] if nguon else df
if chi_co:
    x = x[x[["co_trung_khoa", "co_ngay_loi", "co_gio_loi", "co_lech_thuong_vong"]].any(axis=1)]
st.caption(f"{len(x):,} bản ghi".replace(",", "."))
st.dataframe(
    x.drop(columns=["gio_so", "thang", "cap_duong"]).head(1000),
    hide_index=True,
    column_config={"ngay": st.column_config.DateColumn("ngay", format="DD/MM/YYYY")},
    alt="1.000 bản ghi đầu của nguồn đã chọn",
)
st.caption("Hiển thị tối đa 1.000 dòng.")

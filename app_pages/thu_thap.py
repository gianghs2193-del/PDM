import altair as alt
import pandas as pd
import streamlit as st

from du_lieu import doc_tu_dien

td = doc_tu_dien()

st.title(":material/download: Thu thập & tích hợp")
st.markdown("Quy trình 7 bước. Bước 4–7 được tự động hoá bằng script `thu_thap.py` "
            "(pandas, openpyxl, gdown, pyarrow); chạy lại sẽ cho cùng kết quả.")

BUOC = [
    (":material/help:", "1. Xác định bài toán", "4 câu hỏi phân tích; cần dữ liệu mức từng vụ."),
    (":material/source:", "2. Chọn nguồn", "3 file Excel báo cáo TNGT của CSGT trên Google Drive."),
    (":material/checklist:", "3. Chọn thuộc tính", "55/138 cột, chia 10 nhóm; bỏ thông tin cá nhân."),
    (":material/cloud_download:", "4. Thu thập", "Tải bằng gdown, đọc tiêu đề 2 tầng, sửa lỗi file gốc."),
    (":material/merge:", "5. Tích hợp", "Đưa 3 năm về 1 lược đồ, đổi tên, đồng nhất kiểu, ghi nguồn gốc."),
    (":material/fact_check:", "6. Kiểm tra", "Thiếu, trùng, hợp lệ, đối chiếu; chỉ đánh dấu, không xoá."),
    (":material/save:", "7. Lưu trữ", "CSV + Parquet + từ điển dữ liệu + báo cáo kiểm tra."),
]
with st.grid(4, gap="small"):
    for icon, ten, mo_ta in BUOC:
        with st.container(border=True, height="stretch"):
            st.markdown(f"{icon} **{ten}**")
            st.caption(mo_ta)

st.subheader("Thuộc tính được chọn")
trai, phai = st.columns([3, 2], gap="large")
with trai:
    dem = td.groupby("nhom", sort=False).size().reset_index(name="so_thuoc_tinh")
    st.altair_chart(
        alt.Chart(dem)
        .mark_bar()
        .encode(
            x=alt.X("so_thuoc_tinh:Q", title="Số thuộc tính"),
            y=alt.Y("nhom:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("nhom:N", title="Nhóm"), alt.Tooltip("so_thuoc_tinh:Q", title="Số thuộc tính")],
        ),
        alt="Số thuộc tính được chọn theo 10 nhóm",
    )
with phai:
    with st.container(border=True):
        st.markdown("**Không thu thập**")
        st.markdown(
            "- **Thông tin cá nhân**: họ tên, ngày sinh, địa chỉ, SĐT, biển số, chủ phương tiện. "
            "Gần như trống và không cần cho phân tích; bỏ đi để bảo vệ quyền riêng tư.\n"
            "- **Văn bản tự do, cột trùng nghĩa**: tóm tắt diễn biến, thiệt hại bằng chữ.\n"
            "- **Hành chính**: STT, đơn vị gửi/nhận báo cáo, bảo hiểm, tiền phạt."
        )
    st.warning(
        "*Phân loại tai nạn* được suy ra từ số người chết và bị thương. "
        "Không dùng làm đầu vào khi dự đoán vụ có người chết.",
        title="Rò rỉ nhãn",
        icon=":material/warning:",
    )

st.subheader("Xử lý các điểm bất thường của file gốc")
st.dataframe(
    pd.DataFrame(
        [
            ("Tiêu đề 2 tầng, tên nhóm nằm trong ô gộp", "Điền tên nhóm sang phải, ghép thành `Nhóm | Cột con`"),
            ("File 2021 có thêm dòng tên bảng ở trên cùng", "Tự tìm dòng tiêu đề thật"),
            ("File 2021 ghi nhầm nhóm phương tiện thứ 5 thành PT1", "Đánh số lại PT1…PT5 theo thứ tự"),
            ("File 2020 có 2 cột Thời gian (HH:MM:SS và số thập phân)", "Giữ cột HH:MM:SS"),
            ("Tên cột có khoảng trắng thừa", "Cắt khoảng trắng"),
            ("Sheet TPHCM (1.766 vụ) và QL1 (163 vụ) trong file 2021", "Đối chiếu: 100% đã có trong DSTN → không lấy thêm"),
            ("Thiệt hại không ghi đơn vị", "Đối chiếu cột 'Bằng chữ' → đơn vị là triệu đồng"),
            ("File 2022 chỉ có 12 cột", "Giữ đủ 55 cột, thuộc tính không có để trống, không suy diễn"),
        ],
        columns=["Vấn đề", "Cách xử lý"],
    ),
    hide_index=True,
    alt="Các điểm bất thường của file gốc và cách xử lý",
)

st.subheader("Kết quả tích hợp")
with st.container(horizontal=True):
    st.metric("Bản ghi", "33.086", border=True)
    st.metric("Thuộc tính", 55, border=True)
    st.metric("Cột truy vết", 3, help="nguon, file_goc, dong_goc (số dòng Excel)", border=True)
    st.metric("Cột cờ kiểm tra", 4, help="Đánh dấu bản ghi nghi vấn để bước tiền xử lý quyết định", border=True)

co = td.assign(**{m: td["co_o_nguon"].str.contains(m) for m in ["2020", "2021", "8T2022"]})
st.markdown("**Thuộc tính có ở từng nguồn**")
st.dataframe(
    co[["nhom", "ten_cot", "mo_ta", "2020", "2021", "8T2022"]],
    hide_index=True,
    height=300,
    column_config={
        "nhom": "Nhóm",
        "ten_cot": "Tên cột",
        "mo_ta": st.column_config.TextColumn("Mô tả", width="large"),
        "2020": st.column_config.CheckboxColumn("2020"),
        "2021": st.column_config.CheckboxColumn("2021"),
        "8T2022": st.column_config.CheckboxColumn("8T2022"),
    },
    alt="Thuộc tính nào có ở nguồn nào",
)

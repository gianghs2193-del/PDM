import pandas as pd
import streamlit as st

from du_lieu import doc_du_lieu

df = doc_du_lieu()

st.title(":material/traffic: Tai nạn giao thông đường bộ Việt Nam")
st.markdown(
    "Dự án khai phá dữ liệu trên **33 nghìn vụ tai nạn** được ghi nhận từ 15/12/2019 đến 14/08/2022. "
    "Mục tiêu là tìm hiểu tai nạn thường xảy ra ở đâu, khi nào, trong điều kiện nào, "
    "và yếu tố nào làm một vụ tai nạn dễ có người chết hơn."
)

with st.container(horizontal=True):
    st.metric("Số vụ tai nạn", f"{len(df):,}".replace(",", "."), border=True)
    st.metric("Người chết", f"{int(df['so_chet'].sum()):,}".replace(",", "."), border=True)
    st.metric("Người bị thương", f"{int(df['so_bi_thuong'].sum()):,}".replace(",", "."), border=True)
    st.metric("Tỉnh/thành", df["tinh"].nunique(), border=True)
    st.metric("Thuộc tính thu thập", 55, border=True)

trai, phai = st.columns([3, 2], gap="large")

with trai:
    st.subheader("Câu hỏi phân tích")
    st.table(
        pd.DataFrame(
            {
                "Câu hỏi": [
                    "Tai nạn phân bố thế nào theo thời gian, địa bàn, loại đường, phương tiện?",
                    "Có dự đoán được một vụ tai nạn có người chết hay không?",
                    "Tổ hợp điều kiện nào thường đi kèm vụ có người chết?",
                    "Các tỉnh/thành có chia được thành nhóm có đặc điểm giống nhau?",
                ],
                "Kỹ thuật dự kiến": ["Thống kê mô tả, trực quan hoá", "Phân lớp", "Luật kết hợp", "Phân cụm"],
            },
            index=range(1, 5),
        ),
        alt="Bốn câu hỏi phân tích và kỹ thuật khai phá dự kiến",
    )

with phai:
    st.subheader("Tiến độ")
    with st.container(border=True):
        for ten, mo_ta, xong in [
            ("Bước 1 · Thu thập dữ liệu", "Tải, đọc, tích hợp 3 file Excel; kiểm tra chất lượng; lưu trữ", True),
            ("Bước 2 · Tiền xử lý", "Làm sạch, xử lý thiếu và ngoại lai, chuẩn hoá nhãn", False),
            ("Bước 3 · Khai phá", "Phân lớp, luật kết hợp, phân cụm", False),
            ("Bước 4 · Đánh giá & báo cáo", "Đánh giá mô hình, rút ra kết luận", False),
        ]:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{ten}**")
                if xong:
                    st.badge("Hoàn thành", icon=":material/check:", color="green")
                else:
                    st.badge("Chưa làm", icon=":material/schedule:", color="gray")
            st.caption(mo_ta)

st.subheader("Nguồn dữ liệu")
st.dataframe(
    pd.DataFrame(
        {
            "File": ["bcthdb_thong_tin_tong_hop(2020).xlsx", "bcthdb_thong_tin_tong_hop(2021).xlsx",
                     "bcthdb_thong_tin_tong_hop(8T2022).xlsx"],
            "Kỳ báo cáo": ["15/12/2019 – 14/12/2020", "15/12/2020 – 14/12/2021", "15/12/2021 – 14/08/2022"],
            "Số vụ": [14276, 11408, 7402],
            "Số cột gốc": [138, 138, 12],
            "Thuộc tính có": ["55/55", "55/55", "11/55"],
        }
    ),
    hide_index=True,
    alt="Ba file Excel nguồn",
)
st.caption(
    "Dữ liệu ở mức từng vụ, phạm vi toàn quốc. File 2022 chỉ có thông tin cơ bản (thời gian, địa điểm, "
    "thương vong), nên các phân tích về điều kiện xảy ra tai nạn dùng dữ liệu 2020–2021."
)
st.link_button("Thư mục Google Drive gốc", "https://drive.google.com/drive/folders/1V--WoKpABVX5qaAwLwhHRF_dTOqM-1LI",
               icon=":material/folder_open:")

"""App Streamlit giới thiệu dự án: Tai nạn giao thông đường bộ Việt Nam (2020 – 8 tháng 2022).

Chạy (từ thư mục D:/PDM):
    .venv/Scripts/streamlit run TNGT.py
"""
import sys

import streamlit as st

# Bấm Run / gõ `python TNGT.py` -> tự chuyển sang `streamlit run TNGT.py`
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(cli.main())

st.set_page_config(page_title="Dự án TNGT Việt Nam", page_icon=":material/traffic:", layout="wide")

trang = st.navigation(
    {
        "Dự án": [
            st.Page("app_pages/tong_quan.py", title="Tổng quan", icon=":material/home:", default=True),
        ],
        "Bước 1 – Thu thập dữ liệu": [
            st.Page("app_pages/thu_thap.py", title="Thu thập & tích hợp", icon=":material/download:"),
            st.Page("app_pages/chat_luong.py", title="Chất lượng dữ liệu", icon=":material/fact_check:"),
            st.Page("app_pages/kham_pha.py", title="Khám phá nhanh", icon=":material/bar_chart:"),
            st.Page("app_pages/tu_dien.py", title="Từ điển & tải về", icon=":material/menu_book:"),
        ],
    }
)

with st.sidebar:
    st.caption("Môn Khai phá dữ liệu · Bài tập lớn")
    st.caption("Nguồn: hệ thống báo cáo TNGT đường bộ của lực lượng CSGT")

trang.run()

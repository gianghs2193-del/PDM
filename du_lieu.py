"""Hàm đọc dữ liệu dùng chung cho các trang của app TNGT (có cache)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

THU_MUC = Path(__file__).resolve().parent / "bai_tap_lon" / "1_thu_thap_du_lieu"
NGUON = ["2020", "2021", "8T2022"]
COT_CO = ["co_trung_khoa", "co_ngay_loi", "co_gio_loi", "co_lech_thuong_vong"]


@st.cache_data(show_spinner="Đang đọc dữ liệu tích hợp...")
def doc_du_lieu() -> pd.DataFrame:
    df = pd.read_parquet(THU_MUC / "du_lieu_tich_hop.parquet")
    df["gio_so"] = pd.to_numeric(df["gio"].str[:2], errors="coerce")  # giờ trong ngày 0–23
    df["thang"] = df["ngay"].dt.to_period("M").dt.to_timestamp()
    df["cap_duong"] = df["loai_duong"].str.split(" - ").str[0]  # bỏ hậu tố Nội thị / Nông thôn
    return df


@st.cache_data
def doc_tu_dien() -> pd.DataFrame:
    return pd.read_csv(THU_MUC / "tu_dien_du_lieu.csv", encoding="utf-8-sig")


@st.cache_data
def doc_kiem_tra() -> str:
    return (THU_MUC / "kiem_tra.txt").read_text(encoding="utf-8")


@st.cache_data
def doc_file(ten: str) -> bytes:
    return (THU_MUC / ten).read_bytes()


@st.cache_data
def ty_le_thieu() -> pd.DataFrame:
    """% ô trống của từng thuộc tính theo nguồn — chỉ tính trên nguồn CÓ cột đó (giống bước 6.3)."""
    df, td = doc_du_lieu(), doc_tu_dien()
    dong = []
    for _, r in td.iterrows():
        co = [m.strip() for m in str(r["co_o_nguon"]).split(",") if m.strip() in NGUON]
        for m in co:
            dong.append({"ten_cot": r["ten_cot"], "nhom": r["nhom"], "nguon": m,
                         "thieu": round(df.loc[df["nguon"] == m, r["ten_cot"]].isna().mean() * 100, 1)})
    return pd.DataFrame(dong)

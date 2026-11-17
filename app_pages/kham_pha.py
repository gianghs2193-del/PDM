import altair as alt
import pandas as pd
import streamlit as st

from du_lieu import NGUON, doc_du_lieu

df = doc_du_lieu()

st.title(":material/bar_chart: Khám phá nhanh")
st.caption("Thống kê mô tả trên dữ liệu thô (chưa làm sạch), để kiểm tra các nhận định trong báo cáo thu thập.")

chon = st.pills("Nguồn", NGUON, selection_mode="multi", default=NGUON)
x = df[df["nguon"].isin(chon)] if chon else df
co_dk = x[x["nguon"] != "8T2022"]  # thuộc tính điều kiện chỉ có ở 2020–2021

with st.container(horizontal=True):
    st.metric("Số vụ", f"{len(x):,}".replace(",", "."), border=True)
    st.metric("Người chết", f"{int(x['so_chet'].sum()):,}".replace(",", "."), border=True)
    st.metric("Vụ có người chết", f"{(x['so_chet'] > 0).mean():.1%}".replace(".", ","), border=True)
    st.metric("Chết / vụ", f"{x['so_chet'].mean():.2f}".replace(".", ","), border=True)

# --- Theo tháng ---
with st.container(border=True):
    st.subheader("Số vụ theo tháng")
    thang = x.groupby("thang").agg(vu=("ngay", "size"), chet=("so_chet", "sum")).reset_index()
    chi_so = st.segmented_control("Chỉ số", ["Số vụ", "Người chết"], default="Số vụ", key="chi_so_thang")
    cot = "chet" if chi_so == "Người chết" else "vu"
    duong = alt.Chart(thang).mark_line(point=True).encode(
        x=alt.X("thang:T", title=None, axis=alt.Axis(format="%m/%Y")),
        y=alt.Y(f"{cot}:Q", title=chi_so or "Số vụ"),
        tooltip=[alt.Tooltip("thang:T", title="Tháng", format="%m/%Y"), alt.Tooltip("vu:Q", title="Số vụ"),
                 alt.Tooltip("chet:Q", title="Người chết")],
    )
    covid = alt.Chart(pd.DataFrame({"bd": [pd.Timestamp("2021-07-01")], "kt": [pd.Timestamp("2021-09-30")]})).mark_rect(
        opacity=0.15, color="gray").encode(x="bd:T", x2="kt:T")
    st.altair_chart(covid + duong, alt="Số vụ tai nạn theo tháng, vùng xám là giãn cách COVID-19")
    st.caption("Vùng xám: giãn cách COVID-19 (07–09/2021). Tháng 12/2019 và 08/2022 chỉ có nửa tháng dữ liệu. "
               "Tháng 2 thường giảm do Tết.")

trai, phai = st.columns(2, gap="medium")

with trai, st.container(border=True, height="stretch"):
    st.subheader("Theo giờ trong ngày")
    gio = x.dropna(subset=["gio_so"]).groupby("gio_so").agg(
        vu=("ngay", "size"), ty_le_chet=("so_chet", lambda s: (s > 0).mean())).reset_index()
    nen = alt.Chart(gio).encode(x=alt.X("gio_so:O", title="Giờ"))
    st.altair_chart(
        nen.mark_bar().encode(
            y=alt.Y("vu:Q", title="Số vụ"),
            tooltip=[alt.Tooltip("gio_so:O", title="Giờ"), alt.Tooltip("vu:Q", title="Số vụ"),
                     alt.Tooltip("ty_le_chet:Q", title="% vụ có người chết", format=".1%")]),
        alt="Số vụ tai nạn theo giờ trong ngày",
    )
    if not gio.empty:
        dinh = gio.nlargest(3, "vu")["gio_so"].astype(int).sort_values().tolist()
        st.caption(f"3 khung giờ nhiều vụ nhất: {', '.join(f'{g}h' for g in dinh)}. "
                   "Rê chuột lên cột để xem % vụ có người chết.")

with phai, st.container(border=True, height="stretch"):
    st.subheader("Theo cấp đường")
    duong_df = x.dropna(subset=["cap_duong"]).groupby("cap_duong").agg(
        vu=("ngay", "size"), ty_le_chet=("so_chet", lambda s: (s > 0).mean())).reset_index()
    st.altair_chart(
        alt.Chart(duong_df).mark_bar().encode(
            x=alt.X("vu:Q", title="Số vụ"),
            y=alt.Y("cap_duong:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("cap_duong:N", title="Cấp đường"), alt.Tooltip("vu:Q", title="Số vụ"),
                     alt.Tooltip("ty_le_chet:Q", title="% vụ có người chết", format=".1%")]),
        alt="Số vụ tai nạn theo cấp đường",
    )
    st.caption("Gộp hậu tố \"Nội thị / Nông thôn\". Trống ~12% loại đường.")

# --- Theo tỉnh ---
with st.container(border=True):
    st.subheader("Theo tỉnh/thành")
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    with c1:
        tieu_chi = st.segmented_control("Xếp theo", ["Số vụ", "Người chết", "% va chạm nhẹ"], default="Số vụ",
                                        key="tieu_chi_tinh")
    with c2:
        top = st.slider("Số tỉnh hiển thị", 5, 63, 15)
    tinh = x.groupby("tinh").agg(vu=("ngay", "size"), chet=("so_chet", "sum")).reset_index()
    vc = co_dk.groupby("tinh")["phan_loai"].apply(lambda s: (s == "Va chạm giao thông").mean()).rename("va_cham")
    tinh = tinh.merge(vc, on="tinh", how="left")
    cot = {"Người chết": "chet", "% va chạm nhẹ": "va_cham"}.get(tieu_chi or "", "vu")
    tinh = tinh.dropna(subset=[cot]).nlargest(top, cot)
    st.altair_chart(
        alt.Chart(tinh).mark_bar().encode(
            x=alt.X(f"{cot}:Q", title=tieu_chi or "Số vụ", axis=alt.Axis(format="%" if cot == "va_cham" else "d")),
            y=alt.Y("tinh:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("tinh:N", title="Tỉnh"), alt.Tooltip("vu:Q", title="Số vụ"),
                     alt.Tooltip("chet:Q", title="Người chết"),
                     alt.Tooltip("va_cham:Q", title="% va chạm nhẹ (2020–2021)", format=".0%")]),
        alt="Xếp hạng tỉnh/thành theo tiêu chí đã chọn",
    )
    st.caption("TP.HCM có nhiều vụ nhất nhưng 71% là va chạm nhẹ (chung cả nước 39%): mức độ báo cáo khác nhau "
               "giữa các địa phương, cần thận trọng khi so sánh.")

# --- Điều kiện (2020–2021) ---
st.subheader("Điều kiện xảy ra tai nạn (2020–2021)")
if co_dk.empty:
    st.info("File 2022 không có các thuộc tính điều kiện. Chọn thêm nguồn 2020 hoặc 2021.")
else:
    THUOC_TINH = {"Nhóm phương tiện PT1": "pt1_nhom", "Hình thức va chạm": "hinh_thuc_va_cham",
                  "Thời tiết / ánh sáng": "thoi_tiet", "Phân loại tai nạn": "phan_loai",
                  "Dạng đường": "dang_duong", "Nguyên nhân": "nguyen_nhan"}
    ten = st.selectbox("Thuộc tính", list(THUOC_TINH))
    c = THUOC_TINH[ten]
    bang = co_dk.assign(gia_tri=co_dk[c].fillna("(trống)")).groupby("gia_tri").agg(
        vu=("ngay", "size"), ty_le_chet=("so_chet", lambda s: (s > 0).mean())).reset_index().nlargest(12, "vu")
    with st.container(border=True):
        st.altair_chart(
            alt.Chart(bang).mark_bar().encode(
                x=alt.X("vu:Q", title="Số vụ"),
                y=alt.Y("gia_tri:N", sort="-x", title=None),
                color=alt.Color("ty_le_chet:Q", title="% có người chết", scale=alt.Scale(scheme="oranges"),
                                legend=alt.Legend(format=".0%")),
                tooltip=[alt.Tooltip("gia_tri:N", title=ten), alt.Tooltip("vu:Q", title="Số vụ"),
                         alt.Tooltip("ty_le_chet:Q", title="% vụ có người chết", format=".1%")]),
            alt=f"Số vụ tai nạn theo {ten.lower()}, tô màu theo tỉ lệ có người chết",
        )
        if c == "phan_loai":
            st.caption("Phân loại tai nạn suy ra từ thương vong: không dùng làm đầu vào dự đoán (rò rỉ nhãn).")
        elif c == "thoi_tiet":
            st.caption("Thuộc tính này trộn ánh sáng (\"Đêm tối\") với thời tiết; bước tiền xử lý sẽ tách ra.")

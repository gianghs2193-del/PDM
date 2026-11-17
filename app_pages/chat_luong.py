import altair as alt
import pandas as pd
import streamlit as st

from du_lieu import COT_CO, doc_du_lieu, doc_kiem_tra, ty_le_thieu

df = doc_du_lieu()
thieu = ty_le_thieu()

st.title(":material/fact_check: Chất lượng dữ liệu")
st.markdown("Bước thu thập **không xoá, không sửa** giá trị nào. Bản ghi nghi vấn chỉ được đánh dấu bằng cột cờ "
            "để bước tiền xử lý quyết định.")

co_bat_ky = df[COT_CO].any(axis=1)
with st.container(horizontal=True):
    st.metric("Trùng theo khoá", int(df["co_trung_khoa"].sum()), border=True,
              help="Trùng (ngày, giờ, tỉnh, huyện, địa điểm, số chết, số bị thương) với bản ghi đứng trước")
    st.metric("Giờ sai định dạng", int(df["co_gio_loi"].sum()), border=True, help="Ví dụ 18:0000 thay vì 18:00:00")
    st.metric("Lệch thương vong", int(df["co_lech_thuong_vong"].sum()), border=True)
    st.metric("Ngày lỗi", int(df["co_ngay_loi"].sum()), border=True)
    st.metric("Có ít nhất 1 cờ", f"{int(co_bat_ky.sum())} ({co_bat_ky.mean():.1%})".replace(".", ","), border=True)

st.subheader("Dữ liệu thiếu")
st.caption("Tỉ lệ ô trống chỉ tính trên nguồn **có** cột đó. File 2022 không có cột thì không tính là thiếu.")
nhom = st.pills("Nhóm thuộc tính", sorted(thieu["nhom"].unique()), selection_mode="multi",
                default=["Hạ tầng", "Môi trường", "Nguyên nhân", "Va chạm", "Báo hiệu"])
xem = thieu[thieu["nhom"].isin(nhom)] if nhom else thieu
if xem.empty:
    st.info("Chọn ít nhất một nhóm thuộc tính.")
else:
    thu_tu = xem.groupby("ten_cot")["thieu"].mean().sort_values(ascending=False).index.tolist()
    st.altair_chart(
        alt.Chart(xem)
        .mark_bar()
        .encode(
            x=alt.X("thieu:Q", title="% ô trống", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("ten_cot:N", sort=thu_tu, title=None),
            yOffset="nguon:N",
            color=alt.Color("nguon:N", title="Nguồn"),
            tooltip=[alt.Tooltip("ten_cot:N", title="Thuộc tính"), alt.Tooltip("nguon:N", title="Nguồn"),
                     alt.Tooltip("thieu:Q", title="% trống", format=".1f")],
        )
        .properties(height=alt.Step(9)),
        alt="Tỉ lệ ô trống theo thuộc tính và nguồn",
    )
with st.container(border=True):
    st.markdown("**Nhận xét**")
    st.markdown(
        "- PT3–PT5 trống > 92% vì phần lớn vụ chỉ có 1–2 phương tiện, không phải lỗi ghi chép.\n"
        "- *Bỏ trốn*, *đèn tín hiệu*, *giao cắt đường sắt*, *số bị can* nhiều khả năng chỉ được điền khi có sự việc: "
        "ô trống mang nghĩa \"không có\".\n"
        "- Thuộc tính hạ tầng và môi trường trống 40–60%: cần xem ô trống là nhóm \"Không rõ\" hoặc phân tích "
        "trên tập con."
    )

trai, phai = st.columns(2, gap="large")
with trai:
    st.subheader("Tính hợp lệ")
    th = df["thiet_hai"]
    hop_le = pd.DataFrame(
        [
            ("Thiếu / sai định dạng ngày", int(df["ngay"].isna().sum())),
            ("Ngày nằm ngoài kỳ báo cáo", int(df["co_ngay_loi"].sum())),
            ("Giờ sai định dạng", int(df["co_gio_loi"].sum())),
            ("Bị thương ≠ thương nặng + thương nhẹ",
             int((df["so_thuong_nang"].notna() & df["so_thuong_nhe"].notna()
                  & (df["so_thuong_nang"] + df["so_thuong_nhe"] != df["so_bi_thuong"])).sum())),
            ("Phân loại nhẹ nhưng có người chết",
             int((df["phan_loai"].isin(["Va chạm giao thông", "Ít nghiêm trọng"]) & (df["so_chet"] > 0)).sum())),
            ("Thiệt hại > 10 tỷ đồng (ngoại lai)", int((th > 10_000).sum())),
        ],
        columns=["Kiểm tra", "Số bản ghi"],
    )
    st.dataframe(hop_le, hide_index=True, alt="Kết quả kiểm tra tính hợp lệ")
with phai:
    st.subheader("Trùng lặp")
    st.dataframe(
        pd.DataFrame(
            [("Trùng hoàn toàn mọi thuộc tính", 7), ("Trùng theo khoá", 22), ("Trùng giữa 2 nguồn khác nhau", 0)],
            columns=["Loại", "Số bản ghi"],
        ),
        hide_index=True,
        alt="Số bản ghi trùng lặp",
    )
    st.caption("Tỉ lệ trùng 0,07%. 7 bản ghi trùng hoàn toàn đều nằm trong 22 bản ghi trùng theo khoá.")

st.subheader("Ngoại lai thiệt hại")
st.markdown(f"Thiệt hại có trung vị **{th.median():g} triệu đồng**, 99% số vụ không quá "
            f"**{th.quantile(.99):g} triệu**. Hai vụ dưới đây nhiều khả năng do nhập sai:")
st.dataframe(
    df.loc[th > 10_000, ["nguon", "dong_goc", "ngay", "tinh", "huyen", "phan_loai", "so_chet", "so_bi_thuong",
                         "thiet_hai"]],
    hide_index=True,
    column_config={
        "nguon": "Nguồn", "dong_goc": "Dòng Excel", "ngay": st.column_config.DateColumn("Ngày", format="DD/MM/YYYY"),
        "tinh": "Tỉnh", "huyen": "Huyện", "phan_loai": "Phân loại", "so_chet": "Chết", "so_bi_thuong": "Bị thương",
        "thiet_hai": st.column_config.NumberColumn("Thiệt hại (triệu đồng)", format="localized"),
    },
    alt="Hai vụ có thiệt hại bất thường",
)

with st.expander("Báo cáo kiểm tra đầy đủ (kiem_tra.txt)", icon=":material/description:"):
    st.code(doc_kiem_tra(), language=None)

"""HỒI QUY — Dự báo SỐ VỤ tai nạn theo thời điểm, tìm "thời điểm đen" — Chương 5: Modeling (Regression)

Đầu vào  : bảng tien_xu_ly.tai_nan_loi_sach (PostgreSQL)
Đầu ra   : bang_so_vu_theo_o.csv (dữ liệu hồi quy), ket_qua_hoi_quy.csv, bang_thoi_diem_den.csv,
           kiem_tra_hoi_quy.txt (cũng in ra màn hình)

Câu hỏi: trong một tháng, ở một vùng, một khung giờ của một thứ trong tuần có bao nhiêu vụ?
Mỗi "ô" = tháng × vùng × khung giờ × thứ  ->  biến mục tiêu = số vụ (giá trị liên tục, có thể bằng 0).

Các bước:
    1. Gộp dữ liệu thành bảng ô (cả các ô 0 vụ); bỏ tháng không trọn, giờ trống, giờ mặc định 00:00:00
    2. Đặc trưng của ô: vùng, khung giờ, thứ, tháng trong năm, số lần thứ đó có trong tháng (4 hay 5),
       tỉ lệ ngày giãn cách, tỉ lệ ngày nghỉ Tết, xu hướng thời gian
    3. Chia theo THỜI GIAN: học 2020–2021, kiểm tra 7 tháng trọn của 2022 (dự báo giai đoạn chưa thấy)
    4. Mô hình: trung bình (mốc), Linear, Polynomial bậc 2, Ridge, Lasso, Random Forest Regressor;
       tinh chỉnh alpha / min_samples_leaf bằng GridSearchCV với TimeSeriesSplit trên train
    5. Đánh giá trên tập kiểm tra: MSE, RMSE, MAE, MAPE (trên ô có vụ), R²
    6. Bảng "thời điểm đen": số vụ trung bình MỖI GIỜ theo khung giờ × thứ (chia cho độ dài khung để so công bằng)

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/hoi_quy.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
TX = GOC.parent / "2_tien_xu_ly"
BC: list[str] = ["BÁO CÁO HỒI QUY SỐ VỤ THEO THỜI ĐIỂM", "=" * 70]
HAT = 42
VUNG = ["Bắc", "Duyên hải miền Trung", "Tây Nguyên", "Nam Bộ"]
KHUNG = ["00-04", "04-06", "06-09", "09-16", "16-19", "19-22", "22-24"]
DO_DAI = {"00-04": 4, "04-06": 2, "06-09": 3, "09-16": 7, "16-19": 3, "19-22": 3, "22-24": 2}
THU = {1: "T2", 2: "T3", 3: "T4", 4: "T5", 5: "T6", 6: "T7", 7: "CN"}
GIAN_CACH = [("2020-04-01", "2020-04-22"), ("2021-07-19", "2021-09-30")]          # như bước 1 (thu_thap.py)
NGHI_TET = [("2020-01-23", "2020-01-29"), ("2021-02-10", "2021-02-16"), ("2022-01-29", "2022-02-06")]
MOC_KIEM_TRA = pd.Period("2022-01", "M")
PHAN_LOAI = ["vung", "khung_gio", "thu", "thang"]
SO = ["so_ngay", "ti_le_gian_cach", "ti_le_tet", "xu_huong"]


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def doc(thu_muc: Path, ten: str) -> pd.DataFrame:
    return csdl.doc(thu_muc, ten)               # thư mục 2_tien_xu_ly -> schema tien_xu_ly


def ti_le_ngay(ky: pd.Period, dot: list[tuple[str, str]]) -> float:
    ngay = pd.date_range(ky.start_time, ky.end_time.normalize(), freq="D")
    trong = np.zeros(len(ngay), dtype=bool)
    for a, b in dot:
        trong |= (ngay >= pd.Timestamp(a)) & (ngay <= pd.Timestamp(b))
    return float(trong.mean())


def tao_bang_o(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(pd.to_numeric(df["thang_khong_tron"]) == 0) & df["gio_so"].notna()
           & (pd.to_numeric(df["co_gio_mac_dinh"]) == 0) & df["vung"].isin(VUNG) & df["khung_gio"].isin(KHUNG)].copy()
    d["ky"] = pd.to_datetime(d["ngay"]).dt.to_period("M")
    d["thu"] = pd.to_numeric(d["thu"]).astype(int)
    dem = d.groupby(["ky", "vung", "khung_gio", "thu"]).size()
    ky = sorted(d["ky"].unique())
    luoi = pd.MultiIndex.from_product([ky, VUNG, KHUNG, list(THU)], names=["ky", "vung", "khung_gio", "thu"])
    o = dem.reindex(luoi, fill_value=0).rename("so_vu").reset_index()
    thong_tin = pd.DataFrame({"ky": ky})
    thong_tin["thang"] = [k.month for k in ky]
    thong_tin["xu_huong"] = range(len(ky))
    thong_tin["ti_le_gian_cach"] = [ti_le_ngay(k, GIAN_CACH) for k in ky]
    thong_tin["ti_le_tet"] = [ti_le_ngay(k, NGHI_TET) for k in ky]
    o = o.merge(thong_tin, on="ky")
    so_ngay = {(k, t): int(((pd.date_range(k.start_time, k.end_time.normalize()).dayofweek + 1) == t).sum())
               for k in ky for t in THU}
    o["so_ngay"] = [so_ngay[(k, t)] for k, t in zip(o["ky"], o["thu"])]
    for c in ("thu", "thang"):
        o[c] = o[c].astype(str)
    return o, d


def bo_ma_hoa(bac2: bool) -> ColumnTransformer | Pipeline:
    ct = ColumnTransformer([("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PHAN_LOAI),
                            ("so", StandardScaler(), SO)])
    if not bac2:
        return ct
    return Pipeline([("ma_hoa", ct), ("bac2", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))])


def danh_gia(y, du) -> dict:
    co = y > 0
    return {"MSE": mean_squared_error(y, du), "RMSE": float(np.sqrt(mean_squared_error(y, du))),
            "MAE": mean_absolute_error(y, du), "MAPE": float(np.mean(np.abs((y[co] - du[co]) / y[co]))),
            "R2": r2_score(y, du)}


def main() -> None:
    p(f"[0] Đọc bảng lõi sạch | scikit-learn {sklearn.__version__}")
    df = doc(TX, "tai_nan_loi_sach")

    p("\n[1] Gộp thành bảng ô: tháng × vùng × khung giờ × thứ")
    o, d = tao_bang_o(df)
    p(f"    Dùng {len(d):,d}/{len(df):,d} vụ (bỏ tháng không trọn, giờ trống, giờ mặc định 00:00:00)")
    p(f"    {o['ky'].nunique()} tháng × {len(VUNG)} vùng × {len(KHUNG)} khung giờ × 7 thứ = {len(o):,d} ô"
      f" | trung bình {o['so_vu'].mean():.2f} vụ/ô, {(o['so_vu'] == 0).mean():.0%} ô bằng 0, lớn nhất {o['so_vu'].max()}")
    o_ra = o.assign(ky=o["ky"].astype(str))
    o_ra.to_csv(GOC / "bang_so_vu_theo_o.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(o_ra, "mo_hinh", "bang_so_vu_theo_o")

    p("\n[2] Đặc trưng: vùng, khung giờ, thứ, tháng (one-hot); số lần thứ đó trong tháng (4/5), tỉ lệ ngày giãn cách,"
      " tỉ lệ ngày nghỉ Tết, xu hướng (chuẩn hoá)")

    p("\n[3] Chia theo thời gian")
    hoc = o[o["ky"] < MOC_KIEM_TRA].sort_values(["ky", "vung", "khung_gio", "thu"]).reset_index(drop=True)
    thu_ = o[o["ky"] >= MOC_KIEM_TRA].reset_index(drop=True)
    X_h, y_h = hoc[PHAN_LOAI + SO], hoc["so_vu"].to_numpy(float)
    X_t, y_t = thu_[PHAN_LOAI + SO], thu_["so_vu"].to_numpy(float)
    p(f"    Học: {hoc['ky'].min()} – {hoc['ky'].max()} ({len(hoc):,d} ô) | kiểm tra: {thu_['ky'].min()} – {thu_['ky'].max()}"
      f" ({len(thu_):,d} ô) — không trộn tương lai vào quá khứ")

    p("\n[4] Mô hình và tinh chỉnh (GridSearchCV + TimeSeriesSplit 4 lần trên phần học, chấm theo RMSE)")
    tss = TimeSeriesSplit(4)
    mo_hinh = [
        ("Trung bình (mốc)", Pipeline([("ma_hoa", bo_ma_hoa(False)), ("mh", DummyRegressor())]), None),
        ("Linear Regression", Pipeline([("ma_hoa", bo_ma_hoa(False)), ("mh", LinearRegression())]), None),
        ("Polynomial bậc 2", Pipeline([("ma_hoa", bo_ma_hoa(True)), ("mh", LinearRegression())]), None),
        ("Ridge (bậc 2)", Pipeline([("ma_hoa", bo_ma_hoa(True)), ("mh", Ridge())]), {"mh__alpha": [1, 10, 100, 1000]}),
        ("Lasso (bậc 2)", Pipeline([("ma_hoa", bo_ma_hoa(True)), ("mh", Lasso(max_iter=50000))]),
         {"mh__alpha": [0.003, 0.01, 0.03, 0.1]}),
        ("Random Forest", Pipeline([("ma_hoa", bo_ma_hoa(False)),
                                    ("mh", RandomForestRegressor(n_estimators=300, random_state=HAT, n_jobs=-1))]),
         {"mh__min_samples_leaf": [1, 5, 20]}),
    ]
    dong, da_hoc = [], {}
    for ten, pl, luoi in mo_hinh:
        t0 = time.perf_counter()
        if luoi:
            gs = GridSearchCV(pl, luoi, cv=tss, scoring="neg_root_mean_squared_error").fit(X_h, y_h)
            pl = gs.best_estimator_
            p(f"    {ten:20s}: tốt nhất {gs.best_params_} | RMSE kiểm định {-gs.best_score_:.3f}"
              f" | {time.perf_counter() - t0:.0f} giây")
        else:
            pl.fit(X_h, y_h)
        da_hoc[ten] = pl
        kq = {"mo_hinh": ten, **danh_gia(y_t, np.clip(pl.predict(X_t), 0, None))}
        dong.append(kq)
    so_he_so = int((da_hoc["Lasso (bậc 2)"].named_steps["mh"].coef_ != 0).sum())
    tong = da_hoc["Lasso (bậc 2)"].named_steps["mh"].coef_.size
    p(f"    Lasso giữ {so_he_so}/{tong} hệ số khác 0 (tự chọn đặc trưng); Polynomial bậc 2 có {tong} cột")

    p("\n[5] Đánh giá trên 7 tháng 2022 (dự báo âm được cắt về 0)")
    bang = pd.DataFrame(dong)
    p(f"    {'mô hình':20s} {'MSE':>8s} {'RMSE':>7s} {'MAE':>7s} {'MAPE':>7s} {'R²':>7s}")
    for r in bang.itertuples(index=False):
        p(f"    {r.mo_hinh:20s} {r.MSE:8.2f} {r.RMSE:7.3f} {r.MAE:7.3f} {r.MAPE:7.1%} {r.R2:7.3f}")
    p("    MAPE chỉ tính trên ô có ít nhất 1 vụ (chia cho 0 không xác định); ô ít vụ làm MAPE phình to")
    tot = bang.loc[bang["RMSE"].idxmin(), "mo_hinh"]
    p(f"    -> RMSE thấp nhất: {tot}")
    bang.to_csv(GOC / "ket_qua_hoi_quy.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(bang, "mo_hinh", "ket_qua_hoi_quy")

    lin = da_hoc["Linear Regression"]
    he_so = pd.Series(lin.named_steps["mh"].coef_, index=lin.named_steps["ma_hoa"].get_feature_names_out())
    p("    Linear Regression — hệ số của khung giờ (số vụ thêm mỗi ô so với trung bình các khung):")
    kg = he_so[he_so.index.str.startswith("one_hot__khung_gio_")]
    kg = (kg - kg.mean()).sort_values(ascending=False)
    p("      " + " | ".join(f"{i.replace('one_hot__khung_gio_', '')} {v:+.2f}" for i, v in kg.items()))
    p("      (hệ số theo CẢ khung, nên khung dài như 09-16 với 7 tiếng tự nhiên cao; bảng [6] so theo từng giờ)")

    p("\n[6] Thời điểm đen: số vụ trung bình MỖI GIỜ, cả nước, theo khung giờ × thứ (tháng trọn 2020 – 07/2022)")
    d["thu"] = d["thu"].astype(int)
    ngay = pd.to_datetime(d["ngay"])
    so_ngay_thu = (pd.Series(pd.date_range(ngay.min(), ngay.max(), freq="D")).dt.dayofweek + 1).value_counts()
    dem = d.groupby(["khung_gio", "thu"]).size().rename("so_vu").reset_index()
    dem["vu_moi_gio"] = dem["so_vu"] / dem["thu"].map(so_ngay_thu) / dem["khung_gio"].map(DO_DAI)
    tb = dem.pivot(index="khung_gio", columns="thu", values="vu_moi_gio").reindex(KHUNG).rename(columns=THU)
    p("    (đã chia cho số ngày của mỗi thứ và độ dài khung giờ, nên khung 7 tiếng không bị lợi thế)")
    p("    " + f"{'khung':>6s} " + " ".join(f"{c:>6s}" for c in tb.columns))
    for k, r in tb.iterrows():
        p("    " + f"{k:>6s} " + " ".join(f"{v:6.2f}" for v in r))
    xep = dem.sort_values("vu_moi_gio", ascending=False)
    trung_binh = dem["so_vu"].sum() / so_ngay_thu.sum() / 24
    p(f"    Trung bình cả tuần: {trung_binh:.2f} vụ mỗi giờ. Năm ô cao nhất:")
    for r in xep.head(5).itertuples(index=False):
        p(f"      {THU[r.thu]} {r.khung_gio}h: {r.vu_moi_gio:.2f} vụ/giờ (gấp {r.vu_moi_gio / trung_binh:.1f} lần trung bình)")
    p("    Thấp nhất: " + ", ".join(f"{THU[r.thu]} {r.khung_gio}h ({r.vu_moi_gio:.2f})"
                                    for r in xep.tail(3)[::-1].itertuples(index=False)))
    den = dem.assign(thu=dem["thu"].map(THU))
    den.to_csv(GOC / "bang_thoi_diem_den.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(den, "mo_hinh", "bang_thoi_diem_den")
    p("    Lưu ý: đây là số vụ được ghi nhận, chưa chia cho lưu lượng xe — giờ đông xe tự nhiên có nhiều vụ hơn")

    (GOC / "kiem_tra_hoi_quy.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu bang_so_vu_theo_o.csv, ket_qua_hoi_quy.csv, bang_thoi_diem_den.csv, kiem_tra_hoi_quy.txt")


if __name__ == "__main__":
    bat_dau = time.perf_counter()
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"KẾT THÚC: LỖI sau {time.perf_counter() - bat_dau:.1f} giây")
        sys.exit(1)
    print(f"KẾT THÚC: thành công sau {time.perf_counter() - bat_dau:.1f} giây")

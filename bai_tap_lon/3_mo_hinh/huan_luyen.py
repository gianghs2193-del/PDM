"""BƯỚC 3 – XÂY DỰNG MÔ HÌNH · BƯỚC 3: HUẤN LUYỆN — Chương 5: Modeling (Bước 3 – Huấn luyện mô hình)

Học (fit) các mô hình đã chọn ở bước 2 (chon_mo_hinh.py) trên phần dữ liệu học đã chia ở bước 1 (chia_du_lieu.py):
    PHÂN LOẠI   : học trên phần 1–4 của train (~21.159 vụ), GIỮ phần 5 làm tập kiểm định cho bước 4.
                  Tập test (6.613 vụ) không được đụng tới.
    HỒI QUY     : học trên khối 0–3 (01/2020 – 08/2021), giữ khối 4 (09–12/2021) để kiểm định ở bước 4.
    PHÂN CỤM    : K-Means trên 63 tỉnh × 6 chỉ số, k = 3 tạm thời (bước 6 chọn lại k).
    LUẬT KẾT HỢP: Apriori (tự viết) trên toàn bộ giao_dich -> tập phổ biến -> luật có vế phải là nhãn.

Script chỉ báo cáo mô hình đã HỌC được gì (hệ số, độ sâu cây, tầm quan trọng, tâm cụm, luật) và điểm trên CHÍNH
dữ liệu học; việc chấm trên dữ liệu chưa thấy là của bước 4.

PHÂN LOẠI học từ cột GỐC (tao_khung của ky_nghe_dac_trung.py), bộ mã hoá đặc trưng (One-Hot, TargetEncoder
tỉnh/huyện/đường, tần suất) nằm TRONG pipeline của từng mô hình nên chỉ học từ phần dữ liệu học của lần đó.
Không dùng bảng train_fe đã mã hoá sẵn: bảng đó mã hoá bằng nhãn của TOÀN BỘ train, kể cả phần kiểm định,
làm điểm kiểm định của Random Forest cao hơn thật khoảng 0,01 AUC (rò rỉ nhãn).

Đầu vào  : bảng mo_hinh.chia_phan_loai, mo_hinh.chia_hoi_quy, tien_xu_ly.tai_nan_loi_sach, tien_xu_ly.ho_so_tinh,
           tien_xu_ly.giao_dich; 2_tien_xu_ly/bo_dac_trung.joblib (bộ cột và cách mã hoá của từng hướng)
Đầu ra   : mô hình đã học: huan_luyen_phan_loai_du_bao.joblib, huan_luyen_phan_loai_khong_dia_ly.joblib,
           huan_luyen_hoi_quy.joblib, huan_luyen_phan_cum.joblib
           bảng mo_hinh.huan_luyen, mo_hinh.cum_tinh, mo_hinh.tap_pho_bien, mo_hinh.luat_ket_hop (+ .csv)
           kiem_tra_huan_luyen.txt (cũng in ra màn hình)

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/huan_luyen.py
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
sys.path.insert(0, str(GOC.parent / "2_tien_xu_ly"))  # để dùng lại cách tạo và mã hoá đặc trưng của bước tiền xử lý
import ky_nghe_dac_trung as kn  # noqa: E402
from chon_mo_hinh import (CAU_HINH_APRIORI, COT_PHAN_CUM, HUONG, ung_vien_hoi_quy, ung_vien_phan_cum,  # noqa: E402
                          ung_vien_phan_loai)
from hoi_quy import PHAN_LOAI as COT_PL_HQ, SO as COT_SO_HQ, tao_bang_o  # noqa: E402

TX = "tien_xu_ly"
KHOA = ["nguon", "dong_goc"]
NHAN = "co_nguoi_chet"
PHAN_KIEM_DINH = 5            # phần 5 của train: giữ lại cho bước 4
KHOI_KIEM_DINH = 4            # khối 4 (09–12/2021): giữ lại cho bước 4
SO_CUM_TAM = 3
BC: list[str] = ["BÁO CÁO HUẤN LUYỆN", "=" * 70]
TOM_TAT: list[dict] = []


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


# ---------------------------------------------------------------- DỮ LIỆU (dùng chung cho bước 4–6)

_CACHE: dict = {}


def bo_dac_trung(huong: str) -> dict:
    """Bộ cột (nhom) và loại mã hoá (bo) mà ky_nghe_dac_trung.py đã chọn cho hướng này."""
    if "bo_dt" not in _CACHE:
        _CACHE["bo_dt"] = joblib.load(GOC.parent / "2_tien_xu_ly" / "bo_dac_trung.joblib")["huong"]
    return _CACHE["bo_dt"][huong]


def mo_hinh_phan_loai(huong: str) -> dict[str, Pipeline]:
    """Các mô hình của bước 2, mỗi mô hình đặt sau bộ mã hoá đặc trưng CHƯA học: mỗi lần fit, bộ mã hoá
    (TargetEncoder, One-Hot, tần suất) chỉ học từ dữ liệu được đưa vào fit -> không rò nhãn của phần kiểm định."""
    h = bo_dac_trung(huong)
    return {ten: Pipeline([("dac_trung", kn.tao_bo_bien_doi(h["bo"], h["nhom"])), ("mo_hinh", mh)])
            for ten, mh in ung_vien_phan_loai().items()}


def du_doan_tren_du_lieu_hoc(pl: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Nhãn và xác suất dự đoán trên CHÍNH dữ liệu đã học, bằng đúng các cột mô hình đã thấy lúc fit.
    TargetEncoder mã hoá dữ liệu học bằng chia chéo nội bộ (fit_transform) nhưng mã hoá dữ liệu mới bằng tỉ lệ
    của cả phần học (transform); gọi pl.predict(X_học) sẽ dùng cách thứ hai nên không đo được mức học thuộc.
    fit_transform có hạt giống cố định nên chạy lại cho đúng các cột lúc fit."""
    Z = clone(pl.named_steps["dac_trung"]).fit_transform(X, y)
    mh = pl.named_steps["mo_hinh"]
    return mh.predict(Z), mh.predict_proba(Z)[:, 1]


def du_lieu_phan_loai(huong: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X (cột GỐC, chưa mã hoá), y và số phần (fold 1–5) của các vụ TRAIN, theo cách chia ở mo_hinh.chia_phan_loai."""
    if "khung" not in _CACHE:
        _CACHE["khung"] = kn.tao_khung(csdl.doc(TX, "tai_nan_loi_sach")).astype({"dong_goc": "int64"})
    chia = csdl.doc("mo_hinh", "chia_phan_loai")
    chia = chia[chia["tap"] == "train"][KHOA + ["fold"]].astype({"dong_goc": "int64"})
    d = _CACHE["khung"].merge(chia, on=KHOA, validate="1:1").sort_values(KHOA).reset_index(drop=True)
    cot = [c for cs in bo_dac_trung(huong)["nhom"].values() for c in cs]
    return d[cot], d[NHAN].astype(int), d["fold"].astype(int)


def du_lieu_hoi_quy() -> pd.DataFrame:
    """Bảng ô (tháng × vùng × khung giờ × thứ) kèm cột tap (train/test) và khoi (0–4, -1 = test) của bước 1."""
    o, _ = tao_bang_o(csdl.doc(TX, "tai_nan_loi_sach"))
    chia = csdl.doc("mo_hinh", "chia_hoi_quy").astype({"ky": str})
    o = o.assign(ky_chu=o["ky"].astype(str)).merge(chia, left_on="ky_chu", right_on="ky", suffixes=("", "_c"))
    return o.drop(columns=["ky_chu", "ky_c"]).sort_values(["ky", "vung", "khung_gio", "thu"]).reset_index(drop=True)


# ---------------------------------------------------------------- APRIORI (tự viết)

def apriori(I: pd.DataFrame, min_support: float, max_len: int) -> pd.DataFrame:
    """Tìm mọi tập item có độ hỗ trợ >= min_support, tối đa max_len item.
    Bước k: ghép hai tập phổ biến cỡ k-1 có chung k-2 item đầu -> ứng viên cỡ k; bỏ ứng viên có tập con cỡ k-1
    không phổ biến (tính chất Apriori); đếm độ hỗ trợ của ứng viên còn lại trên dữ liệu."""
    X = I.to_numpy(dtype=bool)
    ten = np.array(I.columns)
    ht = X.mean(axis=0)
    cap_k = [(i,) for i in np.flatnonzero(ht >= min_support)]
    ket_qua = [((i,), ht[i]) for (i,) in cap_k]
    p(f"      cỡ 1: {X.shape[1]} item -> {len(cap_k)} phổ biến")
    for k in range(2, max_len + 1):
        truoc = set(cap_k)
        ung_vien = [a + (b[-1],) for a, b in combinations(cap_k, 2) if a[:-1] == b[:-1]]
        ung_vien = [c for c in ung_vien if all(s in truoc for s in combinations(c, k - 1))]
        cap_k = []
        for c in ung_vien:
            s = np.logical_and.reduce(X[:, c], axis=1).mean()
            if s >= min_support:
                cap_k.append(c)
                ket_qua.append((c, s))
        p(f"      cỡ {k}: {len(ung_vien)} ứng viên -> {len(cap_k)} phổ biến")
        if not cap_k:
            break
    return pd.DataFrame({"tap": [tuple(ten[list(c)]) for c, _ in ket_qua],
                         "so_item": [len(c) for c, _ in ket_qua], "support": [s for _, s in ket_qua]})


def sinh_luat(tap: pd.DataFrame, ve_phai: list[str], min_conf: float) -> pd.DataFrame:
    """Luật A -> B với B là một item nhãn: confidence = sup(A ∪ B) / sup(A), lift = confidence / sup(B)."""
    ht = dict(zip(tap["tap"].map(frozenset), tap["support"]))
    dong = []
    for t, s in zip(tap["tap"], tap["support"]):
        for b in ve_phai:
            if b in t and len(t) > 1:
                a = frozenset(t) - {b}
                conf = s / ht[a]
                dong.append({"ve_trai": " + ".join(sorted(a)), "ve_phai": b, "so_item_trai": len(a),
                             "support": s, "confidence": conf, "lift": conf / ht[frozenset([b])]})
    luat = pd.DataFrame(dong)
    luat = luat[luat["confidence"] >= min_conf].reset_index(drop=True)
    # Luật thừa: thêm item vào vế trái mà confidence không tăng ít nhất 1 điểm % so với luật con của nó
    # (vd "TP.HCM + Nam Bộ -> không chết" thừa vì TP.HCM đã nằm trong Nam Bộ).
    conf = {(frozenset(a.split(" + ")), b): c for a, b, c in zip(luat["ve_trai"], luat["ve_phai"], luat["confidence"])}
    tot_hon = []
    for a, b in zip(luat["ve_trai"], luat["ve_phai"]):
        A = frozenset(a.split(" + "))
        con = [conf.get((frozenset(s), b), ht[frozenset([b])]) for k in range(len(A)) for s in combinations(A, k)
               if (frozenset(s), b) in conf or k == 0]
        tot_hon.append(conf[(A, b)] - max(con))
    luat["cai_thien"] = tot_hon
    return luat.sort_values("lift", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- BỐN BÀI TOÁN

def hoc_phan_loai() -> None:
    p(f"\n[1] PHÂN LOẠI — học trên phần 1–4, giữ phần {PHAN_KIEM_DINH} cho bước 4")
    for huong in HUONG:
        X, y, fold = du_lieu_phan_loai(huong)
        hoc = fold != PHAN_KIEM_DINH
        Xh, yh = X[hoc], y[hoc]
        p(f"\n    Hướng {huong}: học {len(Xh):,d} vụ × {X.shape[1]} cột gốc (giữ lại {int((~hoc).sum()):,d} vụ)")
        da_hoc = {}
        for ten, mh in mo_hinh_phan_loai(huong).items():
            t = time.perf_counter()
            mh.fit(Xh, yh)
            giay = time.perf_counter() - t
            acc = accuracy_score(yh, du_doan_tren_du_lieu_hoc(mh, Xh, yh)[0])
            da_hoc[ten] = mh
            TOM_TAT.append({"bai_toan": "Phân loại", "huong": huong, "mo_hinh": ten, "so_dong_hoc": len(Xh),
                            "giay": round(giay, 2), "diem_tren_du_lieu_hoc": round(acc, 4), "thuoc_do": "Accuracy"})
            p(f"      {ten:24s}: {giay:5.2f} giây | Accuracy trên chính dữ liệu học {acc:.3f}")
        ten_cot = da_hoc["Logistic Regression"].named_steps["dac_trung"].get_feature_names_out()
        p(f"      Sau mã hoá (học trên phần học): {len(ten_cot)} cột")
        lr = da_hoc["Logistic Regression"].named_steps["mo_hinh"].named_steps["mo_hinh"]
        he_so = pd.Series(lr.coef_[0], index=ten_cot).sort_values(key=abs, ascending=False)
        p("      Logistic Regression học được (hệ số sau chuẩn hoá, 5 lớn nhất; dương = tăng khả năng có người chết):")
        for c, v in he_so.head(5).items():
            p(f"        {c:32s} {v:+.3f}")
        cay = da_hoc["Cây quyết định"].named_steps["mo_hinh"]
        p(f"      Cây quyết định: sâu {cay.get_depth()} tầng, {cay.get_n_leaves():,d} lá"
          f" (trung bình {len(Xh) / cay.get_n_leaves():.1f} vụ mỗi lá) -> học gần như thuộc lòng dữ liệu học")
        rung = da_hoc["Random Forest"].named_steps["mo_hinh"]
        qt = pd.Series(rung.feature_importances_, index=ten_cot).sort_values(ascending=False)
        p(f"      Random Forest: {len(rung.estimators_)} cây, sâu trung bình "
          f"{np.mean([c.get_depth() for c in rung.estimators_]):.0f} tầng; tầm quan trọng (5 cột đầu):")
        for c, v in qt.head(5).items():
            p(f"        {c:32s} {v:.3f}")
        joblib.dump({"mo_hinh": da_hoc, "cot": list(X.columns)}, GOC / f"huan_luyen_phan_loai_{huong}.joblib", compress=3)
        p(f"      -> huan_luyen_phan_loai_{huong}.joblib")


def hoc_hoi_quy() -> None:
    p(f"\n[2] HỒI QUY — học trên khối 0–3, giữ khối {KHOI_KIEM_DINH} (09–12/2021) cho bước 4")
    o = du_lieu_hoi_quy()
    hoc = o[(o["tap"] == "train") & (o["khoi"] < KHOI_KIEM_DINH)]
    p(f"    Học: {hoc['ky'].min()} – {hoc['ky'].max()} ({hoc['ky'].nunique()} tháng, {len(hoc):,d} ô)")
    da_hoc = {}
    for ten, mh in ung_vien_hoi_quy().items():
        t = time.perf_counter()
        mh.fit(hoc, hoc["so_vu"])
        giay = time.perf_counter() - t
        du = np.clip(mh.predict(hoc), 0, None)
        rmse, r2 = float(np.sqrt(mean_squared_error(hoc["so_vu"], du))), r2_score(hoc["so_vu"], du)
        da_hoc[ten] = mh
        TOM_TAT.append({"bai_toan": "Hồi quy", "huong": "", "mo_hinh": ten, "so_dong_hoc": len(hoc),
                        "giay": round(giay, 2), "diem_tren_du_lieu_hoc": round(r2, 4), "thuoc_do": "R2"})
        p(f"      {ten:20s}: {giay:5.2f} giây | trên chính dữ liệu học RMSE {rmse:.2f} vụ, R² {r2:.3f}")
    for ten in ("Hồi quy tuyến tính", "Ridge"):
        pl = da_hoc[ten]
        hs = pd.Series(pl.named_steps["mo_hinh"].coef_, index=pl.named_steps["ma_hoa"].get_feature_names_out())
        p(f"    {ten}: hệ số lớn nhất / nhỏ nhất = {hs.abs().max():.2f} / {hs.abs().min():.3f}"
          f" | 3 hệ số lớn nhất: " + ", ".join(f"{c.split('__')[-1]} {v:+.2f}" for c, v in
                                              hs.sort_values(key=abs, ascending=False).head(3).items()))
    joblib.dump(da_hoc, GOC / "huan_luyen_hoi_quy.joblib", compress=3)
    p("    -> huan_luyen_hoi_quy.joblib")


def hoc_phan_cum() -> None:
    p(f"\n[3] PHÂN CỤM — K-Means, k = {SO_CUM_TAM} tạm thời, trên 63 tỉnh × {len(COT_PHAN_CUM)} chỉ số")
    hs = csdl.doc(TX, "ho_so_tinh")
    km = ung_vien_phan_cum(SO_CUM_TAM).fit(hs[COT_PHAN_CUM].astype(float))
    hs["cum"] = km.labels_ + 1
    p(f"    Tổng bình phương khoảng cách tới tâm (SSE): {km.inertia_:.1f} | {km.n_iter_} vòng lặp")
    tam = pd.DataFrame(km.cluster_centers_, columns=[c.removesuffix("_z") for c in COT_PHAN_CUM],
                       index=range(1, SO_CUM_TAM + 1))
    for k, r in tam.iterrows():
        tinh = hs.loc[hs["cum"] == k, "tinh"].tolist()
        noi_bat = r.sort_values(key=abs, ascending=False).head(2)
        p(f"    Cụm {k}: {len(tinh)} tỉnh | nổi bật: " + ", ".join(f"{c} {v:+.1f} độ lệch" for c, v in noi_bat.items())
          + f" | ví dụ: {', '.join(tinh[:4])}")
    ra = hs[["tinh", "cum"]]
    ra.to_csv(GOC / "cum_tinh.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(ra, "mo_hinh", "cum_tinh")
    joblib.dump(km, GOC / "huan_luyen_phan_cum.joblib")
    TOM_TAT.append({"bai_toan": "Phân cụm", "huong": "", "mo_hinh": f"K-Means (k={SO_CUM_TAM})", "so_dong_hoc": len(hs),
                    "giay": None, "diem_tren_du_lieu_hoc": round(km.inertia_, 2), "thuoc_do": "SSE"})
    p("    -> huan_luyen_phan_cum.joblib, bảng mo_hinh.cum_tinh")


def hoc_apriori() -> None:
    ch = CAU_HINH_APRIORI
    p(f"\n[4] LUẬT KẾT HỢP — Apriori tự viết, min_support {ch['min_support']:.0%}, tối đa {ch['max_len']} item")
    gd = csdl.doc(TX, "giao_dich")
    I = gd.drop(columns=KHOA + ch["bo_item"]).astype(bool)
    t = time.perf_counter()
    tap = apriori(I, ch["min_support"], ch["max_len"])
    giay = time.perf_counter() - t
    p(f"    {len(tap)} tập phổ biến trong {giay:.2f} giây")

    p("    Kiểm tra: đếm lại độ hỗ trợ trực tiếp bằng pandas cho 20 tập bất kỳ")
    mau = tap.sample(min(20, len(tap)), random_state=42)
    lech = max(abs(I[list(t)].all(axis=1).mean() - s) for t, s in zip(mau["tap"], mau["support"]))
    p(f"      chênh lớn nhất {lech:.2e} -> {'ĐẠT' if lech < 1e-12 else 'LỖI'}")
    if lech >= 1e-12:
        raise RuntimeError("Apriori đếm sai độ hỗ trợ")

    tat_ca = sinh_luat(tap, ch["ve_phai"], ch["min_confidence"])
    luat = tat_ca[tat_ca["cai_thien"] >= 0.01].reset_index(drop=True)
    p(f"    {len(tat_ca)} luật có vế phải là nhãn và confidence ≥ {ch['min_confidence']};"
      f" bỏ {len(tat_ca) - len(luat)} luật thừa (thêm item mà confidence tăng < 1 điểm %) -> còn {len(luat)} luật")
    for b in ch["ve_phai"]:
        nhom = luat[luat["ve_phai"] == b]
        p(f"    5 luật lift cao nhất cho '{b}' (độ hỗ trợ của riêng nhãn: {I[b].mean():.1%}):")
        for _, r in nhom.head(5).iterrows():
            p(f"      {r['ve_trai']}  ->  {b} | sup {r['support']:.1%}, conf {r['confidence']:.1%}, lift {r['lift']:.2f}")
    tap_ra = tap.assign(tap=tap["tap"].map(" + ".join))
    for bang, ten in ((tap_ra, "tap_pho_bien"), (luat, "luat_ket_hop")):
        bang.to_csv(GOC / f"{ten}.csv", index=False, encoding="utf-8-sig")
        csdl.ghi(bang, "mo_hinh", ten)
    TOM_TAT.append({"bai_toan": "Luật kết hợp", "huong": "", "mo_hinh": "Apriori", "so_dong_hoc": len(I),
                    "giay": round(giay, 2), "diem_tren_du_lieu_hoc": len(luat), "thuoc_do": "Số luật"})
    p("    -> bảng mo_hinh.tap_pho_bien, mo_hinh.luat_ket_hop")


def main() -> None:
    hoc_phan_loai()
    hoc_hoi_quy()
    hoc_phan_cum()
    hoc_apriori()
    p("\n[5] Lưu tóm tắt")
    tt = pd.DataFrame(TOM_TAT)
    tt.to_csv(GOC / "huan_luyen.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(tt, "mo_hinh", "huan_luyen")
    p(f"    {len(tt)} mô hình đã học -> bảng mo_hinh.huan_luyen + huan_luyen.csv")
    (GOC / "kiem_tra_huan_luyen.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu kiem_tra_huan_luyen.txt")


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

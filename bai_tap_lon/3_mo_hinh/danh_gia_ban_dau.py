"""BƯỚC 3 – XÂY DỰNG MÔ HÌNH · BƯỚC 4: DỰ ĐOÁN VÀ ĐÁNH GIÁ BAN ĐẦU — Chương 5: Modeling (Bước 4)

Nạp các mô hình đã học ở bước 3 (huan_luyen.py), dự đoán trên phần dữ liệu mô hình CHƯA THẤY và chấm điểm:
    PHÂN LOẠI   : phần 5 của train (5.289 vụ). Accuracy, Precision, Recall, F1 (lớp 1 = có người chết), AUC,
                  ma trận nhầm lẫn; so với điểm trên dữ liệu học để thấy khoảng cách (quá khớp) và so với mốc.
    HỒI QUY     : khối 4 = 09–12/2021 (784 ô). MSE, RMSE, MAE, MAPE (bỏ ô 0 vụ), R²; dự đoán âm cắt về 0.
    PHÂN CỤM    : không có nhãn -> chỉ số nội tại: silhouette, Davies–Bouldin, Calinski–Harabasz.
    LUẬT KẾT HỢP: độ bền — tính lại support, confidence, lift của từng luật trên mỗi năm 2020, 2021, 8T2022.
Tập test (6.613 vụ) và 7 tháng 2022 của hồi quy KHÔNG được dùng ở đây.

Đầu vào  : các file huan_luyen_*.joblib; bảng mo_hinh.chia_phan_loai, mo_hinh.chia_hoi_quy, mo_hinh.luat_ket_hop,
           tien_xu_ly.* (qua các hàm du_lieu_* của huan_luyen.py)
Đầu ra   : bảng mo_hinh.danh_gia_ban_dau (dạng dài: bài toán, hướng, mô hình, tập, chỉ số, giá trị),
           mo_hinh.luat_theo_nam (+ .csv), kiem_tra_danh_gia_ban_dau.txt (cũng in ra màn hình)

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/danh_gia_ban_dau.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, calinski_harabasz_score, confusion_matrix, davies_bouldin_score,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, silhouette_score)

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
from chon_mo_hinh import CAU_HINH_APRIORI, COT_PHAN_CUM, HUONG  # noqa: E402
from huan_luyen import (KHOI_KIEM_DINH, PHAN_KIEM_DINH, du_doan_tren_du_lieu_hoc, du_lieu_hoi_quy,  # noqa: E402
                        du_lieu_phan_loai)

TX = "tien_xu_ly"
KHOA = ["nguon", "dong_goc"]
BC: list[str] = ["BÁO CÁO DỰ ĐOÁN VÀ ĐÁNH GIÁ BAN ĐẦU", "=" * 70]
KQ: list[dict] = []


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def ghi_kq(bai_toan: str, huong: str, mo_hinh: str, tap: str, diem: dict) -> None:
    for chi_so, v in diem.items():
        KQ.append({"bai_toan": bai_toan, "huong": huong, "mo_hinh": mo_hinh, "tap": tap, "chi_so": chi_so,
                   "gia_tri": float(v)})


def diem_phan_loai(y, du, xs) -> dict:
    return {"Accuracy": accuracy_score(y, du), "Precision": precision_score(y, du, zero_division=0),
            "Recall": recall_score(y, du), "F1": f1_score(y, du), "AUC": roc_auc_score(y, xs)}


def diem_hoi_quy(y, du) -> dict:
    co = y > 0
    return {"MSE": mean_squared_error(y, du), "RMSE": float(np.sqrt(mean_squared_error(y, du))),
            "MAE": mean_absolute_error(y, du), "MAPE": float(np.mean(np.abs((y[co] - du[co]) / y[co]))),
            "R2": r2_score(y, du)}


def danh_gia_phan_loai() -> None:
    p(f"\n[1] PHÂN LOẠI — chấm trên phần {PHAN_KIEM_DINH} (mô hình chưa thấy), lớp 1 = có người chết")
    for huong in HUONG:
        goi = joblib.load(GOC / f"huan_luyen_phan_loai_{huong}.joblib")
        X, y, fold = du_lieu_phan_loai(huong)
        assert list(X.columns) == goi["cot"], "cột của dữ liệu khác lúc học"
        kd, hoc = fold == PHAN_KIEM_DINH, fold != PHAN_KIEM_DINH
        p(f"\n    Hướng {huong}: kiểm định {int(kd.sum()):,d} vụ ({y[kd].mean():.1%} có người chết)")
        p(f"      {'Mô hình':24s} {'Accuracy':>8s} {'Precision':>9s} {'Recall':>7s} {'F1':>6s} {'AUC':>6s}"
          f" | {'AUC học':>7s} {'chênh':>6s}")
        for ten, mh in goi["mo_hinh"].items():
            d_kd = diem_phan_loai(y[kd], mh.predict(X[kd]), mh.predict_proba(X[kd])[:, 1])
            d_hoc = diem_phan_loai(y[hoc], *du_doan_tren_du_lieu_hoc(mh, X[hoc], y[hoc]))
            ghi_kq("Phân loại", huong, ten, "kiem_dinh", d_kd)
            ghi_kq("Phân loại", huong, ten, "hoc", d_hoc)
            p(f"      {ten:24s} {d_kd['Accuracy']:8.3f} {d_kd['Precision']:9.3f} {d_kd['Recall']:7.3f}"
              f" {d_kd['F1']:6.3f} {d_kd['AUC']:6.3f} | {d_hoc['AUC']:7.3f} {d_hoc['AUC'] - d_kd['AUC']:+6.3f}")
        rung = goi["mo_hinh"]["Random Forest"]
        tn, fp, fn, tp = confusion_matrix(y[kd], rung.predict(X[kd])).ravel()
        ghi_kq("Phân loại", huong, "Random Forest", "kiem_dinh", {"TN": tn, "FP": fp, "FN": fn, "TP": tp})
        p(f"      Ma trận nhầm lẫn Random Forest: đoán đúng không chết {tn:,d} | báo nhầm có chết {fp:,d}"
          f" | bỏ sót vụ có chết {fn:,d} | đoán đúng có chết {tp:,d}")


def danh_gia_hoi_quy() -> None:
    p(f"\n[2] HỒI QUY — chấm trên khối {KHOI_KIEM_DINH} (mô hình chưa thấy), dự đoán âm cắt về 0")
    goi = joblib.load(GOC / "huan_luyen_hoi_quy.joblib")
    o = du_lieu_hoi_quy()
    kd = o[(o["tap"] == "train") & (o["khoi"] == KHOI_KIEM_DINH)]
    hoc = o[(o["tap"] == "train") & (o["khoi"] < KHOI_KIEM_DINH)]
    p(f"    Kiểm định: {kd['ky'].min()} – {kd['ky'].max()} ({len(kd):,d} ô, trung bình {kd['so_vu'].mean():.2f} vụ/ô;"
      f" phần học trung bình {hoc['so_vu'].mean():.2f})")
    p(f"      {'Mô hình':20s} {'MSE':>7s} {'RMSE':>6s} {'MAE':>6s} {'MAPE':>6s} {'R²':>6s} | {'R² học':>6s}")
    for ten, mh in goi.items():
        d_kd = diem_hoi_quy(kd["so_vu"].to_numpy(), np.clip(mh.predict(kd), 0, None))
        d_hoc = diem_hoi_quy(hoc["so_vu"].to_numpy(), np.clip(mh.predict(hoc), 0, None))
        ghi_kq("Hồi quy", "", ten, "kiem_dinh", d_kd)
        ghi_kq("Hồi quy", "", ten, "hoc", d_hoc)
        p(f"      {ten:20s} {d_kd['MSE']:7.2f} {d_kd['RMSE']:6.2f} {d_kd['MAE']:6.2f} {d_kd['MAPE']:6.1%}"
          f" {d_kd['R2']:6.3f} | {d_hoc['R2']:6.3f}")
    rid = goi["Ridge"]
    theo_thang = kd.assign(du=np.clip(rid.predict(kd), 0, None)).groupby("ky")[["so_vu", "du"]].sum()
    p("    Tổng số vụ cả nước theo tháng (thật / Ridge dự đoán):")
    for ky, r in theo_thang.iterrows():
        p(f"      {ky}: {int(r['so_vu']):5,d} / {r['du']:7,.0f} ({r['du'] / r['so_vu'] - 1:+.0%})")


def danh_gia_phan_cum() -> None:
    p("\n[3] PHÂN CỤM — không có nhãn, dùng chỉ số nội tại")
    km = joblib.load(GOC / "huan_luyen_phan_cum.joblib")
    hs = csdl.doc(TX, "ho_so_tinh")
    Z = hs[COT_PHAN_CUM].astype(float)
    nhan = km.predict(Z)
    d = {"Silhouette": silhouette_score(Z, nhan), "Davies-Bouldin": davies_bouldin_score(Z, nhan),
         "Calinski-Harabasz": calinski_harabasz_score(Z, nhan), "SSE": km.inertia_}
    ghi_kq("Phân cụm", "", f"K-Means (k={km.n_clusters})", "toan_bo", d)
    p(f"    k = {km.n_clusters}: silhouette {d['Silhouette']:.3f} (−1 đến 1, càng cao càng tách bạch)"
      f" | Davies–Bouldin {d['Davies-Bouldin']:.2f} (càng thấp càng tốt) | Calinski–Harabasz {d['Calinski-Harabasz']:.1f}")
    muc = "yếu, các cụm chồng lấn" if d["Silhouette"] < 0.25 else ("vừa phải" if d["Silhouette"] < 0.5 else "rõ")
    p(f"    -> cấu trúc cụm {muc}; số cụm sẽ chọn lại ở bước 6")


def danh_gia_luat() -> None:
    p("\n[4] LUẬT KẾT HỢP — độ bền theo năm: tính lại từng luật trên mỗi năm báo cáo")
    luat = csdl.doc("mo_hinh", "luat_ket_hop")
    gd = csdl.doc(TX, "giao_dich")
    I = gd.drop(columns=KHOA + CAU_HINH_APRIORI["bo_item"]).astype(bool)
    dong = []
    for _, r in luat.iterrows():
        trai = r["ve_trai"].split(" + ")
        for nam in ["Toàn bộ", *sorted(gd["nguon"].unique())]:
            m = np.ones(len(I), bool) if nam == "Toàn bộ" else (gd["nguon"] == nam).to_numpy()
            a, b = I.loc[m, trai].all(axis=1), I.loc[m, r["ve_phai"]]
            conf = (a & b).sum() / a.sum() if a.sum() else np.nan
            dong.append({"ve_trai": r["ve_trai"], "ve_phai": r["ve_phai"], "nam": nam, "so_vu": int(m.sum()),
                         "support": (a & b).mean(), "confidence": conf, "lift": conf / b.mean()})
    theo_nam = pd.DataFrame(dong)
    bang = theo_nam[theo_nam["nam"] != "Toàn bộ"].pivot_table(index=["ve_trai", "ve_phai"], columns="nam", values="lift")
    ben = bang[(bang > 1).all(axis=1)]
    p(f"    {len(ben)}/{len(bang)} luật có lift > 1 ở cả ba năm (bền); {len(bang) - len(ben)} luật đổi chiều ở ít nhất một năm")
    for (trai, phai), r in bang.sort_values(list(bang.columns)[0], ascending=False).head(8).iterrows():
        p(f"      {trai} -> {phai}: lift " + " | ".join(f"{n} {v:.2f}" for n, v in r.items())
          + ("" if (r > 1).all() else "  <- không bền"))
    theo_nam.to_csv(GOC / "luat_theo_nam.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(theo_nam, "mo_hinh", "luat_theo_nam")
    ghi_kq("Luật kết hợp", "", "Apriori", "theo_nam", {"so_luat": len(bang), "so_luat_ben": len(ben)})


def main() -> None:
    danh_gia_phan_loai()
    danh_gia_hoi_quy()
    danh_gia_phan_cum()
    danh_gia_luat()
    p("\n[5] Lưu")
    kq = pd.DataFrame(KQ)
    kq.to_csv(GOC / "danh_gia_ban_dau.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(kq, "mo_hinh", "danh_gia_ban_dau")
    p(f"    {len(kq)} con số -> bảng mo_hinh.danh_gia_ban_dau, mo_hinh.luat_theo_nam (+ .csv)")
    (GOC / "kiem_tra_danh_gia_ban_dau.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu kiem_tra_danh_gia_ban_dau.txt")


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

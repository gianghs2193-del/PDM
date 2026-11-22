"""BƯỚC 3 – XÂY DỰNG MÔ HÌNH · BƯỚC 5: KIỂM ĐỊNH CHÉO — Chương 5: Modeling (Bước 5)

Bước 4 chấm mỗi mô hình MỘT lần (phần 5 / khối 4). Bước này chấm lại nhiều lần trên các phần khác nhau của train
để biết điểm dao động bao nhiêu và khoảng cách giữa các mô hình có thật hay chỉ do một lần chia:
    PHÂN LOẠI   : 5 phần đã lưu ở mo_hinh.chia_phan_loai (StratifiedKFold). Lần k: học 4 phần còn lại, chấm phần k.
                  Trung bình ± độ lệch chuẩn của Accuracy, Precision, Recall, F1, AUC; so cặp Random Forest với
                  Logistic Regression bằng kiểm định t có hiệu chỉnh (Nadeau & Bengio) vì các lần chia dùng chung dữ liệu.
    HỒI QUY     : các khối thời gian ở mo_hinh.chia_hoi_quy (TimeSeriesSplit). Lần k: học khối 0..k-1, chấm khối k
                  — luôn học quá khứ, đoán tương lai.
    PHÂN CỤM    : không có nhãn để chấm chéo -> đo độ ổn định: lấy mẫu lặp (bootstrap) 63 tỉnh 100 lần, học lại
                  K-Means, so nhãn với lần học trên đủ 63 tỉnh bằng chỉ số Rand hiệu chỉnh (ARI).
    LUẬT KẾT HỢP: như bước 3, dùng toàn bộ giao_dich (luật để mô tả, không để đoán vụ mới); chia ngẫu nhiên 5 phần,
                  tìm luật trên 4 phần, tính lại confidence và lift trên phần còn lại.
Phân loại học từ cột gốc, bộ mã hoá đặc trưng học lại trong từng lần (xem mo_hinh_phan_loai ở huan_luyen.py).
Tập test phân loại (6.613 vụ) và 7 tháng 2022 của hồi quy KHÔNG được dùng ở đây.

Đầu vào  : bảng mo_hinh.chia_phan_loai, mo_hinh.chia_hoi_quy, tien_xu_ly.* (qua các hàm của huan_luyen.py)
Đầu ra   : bảng mo_hinh.kiem_dinh_cheo (dạng dài: bài toán, hướng, mô hình, lần, tập, chỉ số, giá trị) (+ .csv),
           kiem_tra_kiem_dinh_cheo.txt (cũng in ra màn hình)

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/kiem_dinh_cheo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
from chon_mo_hinh import CAU_HINH_APRIORI, COT_PHAN_CUM, HUONG, ung_vien_hoi_quy, ung_vien_phan_cum  # noqa: E402
from danh_gia_ban_dau import diem_hoi_quy, diem_phan_loai  # noqa: E402
import huan_luyen  # noqa: E402
from huan_luyen import (SO_CUM_TAM, apriori, du_doan_tren_du_lieu_hoc, du_lieu_hoi_quy, du_lieu_phan_loai, mo_hinh_phan_loai,  # noqa: E402
                        sinh_luat)

TX = "tien_xu_ly"
KHOA = ["nguon", "dong_goc"]
SO_LAN_BOOTSTRAP = 100
BC: list[str] = ["BÁO CÁO KIỂM ĐỊNH CHÉO", "=" * 70]
KQ: list[dict] = []


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def ghi_kq(bai_toan: str, huong: str, mo_hinh: str, lan: int, tap: str, diem: dict) -> None:
    for chi_so, v in diem.items():
        KQ.append({"bai_toan": bai_toan, "huong": huong, "mo_hinh": mo_hinh, "lan": int(lan), "tap": tap,
                   "chi_so": chi_so, "gia_tri": float(v)})


def t_hieu_chinh(chenh: np.ndarray, n_kiem: int, n_hoc: int) -> tuple[float, float]:
    """Kiểm định t cặp có hiệu chỉnh (Nadeau & Bengio, 2003): các lần học dùng chung ~3/4 dữ liệu nên điểm giữa
    các lần không độc lập; phương sai được nhân thêm (1/k + n_kiem/n_hoc) thay vì 1/k."""
    k = len(chenh)
    ps = chenh.var(ddof=1)
    if ps == 0:
        return np.inf, 0.0
    t = chenh.mean() / np.sqrt((1 / k + n_kiem / n_hoc) * ps)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


# ---------------------------------------------------------------- BỐN BÀI TOÁN

def cheo_phan_loai() -> None:
    p("\n[1] PHÂN LOẠI — 5 lần: học 4 phần, chấm phần còn lại (lớp 1 = có người chết)")
    for huong in HUONG:
        X, y, fold = du_lieu_phan_loai(huong)
        phan = sorted(fold.unique())
        p(f"\n    Hướng {huong}: {len(X):,d} vụ × {X.shape[1]} cột gốc, {len(phan)} phần; mã hoá học lại trong từng lần")
        auc = {}
        t0 = time.perf_counter()
        for ten in mo_hinh_phan_loai(huong):
            auc[ten] = []
            for k in phan:
                kd, hoc = fold == k, fold != k
                mh = mo_hinh_phan_loai(huong)[ten].fit(X[hoc], y[hoc])
                d_kd = diem_phan_loai(y[kd], mh.predict(X[kd]), mh.predict_proba(X[kd])[:, 1])
                d_hoc = {"AUC": diem_phan_loai(y[hoc], *du_doan_tren_du_lieu_hoc(mh, X[hoc], y[hoc]))["AUC"]}
                ghi_kq("Phân loại", huong, ten, k, "kiem_dinh", d_kd)
                ghi_kq("Phân loại", huong, ten, k, "hoc", d_hoc)
                auc[ten].append(d_kd["AUC"])
        p(f"      ({time.perf_counter() - t0:.0f} giây) trung bình ± độ lệch chuẩn qua {len(phan)} lần:")
        p(f"      {'Mô hình':24s} {'Accuracy':>13s} {'Precision':>13s} {'Recall':>13s} {'F1':>13s} {'AUC':>13s}"
          f" | {'AUC học':>7s}")
        kq = pd.DataFrame(KQ)
        kq = kq[(kq["bai_toan"] == "Phân loại") & (kq["huong"] == huong)]
        for ten in auc:
            m = kq[(kq["mo_hinh"] == ten) & (kq["tap"] == "kiem_dinh")].groupby("chi_so")["gia_tri"]
            tb, dl = m.mean(), m.std()
            auc_hoc = kq[(kq["mo_hinh"] == ten) & (kq["tap"] == "hoc")]["gia_tri"].mean()
            p(f"      {ten:24s} " + " ".join(f"{tb[c]:6.3f}±{dl[c]:.3f}" for c in
                                            ["Accuracy", "Precision", "Recall", "F1", "AUC"]) + f" | {auc_hoc:7.3f}")
        p("      AUC từng lần (phần 1 → 5):")
        for ten, v in auc.items():
            p(f"        {ten:24s} " + "  ".join(f"{a:.3f}" for a in v))
        chenh = np.array(auc["Random Forest"]) - np.array(auc["Logistic Regression"])
        n_kd = int((fold == phan[0]).sum())
        t, pv = t_hieu_chinh(chenh, n_kd, len(X) - n_kd)
        thang = int((chenh > 0).sum())
        p(f"      Random Forest − Logistic Regression: AUC chênh {chenh.mean():+.4f} ± {chenh.std(ddof=1):.4f};"
          f" RF hơn ở {thang}/{len(chenh)} lần; t hiệu chỉnh = {t:.2f}, p = {pv:.3f}")
        if pv >= 0.05:
            p("      -> khác biệt KHÔNG có ý nghĩa thống kê ở mức 5%: chưa đủ căn cứ nói Random Forest tốt hơn")
        else:
            p(f"      -> khác biệt có ý nghĩa thống kê ở mức 5%: {'Random Forest' if chenh.mean() > 0 else 'Logistic Regression'} tốt hơn")
        ghi_kq("Phân loại", huong, "RF − LR", 0, "so_sanh", {"AUC_chenh_tb": chenh.mean(), "t_hieu_chinh": t,
                                                              "p": pv, "so_lan_RF_hon": thang})


def cheo_hoi_quy() -> None:
    p("\n[2] HỒI QUY — học các khối trước, chấm khối sau (dự đoán âm cắt về 0)")
    o = du_lieu_hoi_quy()
    o = o[o["tap"] == "train"]
    khoi = sorted(k for k in o["khoi"].unique() if k > 0)
    p(f"      {'Lần':4s} {'Học':17s} {'Chấm':17s} | " + " | ".join(f"{t:^19s}" for t in ung_vien_hoi_quy()))
    p(f"      {'':40s} | " + " | ".join(f"{'RMSE':>6s} {'MAE':>5s} {'R²':>6s}" for _ in ung_vien_hoi_quy()))
    rmse = {t: [] for t in ung_vien_hoi_quy()}
    for k in khoi:
        hoc, kd = o[o["khoi"] < k], o[o["khoi"] == k]
        thang = lambda d: set(pd.PeriodIndex(d["ky"].astype(str), freq="M").month)  # noqa: E731
        thang_moi = sorted(thang(kd) - thang(hoc))
        o_dong = []
        for ten, mh in ung_vien_hoi_quy().items():
            mh.fit(hoc, hoc["so_vu"])
            d = diem_hoi_quy(kd["so_vu"].to_numpy(), np.clip(mh.predict(kd), 0, None))
            ghi_kq("Hồi quy", "", ten, k, "kiem_dinh", d)
            rmse[ten].append(d["RMSE"])
            o_dong.append(f"{d['RMSE']:6.2f} {d['MAE']:5.2f} {d['R2']:6.3f}")
        p(f"      {k:<4d} {str(hoc['ky'].min())}–{str(hoc['ky'].max()):9s} {str(kd['ky'].min())}–{str(kd['ky'].max()):9s} | "
          + " | ".join(o_dong) + (f"  <- tháng {thang_moi} chưa từng có trong phần học" if thang_moi else ""))
    p(f"      {'TB':4s} {'':35s} | " + " | ".join(f"{np.mean(v):6.2f} {'':5s} {'':6s}" for v in rmse.values()))
    lr, rd = np.array(rmse["Hồi quy tuyến tính"]), np.array(rmse["Ridge"])
    p(f"      Ridge − tuyến tính: RMSE chênh trung bình {np.mean(rd - lr):+.4f} vụ/ô -> "
      + ("gần như trùng nhau; alpha = 1 quá nhỏ để thấy tác dụng, sẽ thử ở bước 6" if abs(np.mean(rd - lr)) < 0.01
         else "có khác biệt"))


def cheo_phan_cum() -> None:
    p(f"\n[3] PHÂN CỤM — độ ổn định: bootstrap {SO_LAN_BOOTSTRAP} lần, K-Means k = {SO_CUM_TAM}")
    hs = csdl.doc(TX, "ho_so_tinh")
    Z = hs[COT_PHAN_CUM].astype(float).to_numpy()
    goc = ung_vien_phan_cum(SO_CUM_TAM).fit(Z).labels_
    rng = np.random.default_rng(42)
    ari = []
    for lan in range(1, SO_LAN_BOOTSTRAP + 1):
        chon = rng.choice(len(Z), size=len(Z), replace=True)
        nhan = ung_vien_phan_cum(SO_CUM_TAM).fit(Z[chon]).predict(Z)
        ari.append(adjusted_rand_score(goc, nhan))
        ghi_kq("Phân cụm", "", f"K-Means (k={SO_CUM_TAM})", lan, "bootstrap", {"ARI": ari[-1]})
    ari = np.array(ari)
    p(f"    ARI so với lần học trên đủ 63 tỉnh (1 = chia y hệt, 0 = như chia ngẫu nhiên):")
    p(f"      trung bình {ari.mean():.3f} | trung vị {np.median(ari):.3f} | thấp nhất {ari.min():.3f}"
      f" | {np.mean(ari >= 0.8):.0%} số lần ≥ 0,8")
    muc = "ổn định" if ari.mean() >= 0.8 else ("tạm ổn định" if ari.mean() >= 0.6 else "không ổn định")
    p(f"    -> cách chia 3 cụm {muc}: đổi vài tỉnh trong dữ liệu là ranh giới cụm thay đổi"
      if ari.mean() < 0.8 else f"    -> cách chia 3 cụm {muc}")


def cheo_luat() -> None:
    ch = CAU_HINH_APRIORI
    p("\n[4] LUẬT KẾT HỢP — chia toàn bộ giao_dich (như bước 3) thành 5 phần ngẫu nhiên;"
      " tìm luật trên 4 phần, kiểm lại trên phần còn lại")
    gd = csdl.doc(TX, "giao_dich").astype({"dong_goc": "int64"}).sort_values(KHOA).reset_index(drop=True)
    gd["fold"] = 0
    for k, (_, kd) in enumerate(KFold(5, shuffle=True, random_state=42).split(gd), 1):
        gd.loc[kd, "fold"] = k
    I = gd.drop(columns=KHOA + ["fold"] + ch["bo_item"]).astype(bool)
    tap_luat = {}
    for k in sorted(gd["fold"].unique()):
        hoc, kd = (gd["fold"] != k).to_numpy(), (gd["fold"] == k).to_numpy()
        huan_luyen.p = lambda *_: None  # apriori() in từng cỡ tập ra màn hình; ở đây chạy 5 lần nên tắt
        tap = apriori(I[hoc], ch["min_support"], ch["max_len"])
        luat = sinh_luat(tap, ch["ve_phai"], ch["min_confidence"])
        luat = luat[luat["cai_thien"] >= 0.01]
        lift_kd, conf_kd = [], []
        for trai, phai in zip(luat["ve_trai"], luat["ve_phai"]):
            a, b = I.loc[kd, trai.split(" + ")].all(axis=1), I.loc[kd, phai]
            conf = (a & b).sum() / a.sum()
            conf_kd.append(conf)
            lift_kd.append(conf / b.mean())
        lift_kd, conf_kd = np.array(lift_kd), np.array(conf_kd)
        tap_luat[k] = set(zip(luat["ve_trai"], luat["ve_phai"]))
        d = {"so_luat": len(luat), "lift_hoc_tb": luat["lift"].mean(), "lift_kiem_dinh_tb": lift_kd.mean(),
             "ti_le_lift_tren_1": np.mean(lift_kd > 1), "ti_le_conf_dat": np.mean(conf_kd >= ch["min_confidence"])}
        ghi_kq("Luật kết hợp", "", "Apriori", k, "kiem_dinh", d)
        p(f"      lần {k}: {len(luat)} luật | lift trên phần học {d['lift_hoc_tb']:.3f} → trên phần {k}"
          f" {d['lift_kiem_dinh_tb']:.3f} | lift > 1: {d['ti_le_lift_tren_1']:.0%}"
          f" | confidence vẫn ≥ {ch['min_confidence']}: {d['ti_le_conf_dat']:.0%}")
    chung = set.intersection(*tap_luat.values())
    hop = set.union(*tap_luat.values())
    p(f"    {len(chung)} luật xuất hiện ở cả 5 lần / {len(hop)} luật xuất hiện ít nhất một lần"
      f" (Jaccard {len(chung) / len(hop):.2f})")
    ghi_kq("Luật kết hợp", "", "Apriori", 0, "so_sanh", {"so_luat_ca_5_lan": len(chung), "so_luat_it_nhat_1_lan": len(hop)})


def main() -> None:
    cheo_phan_loai()
    cheo_hoi_quy()
    cheo_phan_cum()
    cheo_luat()
    p("\n[5] Lưu")
    kq = pd.DataFrame(KQ)
    kq.to_csv(GOC / "kiem_dinh_cheo.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(kq, "mo_hinh", "kiem_dinh_cheo")
    p(f"    {len(kq)} con số -> bảng mo_hinh.kiem_dinh_cheo (+ kiem_dinh_cheo.csv)")
    (GOC / "kiem_tra_kiem_dinh_cheo.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu kiem_tra_kiem_dinh_cheo.txt")


if __name__ == "__main__":
    bat_dau = time.perf_counter()
    try:
        main()
        print(f"KẾT THÚC: XONG sau {time.perf_counter() - bat_dau:.1f} giây")
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"KẾT THÚC: LỖI sau {time.perf_counter() - bat_dau:.1f} giây")

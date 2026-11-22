"""BƯỚC 3 – XÂY DỰNG MÔ HÌNH · BƯỚC 6: TINH CHỈNH VÀ CHẤM TẬP TEST — Chương 5: Modeling (Bước 6)

Tinh chỉnh siêu tham số trên ĐÚNG các cách chia của bước 1 (không chia lại), chọn cấu hình theo điểm kiểm định chéo,
học lại cấu hình đã chọn trên toàn bộ train rồi chấm tập test MỘT lần:
    PHÂN LOẠI   : GridSearchCV với PredefinedSplit = cột fold của mo_hinh.chia_phan_loai, thước đo AUC.
                  Logistic Regression: C; Cây quyết định: max_depth, min_samples_leaf;
                  Random Forest: max_depth, min_samples_leaf, max_features.
                  Chọn mô hình có AUC kiểm định chéo cao nhất; chênh < 0,002 thì chọn mô hình đơn giản hơn.
                  Học lại trên 26.449 vụ train, chấm 6.613 vụ test.
    HỒI QUY     : Ridge alpha, kiểm định theo 4 khối thời gian (học khối trước, chấm khối sau), thước đo RMSE.
                  Học lại trên 01/2020–12/2021, chấm 01–07/2022.
    PHÂN CỤM    : k = 2..8: SSE (khuỷu tay), silhouette, Davies–Bouldin, độ ổn định bootstrap (ARI).
                  Chọn k có silhouette cao nhất trong các k có ARI trung bình ≥ 0,6.
    LUẬT KẾT HỢP: min_support × min_confidence: số luật, độ ổn định qua 5 phần (Jaccard), lift trên phần kiểm.
                  Chọn cấu hình ổn định nhất trong các cấu hình cho ≥ 10 luật.
Tập test chỉ được đọc ở cuối mục [1] và [2], sau khi cấu hình đã chọn xong.

Đầu vào  : như bước 3–5 (qua các hàm của huan_luyen.py), thêm vụ test của mo_hinh.chia_phan_loai và 7 tháng 2022
Đầu ra   : bảng mo_hinh.tinh_chinh (mọi cấu hình đã thử), mo_hinh.ket_qua_test, mo_hinh.cum_tinh_cuoi,
           mo_hinh.luat_cuoi (+ .csv); mo_hinh_cuoi_*.joblib; kiem_tra_tinh_chinh.txt (cũng in ra màn hình)

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/tinh_chinh.py    (khoảng 5–10 phút)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, davies_bouldin_score, mean_squared_error,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
import huan_luyen as hl  # noqa: E402
from chon_mo_hinh import CAU_HINH_APRIORI, COT_PHAN_CUM, HUONG, SO_CUM_THU, ung_vien_hoi_quy, ung_vien_phan_cum  # noqa: E402
from danh_gia_ban_dau import diem_hoi_quy, diem_phan_loai  # noqa: E402
from kiem_dinh_cheo import t_hieu_chinh  # noqa: E402

TX = "tien_xu_ly"
KHOA = ["nguon", "dong_goc"]
NHAN = "co_nguoi_chet"
BC: list[str] = ["BÁO CÁO TINH CHỈNH VÀ CHẤM TẬP TEST", "=" * 70]
THU: list[dict] = []    # mọi cấu hình đã thử -> mo_hinh.tinh_chinh
TEST: list[dict] = []   # điểm trên tập test -> mo_hinh.ket_qua_test

LUOI_PHAN_LOAI = {
    "Logistic Regression": {"mo_hinh__mo_hinh__C": [0.01, 0.1, 1, 10]},
    "Cây quyết định": {"mo_hinh__max_depth": [4, 6, 8, 12], "mo_hinh__min_samples_leaf": [20, 50, 100]},
    "Random Forest": {"mo_hinh__max_depth": [8, 12, 20], "mo_hinh__min_samples_leaf": [5, 20],
                      "mo_hinh__max_features": ["sqrt", 0.5]},
}
DON_GIAN = ["Logistic Regression", "Cây quyết định", "Random Forest"]   # thứ tự ưu tiên khi điểm gần bằng nhau
LUOI_ALPHA = [0.1, 1, 10, 30, 100, 300, 1000]
LUOI_APRIORI = {"min_support": [0.03, 0.05, 0.08], "min_confidence": [0.5, 0.55, 0.6]}
SO_LAN_BOOTSTRAP = 50


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def gon(tham_so: dict) -> str:
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in tham_so.items())


def du_lieu_test_phan_loai(huong: str) -> tuple[pd.DataFrame, pd.Series]:
    """Cột gốc và nhãn của 6.613 vụ TEST (fold 0 ở mo_hinh.chia_phan_loai)."""
    hl.du_lieu_phan_loai(huong)  # nạp bảng cột gốc vào bộ nhớ đệm của huan_luyen
    chia = csdl.doc("mo_hinh", "chia_phan_loai")
    chia = chia[chia["tap"] == "test"][KHOA].astype({"dong_goc": "int64"})
    d = hl._CACHE["khung"].merge(chia, on=KHOA, validate="1:1").sort_values(KHOA).reset_index(drop=True)
    cot = [c for cs in hl.bo_dac_trung(huong)["nhom"].values() for c in cs]
    return d[cot], d[NHAN].astype(int)


# ---------------------------------------------------------------- [1] PHÂN LOẠI

def tinh_chinh_phan_loai() -> None:
    p("\n[1] PHÂN LOẠI — GridSearchCV trên 5 phần của bước 1, thước đo AUC")
    for huong in HUONG:
        X, y, fold = hl.du_lieu_phan_loai(huong)
        cv = PredefinedSplit(fold.to_numpy() - 1)   # phần 1..5 -> 0..4
        p(f"\n    Hướng {huong}: {len(X):,d} vụ train")
        tot = {}
        for ten, luoi in LUOI_PHAN_LOAI.items():
            t0 = time.perf_counter()
            gs = GridSearchCV(hl.mo_hinh_phan_loai(huong)[ten], luoi, scoring="roc_auc", cv=cv, refit=True)
            gs.fit(X, y)
            kq = pd.DataFrame(gs.cv_results_)
            for _, r in kq.iterrows():
                THU.append({"bai_toan": "Phân loại", "huong": huong, "mo_hinh": ten, "tham_so": gon(r["params"]),
                            "chi_so": "AUC_tb", "gia_tri": r["mean_test_score"]})
                THU.append({"bai_toan": "Phân loại", "huong": huong, "mo_hinh": ten, "tham_so": gon(r["params"]),
                            "chi_so": "AUC_dl", "gia_tri": r["std_test_score"]})
            tot[ten] = gs
            p(f"      {ten:20s}: {len(kq):2d} cấu hình × 5 phần ({time.perf_counter() - t0:5.0f} giây)"
              f" | tốt nhất {gon(gs.best_params_)} -> AUC {gs.best_score_:.3f}"
              f" ± {kq.loc[gs.best_index_, 'std_test_score']:.3f}")
        # so cặp Random Forest đã chỉnh với Logistic Regression đã chỉnh trên cùng 5 phần (như bước 5)
        tung = {t: np.array([g.cv_results_[f"split{i}_test_score"][g.best_index_] for i in range(5)])
                for t, g in tot.items()}
        chenh = tung["Random Forest"] - tung["Logistic Regression"]
        n_kd = int((fold == 1).sum())
        t, pv = t_hieu_chinh(chenh, n_kd, len(X) - n_kd)
        p(f"      Random Forest − Logistic Regression (đã chỉnh): AUC chênh {chenh.mean():+.4f}; RF hơn ở"
          f" {int((chenh > 0).sum())}/5 lần; t hiệu chỉnh = {t:.2f}, p = {pv:.3f}")
        THU.append({"bai_toan": "Phân loại", "huong": huong, "mo_hinh": "RF − LR", "tham_so": "đã chỉnh",
                    "chi_so": "p", "gia_tri": pv})
        # chọn: AUC cao nhất, chênh < 0,002 thì lấy mô hình đơn giản hơn
        diem = {t: g.best_score_ for t, g in tot.items()}
        cao = max(diem.values())
        chon = next(t for t in DON_GIAN if diem[t] >= cao - 0.002)
        p(f"      -> CHỌN {chon} ({gon(tot[chon].best_params_)}), AUC kiểm định chéo {diem[chon]:.3f}"
          + ("" if diem[chon] == cao else f" (cao nhất {cao:.3f}, chênh < 0,002 nên chọn mô hình đơn giản hơn)"))

        # mức học thuộc của mô hình đã chỉnh, trên toàn bộ train
        for ten, gs in tot.items():
            _, xs = hl.du_doan_tren_du_lieu_hoc(gs.best_estimator_, X, y)
            p(f"        {ten:20s} AUC trên chính train {roc_auc_score(y, xs):.3f} | kiểm định chéo {gs.best_score_:.3f}")

        # ---- TẬP TEST: đọc ở đây, một lần, sau khi đã chọn xong
        Xt, yt = du_lieu_test_phan_loai(huong)
        p(f"      TẬP TEST ({len(Xt):,d} vụ, {yt.mean():.1%} có người chết) — mô hình học lại trên toàn bộ train:")
        p(f"        {'Mô hình':24s} {'Accuracy':>8s} {'Precision':>9s} {'Recall':>7s} {'F1':>6s} {'AUC':>6s}")
        moc = clone(hl.mo_hinh_phan_loai(huong)["Mốc (đoán theo tỉ lệ)"]).fit(X, y)
        for ten, mh in [("Mốc (đoán theo tỉ lệ)", moc)] + [(t, g.best_estimator_) for t, g in tot.items()]:
            d = diem_phan_loai(yt, mh.predict(Xt), mh.predict_proba(Xt)[:, 1])
            for c, v in d.items():
                TEST.append({"bai_toan": "Phân loại", "huong": huong, "mo_hinh": ten,
                             "duoc_chon": ten == chon, "chi_so": c, "gia_tri": float(v)})
            p(f"        {ten + (' *' if ten == chon else ''):24s} {d['Accuracy']:8.3f} {d['Precision']:9.3f}"
              f" {d['Recall']:7.3f} {d['F1']:6.3f} {d['AUC']:6.3f}")
        mh = tot[chon].best_estimator_
        tn, fp, fn, tp = confusion_matrix(yt, mh.predict(Xt)).ravel()
        p(f"        Ma trận nhầm lẫn ({chon}): đúng không chết {tn:,d} | báo nhầm {fp:,d} | bỏ sót {fn:,d}"
          f" | đúng có chết {tp:,d}")
        p("        (* = mô hình đã chọn TRƯỚC khi xem test; các dòng khác chỉ để tham khảo)")
        if chon == "Logistic Regression":
            ten_cot = mh.named_steps["dac_trung"].get_feature_names_out()
            he_so = pd.Series(mh.named_steps["mo_hinh"].named_steps["mo_hinh"].coef_[0], index=ten_cot)
            p("        Hệ số lớn nhất của mô hình cuối (sau chuẩn hoá; dương = tăng khả năng có người chết):")
            for c, v in he_so.sort_values(key=abs, ascending=False).head(6).items():
                p(f"          {c:32s} {v:+.3f}")
        joblib.dump({"mo_hinh": mh, "ten": chon, "tham_so": tot[chon].best_params_, "cot": list(X.columns)},
                    GOC / f"mo_hinh_cuoi_phan_loai_{huong}.joblib", compress=3)


# ---------------------------------------------------------------- [2] HỒI QUY

def tinh_chinh_hoi_quy() -> None:
    p("\n[2] HỒI QUY — Ridge alpha, kiểm định theo 4 khối thời gian, thước đo RMSE")
    o = hl.du_lieu_hoi_quy()
    tr = o[o["tap"] == "train"].reset_index(drop=True)
    te = o[o["tap"] == "test"].reset_index(drop=True)
    cv = [(np.flatnonzero(tr["khoi"] < k), np.flatnonzero(tr["khoi"] == k)) for k in range(1, 5)]
    gs = GridSearchCV(ung_vien_hoi_quy()["Ridge"], {"mo_hinh__alpha": LUOI_ALPHA},
                      scoring="neg_root_mean_squared_error", cv=cv, refit=True).fit(tr, tr["so_vu"])
    kq = pd.DataFrame(gs.cv_results_)
    p(f"      {'alpha':>7s} {'RMSE tb':>8s}   RMSE từng khối 1 → 4")
    for _, r in kq.iterrows():
        tung = [-r[f"split{i}_test_score"] for i in range(4)]
        THU.append({"bai_toan": "Hồi quy", "huong": "", "mo_hinh": "Ridge", "tham_so": gon(r["params"]),
                    "chi_so": "RMSE_tb", "gia_tri": -r["mean_test_score"]})
        p(f"      {r['params']['mo_hinh__alpha']:7g} {-r['mean_test_score']:8.3f}   "
          + "  ".join(f"{v:.2f}" for v in tung))
    lin = [np.sqrt(mean_squared_error(tr.loc[kd, "so_vu"], np.clip(
        clone(ung_vien_hoi_quy()["Hồi quy tuyến tính"]).fit(tr.loc[h], tr.loc[h, "so_vu"]).predict(tr.loc[kd]), 0, None)))
        for h, kd in cv]
    p(f"      Hồi quy tuyến tính (không phạt): RMSE tb {np.mean(lin):.3f}")
    a = gs.best_params_["mo_hinh__alpha"]
    p(f"      -> CHỌN Ridge alpha = {a:g} (RMSE kiểm định {-gs.best_score_:.3f}; chưa cắt dự đoán âm)")

    p(f"      TẬP TEST: {te['ky'].astype(str).min()} – {te['ky'].astype(str).max()}, {len(te):,d} ô"
      f" — học lại trên {tr['ky'].astype(str).min()} – {tr['ky'].astype(str).max()}")
    moc = clone(ung_vien_hoi_quy()["Mốc (trung bình)"]).fit(tr, tr["so_vu"])
    lr = clone(ung_vien_hoi_quy()["Hồi quy tuyến tính"]).fit(tr, tr["so_vu"])
    p(f"        {'Mô hình':24s} {'RMSE':>6s} {'MAE':>6s} {'MAPE':>7s} {'R²':>7s}")
    du_chon = None
    for ten, mh in [("Mốc (trung bình)", moc), ("Hồi quy tuyến tính", lr), (f"Ridge (alpha={a:g})", gs.best_estimator_)]:
        du = np.clip(mh.predict(te), 0, None)
        d = diem_hoi_quy(te["so_vu"].to_numpy(), du)
        chon = ten.startswith("Ridge")
        du_chon = du if chon else du_chon
        for c, v in d.items():
            TEST.append({"bai_toan": "Hồi quy", "huong": "", "mo_hinh": ten, "duoc_chon": chon,
                         "chi_so": c, "gia_tri": float(v)})
        p(f"        {ten + (' *' if chon else ''):24s} {d['RMSE']:6.2f} {d['MAE']:6.2f} {d['MAPE']:7.1%} {d['R2']:7.3f}")
    thang = te.assign(du=du_chon).groupby(te["ky"].astype(str))[["so_vu", "du"]].sum()
    p("        Tổng số vụ cả nước theo tháng (thật / dự báo):")
    for k, r in thang.iterrows():
        p(f"          {k}: {r['so_vu']:5,.0f} / {r['du']:7,.0f} ({r['du'] / r['so_vu'] - 1:+.0%})")
    joblib.dump({"mo_hinh": gs.best_estimator_, "alpha": a}, GOC / "mo_hinh_cuoi_hoi_quy.joblib")


# ---------------------------------------------------------------- [3] PHÂN CỤM

def tinh_chinh_phan_cum() -> None:
    p(f"\n[3] PHÂN CỤM — thử k = {SO_CUM_THU.start}..{SO_CUM_THU.stop - 1} trên 63 tỉnh"
      f" (ổn định: bootstrap {SO_LAN_BOOTSTRAP} lần)")
    hs = csdl.doc(TX, "ho_so_tinh")
    Z = hs[COT_PHAN_CUM].astype(float).to_numpy()
    rng = np.random.default_rng(42)
    mau = [rng.choice(len(Z), size=len(Z), replace=True) for _ in range(SO_LAN_BOOTSTRAP)]
    p(f"      {'k':>2s} {'SSE':>7s} {'silhouette':>10s} {'Davies–Bouldin':>14s} {'ARI tb':>7s} {'cụm nhỏ nhất':>12s}")
    bang = []
    for k in SO_CUM_THU:
        km = ung_vien_phan_cum(k).fit(Z)
        ari = np.mean([adjusted_rand_score(km.labels_, ung_vien_phan_cum(k).fit(Z[c]).predict(Z)) for c in mau])
        d = {"k": k, "SSE": km.inertia_, "silhouette": silhouette_score(Z, km.labels_),
             "davies_bouldin": davies_bouldin_score(Z, km.labels_), "ARI_tb": ari,
             "cum_nho_nhat": int(np.bincount(km.labels_).min())}
        bang.append(d)
        for c in ["SSE", "silhouette", "davies_bouldin", "ARI_tb"]:
            THU.append({"bai_toan": "Phân cụm", "huong": "", "mo_hinh": "K-Means", "tham_so": f"k={k}",
                        "chi_so": c, "gia_tri": float(d[c])})
        p(f"      {k:2d} {d['SSE']:7.1f} {d['silhouette']:10.3f} {d['davies_bouldin']:14.2f} {ari:7.2f}"
          f" {d['cum_nho_nhat']:12d}")
    b = pd.DataFrame(bang)
    on_dinh = b[b["ARI_tb"] >= 0.6]
    if len(on_dinh):
        k = int(on_dinh.loc[on_dinh["silhouette"].idxmax(), "k"])
        p(f"      -> CHỌN k = {k}: silhouette cao nhất trong các k có ARI ≥ 0,6")
    else:
        k = int(b.loc[b["ARI_tb"].idxmax(), "k"])
        p(f"      -> không k nào có ARI ≥ 0,6; CHỌN k = {k} (ổn định nhất). Cấu trúc cụm của 63 tỉnh yếu.")
    km = ung_vien_phan_cum(k).fit(Z)
    hs["cum"] = km.labels_ + 1
    tam = pd.DataFrame(km.cluster_centers_, columns=[c.removesuffix("_z") for c in COT_PHAN_CUM],
                       index=range(1, k + 1))
    for c, r in tam.iterrows():
        tinh = hs.loc[hs["cum"] == c, "tinh"].tolist()
        noi_bat = r.sort_values(key=abs, ascending=False).head(3)
        p(f"      Cụm {c}: {len(tinh)} tỉnh | " + ", ".join(f"{t} {v:+.1f}" for t, v in noi_bat.items())
          + f" | {', '.join(tinh[:6])}{'…' if len(tinh) > 6 else ''}")
    ra = hs[["tinh", "cum"]]
    ra.to_csv(GOC / "cum_tinh_cuoi.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(ra, "mo_hinh", "cum_tinh_cuoi")
    joblib.dump({"mo_hinh": km, "k": k, "cot": COT_PHAN_CUM}, GOC / "mo_hinh_cuoi_phan_cum.joblib")


# ---------------------------------------------------------------- [4] LUẬT KẾT HỢP

def tim_luat(I: pd.DataFrame, ms: float, mc: float) -> pd.DataFrame:
    tap = hl.apriori(I, ms, CAU_HINH_APRIORI["max_len"])
    luat = hl.sinh_luat(tap, CAU_HINH_APRIORI["ve_phai"], mc)
    return luat[luat["cai_thien"] >= 0.01]


def tinh_chinh_luat() -> None:
    ch = CAU_HINH_APRIORI
    p("\n[4] LUẬT KẾT HỢP — thử min_support × min_confidence; ổn định qua 5 phần ngẫu nhiên của toàn bộ giao_dich")
    hl.p = lambda *_: None  # apriori() in từng cỡ tập; chạy nhiều lần nên tắt
    gd = csdl.doc(TX, "giao_dich").astype({"dong_goc": "int64"}).sort_values(KHOA).reset_index(drop=True)
    I = gd.drop(columns=KHOA + ch["bo_item"]).astype(bool)
    phan = list(KFold(5, shuffle=True, random_state=42).split(I))
    p(f"      {'support':>7s} {'conf':>5s} {'số luật':>7s} {'ở cả 5 lần':>10s} {'Jaccard':>7s} {'lift phần kiểm':>14s}")
    bang = []
    for ms in LUOI_APRIORI["min_support"]:
        for mc in LUOI_APRIORI["min_confidence"]:
            luat = tim_luat(I, ms, mc)
            bo, lift_kd = [], []
            for hoc, kd in phan:
                l = tim_luat(I.iloc[hoc], ms, mc)
                bo.append(set(zip(l["ve_trai"], l["ve_phai"])))
                Ik = I.iloc[kd]
                for trai, phai in zip(l["ve_trai"], l["ve_phai"]):
                    a = Ik[trai.split(" + ")].all(axis=1)
                    lift_kd.append(((a & Ik[phai]).sum() / a.sum()) / Ik[phai].mean())
            chung, hop = set.intersection(*bo), set.union(*bo)
            d = {"min_support": ms, "min_confidence": mc, "so_luat": len(luat), "ca_5_lan": len(chung),
                 "jaccard": len(chung) / len(hop) if hop else 0.0, "lift_kd": float(np.mean(lift_kd)) if lift_kd else np.nan}
            bang.append(d)
            for c in ["so_luat", "jaccard", "lift_kd"]:
                THU.append({"bai_toan": "Luật kết hợp", "huong": "", "mo_hinh": "Apriori",
                            "tham_so": f"min_support={ms}, min_confidence={mc}", "chi_so": c, "gia_tri": float(d[c])})
            p(f"      {ms:7.2f} {mc:5.2f} {d['so_luat']:7d} {d['ca_5_lan']:10d} {d['jaccard']:7.2f} {d['lift_kd']:14.3f}")
    b = pd.DataFrame(bang)
    du = b[b["so_luat"] >= 10]
    r = du.sort_values(["jaccard", "so_luat"], ascending=False).iloc[0]
    ms, mc = float(r["min_support"]), float(r["min_confidence"])
    p(f"      -> CHỌN min_support = {ms:g}, min_confidence = {mc:g}: ổn định nhất trong các cấu hình cho ≥ 10 luật")
    luat = tim_luat(I, ms, mc).sort_values("lift", ascending=False)
    for phai in ch["ve_phai"]:
        p(f"      Luật cho '{phai}' (lift cao nhất):")
        for _, l in luat[luat["ve_phai"] == phai].head(4).iterrows():
            p(f"        {l['ve_trai']}  ->  {phai} | sup {l['support']:.1%}, conf {l['confidence']:.1%}, lift {l['lift']:.2f}")
    ra = luat[["ve_trai", "ve_phai", "support", "confidence", "lift", "cai_thien"]]
    ra.to_csv(GOC / "luat_cuoi.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(ra, "mo_hinh", "luat_cuoi")


def main() -> None:
    tinh_chinh_phan_loai()
    tinh_chinh_hoi_quy()
    tinh_chinh_phan_cum()
    tinh_chinh_luat()
    p("\n[5] Lưu")
    thu = pd.DataFrame(THU)
    test = pd.DataFrame(TEST)
    thu.to_csv(GOC / "tinh_chinh.csv", index=False, encoding="utf-8-sig")
    test.to_csv(GOC / "ket_qua_test.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(thu, "mo_hinh", "tinh_chinh")
    csdl.ghi(test, "mo_hinh", "ket_qua_test")
    p(f"    {len(thu)} dòng -> mo_hinh.tinh_chinh | {len(test)} dòng -> mo_hinh.ket_qua_test"
      " | mo_hinh.cum_tinh_cuoi, mo_hinh.luat_cuoi | mo_hinh_cuoi_*.joblib")
    (GOC / "kiem_tra_tinh_chinh.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu kiem_tra_tinh_chinh.txt")


if __name__ == "__main__":
    bat_dau = time.perf_counter()
    try:
        main()
        print(f"KẾT THÚC: XONG sau {time.perf_counter() - bat_dau:.1f} giây")
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"KẾT THÚC: LỖI sau {time.perf_counter() - bat_dau:.1f} giây")

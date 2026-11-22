"""BƯỚC 3 – XÂY DỰNG MÔ HÌNH · BƯỚC 2: CHỌN MÔ HÌNH — Chương 5: Modeling (Bước 2 – Chọn mô hình)

Khai báo MỘT LẦN các mô hình ứng viên của bốn bài toán (kèm một mô hình mốc để so sánh), để các bước sau
(huấn luyện, đánh giá, kiểm định chéo, tinh chỉnh) import dùng chung:
    from chon_mo_hinh import ung_vien_phan_loai, ung_vien_hoi_quy, ung_vien_phan_cum, CAU_HINH_APRIORI

Khi chạy trực tiếp, script KHÔNG chấm điểm mô hình (đó là bước 4). Nó kiểm tra dữ liệu có hợp với từng mô hình không:
    PHÂN LOẠI   : không có ô trống, toàn số, thang đo các cột lệch nhau (-> Logistic Regression cần chuẩn hoá),
                  hai lớp có cân bằng không; đo thời gian học của từng ứng viên và của SVM (lý do loại SVM)
    HỒI QUY     : phân bố số vụ mỗi ô; ma trận đặc trưng sau One-Hot có bị đa cộng tuyến không (-> thêm Ridge)
    PHÂN CỤM    : 7 chỉ số z-score có trung bình 0, độ lệch 1 không; cặp chỉ số trùng thông tin; tỉnh ngoại lai
    LUẬT KẾT HỢP: độ hỗ trợ của các item, item gần như luôn có, số tập ứng viên Apriori phải đếm

Đầu vào  : bảng tien_xu_ly.train_fe, tien_xu_ly.train_fe_khong_dia_ly, tien_xu_ly.tai_nan_loi_sach,
           tien_xu_ly.ho_so_tinh, tien_xu_ly.giao_dich; mo_hinh.chia_phan_loai, mo_hinh.chia_hoi_quy (bước 1)
Đầu ra   : bảng mo_hinh.ung_vien + ung_vien.csv, kiem_tra_chon_mo_hinh.txt (cũng in ra màn hình)

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/chon_mo_hinh.py
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
from hoi_quy import MOC_KIEM_TRA, PHAN_LOAI as COT_PL_HOI_QUY, SO as COT_SO_HOI_QUY, bo_ma_hoa, tao_bang_o  # noqa: E402

TX = "tien_xu_ly"
KHOA = ["nguon", "dong_goc"]
NHAN = "co_nguoi_chet"
HAT = 42
HUONG = {"du_bao": "train_fe", "khong_dia_ly": "train_fe_khong_dia_ly"}
BC: list[str] = ["BÁO CÁO CHỌN MÔ HÌNH", "=" * 70]

# ---------------------------------------------------------------- CÁC MÔ HÌNH ỨNG VIÊN (dùng chung cho các bước sau)

def ung_vien_phan_loai() -> dict[str, object]:
    """Ba mô hình của Chương 5 + một mốc. Tham số để mặc định; bước 6 mới tinh chỉnh."""
    return {
        "Mốc (đoán theo tỉ lệ)": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline([("chuan_hoa", StandardScaler()),
                                         ("mo_hinh", LogisticRegression(max_iter=2000, random_state=HAT))]),
        "Cây quyết định": DecisionTreeClassifier(random_state=HAT),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=HAT),
    }


def ung_vien_hoi_quy() -> dict[str, object]:
    """Đặc trưng của ô: vùng, khung giờ, thứ, tháng (One-Hot) + 4 cột số (chuẩn hoá) — như hoi_quy.py."""
    return {
        "Mốc (trung bình)": DummyRegressor(strategy="mean"),
        "Hồi quy tuyến tính": Pipeline([("ma_hoa", bo_ma_hoa(False)), ("mo_hinh", LinearRegression())]),
        "Ridge": Pipeline([("ma_hoa", bo_ma_hoa(False)), ("mo_hinh", Ridge(alpha=1.0))]),
    }


CHI_SO_Z = ["log_so_vu_z", "ti_le_co_chet_z", "chet_moi_vu_z", "ti_le_dem_z", "ti_le_cuoi_tuan_z",
            "ti_le_quoc_lo_z", "ti_le_dip_tet_z"]
COT_PHAN_CUM = [c for c in CHI_SO_Z if c != "chet_moi_vu_z"]   # trùng thông tin với ti_le_co_chet_z (r = 0,99)
SO_CUM_THU = range(2, 9)


def ung_vien_phan_cum(k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=HAT)


CAU_HINH_APRIORI = {
    "min_support": 0.05,            # item / tập item phải có ở ít nhất 5% số vụ (~1.650 vụ)
    "min_confidence": 0.5,
    "max_len": 3,                   # luật tối đa 3 item: dễ đọc, đủ để thấy "đêm + quốc lộ -> có chết"
    "bo_item": ["giai_doan_covid=Không", "dip_tet=Không"],    # có ở ~94% số vụ, sinh luật hiển nhiên
    "ve_phai": ["co_nguoi_chet=1", "co_nguoi_chet=0"],
    "cai_dat": "tự viết theo thuật toán Apriori (pandas/numpy), không cần thư viện mới",
}

UNG_VIEN = [  # bai_toan, mo_hinh, lop, tien_xu_ly_them, sieu_tham_so_se_chinh, ly_do
    ("Phân loại", "Mốc (đoán theo tỉ lệ)", "DummyClassifier(strategy='prior')", "—", "—",
     "Điểm sàn: mô hình nào không hơn mốc thì không học được gì"),
    ("Phân loại", "Logistic Regression", "LogisticRegression", "StandardScaler", "C",
     "Đơn giản, nhanh, hệ số đọc được; mốc tuyến tính"),
    ("Phân loại", "Cây quyết định", "DecisionTreeClassifier", "Không cần chuẩn hoá", "max_depth, min_samples_leaf",
     "Dễ giải thích: in ra được luật nếu–thì; xử lý quan hệ phi tuyến"),
    ("Phân loại", "Random Forest", "RandomForestClassifier", "Không cần chuẩn hoá",
     "max_depth, min_samples_leaf, max_features", "Nhiều cây bỏ phiếu: giảm quá khớp của một cây, thường mạnh nhất"),
    ("Hồi quy", "Mốc (trung bình)", "DummyRegressor(strategy='mean')", "—", "—", "Điểm sàn: luôn đoán số vụ trung bình"),
    ("Hồi quy", "Hồi quy tuyến tính", "LinearRegression", "One-Hot + StandardScaler", "—",
     "Mô hình hồi quy cơ bản của Chương 5, hệ số đọc được"),
    ("Hồi quy", "Ridge", "Ridge", "One-Hot + StandardScaler", "alpha",
     "Phạt L2 làm ổn định hệ số khi các cột One-Hot phụ thuộc tuyến tính"),
    ("Phân cụm", "K-Means", "KMeans(n_init=10)", "z-score (đã làm ở bước 2)", "k (2–8)",
     "Đơn giản, nhanh, dễ diễn giải tâm cụm; hợp với 63 điểm số liên tục. Dùng 6 chỉ số, bỏ chet_moi_vu (trùng)"),
    ("Luật kết hợp", "Apriori", "tự viết", "Dạng giao dịch (đã làm ở bước 2)", "min_support, min_confidence",
     "Thuật toán luật kết hợp kinh điển, Chương 5 nêu tên"),
]


# ---------------------------------------------------------------- KIỂM TRA DỮ LIỆU CHO TỪNG MÔ HÌNH

def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def bam_gio(mo_hinh, X, y) -> float:
    t = time.perf_counter()
    mo_hinh.fit(X, y)
    return time.perf_counter() - t


def kiem_tra_phan_loai() -> None:
    p("\n[1] PHÂN LOẠI — dữ liệu có hợp với Logistic Regression, Cây quyết định, Random Forest không?")
    chia = csdl.doc("mo_hinh", "chia_phan_loai")
    k_train = chia.loc[chia["tap"] == "train", KHOA]
    for huong, bang in HUONG.items():
        d = csdl.doc(TX, bang).merge(k_train.astype({"dong_goc": "int64"}), on=KHOA)
        X, y = d.drop(columns=KHOA + [NHAN]).astype(float), d[NHAN].astype(int)
        lech = X.std().replace(0, np.nan)
        p(f"\n    Hướng {huong} ({bang}): {len(X):,d} vụ × {X.shape[1]} cột")
        p(f"      ô trống: {int(X.isna().sum().sum())} | cột không phải số: 0 | khoảng giá trị: "
          f"{X.min().min():.2f} đến {X.max().max():.2f}")
        p(f"      độ lệch chuẩn các cột từ {lech.min():.2f} ({lech.idxmin()}) đến {lech.max():.2f} ({lech.idxmax()})"
          f" -> gấp {lech.max() / lech.min():.0f} lần: Logistic Regression cần StandardScaler, cây thì không")
        p(f"      có người chết: {y.mean():.1%} -> hai lớp gần cân bằng, không cần cân bằng lại; Accuracy đọc được")
        p("      thời gian học trên toàn bộ train (tham số mặc định):")
        for ten, mh in ung_vien_phan_loai().items():
            p(f"        {ten:24s}: {bam_gio(mh, X, y):6.2f} giây")
        if huong == "du_bao":
            p("      Vì sao không chọn SVM: đo thời gian học của SVC (nhân RBF) khi số vụ tăng gấp đôi")
            pipe = lambda: Pipeline([("chuan_hoa", StandardScaler()), ("svm", SVC(kernel="rbf"))])  # noqa: E731
            truoc = None
            for n in (2000, 4000, 8000):
                mau = X.sample(n, random_state=HAT)
                t = bam_gio(pipe(), mau, y.loc[mau.index])
                p(f"        {n:6,d} vụ: {t:6.2f} giây" + (f" (gấp {t / truoc:.1f} lần)" if truoc else ""))
                truoc = t
            uoc = t * (len(X) / 8000) ** 2
            p(f"        ước lượng cho {len(X):,d} vụ (thời gian tăng ít nhất theo bình phương số vụ): ~{uoc:.0f} giây"
              f" mỗi lần học; tinh chỉnh 5 phần × 12 bộ tham số = 60 lần -> ~{uoc:.0f} phút")
            p("        -> chạy được, nhưng SVM không cho luật đọc được như cây và không cho tầm quan trọng đặc trưng như"
              " Random Forest; ba mô hình đã đủ vai trò mốc / dễ giải thích / mạnh, nên không thêm SVM")


def kiem_tra_hoi_quy() -> None:
    p("\n[2] HỒI QUY — dữ liệu có hợp với hồi quy tuyến tính và Ridge không?")
    o, _ = tao_bang_o(csdl.doc(TX, "tai_nan_loi_sach"))
    hoc = o[o["ky"] < MOC_KIEM_TRA]
    y = hoc["so_vu"]
    p(f"    Ô học: {len(hoc):,d} | số vụ mỗi ô: trung bình {y.mean():.2f}, trung vị {y.median():.0f},"
      f" lớn nhất {y.max()}, {(y == 0).mean():.0%} ô bằng 0")
    p(f"    Phương sai / trung bình = {y.var() / y.mean():.1f} (dữ liệu đếm, lệch phải) -> mô hình tuyến tính có thể"
      " đoán số âm: cắt về 0 khi đánh giá")
    ma_hoa = bo_ma_hoa(False).fit(hoc[COT_PL_HOI_QUY + COT_SO_HOI_QUY])
    X = ma_hoa.transform(hoc[COT_PL_HOI_QUY + COT_SO_HOI_QUY])
    hang = np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X]))
    p(f"    Sau One-Hot + chuẩn hoá: {X.shape[1]} cột; ma trận (kèm hệ số chặn) có hạng {hang} < {X.shape[1] + 1}")
    p(f"      -> {X.shape[1] + 1 - hang} cột là tổ hợp tuyến tính của cột khác: mỗi nhóm One-Hot (vùng, khung giờ,"
      " thứ, tháng) cộng lại luôn bằng 1, trùng với hệ số chặn")
    p("      -> hồi quy tuyến tính thường không có nghiệm duy nhất; Ridge (phạt L2) cho hệ số ổn định")
    r = hoc[COT_SO_HOI_QUY].corr().abs().where(lambda m: ~np.eye(len(m), dtype=bool)).stack()
    a, b = r.idxmax()
    p(f"    Hai cột số tương quan mạnh nhất: {a} – {b}, |r| = {r.max():.2f}")
    for ten, mh in ung_vien_hoi_quy().items():
        p(f"    {ten:20s}: học {bam_gio(mh, hoc, y):.2f} giây")


def kiem_tra_phan_cum() -> None:
    p("\n[3] PHÂN CỤM — dữ liệu có hợp với K-Means không?")
    hs = csdl.doc(TX, "ho_so_tinh")
    Z = hs[CHI_SO_Z].astype(float)
    p(f"    {len(Z)} tỉnh × {Z.shape[1]} chỉ số z-score | trung bình {Z.mean().abs().max():.2f},"
      f" độ lệch chuẩn {Z.std().min():.2f}–{Z.std().max():.2f} -> cùng thang đo, khoảng cách Euclid công bằng")
    r = Z.corr().where(lambda m: ~np.eye(len(m), dtype=bool)).stack()
    cap = r.abs().sort_values(ascending=False)
    for (a, b), v in cap.iloc[:4:2].items():
        p(f"    Cặp chỉ số trùng thông tin: {a} – {b}, r = {r[(a, b)]:+.2f}"
          + (" -> hai cột này cùng kéo cụm về một hướng (tính hai lần)" if abs(r[(a, b)]) > 0.7 else ""))
    ngoai_lai = (Z.abs() > 3).any(axis=1)
    for i in hs.index[ngoai_lai]:
        cot = Z.columns[(Z.loc[i].abs() > 3)].tolist()
        p(f"    Tỉnh ngoại lai (|z| > 3): {hs.at[i, 'tinh']} ở {', '.join(cot)} -> K-Means nhạy với điểm xa, cần xem riêng")
    p(f"    Quyết định: phân cụm trên {len(COT_PHAN_CUM)} chỉ số, bỏ chet_moi_vu_z (trùng thông tin với ti_le_co_chet_z)")
    p("    Lưu ý: ti_le_co_chet phụ thuộc cách mỗi tỉnh báo cáo (22–98%); cụm nào tách chủ yếu theo chỉ số này phải nói rõ")
    p(f"    Số cụm sẽ thử: k = {SO_CUM_THU.start}–{SO_CUM_THU.stop - 1} (chọn ở bước 6 bằng SSE và silhouette)")
    t = bam_gio(ung_vien_phan_cum(3), Z[COT_PHAN_CUM], None)
    p(f"    K-Means k=3, n_init=10: {t:.2f} giây")


def kiem_tra_apriori() -> None:
    p("\n[4] LUẬT KẾT HỢP — dữ liệu có hợp với Apriori không?")
    gd = csdl.doc(TX, "giao_dich")
    I = gd.drop(columns=KHOA).astype(bool)
    ht = I.mean().sort_values(ascending=False)
    p(f"    {len(I):,d} giao dịch × {I.shape[1]} item (True/False), trung bình {I.sum(axis=1).mean():.1f} item mỗi vụ")
    for item in CAU_HINH_APRIORI["bo_item"]:
        p(f"    Bỏ '{item}': có ở {ht[item]:.1%} số vụ -> luật nào cũng chứa nó, không thêm thông tin")
    giu = ht.drop(CAU_HINH_APRIORI["bo_item"])
    pho_bien = giu[giu >= CAU_HINH_APRIORI["min_support"]]
    p(f"    Item có độ hỗ trợ ≥ {CAU_HINH_APRIORI['min_support']:.0%}: {len(pho_bien)}/{len(giu)}")
    cap = sum(1 for a, b in combinations(pho_bien.index, 2) if (I[a] & I[b]).mean() >= CAU_HINH_APRIORI["min_support"])
    p(f"    Không có Apriori phải đếm mọi tổ hợp ≤ 3 trong {len(giu)} item: {comb(len(giu), 2) + comb(len(giu), 3):,d} tập")
    p(f"    Có Apriori: chỉ ghép từ {len(pho_bien)} item phổ biến -> {comb(len(pho_bien), 2):,d} cặp ứng viên,"
      f" {cap} cặp đạt ngưỡng (tính chất Apriori: tập con không phổ biến thì tập lớn hơn cũng không)")
    p(f"    Cài đặt: {CAU_HINH_APRIORI['cai_dat']}")


def main() -> None:
    p("[0] Mô hình ứng viên")
    for bt, mh, lop, *_ in UNG_VIEN:
        p(f"    {bt:13s} {mh:24s} {lop}")
    kiem_tra_phan_loai()
    kiem_tra_hoi_quy()
    kiem_tra_phan_cum()
    kiem_tra_apriori()

    p("\n[5] Lưu")
    bang = pd.DataFrame(UNG_VIEN, columns=["bai_toan", "mo_hinh", "lop", "tien_xu_ly_them", "sieu_tham_so_se_chinh", "ly_do"])
    bang.to_csv(GOC / "ung_vien.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(bang, "mo_hinh", "ung_vien")
    p(f"    {len(bang)} mô hình -> bảng mo_hinh.ung_vien + ung_vien.csv")
    (GOC / "kiem_tra_chon_mo_hinh.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu kiem_tra_chon_mo_hinh.txt")


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

"""PHÂN LOẠI — Dự đoán vụ tai nạn có người chết (co_nguoi_chet) — Chương 5: Modeling

Đầu vào  : bảng tien_xu_ly.tai_nan_loi_sach, tien_xu_ly.train/test (khoá chia),
           pipeline_tien_xu_ly_du_bao.joblib, pipeline_tien_xu_ly_khong_dia_ly.joblib
Đầu ra   : mo_hinh_phan_loai_du_bao.joblib, mo_hinh_phan_loai_khong_dia_ly.joblib (pipeline tiền xử lý + mô hình tốt nhất)
           ket_qua_phan_loai.csv, kiem_tra_phan_loai.txt (cũng in ra màn hình)

Quy trình 6 bước của Chương 5, làm cho từng hướng đặc trưng (du_bao, khong_dia_ly):
    1. Chia dữ liệu      : dùng lại đúng cách chia 80/20 phân tầng của bien_doi.py
    2. Chọn mô hình      : Logistic Regression, Decision Tree, Random Forest, SVM (+ mốc so sánh Dummy)
    3. Huấn luyện        : mỗi mô hình là MỘT pipeline = tiền xử lý (học lại trong từng lần) + bộ phân lớp
    5. Cross-validation  : 5 lần trên train, tham số mặc định -> so sánh ban đầu
    6. Tinh chỉnh        : GridSearchCV (3 lần chia) trên train, chấm bằng ROC-AUC
    4. Đánh giá          : MỘT lần trên test, sau khi đã chọn xong tham số — accuracy, precision, recall, F1, AUC,
                           ma trận nhầm lẫn. Mô hình tốt nhất được chọn theo AUC kiểm định chéo, KHÔNG theo test.

SVM (nhân RBF) có độ phức tạp tăng theo bình phương số dòng, nên kiểm định chéo và tinh chỉnh SVM chạy trên
một mẫu 6.000 vụ phân tầng của train; mô hình SVM cuối vẫn học trên toàn bộ train.

Chạy (từ D:/PDM, sau các script của 2_tien_xu_ly; mất khoảng 10–15 phút):
       .venv/Scripts/python bai_tap_lon/3_mo_hinh/phan_loai.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
TX = GOC.parent / "2_tien_xu_ly"
sys.path.insert(0, str(TX))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

import ky_nghe_dac_trung  # noqa: E402,F401  (để joblib nạp được các lớp tự viết trong pipeline)

BC: list[str] = ["BÁO CÁO PHÂN LOẠI", "=" * 70]
KHOA, NHAN, HAT = ["nguon", "dong_goc"], "co_nguoi_chet", 42
HUONG = ("du_bao", "khong_dia_ly")
MAU_SVM = 6000
CHI_SO = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "AUC"}

# (tên, bộ phân lớp, cần chuẩn hoá?, lưới tham số, chạy trên mẫu?)
MO_HINH = [
    ("Dummy", DummyClassifier(strategy="most_frequent"), False, None, False),
    ("Logistic Regression", LogisticRegression(max_iter=3000), True, {"mo_hinh__C": [0.01, 0.1, 1, 10]}, False),
    ("Decision Tree", DecisionTreeClassifier(random_state=HAT), False,
     {"mo_hinh__max_depth": [4, 6, 8, 12], "mo_hinh__min_samples_leaf": [20, 100]}, False),
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=HAT, n_jobs=-1), False,
     {"mo_hinh__max_depth": [10, 20, None], "mo_hinh__min_samples_leaf": [5, 20]}, False),
    ("SVM (RBF)", SVC(kernel="rbf", random_state=HAT), True,
     {"mo_hinh__C": [0.5, 2], "mo_hinh__gamma": ["scale", 0.02]}, True),
]


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def doc(thu_muc: Path, ten: str) -> pd.DataFrame:
    return csdl.doc(thu_muc, ten)               # thư mục 2_tien_xu_ly -> schema tien_xu_ly


def tach(df: pd.DataFrame, khoa: pd.DataFrame) -> pd.DataFrame:
    k = set(khoa["nguon"].astype(str) + "#" + khoa["dong_goc"].astype(str))
    return df[(df["nguon"].astype(str) + "#" + df["dong_goc"].astype(str)).isin(k)].reset_index(drop=True)


def dung_pipeline(tien_xu_ly: Pipeline, mo_hinh, chuan_hoa: bool) -> Pipeline:
    """Bước ma hoá (chưa học) + [StandardScaler] + bộ phân lớp. Bước tạo đặc trưng theo dòng đã chạy trước
    một lần vì nó không học gì từ dữ liệu, nên kết quả kiểm định chéo không đổi mà chạy nhanh hơn nhiều."""
    buoc = [("chon_ma_hoa", clone(tien_xu_ly.named_steps["chon_ma_hoa"]))]
    if chuan_hoa:
        buoc.append(("chuan_hoa", StandardScaler()))
    buoc.append(("mo_hinh", clone(mo_hinh)))
    return Pipeline(buoc)


def diem(mh, X, y) -> np.ndarray:
    return mh.predict_proba(X)[:, 1] if hasattr(mh, "predict_proba") else mh.decision_function(X)


def mau_phan_tang(X: pd.DataFrame, y: np.ndarray, n: int):
    if len(X) <= n:
        return X, y
    X2, _, y2, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=HAT)
    return X2.reset_index(drop=True), y2


def ten_dac_trung(pl: Pipeline) -> list[str]:
    return list(pl.named_steps["chon_ma_hoa"].get_feature_names_out())


def mot_huong(huong: str, train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    tien = joblib.load(TX / f"pipeline_tien_xu_ly_{huong}.joblib")
    tao = tien.named_steps["dac_trung_dong"]
    X_tr, X_te = tao.transform(train), tao.transform(test)
    y_tr = pd.to_numeric(train[NHAN]).astype(int).to_numpy()
    y_te = pd.to_numeric(test[NHAN]).astype(int).to_numpy()
    p(f"\n{'#' * 70}\nHƯỚNG {huong.upper()} — {len(ten_dac_trung(tien))} cột đặc trưng sau tiền xử lý")

    p("\n[1] Chia dữ liệu: dùng lại cách chia của bien_doi.py")
    p(f"    train {len(X_tr):,d} vụ ({y_tr.mean():.1%} có người chết) | test {len(X_te):,d} vụ ({y_te.mean():.1%})")

    p("\n[2] Chọn mô hình")
    p("    Dummy (luôn đoán lớp nhiều nhất): mốc tối thiểu, mô hình nào không vượt mốc này là vô dụng")
    p("    Logistic Regression: tuyến tính, dễ giải thích (hệ số) — cần chuẩn hoá")
    p("    Decision Tree: luật if-else, xử lý phi tuyến, dễ quá khớp")
    p("    Random Forest: nhiều cây bỏ phiếu, giảm quá khớp, khó giải thích hơn")
    p(f"    SVM nhân RBF: tìm biên phân tách phi tuyến — cần chuẩn hoá, chậm với dữ liệu lớn (mẫu {MAU_SVM:,d} vụ)")

    p("\n[3]+[5] Huấn luyện và kiểm định chéo 5 lần trên train (tham số mặc định)")
    cv5 = StratifiedKFold(5, shuffle=True, random_state=HAT)
    p(f"    {'mô hình':22s} " + " ".join(f"{v:>9s}" for v in CHI_SO.values()) + "   thời gian")
    for ten, mh, chuan, _, mau in MO_HINH:
        X, y = mau_phan_tang(X_tr, y_tr, MAU_SVM) if mau else (X_tr, y_tr)
        t0 = time.perf_counter()
        kq = cross_validate(dung_pipeline(tien, mh, chuan), X, y, cv=cv5, scoring=list(CHI_SO), n_jobs=1 if mau else None)
        p(f"    {ten:22s} " + " ".join(f"{kq['test_' + k].mean():9.3f}" for k in CHI_SO)
          + f"   {time.perf_counter() - t0:5.0f} giây" + ("  (mẫu)" if mau else ""))

    p("\n[6] Tinh chỉnh siêu tham số: GridSearchCV, 3 lần chia, chấm theo ROC-AUC, chỉ trên train")
    cv3 = StratifiedKFold(3, shuffle=True, random_state=HAT)
    tot: dict[str, tuple] = {}
    for ten, mh, chuan, luoi, mau in MO_HINH:
        if luoi is None:
            continue
        X, y = mau_phan_tang(X_tr, y_tr, MAU_SVM) if mau else (X_tr, y_tr)
        t0 = time.perf_counter()
        gs = GridSearchCV(dung_pipeline(tien, mh, chuan), luoi, cv=cv3, scoring="roc_auc", n_jobs=1 if ten.startswith("Random") else -1)
        gs.fit(X, y)
        tham_so = {k.replace("mo_hinh__", ""): v for k, v in gs.best_params_.items()}
        so_to_hop = int(np.prod([len(v) for v in luoi.values()]))
        p(f"    {ten:22s}: {so_to_hop} tổ hợp -> tốt nhất {tham_so} | AUC kiểm định {gs.best_score_:.4f}"
          f" | {time.perf_counter() - t0:.0f} giây" + ("  (mẫu)" if mau else ""))
        tot[ten] = (gs.best_params_, gs.best_score_, chuan, mh)

    p("\n[4] Đánh giá MỘT lần trên test (mô hình đã tinh chỉnh, học lại trên toàn bộ train)")
    p(f"    {'mô hình':22s} " + " ".join(f"{v:>9s}" for v in CHI_SO.values()))
    dong = []
    da_hoc = {}
    for ten, mh, chuan, _, _ in MO_HINH:
        pl = dung_pipeline(tien, mh, chuan)
        if ten in tot:
            pl.set_params(**tot[ten][0])
        pl.fit(X_tr, y_tr)
        du = pl.predict(X_te)
        s = diem(pl, X_te, y_te)
        kq = {"huong": huong, "mo_hinh": ten, "accuracy": accuracy_score(y_te, du),
              "precision": precision_score(y_te, du, zero_division=0), "recall": recall_score(y_te, du),
              "f1": f1_score(y_te, du), "roc_auc": roc_auc_score(y_te, s),
              "auc_kiem_dinh": tot[ten][1] if ten in tot else np.nan}
        dong.append(kq)
        da_hoc[ten] = pl
        p(f"    {ten:22s} " + " ".join(f"{kq[k]:9.3f}" for k in CHI_SO))
    chon = max(tot, key=lambda t: tot[t][1])
    p(f"    -> Mô hình được chọn (AUC kiểm định cao nhất, không dựa vào test): {chon}")
    best = da_hoc[chon]
    du = best.predict(X_te)
    (tn, fp), (fn, tp) = confusion_matrix(y_te, du)
    p("    Ma trận nhầm lẫn của mô hình được chọn (ngưỡng 0,5):")
    p(f"                       đoán KHÔNG chết   đoán CÓ chết")
    p(f"      thật KHÔNG chết   {tn:15,d}   {fp:12,d}")
    p(f"      thật CÓ chết      {fn:15,d}   {tp:12,d}")
    p(f"    Bỏ sót {fn:,d}/{fn + tp:,d} vụ có người chết ({fn / (fn + tp):.0%}); báo nhầm {fp:,d}/{tn + fp:,d} vụ không chết")

    p("\n[7] Diễn giải mô hình")
    ten_cot = ten_dac_trung(best)
    if hasattr(best.named_steps["mo_hinh"], "feature_importances_"):
        qt = pd.Series(best.named_steps["mo_hinh"].feature_importances_, index=ten_cot).sort_values(ascending=False)
        p(f"    {chon}: 10 đặc trưng quan trọng nhất (mức giảm tạp chất Gini):")
        for c, v in qt.head(10).items():
            p(f"      {c:34s} {v:.3f}")
    lr = da_hoc["Logistic Regression"]
    he_so = pd.Series(lr.named_steps["mo_hinh"].coef_[0], index=ten_dac_trung(lr)).sort_values()
    p("    Logistic Regression (đặc trưng đã chuẩn hoá): hệ số dương = tăng khả năng có người chết")
    for c, v in pd.concat([he_so.tail(5)[::-1], he_so.head(5)]).items():
        p(f"      {c:34s} {v:+.3f}")

    joblib.dump(Pipeline([("dac_trung_dong", tao), *best.steps]), GOC / f"mo_hinh_phan_loai_{huong}.joblib")
    p(f"    Lưu mo_hinh_phan_loai_{huong}.joblib = tiền xử lý + {chon} (dùng: .predict_proba(dòng bảng lõi sạch))")
    return dong


def main() -> None:
    p(f"[0] Đọc dữ liệu | scikit-learn {sklearn.__version__}")
    df = doc(TX, "tai_nan_loi_sach")
    train, test = tach(df, doc(TX, "train")[KHOA]), tach(df, doc(TX, "test")[KHOA])
    p(f"    bảng lõi {len(df):,d} vụ -> train {len(train):,d}, test {len(test):,d}")
    p("    Lưu ý: test chỉ được dùng ở bước [4], sau khi đã chọn xong mô hình và tham số")
    dong = []
    for huong in HUONG:
        dong += mot_huong(huong, train, test)
    bang = pd.DataFrame(dong)
    bang.to_csv(GOC / "ket_qua_phan_loai.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(bang, "mo_hinh", "ket_qua_phan_loai")
    p("\n[8] Tóm tắt AUC trên test")
    p(bang.pivot(index="mo_hinh", columns="huong", values="roc_auc").round(3).to_string())
    (GOC / "kiem_tra_phan_loai.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu ket_qua_phan_loai.csv (+ bảng mo_hinh.ket_qua_phan_loai), kiem_tra_phan_loai.txt, "
          "mo_hinh_phan_loai_*.joblib")


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

"""PIPELINE TIỀN XỬ LÝ — Tai nạn giao thông đường bộ Việt Nam (2020 – 8 tháng 2022)

Gói phần BIẾN ĐỔI + KỸ NGHỆ ĐẶC TRƯNG vào MỘT đối tượng scikit-learn Pipeline (KHÔNG có mô hình).
Bước mô hình sau này chỉ cần gắn thêm bộ phân lớp vào cuối:  Pipeline([*pipeline.steps, ("mo_hinh", ...)])

    dòng bảng lõi sạch ──► [1] dac_trung_dong   ──► [2] chon_ma_hoa          ──► ma trận đặc trưng
                            TaoDacTrungDong         ColumnTransformer
                            (sin/cos, cờ văn bản,   (One-Hot, Target Encoding,
                             dịp lễ, tương tác,      tần suất — chỉ các nhóm
                             khoá địa lý)            đã chọn ở kỹ nghệ đặc trưng)

Giai đoạn: 2 – Tiền xử lý (bước cuối, chạy sau ky_nghe_dac_trung.py trong cùng thư mục)
Đầu vào  : tai_nan_loi_sach.parquet (tien_xu_ly.py), train/test.parquet (khoá chia, bien_doi.py),
           bo_dac_trung.joblib + test_fe*.parquet (ky_nghe_dac_trung.py)
Đầu ra   : pipeline_tien_xu_ly_du_bao.joblib, pipeline_tien_xu_ly_khong_dia_ly.joblib,
           kiem_tra_pipeline.txt (cũng in ra màn hình)

Các thao tác (cho mỗi hướng du_bao, khong_dia_ly):
    1. Dựng pipeline 2 bước theo nhóm đặc trưng đã chọn
    2. Học (fit) trên dữ liệu train ở dạng bảng lõi — mọi tham số chỉ học từ train
    3. Kiểm tra khớp: ma trận test do pipeline tạo phải trùng đúng test_fe của ky_nghe_dac_trung.py
    4. Lưu, nạp lại từ đĩa, biến đổi thử một bản ghi MỚI không có nhãn

Chạy (từ D:/PDM, sau ky_nghe_dac_trung.py):
       .venv/Scripts/python bai_tap_lon/2_tien_xu_ly/pipeline.py
Nạp lại ở script khác: thêm thư mục 2_tien_xu_ly vào sys.path rồi joblib.load(...).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from ky_nghe_dac_trung import KHOA, NHAN, VAO, TaoDacTrungDong, doc_bang, tao_bo_bien_doi  # noqa: E402

BC: list[str] = ["BÁO CÁO PIPELINE TIỀN XỬ LÝ", "=" * 70]
HUONG = ("du_bao", "khong_dia_ly")


def p(dong: str = "") -> None:
    print(dong)
    BC.append(dong)


def tao_pipeline(bo: str, nhom: dict[str, list[str]]) -> Pipeline:
    return Pipeline([("dac_trung_dong", TaoDacTrungDong()), ("chon_ma_hoa", tao_bo_bien_doi(bo, nhom))])


def tach(df: pd.DataFrame, khoa: pd.DataFrame) -> pd.DataFrame:
    k = set(khoa["nguon"].astype(str) + "#" + khoa["dong_goc"].astype(str))
    ma = df["nguon"].astype(str) + "#" + df["dong_goc"].astype(str)
    return df[ma.isin(k)].reset_index(drop=True)


def ban_ghi_moi() -> pd.DataFrame:
    """Một vụ giả định ở dạng bảng lõi sạch, KHÔNG có nhãn — để chứng minh pipeline dùng được cho dữ liệu mới."""
    return pd.DataFrame([{
        "ngay": pd.Timestamp("2022-09-02"), "gio": "21:30:00", "gio_so": 21, "khung_gio": "19-22",
        "thu": 5, "cuoi_tuan": 0, "thang": 9, "vung": "Bắc", "mua": "Mùa mưa", "dip_tet": "Không",
        "giai_doan_covid": "Không", "tinh": "Hà Nội", "huyen": "Sóc Sơn", "xa": "Phù Linh",
        "loai_duong": "Quốc lộ - Nông thôn", "cap_duong": "Quốc lộ", "khu_vuc": "Nông thôn",
        "ten_duong": "Quốc lộ 3", "km": "Km 25+300", "dia_diem": "Ngã tư Phù Lỗ, Phù Linh, Sóc Sơn, Hà Nội",
    }])


def main() -> None:
    p(f"[0] Đọc dữ liệu | scikit-learn {sklearn.__version__}")
    df = doc_bang(VAO, "tai_nan_loi_sach")
    train = tach(df, doc_bang(GOC, "train")[KHOA])
    test = tach(df, doc_bang(GOC, "test")[KHOA])
    y_train = pd.to_numeric(train[NHAN]).astype(int).to_numpy()
    cau_hinh = joblib.load(GOC / "bo_dac_trung.joblib")["huong"]
    p(f"    bảng lõi {len(df):,d} vụ -> train {len(train):,d}, test {len(test):,d} (cùng cách chia của bien_doi.py)")
    p("    Pipeline chỉ gồm tiền xử lý; KHÔNG huấn luyện mô hình, KHÔNG đánh giá trên test ở bước này")

    for so, huong in enumerate(HUONG):
        ch = cau_hinh[huong]
        hau_to = "" if huong == "du_bao" else f"_{huong}"
        p(f"\n[{so * 3 + 1}] Hướng {huong}: dựng pipeline 2 bước")
        pl = tao_pipeline(ch["bo"], ch["nhom"])
        for ten, buoc in pl.steps:
            p(f"    {ten:15s}: {type(buoc).__name__}")
        p(f"    Nhóm đặc trưng đầu vào ({len(ch['nhom'])}): {', '.join(ch['nhom'])}")

        p(f"[{so * 3 + 2}] Học (fit) trên {len(train):,d} dòng train, kiểm tra khớp với ky_nghe_dac_trung.py")
        t0 = time.perf_counter()
        pl.fit(train, y_train)
        ten_cot = list(pl.named_steps["chon_ma_hoa"].get_feature_names_out())
        p(f"    xong sau {time.perf_counter() - t0:.1f} giây | đầu ra {len(ten_cot)} cột")
        X_pl = pl.transform(test)
        X_fe = doc_bang(GOC, f"test_fe{hau_to}")[ch["ten_cot"]].astype(float).to_numpy()
        khop = X_pl.shape == X_fe.shape and np.allclose(X_pl, X_fe)
        p(f"    ma trận test từ pipeline {X_pl.shape} vs test_fe{hau_to} {X_fe.shape}: "
          + ("TRÙNG KHỚP" if khop else f"LỆCH (sai khác lớn nhất {np.abs(X_pl - X_fe).max():.4g})"))
        if not khop:
            raise ValueError("Pipeline không tái tạo được đặc trưng của bước kỹ nghệ đặc trưng")

        p(f"[{so * 3 + 3}] Lưu, nạp lại, biến đổi thử một bản ghi mới (không có nhãn)")
        tep = GOC / f"pipeline_tien_xu_ly_{huong}.joblib"
        joblib.dump(pl, tep)
        nap = joblib.load(tep)
        if not np.allclose(nap.transform(test.head(50)), X_pl[:50]):
            raise ValueError("Pipeline nạp lại cho kết quả khác pipeline gốc")
        moi = pd.Series(nap.transform(ban_ghi_moi())[0], index=ten_cot)
        p(f"    Lưu {tep.name} ({tep.stat().st_size / 1024:,.0f} KB); nạp lại cho kết quả trùng bản gốc")
        p("    Bản ghi mới: 21h30 02/09/2022, Quốc lộ 3, Km 25+300, Ngã tư Phù Lỗ, Sóc Sơn, Hà Nội ->")
        for c, v in moi[moi != 0].items():
            p(f"      {c:32s} = {v:.4g}")
        p(f"      ({int((moi == 0).sum())} cột còn lại = 0)")

    (GOC / "kiem_tra_pipeline.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu pipeline_tien_xu_ly_du_bao.joblib, pipeline_tien_xu_ly_khong_dia_ly.joblib, kiem_tra_pipeline.txt")


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

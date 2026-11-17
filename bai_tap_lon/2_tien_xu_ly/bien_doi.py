"""BIẾN ĐỔI DỮ LIỆU — Tai nạn giao thông đường bộ Việt Nam (2020 – 8 tháng 2022)

Giai đoạn: 2 – Tiền xử lý (phần biến đổi, chạy sau tien_xu_ly.py trong cùng thư mục)
Đầu vào  : bảng tien_xu_ly.tai_nan_loi_sach (PostgreSQL) do tien_xu_ly.py tạo — chỉ dùng BẢNG LÕI (cả 3 năm).
Đầu ra   : train / test      ma trận đặc trưng đã mã hoá + nhãn co_nguoi_chet (phân lớp)
           bo_ma_hoa.joblib  bộ mã hoá đã học trên tập train (bước mô hình dùng lại)
           giao_dich         dạng giao dịch True/False (luật kết hợp)
           ho_so_tinh        hồ sơ 63 tỉnh, bản gốc + bản chuẩn hoá z-score (phân cụm)
           kiem_tra_bien_doi.txt (cũng in ra màn hình)

Các thao tác:
    1. Chọn đặc trưng đầu vào; loại cột rò rỉ nhãn, cột văn bản, cột cờ
    2. Biến chu kỳ (giờ, tháng, thứ) -> cặp sin / cos       (không cần học, không rò rỉ)
    3. Chia train / test (phân tầng theo nhãn) TRƯỚC khi học bất cứ gì
    4. Mã hoá One-Hot các biến phân loại: HỌC trên train, áp dụng cho test
    5. Dạng giao dịch cho luật kết hợp
    6. Hồ sơ tỉnh + chuẩn hoá z-score (StandardScaler) cho phân cụm

Chạy (từ D:/PDM, sau tien_xu_ly.py; cần: pip install scikit-learn):
       .venv/Scripts/python bai_tap_lon/2_tien_xu_ly/bien_doi.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
BC: list[str] = ["BÁO CÁO BIẾN ĐỔI DỮ LIỆU", "=" * 70]

KHOA = ["nguon", "dong_goc"]
NHAN = "co_nguoi_chet"
KHONG_RO = "Không rõ"
CHIA = "ngau_nhien"        # "ngau_nhien": 80/20 phân tầng theo nhãn | "thoi_gian": train 2020–2021, test 8T2022
TI_LE_TEST, HAT_GIONG = 0.2, 42

# Đặc trưng đầu vào (chỉ bảng lõi). Biến danh nghĩa -> One-Hot; không biến nào có thứ tự thật ở bảng lõi.
DANH_NGHIA = ["khung_gio", "vung", "mua", "dip_tet", "giai_doan_covid", "cap_duong", "khu_vuc", "tinh"]
CHU_KY = {"gio_so": 24, "thang": 12, "thu": 7}      # cột -> độ dài chu kỳ
NHI_PHAN = ["cuoi_tuan"]
KHONG_DUNG = {   # lý do không đưa vào mô hình
    "so_chet, so_bi_thuong": "rò rỉ nhãn (nhãn được tính từ so_chet)",
    "huyen, xa, ten_duong, km, dia_diem": "văn bản / quá chi tiết (hàng nghìn giá trị)",
    "ngay, gio, loai_duong": "đã có đặc trưng suy ra thay thế",
    "co_*, gio_da_sua, thang_khong_tron, tinh_ghi_ca_vu_nhe": "cờ kiểm tra, chỉ dùng để lọc",
}


def p(dong: str) -> None:
    print(dong)
    BC.append(dong)


def doc_bang(ten: str) -> pd.DataFrame:
    return csdl.doc("tien_xu_ly", ten)


def luu_bang(df: pd.DataFrame, ten: str) -> None:
    df.to_csv(GOC / f"{ten}.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(df, "tien_xu_ly", ten)


# ---------------------------------------------------------------------------
def chuan_bi(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 1–2: chọn cột, điền 'Không rõ' cho biến phân loại, biến chu kỳ -> sin/cos.
    Không có tham số nào phải học từ dữ liệu, nên làm trên toàn bộ trước khi chia vẫn không rò rỉ."""
    x = df[KHOA].copy()
    for c in DANH_NGHIA:
        x[c] = df[c].astype("string").fillna(KHONG_RO)
    for c, chu_ky in CHU_KY.items():
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        if c in ("thang", "thu"):
            v = v - 1                                   # tháng 1–12, thứ 1–7 -> bắt đầu từ 0
        goc = 2 * np.pi * v / chu_ky
        x[f"{c}_sin"] = np.sin(goc).fillna(0.0)
        x[f"{c}_cos"] = np.cos(goc).fillna(0.0)
    x["gio_thieu"] = pd.to_numeric(df["gio_so"], errors="coerce").isna().astype(int)
    for c in NHI_PHAN:
        x[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    x[NHAN] = pd.to_numeric(df[NHAN], errors="coerce").astype(int)
    return x


def chia_train_test(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if CHIA == "thoi_gian":
        train, test = x[x["nguon"] != "8T2022"], x[x["nguon"] == "8T2022"]
        mo_ta = "theo thời gian: train = 2020, 2021; test = 8 tháng 2022"
    else:
        train, test = train_test_split(x, test_size=TI_LE_TEST, stratify=x[NHAN], random_state=HAT_GIONG)
        mo_ta = f"ngẫu nhiên {1 - TI_LE_TEST:.0%}/{TI_LE_TEST:.0%}, phân tầng theo {NHAN}, random_state={HAT_GIONG}"
    return train.reset_index(drop=True), test.reset_index(drop=True), mo_ta


def tao_bo_ma_hoa() -> ColumnTransformer:
    so = [f"{c}_{h}" for c in CHU_KY for h in ("sin", "cos")] + ["gio_thieu", *NHI_PHAN]
    return ColumnTransformer(
        [("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8), DANH_NGHIA),
         ("giu_nguyen", "passthrough", so)],
        verbose_feature_names_out=False)


def ho_so_tinh(df: pd.DataFrame) -> pd.DataFrame:
    """Mỗi tỉnh một dòng: các tỉ lệ mô tả đặc điểm tai nạn, rồi chuẩn hoá z-score cho phân cụm."""
    d = df.assign(
        dem=pd.to_numeric(df["gio_so"], errors="coerce").pipe(lambda g: ((g >= 18) | (g < 6)).astype(float).where(g.notna())),
        quoc_lo=(df["cap_duong"] == "Quốc lộ").astype(float).where(df["cap_duong"].fillna(KHONG_RO) != KHONG_RO),
        tet=(df["dip_tet"] != "Không"),
        co_chet=pd.to_numeric(df[NHAN]),
        chet=pd.to_numeric(df["so_chet"]))
    g = d.dropna(subset=["tinh"]).groupby("tinh")
    hs = pd.DataFrame({
        "so_vu": g.size(),
        "ti_le_co_chet": g["co_chet"].mean(),
        "chet_moi_vu": g["chet"].mean(),
        "ti_le_dem": g["dem"].mean(),
        "ti_le_cuoi_tuan": g["cuoi_tuan"].apply(lambda s: pd.to_numeric(s).mean()),
        "ti_le_quoc_lo": g["quoc_lo"].mean(),
        "ti_le_dip_tet": g["tet"].mean(),
    }).astype(float)
    hs["log_so_vu"] = np.log1p(hs["so_vu"])
    cot = ["log_so_vu", "ti_le_co_chet", "chet_moi_vu", "ti_le_dem", "ti_le_cuoi_tuan", "ti_le_quoc_lo", "ti_le_dip_tet"]
    chuan = StandardScaler().fit_transform(hs[cot].fillna(hs[cot].mean()))
    for i, c in enumerate(cot):
        hs[f"{c}_z"] = chuan[:, i]
    return hs.reset_index(), cot


# ---------------------------------------------------------------------------
def main() -> None:
    p(f"[0] Đọc bảng tien_xu_ly.tai_nan_loi_sach trong {csdl.mo_ta()}")
    df = doc_bang("tai_nan_loi_sach")
    p(f"    {len(df):,d} vụ × {df.shape[1]} cột | scikit-learn {sklearn.__version__}, pandas {pd.__version__}")

    p("\n[1] Chọn đặc trưng đầu vào (chỉ bảng lõi)")
    p(f"    Danh nghĩa -> One-Hot ({len(DANH_NGHIA)}): {', '.join(DANH_NGHIA)}")
    p(f"    Chu kỳ -> sin/cos ({len(CHU_KY)}): {', '.join(f'{c} (chu kỳ {k})' for c, k in CHU_KY.items())}")
    p(f"    Nhị phân giữ nguyên: {', '.join(NHI_PHAN)} | nhãn: {NHAN}")
    for cot, ly_do in KHONG_DUNG.items():
        p(f"    Không dùng {cot}: {ly_do}")

    p("\n[2] Chuẩn bị: điền 'Không rõ' cho biến phân loại, biến chu kỳ -> sin/cos")
    x = chuan_bi(df)
    for c in DANH_NGHIA:
        n = int((x[c] == KHONG_RO).sum())
        if n:
            p(f"    {c}: {n:,d} ô '{KHONG_RO}' -> thành một cột one-hot riêng")
    p(f"    gio_so trống {int(x['gio_thieu'].sum())} vụ -> sin = cos = 0, cờ gio_thieu = 1")
    for h in (0, 6, 12, 23):
        a = 2 * np.pi * h / 24
        p(f"    vd giờ {h:2d}: sin = {np.sin(a):+.3f}, cos = {np.cos(a):+.3f}")
    p("    -> 23 giờ (+0,966 cos) nằm sát 0 giờ (+1,000 cos), đúng thực tế; mã 0..23 thì cách nhau 23 đơn vị")

    p("\n[3] Chia train / test — làm TRƯỚC khi học bộ mã hoá")
    train, test, mo_ta = chia_train_test(x)
    p(f"    Cách chia: {mo_ta}")
    for ten, t in (("train", train), ("test", test)):
        p(f"    {ten:5s}: {len(t):6,d} vụ | có người chết {t[NHAN].mean():.1%}")

    p("\n[4] Mã hoá One-Hot: học (fit) trên train, áp dụng (transform) cho cả train và test")
    bo = tao_bo_ma_hoa()
    X_train = bo.fit_transform(train)
    X_test = bo.transform(test)
    ten_cot = list(bo.get_feature_names_out())
    oh = bo.named_transformers_["one_hot"]
    for c, loai in zip(DANH_NGHIA, oh.categories_):
        p(f"    {c:16s}: {len(loai):2d} giá trị -> {len(loai):2d} cột" + (f" | vd {', '.join(map(str, loai[:4]))}…" if len(loai) > 4 else f" | {', '.join(map(str, loai))}"))
    p(f"    Tổng: {len(ten_cot)} cột đặc trưng = {sum(len(l) for l in oh.categories_)} cột one-hot + {len(ten_cot) - sum(len(l) for l in oh.categories_)} cột số")
    p("    handle_unknown='ignore': giá trị chỉ có ở test (chưa thấy khi học) -> toàn 0, không làm lỗi")
    p("    Không dùng Label Encoding cho biến danh nghĩa: sẽ tạo thứ tự giả (vd Bắc=0 < Nam Bộ=3)")
    p("    Không StandardScaler ở ma trận này: one-hot là 0/1, sin/cos đã nằm trong [-1, 1]")
    for ten, t, X in (("train", train, X_train), ("test", test, X_test)):
        out = pd.concat([t[KHOA], pd.DataFrame(X, columns=ten_cot), t[[NHAN]]], axis=1)
        luu_bang(out, ten)
    joblib.dump({"bo_ma_hoa": bo, "ten_cot": ten_cot, "danh_nghia": DANH_NGHIA, "chu_ky": CHU_KY,
                 "nhi_phan": NHI_PHAN, "khong_ro": KHONG_RO}, GOC / "bo_ma_hoa.joblib")
    p(f"    Lưu train ({X_train.shape[0]:,d} × {X_train.shape[1]}), test ({X_test.shape[0]:,d} × {X_test.shape[1]}), bo_ma_hoa.joblib")

    p("\n[5] Dạng giao dịch cho luật kết hợp (toàn bộ dữ liệu, không cần chia)")
    gd = pd.get_dummies(x[DANH_NGHIA].astype(str), prefix_sep="=").astype(bool)
    bo_cot = [c for c in gd.columns if c.endswith(f"={KHONG_RO}")]
    gd = gd.drop(columns=bo_cot)
    gd["cuoi_tuan=1"] = x["cuoi_tuan"] == 1
    gd[f"{NHAN}=1"] = x[NHAN] == 1
    gd[f"{NHAN}=0"] = x[NHAN] == 0
    gd = pd.concat([x[KHOA].reset_index(drop=True), gd.reset_index(drop=True)], axis=1)
    p(f"    {len(gd):,d} giao dịch × {gd.shape[1] - 2} item; bỏ {len(bo_cot)} item '{KHONG_RO}' (ô trống không phải một điều kiện)")
    p(f"    Trung bình mỗi giao dịch có {gd.drop(columns=KHOA).sum(axis=1).mean():.1f} item")
    luu_bang(gd, "giao_dich")

    p("\n[6] Hồ sơ tỉnh + chuẩn hoá z-score cho phân cụm")
    hs, cot = ho_so_tinh(df)
    p(f"    {len(hs)} tỉnh × {len(cot)} chỉ số: {', '.join(cot)}")
    p("    log_so_vu = log(1 + số vụ): số vụ lệch mạnh (TP.HCM, Hà Nội rất lớn)")
    p("    StandardScaler: z = (x - trung bình) / độ lệch chuẩn -> mọi chỉ số cùng thang đo, trung bình 0, độ lệch 1")
    p("    Vì sao Standard mà không Min-Max: phân cụm dựa trên khoảng cách; Min-Max bị một tỉnh cực trị kéo dãn cả thang")
    for c in cot:
        p(f"      {c:16s}: gốc {hs[c].min():.3f}–{hs[c].max():.3f} -> z {hs[c + '_z'].min():+.2f}–{hs[c + '_z'].max():+.2f}")
    luu_bang(hs, "ho_so_tinh")

    (GOC / "kiem_tra_bien_doi.txt").write_text("\n".join(BC), encoding="utf-8")
    for ten in ("train", "test", "giao_dich", "ho_so_tinh"):
        print(f"    Đã lưu {ten}.csv + bảng tien_xu_ly.{ten}")


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

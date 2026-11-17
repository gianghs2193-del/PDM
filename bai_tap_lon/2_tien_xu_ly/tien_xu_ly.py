"""TIỀN XỬ LÝ DỮ LIỆU — Tai nạn giao thông đường bộ Việt Nam (2020 – 8 tháng 2022)

Đầu vào  : bảng thu_thap.bang_loi, thu_thap.bang_mo_rong (PostgreSQL) của bước 1 (bai_tap_lon/1_thu_thap_du_lieu) — KHÔNG ghi đè.
Đầu ra   : tai_nan_loi_sach      (mọi nguồn, 1 dòng = 1 vụ)
           tai_nan_mo_rong_sach  (2020–2021, nối với bảng lõi qua khoá nguon + dong_goc)
           tu_dien_sach.csv      (vai trò từng cột: đặc trưng / nhãn / rò rỉ nhãn / cờ ...)
           kiem_tra_tien_xu_ly.txt (số bản ghi / giá trị bị ảnh hưởng của từng thao tác; cũng in ra màn hình)

Các thao tác (theo thứ tự):
    1. Xoá bản ghi trùng hoàn toàn (giữ bản đầu)
    2. Sửa giờ sai định dạng; gắn cờ giờ 00:00:00 nghi là giá trị mặc định
    3. Gắn cờ tháng không trọn (đầu / cuối kỳ dữ liệu)
    4. Chuẩn hoá nhãn và văn bản: khoảng trắng, chữ hoa / thường, lỗi chính tả, nhãn đồng nghĩa,
       lỗi lặp trong cột nhiều giá trị; gộp bản ghi trùng thực thể (một vụ nhập nhiều lần, giữ bản đầy đủ nhất);
       xoá bản ghi không có bất kỳ thông tin địa điểm
    5. Ngoại lai: phát hiện bằng IQR và z-score (chỉ báo cáo); giá trị vô lý (thiệt hại > 10 tỷ, số bị can > 10)
       -> để trống, gắn cờ
    6. Cờ theo tỉnh: nhập đủ thông tin điều kiện (2020–2021); ghi nhận cả vụ không có người chết (cả 3 năm)
    7. Ô trống: mang nghĩa "không có" -> điền giá trị tương ứng; do không nhập -> "Không rõ"
    8. Gán vai trò cột, kiểm tra toàn vẹn, lưu

KHÔNG làm ở bước này: rút đặc trưng mới (chỉ dùng 12 đặc trưng nhóm A có sẵn từ bước 1), mã hoá one-hot, chuẩn hoá số (phụ thuộc thuật toán -> để ở bước mô hình).

Chạy (từ thư mục D:/PDM, sau khi đã chạy thu_thap.py):
       .venv/Scripts/python bai_tap_lon/2_tien_xu_ly/tien_xu_ly.py
Giai đoạn 2 – Tiền xử lý gồm 4 script chạy lần lượt trong thư mục này:
       tien_xu_ly.py (làm sạch) -> bien_doi.py -> ky_nghe_dac_trung.py -> pipeline.py
"""
from __future__ import annotations

import logging
import platform
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
log = logging.getLogger("tien_xu_ly")
BC: list[str] = ["BÁO CÁO TIỀN XỬ LÝ DỮ LIỆU", "=" * 70]

KHOA = ["nguon", "dong_goc"]
KHONG_RO = "Không rõ"
NGUONG_THIEU_TINH = 0.20      # tỉnh "nhập đủ" nếu tỉ lệ trống của thoi_tiet < 20%
NGUONG_THIET_HAI = 10_000     # triệu đồng (10 tỷ) — lớn hơn thì coi là nhập sai
NGUONG_BI_CAN = 10            # số bị can trong một vụ; lớn hơn thì coi là nhập sai (giá trị thật: 0–3, rồi 33, 34, 85)
NGUONG_VU_NHE = 0.50          # tỉnh "ghi nhận cả vụ không có người chết" nếu tỉ lệ vụ có người chết của tỉnh <= 50%
# Hai bản ghi trùng các thuộc tính này coi là CÙNG MỘT VỤ được nhập nhiều lần
KHOA_THUC_THE = ["ngay", "gio", "tinh", "huyen", "dia_diem", "so_chet", "so_bi_thuong"]
# Chuẩn hoá nhãn: lỗi chính tả và nhãn đồng nghĩa đã kiểm tra trên dữ liệu (cũ -> mới)
SUA_CHU = {"phần đương": "phần đường", "rẻ phải": "rẽ phải"}
NHAN_DONG_NGHIA = {"kieu_va_cham": {"Các trường hợp khác": "Trường hợp khác"}}
COT_VAN_BAN_LOI = ["tinh", "huyen", "xa", "ten_duong", "km", "dia_diem"]   # gộp biến thể hoa / thường
COT_NHIEU_GIA_TRI = ["nguyen_nhan", "loi_vi_pham"]                        # nhiều giá trị ngăn bởi ";"
DAC_TRUNG_A = ["gio_so", "khung_gio", "thu", "cuoi_tuan", "thang", "vung", "mua", "dip_tet",
               "giai_doan_covid", "cap_duong", "khu_vuc", "co_nguoi_chet"]

# Thuộc tính phân loại: ô trống do đơn vị không nhập -> "Không rõ"
COT_KHONG_RO = ["thoi_tiet", "mat_do_gt", "hinh_thuc_va_cham", "kieu_va_cham", "dang_duong", "loai_mat_duong",
                "do_rong_duong", "tinh_trang_mat_duong", "to_chuc_gt", "nut_gt", "bien_bao", "den_chieu_sang",
                "nguyen_nhan", "hinh_thuc_xu_ly"]
# Vai trò cột (ghi vào tu_dien_sach.csv)
RO_RI = ["phan_loai", "so_chet", "so_bi_thuong", "so_thuong_nang", "so_thuong_nhe", "thiet_hai",
         "thiet_hai_ngoai_lai", "hinh_thuc_xu_ly", "so_bi_can", "co_lech_thuong_vong",
         *[f"pt{i}_tinh_trang" for i in range(1, 6)]]
SAU_DIEU_TRA = ["nguyen_nhan", "loi_vi_pham", "bo_tron", *[f"pt{i}_gay_tn" for i in range(1, 6)]]
CO_KIEM_TRA = ["co_trung_khoa", "co_ngay_loi", "co_gio_loi", "gio_da_sua", "co_gio_mac_dinh", "thang_khong_tron",
               "tinh_nhap_day_du", "tinh_ghi_ca_vu_nhe", "so_bi_can_ngoai_lai"]
VAN_BAN = ["huyen", "xa", "ten_duong", "km", "dia_diem"]


# ---------------------------------------------------------------------------
def cai_dat_nhat_ky() -> None:
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    man_hinh = logging.StreamHandler(sys.stdout)
    man_hinh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(man_hinh)


def p(dong: str, canh_bao: bool = False) -> None:
    """In một dòng ra màn hình và đưa vào báo cáo kiem_tra_tien_xu_ly.txt."""
    (log.warning if canh_bao else log.info)("%s", dong)
    BC.append(dong)


def doc_bang(ten: str) -> pd.DataFrame:
    return csdl.doc("thu_thap", ten)


def luu_bang(df: pd.DataFrame, ten: str) -> None:
    df.to_csv(GOC / f"{ten}.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(df, "tien_xu_ly", ten)


def chuoi(gia_tri) -> pd.arrays.StringArray:
    return pd.array(list(gia_tri), dtype="string")


def anh_xa(cot: pd.Series, ham) -> pd.arrays.StringArray:
    """Ánh xạ từng giá trị; ô trống hoặc đã là 'Không rõ' -> 'Không rõ'."""
    return chuoi(KHONG_RO if pd.isna(v) or v == KHONG_RO else ham(v) for v in cot)


def phan_bo(cot: pd.Series) -> str:
    dem = cot.value_counts(dropna=True)
    trong = int(cot.isna().sum())
    return "; ".join(f"{k}={v:,d}" for k, v in dem.items()) + (f"; (trống)={trong:,d}" if trong else "")


# ---------------------------------------------------------------------------
# 1. Trùng lặp
# ---------------------------------------------------------------------------
def xoa_trung_hoan_toan(loi: pd.DataFrame, mr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p("\n[1] Bản ghi trùng hoàn toàn")
    day_du = loi.merge(mr, on=KHOA, how="left", validate="one_to_one")
    noi_dung = [c for c in day_du.columns if c not in (*KHOA, "file_goc") and not c.startswith("co_")]
    trung = day_du.duplicated(subset=noi_dung, keep="first")
    xoa = day_du.loc[trung, KHOA]
    p(f"    So trên {len(noi_dung)} cột nội dung (bỏ khoá, tên file, cờ): {int(trung.sum())} bản lặp -> XOÁ, giữ bản đầu")
    for ma, dong in xoa.itertuples(index=False):
        p(f"      xoá: nguồn {ma}, dòng Excel {dong}")
    khoa_xoa = set(map(tuple, xoa.to_numpy()))
    giu = lambda df: df[[k not in khoa_xoa for k in zip(df["nguon"], df["dong_goc"])]].reset_index(drop=True)
    loi2, mr2 = giu(loi), giu(mr)
    p(f"    Bảng lõi: {len(loi):,d} -> {len(loi2):,d} | bảng mở rộng: {len(mr):,d} -> {len(mr2):,d}")
    p(f"    Trùng theo khoá (co_trung_khoa = 1) còn lại: {int(loi2['co_trung_khoa'].sum())} bản ghi"
      " -> xử lý ở bước [4d] sau khi chuẩn hoá văn bản")
    return loi2, mr2


# ---------------------------------------------------------------------------
# 2–3. Thời gian
# ---------------------------------------------------------------------------
def xu_ly_thoi_gian(loi: pd.DataFrame) -> pd.DataFrame:
    p("\n[2] Giờ xảy ra")
    hop_le = r"([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?"
    gio = loi["gio"]
    sai = gio.notna() & ~gio.fillna("").str.fullmatch(hop_le)

    def sua(g: str) -> str | None:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})(\d{2})", g)  # thiếu dấu ':' thứ hai, vd 18:0030
        if m and int(m[1]) <= 23 and int(m[2]) <= 59 and int(m[3]) <= 59:
            return f"{int(m[1]):02d}:{m[2]}:{m[3]}"
        return None

    moi, da_sua, con_sai = [], [], []
    for g, s in zip(gio, sai):
        k = sua(g) if s else None
        moi.append(k if k else g)
        da_sua.append(int(bool(k)))
        if s and not k:
            con_sai.append(g)
    loi["gio"] = chuoi(moi)
    loi["gio_da_sua"] = pd.array(da_sua, dtype="Int64")
    vi_du = [f"{a} -> {b}" for a, b, d in zip(gio, moi, da_sua) if d][:5]
    p(f"    Sai định dạng: {int(sai.sum())} | sửa được (chèn dấu ':' thiếu): {sum(da_sua)}, vd {'; '.join(vi_du)}")
    p(f"    Không sửa được, giữ nguyên: {len(con_sai)}" + (f" ({'; '.join(con_sai[:5])})" if con_sai else ""),
      canh_bao=bool(con_sai))
    gio_moi = pd.to_numeric(loi["gio"].str.extract(r"^(\d{1,2}):", expand=False), errors="coerce")
    lech = int((gio_moi.notna() & loi["gio_so"].notna() & (gio_moi != loi["gio_so"].astype("float"))).sum())
    p(f"    Đối chiếu với đặc trưng gio_so đã rút ở bước 1: {lech} giá trị lệch", canh_bao=bool(lech))
    mac_dinh = loi["gio"].isin(["00:00:00", "00:00"])
    loi["co_gio_mac_dinh"] = mac_dinh.astype("Int64")
    p(f"    Giờ đúng 00:00:00 (nghi là giá trị mặc định khi không rõ giờ): {int(mac_dinh.sum())} -> gắn cờ"
      f" co_gio_mac_dinh, không sửa | giờ trống: {int(loi['gio'].isna().sum())} -> giữ trống")

    p("\n[3] Tháng không trọn")
    dau, cuoi = loi["ngay"].min(), loi["ngay"].max()
    ky = loi["ngay"].dt.to_period("M")
    khong_tron = [d.to_period("M") for d, thieu in ((dau, dau.day > 1), (cuoi, cuoi.day < cuoi.days_in_month)) if thieu]
    loi["thang_khong_tron"] = ky.isin(khong_tron).astype("Int64")
    p(f"    Dữ liệu từ {dau:%d/%m/%Y} đến {cuoi:%d/%m/%Y} -> tháng không trọn: "
      f"{', '.join(str(k) for k in khong_tron) or '(không có)'} | {int(loi['thang_khong_tron'].sum()):,d} bản ghi"
      " gắn cờ thang_khong_tron (loại hoặc chú thích khi vẽ theo tháng)")

    for c in ("cap_duong", "khu_vuc"):
        n = int(loi[c].isna().sum())
        loi[c] = loi[c].fillna(KHONG_RO)
        p(f"    {c}: {n:,d} ô trống -> '{KHONG_RO}' (cột gốc loai_duong giữ nguyên)")
    return loi


# ---------------------------------------------------------------------------
# 4–6. Bảng mở rộng: ngoại lai, ô trống
# ---------------------------------------------------------------------------
# Dấu thanh đặt ở nguyên âm đầu ("hòa", "Hòang") hay nguyên âm sau ("hoà", "Hoàng"): coi là một cách viết
_THANH = {"oa": "àáảãạ", "oe": "èéẻẽẹ", "uy": "ỳýỷỹỵ"}
_DAU_DAU = {"oa": "òóỏõọ", "oe": "òóỏõọ", "uy": "ùúủũụ"}
_DOI_THANH = {d + c[0][1]: c[0][0] + t for c in _THANH.items() for d, t in zip(_DAU_DAU[c[0]], c[1])}


_SAI_DAU = re.compile(r"([òóỏõọ])([ae])(?=[a-zđ])|([ùúủũụ])(y)(?=[a-zđ])", re.I)


def sua_vi_tri_dau(gt):
    """Sửa dấu thanh đặt sai khi sau cụm 'oa', 'oe', 'uy' còn phụ âm: 'Hòang' -> 'Hoàng'."""
    if pd.isna(gt):
        return gt
    def doi(m):
        dau, sau = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        bo = unicodedata.normalize("NFD", dau)                 # tách chữ gốc và dấu thanh
        goc, thanh = bo[0], bo[1:]
        return goc + unicodedata.normalize("NFC", sau + thanh)
    return _SAI_DAU.sub(doi, str(gt))


def khoa_so_sanh(gt: str) -> str:
    """Khoá để so hai cách viết: chữ thường, NFC, gom khoảng trắng, thống nhất vị trí dấu thanh."""
    k = " ".join(unicodedata.normalize("NFC", str(gt)).lower().split())
    for cu, moi in _DOI_THANH.items():
        k = k.replace(cu, moi)
    return k


def gop_bien_the(cot: pd.Series) -> tuple[pd.Series, int, list[str]]:
    """Gộp các cách viết chỉ khác nhau ở chữ hoa / thường và khoảng trắng về cách viết hay gặp nhất."""
    sach = cot.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    khoa = pd.Series([pd.NA if pd.isna(v) else khoa_so_sanh(v) for v in sach], index=cot.index, dtype="string")
    dem = sach.value_counts()
    # Ưu tiên: không có từ viết hoa lộn xộn ("GIa"), rồi hay gặp nhất, rồi ít chữ hoa hơn
    lon_xon = lambda t: sum(1 for w in t.split() if any(c.isupper() for c in w[1:]) and any(c.islower() for c in w))
    thu_tu = sorted(dem.items(), key=lambda kv: (lon_xon(kv[0]), -kv[1], sum(c.isupper() for c in kv[0])))
    chuan = {}
    for gt, _ in thu_tu:
        chuan.setdefault(khoa_so_sanh(gt), gt)
    moi = pd.array([pd.NA if pd.isna(k) else chuan[k] for k in khoa], dtype="string")
    moi = pd.Series(moi, index=cot.index)
    doi = cot.notna() & (cot.astype("string") != moi)
    nhom = sach.groupby(khoa).nunique()
    vi_du = []
    for k in nhom[nhom > 1].index[:3]:
        bt = sorted(sach[khoa == k].dropna().unique())
        vi_du.append(" / ".join(f"\u201c{b}\u201d" for b in bt[:4]) + f" -> \u201c{chuan[k]}\u201d")
    return moi, int(doi.sum()), vi_du


def chuan_nhieu_gia_tri(gt):
    """Tách theo ';', sửa chính tả, bỏ phần lặp, ghép lại bằng '; '."""
    if pd.isna(gt):
        return gt
    phan = []
    for x in str(gt).split(";"):
        x = " ".join(x.split())
        for cu, moi in SUA_CHU.items():
            x = x.replace(cu, moi)
        if x and x not in phan:
            phan.append(x)
    return "; ".join(phan) if phan else pd.NA


def chuan_hoa_nhan(loi: pd.DataFrame, mr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p("\n[4] Chuẩn hoá nhãn và văn bản (chỉ đổi cách viết, không đổi nghĩa)")
    p("  a) Gộp cách viết chỉ khác chữ hoa / thường, khoảng trắng, vị trí dấu thanh (Hòa / Hoà) -> cách viết hay gặp nhất:")
    for c in COT_VAN_BAN_LOI:
        truoc = int(loi[c].nunique())
        goc = loi[c].copy()
        loi[c] = pd.array([v if pd.isna(v) else unicodedata.normalize("NFC", v) for v in loi[c]], dtype="string")
        n_nfc = int((goc.notna() & (goc.astype("string") != loi[c])).sum())
        if n_nfc:
            p(f"    {c:10s}: chuẩn hoá mã Unicode (NFC) ở {n_nfc:,d} ô — chữ nhìn giống hệt nhưng máy coi là khác")
        cu = loi[c].copy()
        loi[c] = pd.array([sua_vi_tri_dau(v) for v in loi[c]], dtype="string")
        n_dau = int((cu.notna() & (cu.astype("string") != loi[c])).sum())
        if n_dau:
            vd = sorted({f"\u201c{a}\u201d -> \u201c{b}\u201d" for a, b in zip(cu[cu.astype("string") != loi[c]], loi[c][cu.astype("string") != loi[c]]) if pd.notna(a)})[:2]
            p(f"    {c:10s}: sửa dấu thanh đặt sai ở {n_dau:,d} ô, vd {'; '.join(vd)}")
        loi[c], n, vi_du = gop_bien_the(loi[c])
        p(f"    {c:10s}: {truoc:,d} -> {int(loi[c].nunique()):,d} giá trị khác nhau | {n:,d} ô đổi"
          + (f" | vd {'; '.join(vi_du)}" if vi_du else ""))
    p("  b) Cột nhiều giá trị (ngăn bởi ';'): tách, sửa chính tả, bỏ phần lặp, ghép lại:")
    for c in COT_NHIEU_GIA_TRI:
        truoc = mr[c].copy()
        mr[c] = pd.array([chuan_nhieu_gia_tri(v) for v in mr[c]], dtype="string")
        doi = truoc.notna() & (truoc.astype("string") != mr[c])
        vi_du = [f"\u201c{a}\u201d -> \u201c{b}\u201d" for a, b in zip(truoc[doi], mr[c][doi])][:2]
        p(f"    {c:12s}: {truoc.nunique():,d} -> {mr[c].nunique():,d} giá trị khác nhau | {int(doi.sum()):,d} ô đổi"
          + (f" | vd {'; '.join(vi_du)}" if vi_du else ""))
    p(f"    Sửa chính tả: {'; '.join(f'{a!r} -> {b!r}' for a, b in SUA_CHU.items())}")
    p("  c) Nhãn đồng nghĩa và lỗi chính tả trong cột phân loại:")
    for c, bang in NHAN_DONG_NGHIA.items():
        for cu, moi in bang.items():
            n = int((mr[c] == cu).sum())
            mr[c] = mr[c].replace(cu, moi)
            p(f"    {c}: \u201c{cu}\u201d -> \u201c{moi}\u201d ({n:,d} ô)")
    for c in ("kieu_va_cham", "hinh_thuc_va_cham", "nut_gt", "dang_duong"):
        truoc = mr[c].copy()
        mr[c] = mr[c].astype("string")
        for cu, moi in SUA_CHU.items():
            mr[c] = mr[c].str.replace(cu, moi, regex=False)
        n = int((truoc.notna() & (truoc.astype("string") != mr[c])).sum())
        if n:
            p(f"    {c}: sửa chính tả ở {n:,d} ô")

    return loi, mr


def gop_trung_thuc_the(loi: pd.DataFrame, mr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Một vụ được nhập nhiều lần: trùng ngày, giờ, tỉnh, huyện, địa điểm, số chết, số bị thương.
    Giữ bản ghi có nhiều ô có giá trị nhất (hoà thì giữ bản đầu), xoá các bản còn lại ở cả hai bảng."""
    p(f"  d) Trùng thực thể: trùng {', '.join(KHOA_THUC_THE)} -> coi là MỘT vụ nhập nhiều lần")
    day_du = loi.merge(mr, on=KHOA, how="left", validate="one_to_one")
    noi_dung = [c for c in day_du.columns if c not in (*KHOA, "file_goc") and not c.startswith("co_")]
    co_khoa = day_du[KHOA_THUC_THE].notna().all(axis=1)
    nghi = day_du[co_khoa & day_du.duplicated(KHOA_THUC_THE, keep=False)]
    xoa: list[tuple] = []
    for _, g in nghi.groupby(KHOA_THUC_THE, sort=False):
        so_o = g[noi_dung].notna().sum(axis=1)
        giu = so_o.idxmax()
        khac = [c for c in noi_dung if g[c].astype("string").nunique(dropna=False) > 1]
        bo = g.drop(index=giu)
        xoa += list(zip(bo["nguon"], bo["dong_goc"]))
        r0 = g.loc[giu]
        p(f"      {r0['ngay']:%d/%m/%Y} {r0['gio']}, {r0['huyen']}, {r0['tinh']}: {len(g)} bản ghi, khác ở {len(khac)} cột"
          f" ({', '.join(khac[:3])}{', …' if len(khac) > 3 else ''}) -> giữ dòng {r0['dong_goc']}"
          f" ({int(so_o[giu])} ô có giá trị), xoá dòng {', '.join(str(d) for d in bo['dong_goc'])}")
    p(f"    {len(xoa)} bản ghi bị xoá. Lý do: hai vụ khác nhau gần như không thể trùng đúng phút, đúng địa điểm, đúng số"
      " người chết và bị thương; phần khác nhau là cách mô tả của người nhập (vd đảo thứ tự phương tiện 1 và 2,"
      " một bản ghi bỏ trống ô bản kia có). Giữ bản ghi đầy đủ hơn.")
    khong_dia_diem = loi[["tinh", "huyen", "dia_diem"]].isna().all(axis=1)
    for ma, dong in loi.loc[khong_dia_diem, KHOA].itertuples(index=False):
        p(f"  e) Không có thông tin địa điểm (tỉnh, huyện, địa điểm đều trống): nguồn {ma}, dòng Excel {dong} -> XOÁ"
          " (không gán được cho tỉnh nào, không kiểm chứng được)")
        xoa.append((ma, dong))
    khoa_xoa = set(xoa)
    giu_lai = lambda df: df[[k not in khoa_xoa for k in zip(df["nguon"], df["dong_goc"])]].reset_index(drop=True)
    loi2, mr2 = giu_lai(loi), giu_lai(mr)
    con = loi2[KHOA_THUC_THE].notna().all(axis=1) & loi2.duplicated(KHOA_THUC_THE, keep=False)
    if con.any():
        raise ValueError("Vẫn còn bản ghi trùng thực thể sau khi gộp")
    loi2["co_trung_khoa"] = pd.array([0] * len(loi2), dtype="Int64")
    p(f"    Bảng lõi: {len(loi):,d} -> {len(loi2):,d} | bảng mở rộng: {len(mr):,d} -> {len(mr2):,d}"
      " | không còn bản ghi trùng thực thể (co_trung_khoa = 0)")
    return loi2, mr2


def xu_ly_gia_tri(loi: pd.DataFrame, mr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p("\n[5] Ngoại lai")
    p("  a) Phát hiện bằng thống kê (IQR và z-score) — chỉ báo cáo, không sửa:")
    for c, df in (("so_chet", loi), ("so_bi_thuong", loi), ("thiet_hai", mr), ("so_bi_can", mr)):
        v = pd.to_numeric(df[c], errors="coerce").dropna().astype(float)
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        tren = q3 + 1.5 * (q3 - q1)
        n_iqr = int((v > tren).sum())
        n_z = int((((v - v.mean()) / v.std()).abs() > 3).sum())
        p(f"    {c:12s}: Q1 = {q1:g}, Q3 = {q3:g}, ngưỡng trên IQR = Q3 + 1,5×(Q3 − Q1) = {tren:g} -> {n_iqr:,d} giá trị"
          f" ({n_iqr / len(v):.1%}) | |z| > 3: {n_z:,d} giá trị | lớn nhất {v.max():,g}")
    p("    Không xoá theo IQR / z-score: đây là số đếm lệch phải (phần lớn bằng 0 hoặc 1), nên hai phương pháp đánh dấu"
      " cả những vụ nhiều người chết có thật. Chỉ xử lý giá trị VÔ LÝ ở mục b.")
    p("  b) Giá trị vô lý (nhập sai) -> để trống, gắn cờ:")
    ngoai = mr["thiet_hai"] > NGUONG_THIET_HAI
    for ma, dong, v in mr.loc[ngoai, [*KHOA, "thiet_hai"]].itertuples(index=False):
        p(f"    nguồn {ma}, dòng Excel {dong}: {v:,.0f} triệu đồng -> để trống")
    mr["thiet_hai_ngoai_lai"] = ngoai.astype("Int64")
    mr["thiet_hai"] = mr["thiet_hai"].where(~ngoai)
    p(f"    {int(ngoai.sum())} giá trị > {NGUONG_THIET_HAI:,d} triệu đồng -> trống, gắn cờ thiet_hai_ngoai_lai; giữ bản ghi")
    p(f"    Mâu thuẫn thương vong / phân loại (co_lech_thuong_vong = 1): {int(mr['co_lech_thuong_vong'].sum())} bản ghi"
      " -> KHÔNG sửa số liệu; loại khỏi phân tích dùng phan_loai")
    ngoai = mr["so_bi_can"] > NGUONG_BI_CAN
    for ma, dong, v in mr.loc[ngoai, [*KHOA, "so_bi_can"]].itertuples(index=False):
        p(f"    nguồn {ma}, dòng Excel {dong}: {v:g} bị can trong một vụ -> để trống")
    mr["so_bi_can_ngoai_lai"] = ngoai.astype("Int64")
    mr["so_bi_can"] = mr["so_bi_can"].where(~ngoai)
    p(f"    {int(ngoai.sum())} giá trị so_bi_can > {NGUONG_BI_CAN} -> trống, gắn cờ so_bi_can_ngoai_lai; giữ bản ghi")

    p("\n[6] Cờ theo tỉnh")
    p("  a) Nhập đủ thông tin điều kiện (bảng mở rộng, 2020–2021):")
    tinh = mr[KHOA].merge(loi[[*KHOA, "tinh"]], on=KHOA, how="left", validate="one_to_one")["tinh"]
    ti_le = mr["thoi_tiet"].isna().groupby(tinh.to_numpy()).mean()
    du = sorted(ti_le[ti_le < NGUONG_THIEU_TINH].index)
    mr["tinh_nhap_day_du"] = pd.array([int(t in du) if pd.notna(t) else 0 for t in tinh], dtype="Int64")
    p(f"    Tiêu chí: tỉ lệ trống của thoi_tiet theo tỉnh < {NGUONG_THIEU_TINH:.0%} -> {len(du)}/{ti_le.size} tỉnh,"
      f" {int(mr['tinh_nhap_day_du'].sum()):,d}/{len(mr):,d} bản ghi (cờ tinh_nhap_day_du)")
    p(f"    Tỉnh nhập đủ: {', '.join(du)}")
    p(f"    Tỉnh trống > 90%: {', '.join(sorted(ti_le[ti_le > 0.9].index))}")
    p("  b) Ghi nhận cả vụ không có người chết (bảng lõi, cả 3 năm):")
    ti_le_chet = pd.to_numeric(loi["co_nguoi_chet"]).astype(float).groupby(loi["tinh"]).mean()
    nhe = sorted(ti_le_chet[ti_le_chet <= NGUONG_VU_NHE].index)
    loi["tinh_ghi_ca_vu_nhe"] = pd.array([int(t in nhe) if pd.notna(t) else 0 for t in loi["tinh"]], dtype="Int64")
    p(f"    Tiêu chí: tỉ lệ vụ có người chết của tỉnh <= {NGUONG_VU_NHE:.0%} -> {len(nhe)}/{ti_le_chet.size} tỉnh,"
      f" {int(loi['tinh_ghi_ca_vu_nhe'].sum()):,d}/{len(loi):,d} vụ (cờ tinh_ghi_ca_vu_nhe)")
    so_vu = loi["tinh"].value_counts()
    for ng in (0.40, 0.60):
        ds = ti_le_chet[ti_le_chet <= ng].index
        p(f"    Độ nhạy: ngưỡng {ng:.0%} -> {len(ds)} tỉnh, {int(so_vu[ds].sum()):,d} vụ")
    p(f"    Tỉnh: {', '.join(nhe)}")
    chung = ti_le.index.intersection(ti_le_chet.index)
    p(f"    Tương quan theo tỉnh giữa tỉ lệ trống thoi_tiet và tỉ lệ có người chết: r = "
      f"{ti_le[chung].corr(ti_le_chet[chung]):.2f} -> hai cờ đo hai việc khác nhau (điền đủ ô / ghi đủ loại vụ)")
    p("    Tỉnh có tỉ lệ chết rất cao nhiều khả năng chỉ ghi vụ nghiêm trọng. Dùng cờ này để chạy lại phân tích trên nhóm"
      " tỉnh ghi nhận rộng hơn và so kết luận. Cờ suy ra từ nhãn nên KHÔNG dùng làm đặc trưng dự báo.")

    p("\n[7] Ô trống")
    p("  a) Trống mang nghĩa 'không có' (giả định theo cách nhập liệu của nguồn):")
    trong = mr["so_bi_can"].isna() & (mr["so_bi_can_ngoai_lai"] == 0)
    mr.loc[trong, "so_bi_can"] = 0
    p(f"    so_bi_can: {int(trong.sum()):,d} ô trống -> 0 (không có bị can); 3 ô ngoại lai giữ trống"
      if int(mr["so_bi_can_ngoai_lai"].sum()) == 3 else
      f"    so_bi_can: {int(trong.sum()):,d} ô trống -> 0 (không có bị can); ô ngoại lai giữ trống")
    n = int(mr["bo_tron"].isna().sum())
    mr["bo_tron"] = mr["bo_tron"].fillna("Không bỏ trốn")
    p(f"    bo_tron: {n:,d} ô trống -> 'Không bỏ trốn'")
    p("    pt1..pt5 trống = không có phương tiện thứ i -> giữ trống")
    p("    giao_duong_sat, den_tin_hieu: không phân biệt được 'không có' với 'không nhập' -> giữ trống")
    p(f"  b) Trống do đơn vị không nhập -> '{KHONG_RO}' ({len(COT_KHONG_RO)} thuộc tính phân loại):")
    for c in COT_KHONG_RO:
        n = int(mr[c].isna().sum())
        mr[c] = mr[c].fillna(KHONG_RO)
        p(f"    {c:22s}: {n:6,d} ô ({n / len(mr):5.1%})")
    return loi, mr


# ---------------------------------------------------------------------------
# 7. Vai trò cột, kiểm tra, lưu
# ---------------------------------------------------------------------------
def vai_tro(cot: str) -> str:
    if cot in (*KHOA, "file_goc"):
        return "khoá / truy vết"
    if cot == "co_nguoi_chet":
        return "nhãn"
    if cot in RO_RI:
        return "rò rỉ nhãn — không dùng làm đầu vào dự đoán"
    if cot in SAU_DIEU_TRA:
        return "biết sau điều tra — dùng cho mô tả / luật kết hợp, cân nhắc khi dự đoán"
    if cot in CO_KIEM_TRA:
        return "cờ kiểm tra / lọc"
    if cot in VAN_BAN:
        return "địa bàn chi tiết / văn bản — không dùng trực tiếp"
    if cot in ("ngay", "gio", "loai_duong"):  # đã có đặc trưng nhóm A suy ra từ các cột này
        return "thuộc tính gốc — đã có đặc trưng suy ra thay thế"
    return "đặc trưng"


def main() -> None:
    p(f"[0] Đọc đầu vào: bảng thu_thap.bang_loi, thu_thap.bang_mo_rong trong {csdl.mo_ta()}")
    loi, mr = doc_bang("bang_loi"), doc_bang("bang_mo_rong")
    thieu = [c for c in DAC_TRUNG_A if c not in loi.columns]
    if thieu:
        raise ValueError(f"bang_loi thiếu đặc trưng nhóm A {thieu} -> chạy lại bước 1 (thu_thap.py) trước")
    n_loi, n_mr = len(loi), len(mr)
    p(f"    bang_loi: {n_loi:,d} dòng × {loi.shape[1]} cột | bang_mo_rong: {n_mr:,d} dòng × {mr.shape[1]} cột")

    loi, mr = xoa_trung_hoan_toan(loi, mr)
    loi = xu_ly_thoi_gian(loi)
    loi, mr = chuan_hoa_nhan(loi, mr)
    loi, mr = gop_trung_thuc_the(loi, mr)
    loi, mr = xu_ly_gia_tri(loi, mr)

    p("\n[8] Kiểm tra toàn vẹn và lưu")
    if loi.duplicated(subset=KHOA).any() or mr.duplicated(subset=KHOA).any():
        raise ValueError("Khoá (nguon, dong_goc) không còn duy nhất")
    mo_coi = mr[KHOA].merge(loi[KHOA], on=KHOA, how="left", indicator=True)["_merge"].ne("both").sum()
    if mo_coi:
        raise ValueError(f"{mo_coi} bản ghi bảng mở rộng không có trong bảng lõi")
    p(f"    Khoá duy nhất ở cả hai bảng; mọi bản ghi bảng mở rộng đều có trong bảng lõi")
    p(f"    tai_nan_loi_sach    : {len(loi):,d} dòng × {loi.shape[1]} cột (trước: {n_loi:,d} dòng)")
    p(f"    tai_nan_mo_rong_sach: {len(mr):,d} dòng × {mr.shape[1]} cột (trước: {n_mr:,d} dòng)")
    p(f"    Vụ có người chết: {int(loi['co_nguoi_chet'].sum()):,d}/{len(loi):,d} ({loi['co_nguoi_chet'].mean():.1%})")

    tu_dien = pd.DataFrame(
        [{"bang": ten, "ten_cot": c, "kieu": str(df[c].dtype), "vai_tro": vai_tro(c),
          "ty_le_trong_%": round(df[c].isna().mean() * 100, 1), "so_gia_tri_khac_nhau": int(df[c].nunique())}
         for ten, df in (("tai_nan_loi_sach", loi), ("tai_nan_mo_rong_sach", mr)) for c in df.columns])
    for vt, sub in tu_dien.groupby("vai_tro", sort=False):
        p(f"    [{vt}] {len(sub)} cột: {', '.join(sub['ten_cot'])}")

    luu_bang(loi, "tai_nan_loi_sach")
    luu_bang(mr, "tai_nan_mo_rong_sach")
    tu_dien.to_csv(GOC / "tu_dien_sach.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(tu_dien, "tien_xu_ly", "tu_dien_sach")
    (GOC / "kiem_tra_tien_xu_ly.txt").write_text("\n".join(BC), encoding="utf-8")
    for ten in ("tai_nan_loi_sach.csv", "tai_nan_mo_rong_sach.csv", "tu_dien_sach.csv", "kiem_tra_tien_xu_ly.txt"):
        log.info("    Đã lưu %s (%s byte)", ten, f"{(GOC / ten).stat().st_size:,d}")


if __name__ == "__main__":
    cai_dat_nhat_ky()
    bat_dau = time.perf_counter()
    log.info("=" * 70)
    log.info("BẮT ĐẦU tien_xu_ly.py | Python %s | pandas %s | %s", platform.python_version(), pd.__version__,
             platform.platform())
    try:
        main()
    except Exception:
        log.exception("KẾT THÚC: LỖI sau %.1f giây", time.perf_counter() - bat_dau)
        sys.exit(1)
    log.info("KẾT THÚC: thành công sau %.1f giây", time.perf_counter() - bat_dau)

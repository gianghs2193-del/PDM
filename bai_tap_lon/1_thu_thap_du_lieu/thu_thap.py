"""THU THẬP DỮ LIỆU — Tai nạn giao thông đường bộ Việt Nam (2020 – 8 tháng 2022)

Thực hiện các bước 4–7 của quy trình thu thập dữ liệu (bước 1–3 mô tả trong bao_cao_thu_thap.md):
    4. Thu thập   : tải 3 file Excel từ Google Drive (nếu chưa có), đọc tiêu đề 2 dòng
    5. Tích hợp    : đưa 3 năm về cùng 1 lược đồ cột, ghi nguồn gốc từng bản ghi
    6. Kiểm tra    : số bản ghi, kiểu dữ liệu, dữ liệu thiếu, trùng lặp, tính hợp lệ
    7. Lưu trữ     : CSV + Parquet + từ điển dữ liệu + báo cáo kiểm tra
                     - du_lieu_tich_hop : bảng phẳng 1 dòng = 1 vụ, đủ mọi thuộc tính
                     - bang_loi         : thuộc tính có ở CẢ 3 nguồn (mọi bản ghi)
                     - bang_mo_rong     : thuộc tính chỉ có ở 2020–2021, nối với bang_loi qua (nguon, dong_goc)

Khi chạy, script in ra màn hình từng thao tác (mã băm file gốc, đọc / đổi tên / chọn cột / ép kiểu kèm
số giá trị bị ảnh hưởng, kết quả kiểm tra, file đã lưu, thời gian chạy, lỗi nếu có). Không ghi file nhật ký.

Ở bước này CHƯA làm sạch: không xoá, không điền giá trị thiếu, không gộp nhãn.
Bản ghi nghi vấn chỉ được ĐÁNH DẤU để bước tiền xử lý quyết định.
Riêng bang_loi có thêm các ĐẶC TRƯNG SUY RA (nhóm A: giờ, khung giờ, thứ, tháng, vùng, mùa, dịp Tết,
giãn cách COVID, cấp đường, khu vực, có người chết). Đây là cột THÊM, các cột gốc giữ nguyên giá trị.

Chạy (từ thư mục D:/PDM):
       python -m venv .venv
       .venv/Scripts/python -m pip install -r requirements.txt   # pandas openpyxl gdown sqlalchemy psycopg
       .venv/Scripts/python bai_tap_lon/1_thu_thap_du_lieu/thu_thap.py
"""
from __future__ import annotations

import hashlib
import logging
import platform
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
log = logging.getLogger("thu_thap")
THU_MUC_GOC = GOC.parent.parent / "data_drive"  # D:\PDM\data_drive — nơi lưu file Excel gốc
DRIVE_URL = "https://drive.google.com/drive/folders/1V--WoKpABVX5qaAwLwhHRF_dTOqM-1LI"
NGUON = [  # (mã nguồn, tên file, ngày bắt đầu kỳ, ngày kết thúc kỳ)
    ("2020", "bcthdb_thong_tin_tong_hop(2020).xlsx", "2019-12-15", "2020-12-14"),
    ("2021", "bcthdb_thong_tin_tong_hop(2021).xlsx", "2020-12-15", "2021-12-14"),
    ("8T2022", "bcthdb_thong_tin_tong_hop(8T2022).xlsx", "2021-12-15", "2022-08-14"),
]

# ---------------------------------------------------------------------------
# BƯỚC 3 — Thuộc tính cần thu thập: (tên cột gốc, tên mới, nhóm, kiểu, mô tả)
# Không thu thập: STT, thông tin cá nhân (họ tên, ngày sinh, địa chỉ, SĐT, biển số,
# chủ phương tiện), mô tả tự do, đơn vị hành chính tiếp nhận báo cáo, bảo hiểm.
# ---------------------------------------------------------------------------
_PT = [(f"Phương tiện liên quan - PT{i} | {g}", f"pt{i}_{m}", "Phương tiện", k, f"PT{i}: {d}")
       for i in range(1, 6) for g, m, k, d in [
           ("Nhóm PT", "nhom", "phân loại", "nhóm phương tiện"),
           ("Phương tiện", "loai", "phân loại", "loại phương tiện chi tiết"),
           ("Tình trạng phương tiện", "tinh_trang", "phân loại", "mức hư hỏng"),
           ("Là Phương tiện gây TN", "gay_tn", "nhị phân 0/1", "1 = phương tiện gây tai nạn")]]
THUOC_TINH = [
    ("Ngày xảy ra tai nạn", "ngay", "Thời gian", "ngày", "Ngày xảy ra (dd/mm/yyyy)"),
    ("Thời gian xảy ra tai nạn", "gio", "Thời gian", "giờ", "Giờ xảy ra (HH:MM:SS)"),
    ("Địa điểm xảy ra tai nạn | Tỉnh/Thành phố", "tinh", "Địa điểm", "phân loại", "Tỉnh/thành phố"),
    ("Địa điểm xảy ra tai nạn | Quận/Huyện", "huyen", "Địa điểm", "phân loại", "Quận/huyện"),
    ("Địa điểm xảy ra tai nạn | Phường/Xã", "xa", "Địa điểm", "phân loại", "Phường/xã"),
    ("Địa điểm xảy ra tai nạn | Loại đường", "loai_duong", "Địa điểm", "phân loại",
     "Cấp đường (+ Nội thị/Nông thôn)"),
    ("Địa điểm xảy ra tai nạn | Tên đường", "ten_duong", "Địa điểm", "văn bản", "Tên đường"),
    ("Địa điểm xảy ra tai nạn | KM", "km", "Địa điểm", "văn bản", "Lý trình / số nhà"),
    ("Địa điểm xảy ra tai nạn | Địa điểm", "dia_diem", "Địa điểm", "văn bản", "Địa chỉ đầy đủ"),
    ("Số người chết và bị thương | Số người chết", "so_chet", "Hậu quả", "số nguyên", "Số người chết"),
    ("Số người chết và bị thương | Số người bị thương", "so_bi_thuong", "Hậu quả", "số nguyên",
     "Tổng số người bị thương"),
    ("Số người chết và bị thương | Số người thương nặng", "so_thuong_nang", "Hậu quả", "số nguyên",
     "Số người bị thương nặng"),
    ("Số người chết và bị thương | Số người thương nhẹ", "so_thuong_nhe", "Hậu quả", "số nguyên",
     "Số người bị thương nhẹ"),
    ("Thiệt hại ước tính | Bằng số", "thiet_hai", "Hậu quả", "số thực",
     "Thiệt hại tài sản ước tính (triệu đồng — cột 'Bằng chữ' ghi 'Bốn triệu đồng chẵn' ứng với 4)"),
    ("Phân loại tai nạn", "phan_loai", "Hậu quả", "phân loại có thứ tự", "Mức độ: Va chạm GT → Đặc biệt nghiêm trọng"),
    *_PT,
    ("Tình trạng thời tiết", "thoi_tiet", "Môi trường", "phân loại", "Thời tiết / ánh sáng"),
    ("Mật độ giao thông", "mat_do_gt", "Môi trường", "phân loại có thứ tự", "Vắng / Bình thường / Đông"),
    ("Hình thức va chạm | Hình thức", "hinh_thuc_va_cham", "Va chạm", "phân loại", "Xe-xe, người-xe, tự gây..."),
    ("Hình thức va chạm | Va chạm", "kieu_va_cham", "Va chạm", "phân loại", "Kiểu va chạm chi tiết"),
    ("Gây tai nạn và bỏ trốn", "bo_tron", "Va chạm", "phân loại", "Có bỏ trốn / đã tìm thấy chưa"),
    ("Hiện trạng đường | Dạng đường", "dang_duong", "Hạ tầng", "phân loại", "Thẳng, cong, đèo dốc, cầu..."),
    ("Hiện trạng đường | Loại mặt đường", "loai_mat_duong", "Hạ tầng", "phân loại", "Bê tông nhựa, xi măng, đất..."),
    ("Hiện trạng đường | Độ rộng đường", "do_rong_duong", "Hạ tầng", "phân loại có thứ tự", "Khoảng độ rộng"),
    ("Hiện trạng đường | Tình trạng mặt đường", "tinh_trang_mat_duong", "Hạ tầng", "phân loại", "Khô, ướt trơn..."),
    ("Hiện trạng đường | Tổ chức giao thông", "to_chuc_gt", "Hạ tầng", "phân loại", "1 chiều, 2 chiều, đường đôi"),
    ("Hiện trạng đường | Nút giao thông", "nut_gt", "Hạ tầng", "phân loại", "Ngã 3, ngã 4..."),
    ("Hiện trạng đường | Vòng xuyến", "vong_xuyen", "Hạ tầng", "phân loại", "Có / không"),
    ("Hiện trạng đường | Giao cắt với đường sắt", "giao_duong_sat", "Hạ tầng", "phân loại", "Loại giao cắt"),
    ("Báo hiệu giao thông | Đèn tín hiệu", "den_tin_hieu", "Báo hiệu", "phân loại", "Trạng thái đèn tín hiệu"),
    ("Báo hiệu giao thông | Biển báo hiệu", "bien_bao", "Báo hiệu", "phân loại", "Có / không"),
    ("Báo hiệu giao thông | Đèn chiếu sáng công cộng", "den_chieu_sang", "Báo hiệu", "phân loại", "Có / không"),
    ("Nguyên nhân", "nguyen_nhan", "Nguyên nhân", "phân loại", "Nguyên nhân chính"),
    ("Lỗi vi phạm", "loi_vi_pham", "Nguyên nhân", "đa trị (ngăn bởi ;)", "Các lỗi vi phạm"),
    ("Phương pháp xử lý | Hình thức xử lý", "hinh_thuc_xu_ly", "Xử lý", "phân loại", "Hình sự / hành chính..."),
    ("Phương pháp xử lý | Số bị can", "so_bi_can", "Xử lý", "số nguyên", "Số bị can bị khởi tố"),
]
SO_NGUYEN = [t[1] for t in THUOC_TINH if t[3] in ("số nguyên", "số thực", "nhị phân 0/1")]


# ---------------------------------------------------------------------------
# Báo cáo thao tác (chỉ in ra màn hình)
# ---------------------------------------------------------------------------
def cai_dat_nhat_ky() -> None:
    """In từng thao tác ra màn hình; không ghi file nhật ký vào dự án."""
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    man_hinh = logging.StreamHandler(sys.stdout)
    man_hinh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(man_hinh)


def ma_bam(path: Path) -> str:
    """SHA-256 của file: file gốc bị thay đổi thì mã này đổi theo."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest()


def ghi_file_da_luu(path: Path) -> None:
    log.info("    Đã lưu %s (%s byte)", path.name, f"{path.stat().st_size:,d}")


# ---------------------------------------------------------------------------
# BƯỚC 4 — Thu thập
# ---------------------------------------------------------------------------
def tai_du_lieu() -> None:
    thieu = [f for _, f, _, _ in NGUON if not (THU_MUC_GOC / f).exists()]
    if thieu:
        import gdown
        log.info("    Thiếu %d file gốc %s -> tải thư mục Google Drive: %s", len(thieu), thieu, DRIVE_URL)
        gdown.download_folder(DRIVE_URL, output=str(THU_MUC_GOC), quiet=True)
    else:
        log.info("    Đủ 3 file gốc trong %s -> không tải lại", THU_MUC_GOC)
    for _, f, _, _ in NGUON:
        path = THU_MUC_GOC / f
        if not path.exists():
            raise FileNotFoundError(f"Không có file gốc sau khi tải: {path}")
        log.info("    File gốc %s | %s byte | sha256 %s", f, f"{path.stat().st_size:,d}", ma_bam(path))


def doc_sheet(path: Path, sheet: str = "DSTN") -> pd.DataFrame:
    """Đọc sheet có tiêu đề 2 dòng (ô gộp) -> cột dạng 'Nhóm | Cột con'.

    File 2021 ghi nhầm tiêu đề nhóm phương tiện thứ 5 là 'PT1' -> đánh số lại PT theo thứ tự xuất hiện.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    log.info("    [%s | %s] Đọc thô: %s dòng × %d cột (mọi ô giữ nguyên giá trị gốc)", path.name, sheet,
             f"{len(raw):,d}", raw.shape[1])
    h = next(i for i in range(6) if raw.iloc[i].astype(str).str.contains("Thời gian xảy ra").any())
    log.info("      Tiêu đề 2 tầng ở dòng Excel %d và %d; bỏ qua %d dòng phía trên; dữ liệu bắt đầu từ dòng Excel %d",
             h + 1, h + 2, h, h + 3)
    o_gop = int(raw.iloc[h].isna().sum())
    tren = raw.iloc[h].ffill().astype(str).str.strip()
    duoi = raw.iloc[h + 1]
    log.info("      Điền tên nhóm cho %d ô tiêu đề trống (ô gộp) bằng tên nhóm đứng trước", o_gop)

    so_pt = 0
    doi_pt = {}
    for i, t in enumerate(tren):
        if t.startswith("Phương tiện liên quan - PT"):
            if pd.notna(raw.iloc[h, i]):  # ô đầu của một nhóm (ô gộp chỉ có giá trị ở ô đầu)
                so_pt += 1
            moi = f"Phương tiện liên quan - PT{so_pt}"
            if t != moi:
                doi_pt[t + " -> " + moi] = doi_pt.get(t + " -> " + moi, 0) + 1
            tren.iloc[i] = moi
    if doi_pt:
        log.warning("      Sửa tiêu đề nhóm phương tiện ghi nhầm: %s",
                    "; ".join(f"{k} ({v} cột)" for k, v in doi_pt.items()))
    elif so_pt:
        log.info("      Tiêu đề %d nhóm phương tiện đúng thứ tự PT1..PT%d, không phải sửa", so_pt, so_pt)

    ten = [f"{t} | {str(d).strip()}" if pd.notna(d) and str(d).strip() != t else t
           for t, d in zip(tren, duoi)]
    df = raw.iloc[h + 2:]
    df.index = df.index + 1  # = số dòng Excel (đã đối chiếu với openpyxl), để truy vết về file gốc
    df.columns = ten
    trung_ten = pd.Index(ten).duplicated()
    if trung_ten.any():
        log.warning("      Bỏ %d cột trùng tên, giữ cột xuất hiện trước: %s", int(trung_ten.sum()),
                    "; ".join(f"'{t}' (cột thứ {i + 1})" for i, t in enumerate(ten) if trung_ten[i]))
    df = df.loc[:, ~trung_ten]  # 2020 có 2 cột "Thời gian": giữ cột HH:MM:SS
    log.info("      Kết quả: %s bản ghi (dòng Excel %d–%d) × %d cột", f"{len(df):,d}", df.index.min(),
             df.index.max(), df.shape[1])
    return df


# ---------------------------------------------------------------------------
# BƯỚC 5 — Tích hợp
# ---------------------------------------------------------------------------
def chuan_lucdo(df: pd.DataFrame, ma: str, ten_file: str) -> pd.DataFrame:
    """Chọn thuộc tính, đổi tên, đồng nhất kiểu. Cột không có ở nguồn này -> để trống (NA)."""
    can = {goc for goc, *_ in THUOC_TINH}
    lay = [(goc, moi) for goc, moi, *_ in THUOC_TINH if goc in df.columns]
    khong_co = [moi for goc, moi, *_ in THUOC_TINH if goc not in df.columns]
    bo = [c for c in df.columns if c not in can]
    log.info("    [%s] Chọn %d/%d cột của nguồn, đổi sang tên ngắn: %s", ma, len(lay), df.shape[1],
             "; ".join(f"'{g}' -> {m}" for g, m in lay))
    log.info("    [%s] Không thu thập %d cột: %s", ma, len(bo), "; ".join(f"'{c}'" for c in bo) or "(không có)")
    if khong_co:
        log.warning("    [%s] Nguồn không có %d thuộc tính -> tạo cột trống (NA), không suy diễn: %s", ma,
                    len(khong_co), ", ".join(khong_co))

    out = pd.DataFrame({moi: df[goc] if goc in df.columns else pd.NA for goc, moi, *_ in THUOC_TINH})
    co = out["ngay"].notna().sum()
    out["ngay"] = pd.to_datetime(out["ngay"], format="%d/%m/%Y", errors="coerce")
    hong = int(co - out["ngay"].notna().sum())
    log.log(logging.WARNING if hong else logging.INFO,
            "    [%s] ngay: ép sang kiểu ngày (dd/mm/yyyy) %s giá trị, %d giá trị không đọc được -> NA", ma,
            f"{int(co):,d}", hong)
    truoc = out["gio"].astype("string")
    out["gio"] = out["gio"].astype("string").str.strip().where(out["gio"].notna())
    log.info("    [%s] gio: giữ dạng chuỗi, cắt khoảng trắng thừa ở %d giá trị (chưa sửa định dạng)", ma,
             int((truoc != out["gio"]).fillna(False).sum()))
    mat = {}
    for c in SO_NGUYEN:
        co = out[c].notna().sum()
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if co - out[c].notna().sum():
            mat[c] = int(co - out[c].notna().sum())
    log.log(logging.WARNING if mat else logging.INFO,
            "    [%s] Ép %d cột sang kiểu số (%s); giá trị không phải số -> NA: %s", ma, len(SO_NGUYEN),
            ", ".join(SO_NGUYEN), ", ".join(f"{c}={v}" for c, v in mat.items()) or "0 giá trị")
    cot_chuoi = out.columns.difference(SO_NGUYEN + ["ngay", "gio"])
    da_cat = 0
    for c in cot_chuoi:
        truoc = out[c].astype("string")
        out[c] = truoc.str.strip()
        da_cat += int((truoc != out[c]).fillna(False).sum())
    log.info("    [%s] Ép %d cột phân loại/văn bản sang chuỗi, cắt khoảng trắng đầu/cuối ở %s giá trị"
             " (không gộp, không sửa nhãn)", ma, len(cot_chuoi), f"{da_cat:,d}")
    log.info("    [%s] Thêm 3 cột truy vết: nguon='%s', file_goc='%s', dong_goc=số dòng Excel", ma, ma, ten_file)
    out.insert(0, "nguon", ma)
    out.insert(1, "file_goc", ten_file)
    out.insert(2, "dong_goc", df.index)  # số dòng Excel trong sheet DSTN của file gốc
    return out.reset_index(drop=True)


def kiem_tra_tap_con(path: Path, tap_chinh: pd.DataFrame) -> list[str]:
    """Sheet TPHCM, QL1 (file 2021) có phải tập con của DSTN không -> nếu đúng thì không lấy thêm."""
    k = ["Ngày xảy ra tai nạn", "Thời gian xảy ra tai nạn", "Địa điểm xảy ra tai nạn | Địa điểm",
         "Số người chết và bị thương | Số người chết"]
    dong = []
    for sh in pd.ExcelFile(path).sheet_names:
        if sh == "DSTN":
            continue
        x = doc_sheet(path, sh)
        m = x[k].astype(str).merge(tap_chinh[k].astype(str).drop_duplicates(), how="left", indicator=True)
        khop = int((m["_merge"] == "both").sum())
        dong.append(f"  Sheet '{sh}': {len(x)} bản ghi, {khop} trùng với DSTN"
                    f" -> {'tập con, KHÔNG lấy thêm' if khop == len(x) else 'CÓ bản ghi mới, cần xem lại'}")
    return dong


# ---------------------------------------------------------------------------
# BƯỚC 6 — Kiểm tra & đánh giá
# ---------------------------------------------------------------------------
def kiem_tra(df: pd.DataFrame, cot_goc: dict[str, int], co_cot: dict[str, set[str]]) -> tuple[pd.DataFrame, list[str]]:
    bc = []
    p = bc.append
    p("\n[6.1] Số bản ghi theo nguồn")
    for ma, s, e in [(m, s, e) for m, _, s, e in NGUON]:
        x = df[df["nguon"] == ma]
        p(f"  {ma:7s}: {len(x):6,d} bản ghi | {cot_goc[ma]} cột gốc | ngày {x['ngay'].min():%d/%m/%Y}"
          f" – {x['ngay'].max():%d/%m/%Y} (kỳ báo cáo {pd.Timestamp(s):%d/%m/%Y} – {pd.Timestamp(e):%d/%m/%Y})")
    p(f"  TỔNG   : {len(df):6,d} bản ghi × {df.shape[1]} cột")

    p("\n[6.2] Kiểu dữ liệu sau tích hợp")
    for k, v in df.dtypes.astype(str).value_counts().items():
        p(f"  {k:15s}: {v} cột")

    p("\n[6.3] Dữ liệu thiếu (% ô trống) — chỉ tính trên nguồn CÓ cột đó")
    thieu = pd.DataFrame({ma: df[df["nguon"] == ma].isna().mean() * 100 for ma, *_ in NGUON}).round(1)
    for ma in thieu.columns:  # cột không tồn tại ở nguồn đó -> không tính là "thiếu"
        thieu.loc[~thieu.index.isin(co_cot[ma] | {"nguon", "file_goc", "dong_goc"}), ma] = None
    hoan_toan_trong = [c for c in thieu.index if (thieu.loc[c].dropna() >= 99.9).all() and thieu.loc[c].notna().any()]
    f = lambda v: "   –   " if pd.isna(v) else f"{v:5.1f}%"
    p(f"  {'thuộc tính':22s}   2020    2021  8T2022   ('–' = nguồn không có cột này)")
    for c in thieu.index[3:]:
        r = thieu.loc[c]
        p(f"  {c:22s} {f(r['2020'])} {f(r['2021'])} {f(r['8T2022'])}")
    nhieu = thieu[(thieu[["2020", "2021"]].mean(axis=1) > 50)].index.tolist()
    p(f"  -> {len(nhieu)} thuộc tính thiếu > 50%: {', '.join(nhieu)}")
    if hoan_toan_trong:
        p(f"  -> Trống hoàn toàn: {', '.join(hoan_toan_trong)}")

    p("\n[6.4] Trùng lặp")
    noi_dung = df.columns.drop(["nguon", "file_goc", "dong_goc"])
    trung_hoan_toan = df.duplicated(subset=noi_dung, keep="first")
    khoa = ["ngay", "gio", "tinh", "huyen", "dia_diem", "so_chet", "so_bi_thuong"]
    trung_khoa = df.duplicated(subset=khoa, keep="first") & df["gio"].notna()
    giua_nguon = df[df.duplicated(subset=khoa, keep=False) & df["gio"].notna()].groupby(khoa, dropna=False)[
        "nguon"].nunique().gt(1).sum()
    p(f"  Trùng hoàn toàn mọi thuộc tính      : {trung_hoan_toan.sum()} bản ghi")
    p(f"  Trùng theo khoá (ngày, giờ, tỉnh, huyện, địa điểm, số chết, số bị thương): {trung_khoa.sum()} bản ghi")
    p(f"  Nhóm trùng nằm ở 2 nguồn khác nhau  : {giua_nguon}")

    p("\n[6.5] Tính hợp lệ")
    ky = {m: (pd.Timestamp(s), pd.Timestamp(e)) for m, _, s, e in NGUON}
    ngoai_ky = df.apply(lambda r: pd.notna(r["ngay"]) and not ky[r["nguon"]][0] <= r["ngay"] <= ky[r["nguon"]][1],
                        axis=1)
    gio_sai = df["gio"].notna() & ~df["gio"].fillna("").str.fullmatch(r"([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?")
    am = (df[SO_NGUYEN] < 0).any(axis=1)
    lech_bt = (df["so_thuong_nang"].notna() & df["so_thuong_nhe"].notna()
               & (df["so_thuong_nang"] + df["so_thuong_nhe"] != df["so_bi_thuong"]))
    nhe_ma_chet = df["phan_loai"].isin(["Va chạm giao thông", "Ít nghiêm trọng"]) & (df["so_chet"] > 0)
    nang_ma_khong = (df["phan_loai"].isin(["Rất nghiêm trọng", "Đặc biệt nghiêm trọng"])
                     & (df["so_chet"].fillna(0) + df["so_bi_thuong"].fillna(0) == 0))
    gay_tn_sai = pd.concat([~df[f"pt{i}_gay_tn"].isin([0, 1]) & df[f"pt{i}_gay_tn"].notna() for i in range(1, 6)],
                           axis=1).any(axis=1)
    khong_ngay = df["ngay"].isna()
    for ten, mask in [("Thiếu / sai định dạng ngày", khong_ngay), ("Ngày nằm ngoài kỳ báo cáo", ngoai_ky),
                      ("Giờ sai định dạng", gio_sai), ("Giá trị số âm", am),
                      ("Bị thương ≠ thương nặng + thương nhẹ", lech_bt),
                      ("Phân loại nhẹ nhưng có người chết", nhe_ma_chet),
                      ("Phân loại rất/đặc biệt nghiêm trọng nhưng 0 thương vong", nang_ma_khong),
                      ("'Là PT gây TN' khác 0/1", gay_tn_sai)]:
        p(f"  {ten:58s}: {int(mask.sum())}")
    tinh = df["tinh"].dropna().unique()
    th = df["thiet_hai"]
    p(f"  Thiệt hại (triệu đồng): trung vị {th.median():g}, P99 {th.quantile(.99):g}, lớn nhất {th.max():,.0f}"
      f" | > 10.000 triệu (10 tỷ): {int((th > 10_000).sum())} bản ghi -> nghi nhập sai, xử lý ở bước tiền xử lý")
    p(f"  Số tỉnh/thành xuất hiện: {len(tinh)} (Việt Nam có 63 tỉnh/thành trước 07/2025)")
    p(f"  Giá trị 'Phân loại tai nạn': {df['phan_loai'].value_counts().to_dict()}")

    p("\n[6.6] Đối chiếu số liệu chính")
    t = df.groupby("nguon", sort=False).agg(vu=("ngay", "size"), chet=("so_chet", "sum"),
                                            bi_thuong=("so_bi_thuong", "sum"))
    for ma, r in t.iterrows():
        p(f"  {ma:7s}: {r['vu']:6,d} vụ | {int(r['chet']):5,d} người chết | {int(r['bi_thuong']):6,d} người bị thương")

    # Cờ đánh dấu (không xoá) để bước tiền xử lý quyết định
    co = pd.DataFrame({"co_trung_khoa": trung_khoa, "co_ngay_loi": khong_ngay | ngoai_ky,
                       "co_gio_loi": gio_sai, "co_lech_thuong_vong": lech_bt | nhe_ma_chet | nang_ma_khong})
    return co.astype(int), bc


# ---------------------------------------------------------------------------
# BƯỚC 7 — Lược đồ hai tầng: bảng lõi (mọi nguồn) + bảng mở rộng (nguồn có đủ thuộc tính)
# ---------------------------------------------------------------------------
KHOA = ["nguon", "dong_goc"]  # khoá chính của bang_loi, khoá ngoại của bang_mo_rong
CO_LOI = ["co_trung_khoa", "co_ngay_loi", "co_gio_loi"]  # cờ chỉ dùng thuộc tính chung
CO_MO_RONG = ["co_lech_thuong_vong"]  # cờ cần phân loại tai nạn / thương nặng, nhẹ


def tach_hai_tang(df: pd.DataFrame, co_cot: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Tách bảng phẳng thành 2 bảng, không mất và không sửa giá trị nào.

    bang_loi     : 1 dòng = 1 vụ, MỌI nguồn; chỉ gồm thuộc tính mà nguồn nào cũng có.
    bang_mo_rong : 1 dòng = 1 vụ của các nguồn có thuộc tính mở rộng; các thuộc tính còn lại.
    """
    chung = [t[1] for t in THUOC_TINH if all(t[1] in co_cot[m] for m, *_ in NGUON)]
    rieng = [t[1] for t in THUOC_TINH if t[1] not in chung]
    nguon_mr = [m for m, *_ in NGUON if co_cot[m] & set(rieng)]
    if df.duplicated(subset=KHOA).any():
        raise ValueError("Khoá (nguon, dong_goc) không duy nhất -> không tách được bảng")

    loi = df[["nguon", "file_goc", "dong_goc", *chung, *CO_LOI]].copy()
    mo_rong = df.loc[df["nguon"].isin(nguon_mr), [*KHOA, *rieng, *CO_MO_RONG]].reset_index(drop=True)

    # Kiểm chứng: nối 2 bảng lại phải ra đúng bảng phẳng ban đầu
    noi = loi.merge(mo_rong, on=KHOA, how="left", validate="one_to_one")[df.columns]
    ngoai = df.loc[~df["nguon"].isin(nguon_mr), CO_MO_RONG]
    noi[CO_MO_RONG] = noi[CO_MO_RONG].fillna(0).astype(df[CO_MO_RONG[0]].dtype)
    khop = noi.astype("string").fillna("<NA>").equals(df.astype("string").fillna("<NA>"))
    if not khop or int(ngoai.to_numpy().sum()) != 0:
        raise ValueError("Nối bang_loi + bang_mo_rong không tái tạo được bảng phẳng")

    dong = [
        f"    bang_loi    : {len(loi):,d} dòng × {loi.shape[1]} cột = 3 cột truy vết + {len(chung)} thuộc tính chung"
        f" + {len(CO_LOI)} cờ | nguồn: {', '.join(m for m, *_ in NGUON)}",
        f"                  thuộc tính: {', '.join(chung)}",
        f"    bang_mo_rong: {len(mo_rong):,d} dòng × {mo_rong.shape[1]} cột = 2 cột khoá + {len(rieng)} thuộc tính"
        f" + {len(CO_MO_RONG)} cờ | nguồn: {', '.join(nguon_mr)}",
        f"    Khoá nối (nguon, dong_goc): duy nhất trên {len(loi):,d} bản ghi;"
        f" nối 2 bảng tái tạo đúng bảng phẳng {df.shape[0]:,d} × {df.shape[1]}",
    ]
    return loi, mo_rong, dong


# ---------------------------------------------------------------------------
# Đặc trưng suy ra nhóm A — chỉ dùng thuộc tính có ở cả 3 nguồn, thêm vào bang_loi
# ---------------------------------------------------------------------------
KHUNG_GIO = [0, 4, 6, 9, 16, 19, 22, 24]  # mốc giờ; khung [a, b)
VUNG = {
    "Bắc": ["Hà Nội", "Hải Phòng", "Quảng Ninh", "Bắc Giang", "Bắc Kạn", "Bắc Ninh", "Cao Bằng", "Hà Giang",
            "Hà Nam", "Hòa Bình", "Hưng Yên", "Hải Dương", "Lai Châu", "Lào Cai", "Lạng Sơn", "Nam Định",
            "Ninh Bình", "Phú Thọ", "Sơn La", "Thái Bình", "Thái Nguyên", "Tuyên Quang", "Vĩnh Phúc", "Yên Bái",
            "Điện Biên"],
    "Duyên hải miền Trung": ["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Bình", "Quảng Trị", "Thừa Thiên Huế",
                             "Đà Nẵng", "Quảng Nam", "Quảng Ngãi", "Bình Định", "Phú Yên", "Khánh Hòa",
                             "Ninh Thuận", "Bình Thuận"],
    "Tây Nguyên": ["Kon Tum", "Gia Lai", "Đắk Lắk", "Đắk Nông", "Lâm Đồng"],
    "Nam Bộ": ["Hồ Chí Minh", "Bà Rịa - Vũng Tàu", "Bình Dương", "Bình Phước", "Đồng Nai", "Tây Ninh", "Long An",
               "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long", "Đồng Tháp", "An Giang", "Kiên Giang", "Cần Thơ",
               "Hậu Giang", "Sóc Trăng", "Bạc Liêu", "Cà Mau"],
}
MUA_MUA = {"Bắc": (5, 10), "Duyên hải miền Trung": (9, 12), "Tây Nguyên": (5, 10), "Nam Bộ": (5, 11)}  # tháng, xấp xỉ
NGHI_TET = [("2020-01-23", "2020-01-29"), ("2021-02-10", "2021-02-16"), ("2022-01-29", "2022-02-06")]  # đợt nghỉ Tết
GIAN_CACH = [("2020-04-01", "2020-04-22", "Giãn cách 04/2020"),  # mốc cấp quốc gia, xấp xỉ; không tách theo tỉnh
             ("2021-07-19", "2021-09-30", "Giãn cách 07-09/2021")]
DAC_TRUNG_A = [  # (tên cột, suy ra từ, quy tắc)
    ("gio_so", "gio", "giờ trong ngày 0–23 = số đứng trước dấu ':' đầu tiên; trống nếu gio trống"),
    ("khung_gio", "gio_so", "00-04, 04-06, 06-09, 09-16, 16-19, 19-22, 22-24 (gồm mốc đầu, không gồm mốc cuối)"),
    ("thu", "ngay", "thứ trong tuần: 1 = Thứ Hai … 7 = Chủ nhật"),
    ("cuoi_tuan", "ngay", "1 nếu Thứ Bảy hoặc Chủ nhật"),
    ("thang", "ngay", "tháng dương lịch 1–12"),
    ("vung", "tinh", "Bắc / Duyên hải miền Trung / Tây Nguyên / Nam Bộ"),
    ("mua", "vung, thang", "Mùa mưa nếu tháng thuộc: Bắc 5–10, DH miền Trung 9–12, Tây Nguyên 5–10, Nam Bộ 5–11;"
                           " còn lại Mùa khô (xấp xỉ)"),
    ("dip_tet", "ngay", "Trong đợt nghỉ / Trước Tết 7 ngày / Sau Tết 7 ngày / Không; đợt nghỉ: 23–29/01/2020,"
                        " 10–16/02/2021, 29/01–06/02/2022"),
    ("giai_doan_covid", "ngay", "Giãn cách 04/2020 (01–22/04/2020) / Giãn cách 07-09/2021 (19/07–30/09/2021)"
                                " / Không — mốc cấp quốc gia, xấp xỉ"),
    ("cap_duong", "loai_duong", "phần trước ' - ': Cao tốc, Quốc lộ, Đường tỉnh, Đường huyện, Đường xã, Khác"),
    ("khu_vuc", "loai_duong", "phần sau ' - ': Nội thị / Nông thôn; trống nếu nguồn không ghi"),
    ("co_nguoi_chet", "so_chet", "1 nếu so_chet > 0"),
]


def rut_dac_trung_chung(loi: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Thêm đặc trưng nhóm A vào bang_loi. Không sửa cột gốc; giá trị gốc trống thì đặc trưng cũng trống."""
    x = loi.copy()
    gio = pd.to_numeric(x["gio"].str.extract(r"^(\d{1,2}):", expand=False), errors="coerce")
    x["gio_so"] = gio.where(gio.between(0, 23)).astype("Int64")
    nhan = [f"{a:02d}-{b:02d}" for a, b in zip(KHUNG_GIO, KHUNG_GIO[1:])]
    x["khung_gio"] = pd.cut(x["gio_so"].astype("float"), bins=KHUNG_GIO, right=False, labels=nhan).astype("string")
    x["thu"] = (x["ngay"].dt.dayofweek + 1).astype("Int64")
    x["cuoi_tuan"] = (x["thu"] >= 6).astype("Int64")
    x["thang"] = x["ngay"].dt.month.astype("Int64")

    chuan = lambda t: unicodedata.normalize("NFC", str(t)).strip()
    tinh_vung = {chuan(t): v for v, ds in VUNG.items() for t in ds}
    x["vung"] = pd.array([tinh_vung.get(chuan(t), pd.NA) if pd.notna(t) else pd.NA for t in x["tinh"]],
                         dtype="string")
    la = sorted({chuan(t) for t in x["tinh"].dropna()} - set(tinh_vung))
    x["mua"] = pd.array(
        [pd.NA if pd.isna(v) or pd.isna(m) else ("Mùa mưa" if MUA_MUA[v][0] <= m <= MUA_MUA[v][1] else "Mùa khô")
         for v, m in zip(x["vung"], x["thang"])], dtype="string")

    ngay, mot_tuan = x["ngay"], pd.Timedelta(days=7)
    tet = pd.Series("Không", index=x.index, dtype="string").where(ngay.notna())
    for a, b in NGHI_TET:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        tet[(ngay >= a - mot_tuan) & (ngay < a)] = "Trước Tết 7 ngày"
        tet[(ngay >= a) & (ngay <= b)] = "Trong đợt nghỉ"
        tet[(ngay > b) & (ngay <= b + mot_tuan)] = "Sau Tết 7 ngày"
    x["dip_tet"] = tet
    covid = pd.Series("Không", index=x.index, dtype="string").where(ngay.notna())
    for a, b, ten in GIAN_CACH:
        covid[(ngay >= pd.Timestamp(a)) & (ngay <= pd.Timestamp(b))] = ten
    x["giai_doan_covid"] = covid

    tach = x["loai_duong"].str.split(" - ", n=1)
    x["cap_duong"] = tach.str[0].astype("string")
    x["khu_vuc"] = tach.str[1].astype("string")
    x["co_nguoi_chet"] = (x["so_chet"] > 0).astype("Int64").where(x["so_chet"].notna())

    dong = [f"    Thêm {len(DAC_TRUNG_A)} đặc trưng suy ra vào bang_loi -> {x.shape[0]:,d} dòng × {x.shape[1]} cột"
            " (cột gốc giữ nguyên):"]
    for ten, tu, quy_tac in DAC_TRUNG_A:
        dem = x[ten].value_counts(dropna=True).sort_index()
        trong = int(x[ten].isna().sum())
        dong.append(f"    - {ten} (từ {tu}): {quy_tac}")
        dong.append(f"        phân bố: {'; '.join(f'{k}={v:,d}' for k, v in dem.items())}"
                    + (f"; (trống)={trong:,d}" if trong else ""))
    if la:
        dong.append(f"    CẢNH BÁO: {len(la)} tên tỉnh không có trong bảng vùng -> vung, mua để trống: {', '.join(la)}")
    return x, dong


def luu_bang(df: pd.DataFrame, ten: str) -> None:
    df.to_csv(GOC / f"{ten}.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(df, "thu_thap", ten)                      # PostgreSQL: thu_thap.<ten>


# ---------------------------------------------------------------------------
def main() -> None:
    bc = ["BÁO CÁO KIỂM TRA DỮ LIỆU THU THẬP", "=" * 70]
    log.info("[4] Thu thập ...")
    tai_du_lieu()

    phan, cot_goc, co_cot = [], {}, {}
    doi_ten = {goc: moi for goc, moi, *_ in THUOC_TINH}
    for ma, ten, _, _ in NGUON:
        path = THU_MUC_GOC / ten
        t0 = time.perf_counter()
        log.info("    --- Nguồn %s ---", ma)
        goc = doc_sheet(path)
        cot_goc[ma] = goc.shape[1]
        co_cot[ma] = {doi_ten[c] for c in goc.columns if c in doi_ten}
        bc.append(f"[4] Đọc {ten}: sheet DSTN, {len(goc):,d} dòng × {goc.shape[1]} cột")
        if ma == "2021":
            log.info("    Đối chiếu các sheet phụ với sheet DSTN theo (ngày, giờ, địa điểm, số chết):")
            tap_con = kiem_tra_tap_con(path, goc)
            for d in tap_con:
                log.info("    %s", d)
            bc += tap_con
        phan.append(chuan_lucdo(goc, ma, ten))
        log.info("    Xong nguồn %s: %s bản ghi, %d/%d thuộc tính (%.1f giây)", ma, f"{len(goc):,d}",
                 len(co_cot[ma]), len(THUOC_TINH), time.perf_counter() - t0)

    log.info("[5] Tích hợp ...")
    df = pd.concat(phan, ignore_index=True)
    log.info("    Nối dọc %d nguồn (%s) -> %s bản ghi × %d cột; không xoá, không sửa bản ghi nào",
             len(phan), " + ".join(f"{len(x):,d}" for x in phan), f"{len(df):,d}", df.shape[1])
    bc.append(f"[5] Tích hợp: {len(THUOC_TINH)} thuộc tính chung + 3 cột truy vết (nguon, file_goc, dong_goc)"
              f" -> {len(df):,d} bản ghi")
    for ma, *_ in NGUON:
        bc.append(f"    {ma}: có {len(co_cot[ma])}/{len(THUOC_TINH)} thuộc tính"
                  + (f"; thiếu: {', '.join(t[1] for t in THUOC_TINH if t[1] not in co_cot[ma])}"
                     if len(co_cot[ma]) < len(THUOC_TINH) and len(co_cot[ma]) > 20 else ""))

    log.info("[6] Kiểm tra ...")
    co, dong = kiem_tra(df, cot_goc, co_cot)
    bc += dong
    for d in dong:
        if d.strip():
            log.info("    %s", d.strip("\n"))
    df = pd.concat([df, co], axis=1)
    log.info("    Thêm 4 cột cờ (1 = nghi vấn, chỉ đánh dấu, không xoá): %s | có ít nhất một cờ: %d",
             ", ".join(f"{c}={int(v)}" for c, v in co.sum().items()), int(co.any(axis=1).sum()))

    log.info("[7] Lưu trữ ...")
    luu_bang(df, "du_lieu_tich_hop")
    log.info("    Tách lược đồ hai tầng (không xoá, không sửa giá trị):")
    loi, mo_rong, dong_tach = tach_hai_tang(df, co_cot)
    for d in dong_tach:
        log.info("%s", d)
    loi, dong_dt = rut_dac_trung_chung(loi)
    for d in dong_dt:
        (log.warning if "CẢNH BÁO" in d else log.info)("%s", d)
    luu_bang(loi, "bang_loi")
    luu_bang(mo_rong, "bang_mo_rong")
    tu_dien = pd.DataFrame(THUOC_TINH, columns=["ten_cot_goc", "ten_cot", "nhom", "kieu", "mo_ta"])
    nguon_co = {c: [m for m, *_ in NGUON if c in co_cot[m]] for c in tu_dien["ten_cot"]}
    tu_dien["co_o_nguon"] = tu_dien["ten_cot"].map(lambda c: ", ".join(nguon_co[c]))
    # Như bước 6.3: chỉ tính thiếu trên các nguồn CÓ cột đó (2022 không có cột -> không tính là thiếu)
    tu_dien["ty_le_thieu_%"] = tu_dien["ten_cot"].map(
        lambda c: round(df.loc[df["nguon"].isin(nguon_co[c]), c].isna().mean() * 100, 1))
    tu_dien["bang"] = tu_dien["ten_cot"].map(lambda c: "bang_loi" if c in loi.columns else "bang_mo_rong")
    tu_dien.to_csv(GOC / "tu_dien_du_lieu.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(tu_dien, "thu_thap", "tu_dien_du_lieu")
    bc.append(f"\n[7] Đã lưu: du_lieu_tich_hop.csv, tu_dien_du_lieu.csv, kiem_tra.txt + bảng thu_thap.* trong {csdl.mo_ta()}"
              f" ({df.shape[0]:,d} dòng × {df.shape[1]} cột)")
    bc.append("[7] Lược đồ hai tầng (bang_loi + bang_mo_rong, mỗi bảng có file .csv và bảng thu_thap.<tên> trong CSDL):")
    bc += dong_tach
    bc.append("[7] Đặc trưng suy ra (nhóm A — chỉ từ thuộc tính có ở cả 3 nguồn):")
    bc += dong_dt
    (GOC / "kiem_tra.txt").write_text("\n".join(bc), encoding="utf-8")
    log.info("    Dữ liệu tích hợp: %s dòng × %d cột (CSV UTF-8 có BOM + PostgreSQL); từ điển: %d thuộc tính",
             f"{df.shape[0]:,d}", df.shape[1], len(tu_dien))
    for ten in ("du_lieu_tich_hop.csv", "bang_loi.csv", "bang_mo_rong.csv", "tu_dien_du_lieu.csv", "kiem_tra.txt"):
        ghi_file_da_luu(GOC / ten)
    log.info("    CSDL %s: thu_thap.du_lieu_tich_hop, thu_thap.bang_loi, thu_thap.bang_mo_rong, "
             "thu_thap.tu_dien_du_lieu", csdl.mo_ta())


if __name__ == "__main__":
    cai_dat_nhat_ky()
    bat_dau = time.perf_counter()
    log.info("=" * 70)
    log.info("BẮT ĐẦU thu_thap.py | Python %s | pandas %s | %s", platform.python_version(), pd.__version__,
             platform.platform())
    try:
        main()
    except Exception:
        log.exception("KẾT THÚC: LỖI sau %.1f giây", time.perf_counter() - bat_dau)
        sys.exit(1)
    log.info("KẾT THÚC: thành công sau %.1f giây", time.perf_counter() - bat_dau)

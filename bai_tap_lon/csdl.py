"""CƠ SỞ DỮ LIỆU DÙNG CHUNG — mọi script của dự án đọc / ghi bảng qua module này (thay cho file Parquet).

Dữ liệu nằm trong PostgreSQL, mỗi giai đoạn một schema:
    thu_thap   : du_lieu_tich_hop, bang_loi, bang_mo_rong, tu_dien_du_lieu
    tien_xu_ly : tai_nan_loi_sach, tai_nan_mo_rong_sach, tu_dien_sach, train, test, train_fe, test_fe,
                 train_fe_khong_dia_ly, test_fe_khong_dia_ly, giao_dich, ho_so_tinh
    mo_hinh    : ket_qua_phan_loai, ho_so_tinh_cum, bang_so_vu_theo_o, ket_qua_hoi_quy, bang_thoi_diem_den
Khoá chính, khoá ngoại và chỉ mục được tạo lại mỗi lần ghi (xem KHOA_CHINH, KHOA_NGOAI, CHI_MUC).
Bảng phụ <schema>._kieu_cot lưu kiểu pandas của từng cột để doc() trả về đúng kiểu như lúc ghi.

Kết nối: biến môi trường PDM_CSDL (chuỗi kết nối SQLAlchemy), mặc định
    postgresql+psycopg://postgres@localhost:5432/tngt
Mật khẩu KHÔNG ghi trong code: đặt biến môi trường PGPASSWORD (thư viện psycopg tự đọc).
Chạy thử khi chưa cài PostgreSQL: đặt PDM_CSDL=sqlite:///D:/PDM/tngt.db (không có khoá, schema thành tiền tố tên bảng).

Cài thư viện: .venv/Scripts/python -m pip install sqlalchemy "psycopg[binary]"
Kiểm tra kết nối: .venv/Scripts/python bai_tap_lon/csdl.py
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

URL = os.environ.get("PDM_CSDL", "postgresql+psycopg://postgres@localhost:5432/tngt")
LA_SQLITE = URL.startswith("sqlite:///")

# Thư mục của từng bước -> schema tương ứng
SCHEMA_THEO_THU_MUC = {"1_thu_thap_du_lieu": "thu_thap", "2_tien_xu_ly": "tien_xu_ly", "3_mo_hinh": "mo_hinh"}
KHOA_VU = ["nguon", "dong_goc"]

# (schema, bảng) -> cột khoá chính
KHOA_CHINH: dict[tuple[str, str], list[str]] = {
    **{("thu_thap", t): KHOA_VU for t in ("du_lieu_tich_hop", "bang_loi", "bang_mo_rong")},
    ("thu_thap", "tu_dien_du_lieu"): ["ten_cot"],
    **{("tien_xu_ly", t): KHOA_VU for t in ("tai_nan_loi_sach", "tai_nan_mo_rong_sach", "train", "test", "train_fe",
                                            "test_fe", "train_fe_khong_dia_ly", "test_fe_khong_dia_ly", "giao_dich")},
    ("tien_xu_ly", "tu_dien_sach"): ["bang", "ten_cot"],
    ("tien_xu_ly", "ho_so_tinh"): ["tinh"],
    ("mo_hinh", "ho_so_tinh_cum"): ["tinh"],
    ("mo_hinh", "ket_qua_phan_loai"): ["huong", "mo_hinh"],
    ("mo_hinh", "ket_qua_hoi_quy"): ["mo_hinh"],
    ("mo_hinh", "bang_so_vu_theo_o"): ["ky", "vung", "khung_gio", "thu"],
    ("mo_hinh", "bang_thoi_diem_den"): ["khung_gio", "thu"],
}
# (schema, bảng) -> (cột, bảng được tham chiếu): mỗi vụ của bảng con phải có trong bảng cha
KHOA_NGOAI: dict[tuple[str, str], tuple[list[str], str]] = {
    ("thu_thap", "bang_mo_rong"): (KHOA_VU, "thu_thap.bang_loi"),
    ("tien_xu_ly", "tai_nan_mo_rong_sach"): (KHOA_VU, "tien_xu_ly.tai_nan_loi_sach"),
    **{("tien_xu_ly", t): (KHOA_VU, "tien_xu_ly.tai_nan_loi_sach")
       for t in ("train", "test", "train_fe", "test_fe", "train_fe_khong_dia_ly", "test_fe_khong_dia_ly", "giao_dich")},
}
CHI_MUC: dict[tuple[str, str], list[str]] = {
    ("thu_thap", "bang_loi"): ["tinh", "ngay"],
    ("tien_xu_ly", "tai_nan_loi_sach"): ["tinh", "ngay", "khung_gio"],
}

_ENGINE = None


def schema_cua(noi: str | Path) -> str:
    """Nhận tên schema hoặc thư mục của một bước (vd .../2_tien_xu_ly) và trả về schema."""
    ten = Path(str(noi)).name                     # "D:/PDM/.../2_tien_xu_ly" -> "2_tien_xu_ly"; "mo_hinh" giữ nguyên
    return SCHEMA_THEO_THU_MUC.get(ten, ten)


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from sqlalchemy import create_engine
        _ENGINE = create_engine(URL)
    return _ENGINE


def _sqlite():
    return sqlite3.connect(URL.removeprefix("sqlite:///"))


def _q(ten: str) -> str:
    return '"' + ten.replace('"', '""') + '"'


def _ten_day_du(schema: str, ten: str) -> str:
    return f"{_q(schema)}.{_q(ten)}"


def _chep_copy(bang, cn, cot, dong) -> None:
    """Cách nạp dữ liệu cho DataFrame.to_sql: dùng lệnh COPY của PostgreSQL (nhanh hơn INSERT nhiều lần,
    và tên cột có dấu, khoảng trắng, '=' hay '%' không bị coi là tham số)."""
    ten = f"{_q(bang.schema)}.{_q(bang.name)}" if bang.schema else _q(bang.name)
    with cn.connection.cursor() as cur, cur.copy(f"COPY {ten} ({', '.join(map(_q, cot))}) FROM STDIN") as cp:
        for d in dong:
            cp.write_row(d)


def ghi(df: pd.DataFrame, noi: str | Path, ten: str) -> None:
    """Ghi đè bảng schema.ten bằng df, rồi tạo khoá chính, khoá ngoại, chỉ mục nếu có khai báo."""
    schema = schema_cua(noi)
    df = df.reset_index(drop=True)
    kieu = _bang_kieu(df, ten)
    if LA_SQLITE:
        with _sqlite() as cn:
            df.to_sql(f"{schema}__{ten}", cn, if_exists="replace", index=False)
            bang_k = f"{schema}__{BANG_KIEU}"
            cn.execute(f"CREATE TABLE IF NOT EXISTS {_q(bang_k)} (bang TEXT, cot TEXT, kieu TEXT)")
            cn.execute(f"DELETE FROM {_q(bang_k)} WHERE bang = ?", (ten,))
            kieu.to_sql(bang_k, cn, if_exists="append", index=False)
        return
    from sqlalchemy import text
    with _engine().begin() as cn:
        cn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {_q(schema)}"))
        cn.execute(text(f"DROP TABLE IF EXISTS {_ten_day_du(schema, ten)} CASCADE"))
        df.to_sql(ten, cn, schema=schema, index=False, method=_chep_copy)
        bang = _ten_day_du(schema, ten)
        if (khoa := KHOA_CHINH.get((schema, ten))) and all(c in df.columns for c in khoa):
            cn.execute(text(f"ALTER TABLE {bang} ADD PRIMARY KEY ({', '.join(map(_q, khoa))})"))
        if (kn := KHOA_NGOAI.get((schema, ten))) is not None:
            cot, cha = kn
            s_cha, t_cha = cha.split(".")
            kieu_cot = text("SELECT string_agg(data_type, ',' ORDER BY column_name) FROM information_schema.columns "
                            "WHERE table_schema = :s AND table_name = :t AND column_name = ANY(:c)")
            k_con = cn.execute(kieu_cot, {"s": schema, "t": ten, "c": cot}).scalar()
            k_cha = cn.execute(kieu_cot, {"s": s_cha, "t": t_cha, "c": cot}).scalar()
            if k_cha is not None and k_con != k_cha:
                print(f"    [csdl] bỏ qua khoá ngoại {schema}.{ten} -> {cha}: kiểu cột khoá khác nhau ({k_con} / {k_cha})")
            elif k_cha is not None:
                cn.execute(text(f"ALTER TABLE {bang} ADD FOREIGN KEY ({', '.join(map(_q, cot))}) "
                                f"REFERENCES {_ten_day_du(s_cha, t_cha)} ({', '.join(map(_q, cot))})"))
        for c in CHI_MUC.get((schema, ten), []):
            cn.execute(text(f"CREATE INDEX ON {bang} ({_q(c)})"))
        bang_k = _ten_day_du(schema, BANG_KIEU)
        cn.execute(text(f"CREATE TABLE IF NOT EXISTS {bang_k} (bang TEXT, cot TEXT, kieu TEXT, PRIMARY KEY (bang, cot))"))
        cn.execute(text(f"DELETE FROM {bang_k} WHERE bang = :t"), {"t": ten})
        kieu.to_sql(BANG_KIEU, cn, schema=schema, if_exists="append", index=False, method=_chep_copy)


BANG_KIEU = "_kieu_cot"      # bảng phụ trong mỗi schema: kiểu pandas của từng cột, để đọc ra đúng như lúc ghi


def _bang_kieu(df: pd.DataFrame, ten: str) -> pd.DataFrame:
    return pd.DataFrame({"bang": ten, "cot": df.columns, "kieu": [str(k) for k in df.dtypes]})


def _tra_kieu(df: pd.DataFrame, kieu: pd.DataFrame) -> pd.DataFrame:
    """Đưa từng cột về đúng kiểu pandas đã ghi (số nguyên có ô trống, chuỗi, ngày, đúng/sai…)."""
    for c, k in zip(kieu["cot"], kieu["kieu"]):
        if c not in df.columns or str(df[c].dtype) == k:
            continue
        try:
            if k.startswith("datetime64"):
                df[c] = pd.to_datetime(df[c], errors="coerce").astype(k)
            elif k == "bool":
                df[c] = df[c].astype(bool)
            elif k == "object":
                continue
            else:
                df[c] = df[c].astype(k)
        except (TypeError, ValueError):
            pass                                           # giữ kiểu đọc được nếu không đổi được
    return df


def doc(noi: str | Path, ten: str) -> pd.DataFrame:
    """Đọc toàn bộ bảng schema.ten."""
    schema = schema_cua(noi)
    if LA_SQLITE:
        with _sqlite() as cn:
            df = pd.read_sql(f"SELECT * FROM {_q(schema + '__' + ten)}", cn)
            kieu = pd.read_sql(f"SELECT cot, kieu FROM {_q(schema + '__' + BANG_KIEU)} WHERE bang = ?", cn,
                               params=(ten,))
        return _tra_kieu(df, kieu)
    from sqlalchemy import text
    with _engine().connect() as cn:
        df = pd.read_sql_table(ten, cn, schema=schema)
        kieu = pd.read_sql(text(f"SELECT cot, kieu FROM {_ten_day_du(schema, BANG_KIEU)} WHERE bang = :t"), cn,
                           params={"t": ten})
    return _tra_kieu(df, kieu)


def co_bang(noi: str | Path, ten: str) -> bool:
    schema = schema_cua(noi)
    try:
        if LA_SQLITE:
            with _sqlite() as cn:
                return cn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                  (f"{schema}__{ten}",)).fetchone() is not None
        from sqlalchemy import inspect
        return inspect(_engine()).has_table(ten, schema=schema)
    except Exception:
        return False


def truy_van(cau_lenh: str) -> pd.DataFrame:
    """Chạy một câu SELECT và trả về DataFrame."""
    if LA_SQLITE:
        with _sqlite() as cn:
            return pd.read_sql(cau_lenh, cn)
    from sqlalchemy import text
    with _engine().connect() as cn:
        return pd.read_sql(text(cau_lenh), cn)


def mo_ta() -> str:
    """Chuỗi kết nối đã che mật khẩu, để in vào báo cáo."""
    if LA_SQLITE:
        return URL
    from sqlalchemy.engine import make_url
    return make_url(URL).render_as_string(hide_password=True)


if __name__ == "__main__":
    print(f"Kết nối: {mo_ta()}")
    if LA_SQLITE:
        print(truy_van("SELECT name AS bang FROM sqlite_master WHERE type='table' AND name NOT LIKE '%\\_kieu\\_cot' ESCAPE '\\' ORDER BY name").to_string(index=False))
    else:
        print(truy_van("SELECT version()").iloc[0, 0])
        print(truy_van("""SELECT table_schema AS schema, table_name AS bang
                          FROM information_schema.tables
                          WHERE table_schema IN ('thu_thap', 'tien_xu_ly', 'mo_hinh') AND table_name <> '_kieu_cot'
                          ORDER BY 1, 2""").to_string(index=False))

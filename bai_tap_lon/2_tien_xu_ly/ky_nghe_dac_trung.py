"""KỸ NGHỆ ĐẶC TRƯNG (Feature Engineering) — Tai nạn giao thông đường bộ Việt Nam (2020 – 8 tháng 2022)

Giai đoạn: 2 – Tiền xử lý (phần kỹ nghệ đặc trưng, chạy sau bien_doi.py trong cùng thư mục)
Đầu vào  : bảng tien_xu_ly.tai_nan_loi_sach (tien_xu_ly.py) — chỉ BẢNG LÕI
           bảng tien_xu_ly.train / tien_xu_ly.test (bien_doi.py) — chỉ lấy khoá để giữ ĐÚNG cách chia train/test
Đầu ra   : train_fe / test_fe       hướng "du_bao": bộ đặc trưng dự báo tốt nhất (có mã hoá địa lý) + nhãn
           train_fe_khong_dia_ly / test_fe_khong_dia_ly   hướng giải thích: bỏ hẳn tỉnh/huyện/đường
           bo_dac_trung.joblib      bộ biến đổi đã học trên train (bước mô hình dùng lại)
           kiem_tra_ky_nghe_dac_trung.txt (cũng in ra màn hình)

Các thao tác:
    1. Tạo đặc trưng mới theo từng dòng (không học từ dữ liệu -> không rò rỉ):
         - từ văn bản địa điểm: giao lộ, cầu, đèo/dốc, lý trình km, số nhà, nơi đông người
         - từ ngày: dịp lễ (1/1, 30/4, 1/5, 2/9 ± 2 ngày)
         - tương tác: đêm, đêm × quốc lộ/cao tốc, đêm × cuối tuần
         - khoá địa lý chuẩn hoá: tỉnh|huyện, tỉnh|đường (gộp "QL 1A", "Quốc lộ 1A", "Quoc lo 1A" -> ql1a)
    2. Mã hoá có học (chỉ học trên phần train của từng lần kiểm định):
         - Target Encoding cho tỉnh / huyện / đường (sklearn TargetEncoder, tự chia chéo bên trong)
         - Mã hoá tần suất: số vụ cùng huyện / cùng đường trong tập học (lớp BoDemTanSuat)
    3. So sánh 4 bộ đặc trưng bằng kiểm định chéo 5 lần trên TRAIN (không đụng test), thêm kiểm định theo thời gian
       trong train (học 2020–2021, chấm phần 8T2022 của train)
    4. Chọn lọc đặc trưng: tráo từng nhóm cột (permutation) trên 3 cách chia × 10 lần, bỏ nhóm không đóng góp rõ
    5. Với từng hướng: học bộ biến đổi cuối trên toàn bộ train, áp dụng cho test, lưu

Chạy (từ D:/PDM, sau bien_doi.py):
       .venv/Scripts/python bai_tap_lon/2_tien_xu_ly/ky_nghe_dac_trung.py
Khi nạp lại bo_dac_trung.joblib ở bước sau cần import được lớp BoDemTanSuat từ file này.
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
VAO = GOC                       # cùng thư mục với tien_xu_ly.py
BC: list[str] = ["BÁO CÁO KỸ NGHỆ ĐẶC TRƯNG", "=" * 70]

KHOA = ["nguon", "dong_goc"]
NHAN = "co_nguoi_chet"
KHONG_RO = "Không rõ"
HAT_GIONG, SO_FOLD = 42, 5
NGUONG_BO = 0.0005          # nhóm cột bị bỏ nếu AUC giảm trung bình không vượt quá max(2 × sai số chuẩn, mức này);
                            # 0,0005 AUC = mức đóng góp nhỏ nhất coi là có ý nghĩa thực tế
SO_CACH_CHIA, SO_LAN_TRAO = 3, 10   # tráo cột trên 3 cách chia 80/20 khác nhau × 10 lần mỗi cách -> 30 phép đo
NGUON_THU_TG = "8T2022"     # kiểm định theo thời gian TRONG train: học 2020–2021, thử trên phần 8T2022 của train
DUNG_SAI_AUC = 0.002        # bản rút gọn được chấp nhận nếu AUC không thấp hơn bản đầy đủ quá mức này

DANH_NGHIA = ["khung_gio", "vung", "mua", "dip_tet", "giai_doan_covid", "cap_duong", "khu_vuc"]
CHU_KY = {"gio_so": 24, "thang": 12, "thu": 7}
SO_NEN = [f"{c}_{h}" for c in CHU_KY for h in ("sin", "cos")] + ["gio_thieu", "cuoi_tuan"]
CO_MOI = ["giao_lo", "cau", "deo_doc", "ly_trinh", "so_nha", "noi_dong_nguoi",
          "dip_le", "dem", "dem_x_quoc_lo", "dem_x_cuoi_tuan"]
MA_HOA_MUC_TIEU = ["tinh", "khoa_huyen", "khoa_duong"]
TAN_SUAT = ["khoa_huyen", "khoa_duong"]

MAU = {   # tìm trong phần địa điểm cụ thể (đã bỏ xã/huyện/tỉnh ở cuối) + tên đường + km, chữ thường
    "giao_lo": r"ngã (?:ba|tư|năm|sáu|3|4|5)|giao lộ|giao cắt|giao nhau|vòng xoay|bùng binh|nút giao",
    "cau": r"(?<!\w)cầu(?!\w)",
    "deo_doc": r"(?<!\w)(?:đèo|dốc)(?!\w)",
    "ly_trinh": r"km\s*\d|\d+\s*\+\s*\d{2,3}",
    "so_nha": r"(?:^|,\s*)(?:số nhà\s*|số\s*)?\d+[a-z]?(?:/\d+[a-z]?)*\s+(?!\+)[^\d\s+,]",
    "noi_dong_nguoi": r"chợ|trường|bến xe|khu công nghiệp|kcn|bệnh viện|siêu thị",
}
NGAY_LE = [(1, 1), (4, 30), (5, 1), (9, 2)]   # (tháng, ngày) dương lịch
CUA_SO_LE = 2

BO_DAC_TRUNG = {
    "nen": "bien_doi.py: one-hot (có tỉnh) + sin/cos",
    "nen_moi": "nền + 10 đặc trưng mới theo dòng",
    "ma_hoa_dia_ly": "nền_mới, nhưng tỉnh one-hot -> Target Encoding tỉnh/huyện/đường + tần suất huyện/đường",
    "khong_dia_ly": "nền_mới bỏ hẳn tỉnh (đo phần dự báo KHÔNG nhờ thói quen báo cáo của tỉnh)",
}


def p(dong: str = "") -> None:
    print(dong)
    BC.append(dong)


def doc_bang(thu_muc: Path, ten: str) -> pd.DataFrame:
    return csdl.doc(thu_muc, ten)               # thư mục 2_tien_xu_ly -> schema tien_xu_ly


def luu_bang(df: pd.DataFrame, ten: str) -> None:
    df.to_csv(GOC / f"{ten}.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(df, "tien_xu_ly", ten)


# ---------------------------------------------------------------------------
class BoDemTanSuat(BaseEstimator, TransformerMixin):
    """Mã hoá tần suất: thay mỗi giá trị bằng log(1 + số vụ có cùng giá trị trong TẬP HỌC).
    Giá trị trống hoặc chưa gặp khi học -> 0."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cot_ = list(X.columns)
        self.dem_ = {c: X[c].value_counts() for c in self.cot_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cot_)
        return np.column_stack(
            [np.log1p(X[c].map(self.dem_[c]).astype(float).fillna(0.0)) for c in self.cot_])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"tan_suat_{c}" for c in self.cot_], dtype=object)


def bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def chuan_ten_duong(s) -> str | float:
    if pd.isna(s) or not str(s).strip():
        return np.nan
    t = re.sub(r"\s+", " ", bo_dau(str(s))).strip()
    m = re.search(r"\b(?:quoc lo|ql)\s*\.?\s*(\d+[a-z]?)\b", t)
    if m:
        return "ql" + m.group(1)
    m = re.search(r"\b(?:tinh lo|duong tinh|dt|tl)\s*\.?\s*(\d+[a-z]?)\b", t)
    if m:
        return "dt" + m.group(1)
    return re.sub(r"^(duong|pho)\s+", "", t)


def vi_tri_cu_the(s) -> str:
    """'Ngã Tư Cầu Ông Bố, Bình Hòa, Thuận An, Bình Dương' -> 'Ngã Tư Cầu Ông Bố' (bỏ xã, huyện, tỉnh ở cuối)."""
    if pd.isna(s):
        return ""
    phan = [x.strip() for x in str(s).split(",")]
    return ", ".join(phan[:-3]) if len(phan) > 3 else ""


def doi_tuong(s: pd.Series) -> pd.Series:
    """Cột phân loại -> kiểu object với ô trống là np.nan (định dạng scikit-learn đọc ổn định)."""
    return s.astype(object).where(s.notna(), np.nan)


def tao_khung(df: pd.DataFrame) -> pd.DataFrame:
    """Mọi đặc trưng tính theo từng dòng, không học từ dữ liệu."""
    x = df[KHOA].copy()
    x["dong_goc"] = x["dong_goc"].astype(str)
    for c in DANH_NGHIA:
        x[c] = doi_tuong(df[c].astype("string").fillna(KHONG_RO))
    for c, chu_ky in CHU_KY.items():
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        if c in ("thang", "thu"):
            v = v - 1
        goc = 2 * np.pi * v / chu_ky
        x[f"{c}_sin"], x[f"{c}_cos"] = np.sin(goc).fillna(0.0), np.cos(goc).fillna(0.0)
    gio = pd.to_numeric(df["gio_so"], errors="coerce")
    x["gio_thieu"] = gio.isna().astype(int)
    x["cuoi_tuan"] = pd.to_numeric(df["cuoi_tuan"], errors="coerce").fillna(0).astype(int)

    # văn bản địa điểm
    van_ban = (df["dia_diem"].map(vi_tri_cu_the) + " | " + df["ten_duong"].fillna("").astype(str)
               + " | " + df["km"].fillna("").astype(str)).map(lambda s: unicodedata.normalize("NFC", s.lower()))
    for ten, mau in MAU.items():
        x[ten] = van_ban.str.contains(mau, regex=True).astype(int)

    # dịp lễ dương lịch
    ngay = pd.to_datetime(df["ngay"], errors="coerce")       # tien_xu_ly.py đã lưu dạng ngày chuẩn (yyyy-mm-dd)
    cua_so = {pd.Timestamp(nam, th, ng) + pd.Timedelta(days=k)
              for nam in range(2019, 2024) for th, ng in NGAY_LE for k in range(-CUA_SO_LE, CUA_SO_LE + 1)}
    x["dip_le"] = ngay.isin(cua_so).astype(int)

    # tương tác
    dem = ((gio >= 18) | (gio < 6)).fillna(False).astype(int)
    x["dem"] = dem
    x["dem_x_quoc_lo"] = dem * df["cap_duong"].isin(["Quốc lộ", "Cao tốc"]).astype(int)
    x["dem_x_cuoi_tuan"] = dem * x["cuoi_tuan"]

    # khoá địa lý
    tinh = df["tinh"].astype("string")
    x["tinh"] = doi_tuong(tinh)
    x["khoa_huyen"] = doi_tuong(tinh + "|" + df["huyen"].astype("string"))
    duong = df["ten_duong"].map(chuan_ten_duong).astype("string")
    x["khoa_duong"] = doi_tuong(tinh + "|" + duong)
    x["_ten_duong_goc"] = df["ten_duong"]
    x[NHAN] = pd.to_numeric(df[NHAN], errors="coerce").fillna(0).astype(int)
    return x


class TaoDacTrungDong(BaseEstimator, TransformerMixin):
    """Bước đầu của pipeline: nhận các dòng ở dạng bảng lõi sạch (tien_xu_ly.py), tạo mọi đặc trưng theo dòng
    bằng tao_khung. Không học gì từ dữ liệu; dòng mới không cần có nhãn."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in KHOA:
            if c not in X:
                X[c] = ""
        if NHAN not in X:
            X[NHAN] = 0
        return tao_khung(X).drop(columns=[NHAN, "_ten_duong_goc"])


# ---------------------------------------------------------------------------
def nhom_cua_bo(bo: str) -> dict[str, list[str]]:
    """Nhóm cột đầu vào (tên nhóm -> cột) của từng bộ đặc trưng; dùng cho cả xây bộ mã hoá lẫn tráo cột."""
    g: dict[str, list[str]] = {c: [c] for c in DANH_NGHIA}
    g.update({c: [f"{c}_sin", f"{c}_cos"] for c in CHU_KY})
    g.update({"gio_thieu": ["gio_thieu"], "cuoi_tuan": ["cuoi_tuan"]})
    if bo != "nen":
        g.update({c: [c] for c in CO_MOI})
    if bo in ("nen", "nen_moi"):
        g["tinh"] = ["tinh"]
    if bo == "ma_hoa_dia_ly":
        g.update({"tinh": ["tinh"], "khoa_huyen": ["khoa_huyen"], "khoa_duong": ["khoa_duong"]})
    return g


def tao_bo_bien_doi(bo: str, nhom: dict[str, list[str]]) -> ColumnTransformer:
    cot = [c for cs in nhom.values() for c in cs]
    dn = [c for c in cot if c in DANH_NGHIA or (c == "tinh" and bo != "ma_hoa_dia_ly")]
    mt = [c for c in cot if c in MA_HOA_MUC_TIEU and bo == "ma_hoa_dia_ly"]
    ts = [c for c in TAN_SUAT if c in mt]
    so = [c for c in cot if c not in dn and c not in mt]
    phan = []
    if dn:
        phan.append(("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8), dn))
    if so:
        phan.append(("so", "passthrough", so))
    if mt:
        phan.append(("muc_tieu", bo_ma_hoa_muc_tieu(), mt))
    if ts:
        phan.append(("tan_suat", BoDemTanSuat(), ts))
    return ColumnTransformer(phan, verbose_feature_names_out=False)


def bo_ma_hoa_muc_tieu() -> TargetEncoder:
    """Target Encoding có làm mượt; fit_transform tự chia chéo để mỗi dòng không thấy nhãn của chính nó."""
    phien_ban = tuple(int(v) for v in re.findall(r"\d+", sklearn.__version__)[:2])
    if phien_ban >= (1, 9):
        return TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(SO_FOLD, shuffle=True, random_state=HAT_GIONG))
    return TargetEncoder(target_type="binary", cv=SO_FOLD, shuffle=True, random_state=HAT_GIONG)


def mo_hinh(ten: str, bo: str, nhom: dict[str, list[str]]):
    if ten == "logistic":
        return make_pipeline(tao_bo_bien_doi(bo, nhom), StandardScaler(),
                             LogisticRegression(max_iter=3000))
    return make_pipeline(tao_bo_bien_doi(bo, nhom), HistGradientBoostingClassifier(random_state=HAT_GIONG))


def kiem_dinh_cheo(train: pd.DataFrame, bo: str, nhom: dict[str, list[str]], ten_mh: str) -> dict:
    cot = [c for cs in nhom.values() for c in cs]
    kq = cross_validate(mo_hinh(ten_mh, bo, nhom), train[cot], train[NHAN],
                        cv=StratifiedKFold(SO_FOLD, shuffle=True, random_state=HAT_GIONG),
                        scoring=["roc_auc", "f1"], n_jobs=-1)
    return {"auc": kq["test_roc_auc"].mean(), "auc_sd": kq["test_roc_auc"].std(), "f1": kq["test_f1"].mean()}


def trao_nhom(train: pd.DataFrame, bo: str, nhom: dict[str, list[str]], ten_mh: str) -> tuple[pd.DataFrame, float]:
    """Permutation importance theo NHÓM cột: trên phần giữ lại 20% của train, tráo ngẫu nhiên giá trị của một nhóm
    giữa các vụ rồi đo AUC giảm bao nhiêu. Lặp trên SO_CACH_CHIA cách chia × SO_LAN_TRAO lần tráo để kết quả ổn định.
    Cặp sin/cos được tráo cùng nhau để không phá vỡ ý nghĩa giờ/tháng/thứ."""
    cot = [c for cs in nhom.values() for c in cs]
    rng = np.random.default_rng(HAT_GIONG)
    giam: dict[str, list[float]] = {ten: [] for ten in nhom}
    co_ich: dict[str, int] = {ten: 0 for ten in nhom}
    goc_ds = []
    for k in range(SO_CACH_CHIA):
        hoc, kd = train_test_split(train, test_size=0.2, stratify=train[NHAN], random_state=HAT_GIONG + k)
        mh = mo_hinh(ten_mh, bo, nhom).fit(hoc[cot], hoc[NHAN])
        goc = roc_auc_score(kd[NHAN], mh.predict_proba(kd[cot])[:, 1])
        goc_ds.append(goc)
        for ten, cs in nhom.items():
            lan = []
            for _ in range(SO_LAN_TRAO):
                x = kd[cot].copy()
                x[cs] = x[cs].iloc[rng.permutation(len(x))].to_numpy()
                lan.append(goc - roc_auc_score(kd[NHAN], mh.predict_proba(x)[:, 1]))
            giam[ten] += lan
            co_ich[ten] += int(np.mean(lan) > NGUONG_BO)
    dong = [{"nhom": ten, "auc_giam": float(np.mean(v)), "sd": float(np.std(v, ddof=1) / np.sqrt(len(v))),
             "so_lan_co_ich": co_ich[ten]} for ten, v in giam.items()]
    return pd.DataFrame(dong).sort_values("auc_giam", ascending=False).reset_index(drop=True), float(np.mean(goc_ds))


def kiem_dinh_thoi_gian(train: pd.DataFrame, bo: str, nhom: dict[str, list[str]], ten_mh: str) -> float:
    """Học trên các vụ 2020–2021 của train, chấm trên các vụ 8 tháng 2022 của train (tập test không bị đụng tới).
    Cho biết điểm có giữ được khi dự báo một giai đoạn CHƯA thấy, thay vì các vụ trộn lẫn cùng thời kỳ."""
    cot = [c for cs in nhom.values() for c in cs]
    hoc, kd = train[train["nguon"] != NGUON_THU_TG], train[train["nguon"] == NGUON_THU_TG]
    mh = mo_hinh(ten_mh, bo, nhom).fit(hoc[cot], hoc[NHAN])
    return float(roc_auc_score(kd[NHAN], mh.predict_proba(kd[cot])[:, 1]))


def chon_loc(train: pd.DataFrame, bo: str, mh: str, auc_day_du: float) -> tuple[dict[str, list[str]], pd.DataFrame]:
    nhom = nhom_cua_bo(bo)
    qt, auc_goc = trao_nhom(train, bo, nhom, mh)
    p(f"    AUC trên phần giữ lại khi chưa tráo (trung bình {SO_CACH_CHIA} cách chia): {auc_goc:.4f}")
    p(f"    Mỗi nhóm: {SO_CACH_CHIA} cách chia × {SO_LAN_TRAO} lần tráo = {SO_CACH_CHIA * SO_LAN_TRAO} phép đo;"
      " ± là sai số chuẩn của trung bình")
    qt["bo"] = qt["auc_giam"] <= np.maximum(2 * qt["sd"], NGUONG_BO)
    for _, r in qt.iterrows():
        p(f"    {r['nhom']:16s}: AUC giảm {r['auc_giam']:+.4f} (±{r['sd']:.4f}) | có ích ở {r['so_lan_co_ich']}/{SO_CACH_CHIA}"
          " cách chia" + ("   <- không đóng góp rõ ràng" if r["bo"] else ""))
    bo_di = qt.loc[qt["bo"], "nhom"].tolist()
    if not bo_di:
        p("    Không có nhóm nào dưới ngưỡng -> giữ toàn bộ")
        return nhom, qt
    nhom_gon = {k: v for k, v in nhom.items() if k not in bo_di}
    kq = kiem_dinh_cheo(train, bo, nhom_gon, mh)
    p(f"    Bỏ {len(bo_di)} nhóm: {', '.join(bo_di)}")
    p(f"    Kiểm định chéo lại: AUC {kq['auc']:.4f} (đầy đủ {auc_day_du:.4f})")
    if kq["auc"] >= auc_day_du - DUNG_SAI_AUC:
        p(f"    -> Giữ bản rút gọn (AUC không giảm quá {DUNG_SAI_AUC})")
        return nhom_gon, qt
    p("    -> Giữ bản đầy đủ (bản rút gọn giảm AUC quá mức cho phép)")
    return nhom, qt


def luu_huong(huong: str, bo: str, mh: str, nhom: dict[str, list[str]],
              train: pd.DataFrame, test: pd.DataFrame) -> dict:
    cot = [c for cs in nhom.values() for c in cs]
    bd = tao_bo_bien_doi(bo, nhom)
    X_train = bd.fit_transform(train[cot], train[NHAN])
    X_test = bd.transform(test[cot])
    ten_cot = list(bd.get_feature_names_out())
    hau_to = "" if huong == "du_bao" else f"_{huong}"
    p(f"    Học bộ biến đổi trên toàn bộ train, áp dụng cho test: {len(ten_cot)} cột | "
      f"train {X_train.shape[0]:,d} × {X_train.shape[1]} | test {X_test.shape[0]:,d} × {X_test.shape[1]}")
    p(f"    Nhóm giữ lại: {', '.join(nhom)}")
    if bo == "ma_hoa_dia_ly":
        p("    TargetEncoder: train được mã hoá bằng chia chéo nội bộ (mỗi dòng nhận tỉ lệ có chết tính từ các dòng "
          "KHÁC), test dùng tỉ lệ học từ toàn bộ train -> không rò rỉ nhãn")
    for ten, t, X in ((f"train_fe{hau_to}", train, X_train), (f"test_fe{hau_to}", test, X_test)):
        khoa = t[KHOA].astype({"dong_goc": "int64"})        # cùng kiểu với bảng lõi -> khoá ngoại trong CSDL hợp lệ
        luu_bang(pd.concat([khoa, pd.DataFrame(X, columns=ten_cot), t[[NHAN]]], axis=1), ten)
    return {"bo_dac_trung": bd, "bo": bo, "mo_hinh_goi_y": mh, "nhom": nhom, "ten_cot": ten_cot}


# ---------------------------------------------------------------------------
def main() -> None:
    p(f"[0] Đọc bảng lõi sạch + khoá chia train/test của bien_doi.py | scikit-learn {sklearn.__version__}")
    df = doc_bang(VAO, "tai_nan_loi_sach")
    khoa_train = doc_bang(GOC, "train")[KHOA].astype(str)
    khoa_test = doc_bang(GOC, "test")[KHOA].astype(str)
    p(f"    bảng lõi {len(df):,d} vụ | train {len(khoa_train):,d} | test {len(khoa_test):,d} (giữ nguyên cách chia của bien_doi.py)")

    p("\n[1] Tạo đặc trưng mới theo từng dòng (không học từ dữ liệu)")
    x = tao_khung(df)
    for ten in ("giao_lo", "cau", "deo_doc", "ly_trinh", "so_nha", "noi_dong_nguoi"):
        co = x[ten] == 1
        p(f"    {ten:15s}: {int(co.sum()):6,d} vụ ({co.mean():5.1%}) | tỉ lệ có chết {x.loc[co, NHAN].mean():.1%} "
          f"vs {x.loc[~co, NHAN].mean():.1%} khi không có")
    for ten in ("dip_le", "dem", "dem_x_quoc_lo", "dem_x_cuoi_tuan"):
        co = x[ten] == 1
        p(f"    {ten:15s}: {int(co.sum()):6,d} vụ ({co.mean():5.1%}) | tỉ lệ có chết {x.loc[co, NHAN].mean():.1%} "
          f"vs {x.loc[~co, NHAN].mean():.1%}")
    so_duong_goc = df["ten_duong"].dropna().astype(str).str.strip().nunique()
    so_duong_chuan = df["ten_duong"].map(chuan_ten_duong).dropna().nunique()
    p(f"    Chuẩn hoá tên đường: {so_duong_goc:,d} cách viết -> {so_duong_chuan:,d} đường "
      f"(vd 'QL 1A', 'Quốc lộ 1A', 'Quoc lo 1A' -> 'ql1a')")
    p(f"    Khoá địa lý: {x['khoa_huyen'].nunique():,d} tỉnh|huyện, {x['khoa_duong'].nunique():,d} tỉnh|đường "
      f"({x['khoa_duong'].isna().mean():.0%} vụ không có tên đường)")

    x["_k"] = x["nguon"].astype(str) + "#" + x["dong_goc"]
    k_tr = set(khoa_train["nguon"] + "#" + khoa_train["dong_goc"])
    k_te = set(khoa_test["nguon"] + "#" + khoa_test["dong_goc"])
    train = x[x["_k"].isin(k_tr)].reset_index(drop=True)
    test = x[x["_k"].isin(k_te)].reset_index(drop=True)
    if len(train) != len(k_tr) or len(test) != len(k_te):
        raise ValueError("Khoá train/test của bien_doi.py không khớp bảng lõi — chạy lại bien_doi.py")

    p(f"\n[2] So sánh 4 bộ đặc trưng — kiểm định chéo {SO_FOLD} lần, CHỈ trên train "
      "(mã hoá có học được học lại trong từng lần)")
    for bo, mo_ta in BO_DAC_TRUNG.items():
        p(f"    {bo:14s}: {mo_ta}")
    p(f"    {'bộ đặc trưng':14s} {'số cột':>6s} | {'Logistic AUC':>13s} {'F1':>6s} | {'HistGB AUC':>13s} {'F1':>6s}")
    bang = {}
    for bo in BO_DAC_TRUNG:
        nhom = nhom_cua_bo(bo)
        cot = [c for cs in nhom.values() for c in cs]
        so_cot = len(tao_bo_bien_doi(bo, nhom).fit(train[cot], train[NHAN]).get_feature_names_out())
        kq = {mh: kiem_dinh_cheo(train, bo, nhom, mh) for mh in ("logistic", "histgb")}
        bang[bo] = kq
        p(f"    {bo:14s} {so_cot:6d} | {kq['logistic']['auc']:.4f}±{kq['logistic']['auc_sd']:.3f} "
          f"{kq['logistic']['f1']:.3f} | {kq['histgb']['auc']:.4f}±{kq['histgb']['auc_sd']:.3f} {kq['histgb']['f1']:.3f}")
    p(f"    Kiểm định theo thời gian, vẫn chỉ trong train: học {int((train['nguon'] != NGUON_THU_TG).sum()):,d} vụ"
      f" 2020–2021, chấm {int((train['nguon'] == NGUON_THU_TG).sum()):,d} vụ {NGUON_THU_TG}")
    p(f"    {'bộ đặc trưng':14s} | {'Logistic: ngẫu nhiên -> thời gian':>34s} | {'HistGB: ngẫu nhiên -> thời gian':>32s}")
    for bo in BO_DAC_TRUNG:
        nhom = nhom_cua_bo(bo)
        for mh in ("logistic", "histgb"):
            bang[bo][mh]["auc_tg"] = kiem_dinh_thoi_gian(train, bo, nhom, mh)
        p(f"    {bo:14s} | {bang[bo]['logistic']['auc']:.4f} -> {bang[bo]['logistic']['auc_tg']:.4f}"
          f"{'':16s} | {bang[bo]['histgb']['auc']:.4f} -> {bang[bo]['histgb']['auc_tg']:.4f}")
    giam_tg = {b: bang[b]["histgb"]["auc"] - bang[b]["histgb"]["auc_tg"] for b in bang}
    p(f"    AUC giảm {min(giam_tg.values()):.3f}–{max(giam_tg.values()):.3f} khi chấm giai đoạn chưa thấy; thứ hạng 4 bộ"
      + (" giữ nguyên" if sorted(bang, key=lambda b: bang[b]["histgb"]["auc"]) ==
         sorted(bang, key=lambda b: bang[b]["histgb"]["auc_tg"]) else " THAY ĐỔI")
      + " -> kiểm định ngẫu nhiên hơi lạc quan, nhưng kết luận so sánh không đổi")
    tot = max(((bo, mh) for bo in bang for mh in bang[bo]), key=lambda t: bang[t[0]][t[1]]["auc"])
    bo, mh = tot
    p(f"    -> Bộ tốt nhất: {bo} với {mh} (AUC {bang[bo][mh]['auc']:.4f})")
    p(f"    -> Phần do thông tin tỉnh/huyện/đường: AUC {bang[bo][mh]['auc']:.4f} so với "
      f"{bang['khong_dia_ly'][mh]['auc']:.4f} khi bỏ hẳn địa lý")

    mh_kdl = max(bang["khong_dia_ly"], key=lambda m: bang["khong_dia_ly"][m]["auc"])
    p("    Vì thông tin địa lý chi phối, giữ HAI hướng đầu ra:")
    p(f"      du_bao      : {bo} + {mh} — dự báo tốt nhất, nhưng học cả thói quen báo cáo của địa phương")
    p(f"      khong_dia_ly: khong_dia_ly + {mh_kdl} — chỉ yếu tố thời gian / loại đường / vị trí, dùng để giải thích")

    ket_qua = {}
    for so, (huong, bo_h, mh_h) in enumerate((("du_bao", bo, mh), ("khong_dia_ly", "khong_dia_ly", mh_kdl)), 3):
        p(f"\n[{so}] Chọn lọc đặc trưng — hướng {huong} (bộ {bo_h}, {mh_h}): tráo từng nhóm cột trên phần giữ lại của train")
        nhom, qt = chon_loc(train, bo_h, mh_h, bang[bo_h][mh_h]["auc"])
        ket_qua[huong] = luu_huong(huong, bo_h, mh_h, nhom, train, test)
        ket_qua[huong]["tam_quan_trong"] = qt

    joblib.dump({"huong": ket_qua, "ket_qua_cv": bang}, GOC / "bo_dac_trung.joblib")
    (GOC / "kiem_tra_ky_nghe_dac_trung.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu train_fe, test_fe, train_fe_khong_dia_ly, test_fe_khong_dia_ly (.csv + bảng tien_xu_ly.*), "
          "bo_dac_trung.joblib, kiem_tra_ky_nghe_dac_trung.txt")


def chay() -> None:
    bat_dau = time.perf_counter()
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"KẾT THÚC: LỖI sau {time.perf_counter() - bat_dau:.1f} giây")
        sys.exit(1)
    print(f"KẾT THÚC: thành công sau {time.perf_counter() - bat_dau:.1f} giây")


if __name__ == "__main__":
    # Chạy qua tên module để joblib lưu lớp BoDemTanSuat dưới tên "ky_nghe_dac_trung.BoDemTanSuat"
    # (thay vì "__main__.BoDemTanSuat") -> script khác nạp lại được bo_dac_trung.joblib.
    sys.path.insert(0, str(GOC))
    import ky_nghe_dac_trung as _module
    _module.chay()

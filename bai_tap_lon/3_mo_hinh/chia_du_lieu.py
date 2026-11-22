"""BƯỚC 3 – XÂY DỰNG MÔ HÌNH · BƯỚC 1: CHIA DỮ LIỆU — Chương 5: Modeling (Bước 1 – Chia dữ liệu)

Mỗi bài toán chia một kiểu, vì mỗi bài toán hỏi một câu khác nhau:
    PHÂN LOẠI   (đoán vụ có người chết) : DÙNG LẠI cách chia 80/20 phân tầng của bước 2 (bien_doi.py),
                                           vì các bộ mã hoá (One-Hot, TargetEncoder) đã học trên đúng phần train đó.
                                           Chia lại sẽ làm vụ đã dùng để học mã hoá lọt vào test -> điểm đẹp giả.
                                           Thêm: chia train thành 5 phần (StratifiedKFold) cho kiểm định chéo.
    HỒI QUY     (số vụ theo thời điểm)   : chia THEO THỜI GIAN — học 2020–2021, kiểm tra 7 tháng trọn của 2022;
                                           trong phần học, chia 5 khối tháng liên tiếp cho kiểm định chéo theo
                                           thời gian (khối k được kiểm tra bằng mô hình học trên các khối trước nó).
    PHÂN CỤM    (tỉnh giống tỉnh)        : KHÔNG CHIA — học không giám sát, không có nhãn để chấm trên tập test.
    LUẬT KẾT HỢP (điều kiện đi cùng nhau): KHÔNG CHIA — tìm mẫu để mô tả, không dự đoán vụ mới.

Đầu vào  : bảng tien_xu_ly.tai_nan_loi_sach, tien_xu_ly.train, tien_xu_ly.test, tien_xu_ly.train_fe,
           tien_xu_ly.test_fe, tien_xu_ly.giao_dich, tien_xu_ly.ho_so_tinh (PostgreSQL)
Đầu ra   : bảng mo_hinh.chia_phan_loai (mỗi vụ: tập train/test, số fold), mo_hinh.chia_hoi_quy (mỗi tháng: tập, khối)
           + bản .csv, kiem_tra_chia_du_lieu.txt (cũng in ra màn hình)

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/chia_du_lieu.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
from hoi_quy import MOC_KIEM_TRA, tao_bang_o  # noqa: E402  (cùng cách gộp ô như script hồi quy)

TX = "tien_xu_ly"
KHOA = ["nguon", "dong_goc"]
NHAN = "co_nguoi_chet"
SO_FOLD, HAT = 5, 42
SO_KHOI_THOI_GIAN = 4            # TimeSeriesSplit(4) trên 24 tháng -> 5 khối: khối 0 chỉ để học, khối 1–4 lần lượt được kiểm tra
BC: list[str] = ["BÁO CÁO CHIA DỮ LIỆU CHO BƯỚC MÔ HÌNH", "=" * 70]
LOI: list[str] = []


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def kiem(dieu_kien: bool, mo_ta: str) -> None:
    p(f"    [{'ĐẠT' if dieu_kien else 'LỖI'}] {mo_ta}")
    if not dieu_kien:
        LOI.append(mo_ta)


def bo_khoa(df: pd.DataFrame) -> set[tuple[str, int]]:
    """Tập khoá (nguon, dong_goc), đưa dong_goc về số nguyên để so được giữa các bảng."""
    return set(zip(df["nguon"].astype(str), pd.to_numeric(df["dong_goc"]).astype(int)))


def ti_le(s: pd.Series) -> str:
    return f"{s.mean():.2%}"


def chia_phan_loai(loi: pd.DataFrame) -> pd.DataFrame:
    p("\n[1] PHÂN LOẠI — dùng lại cách chia của bước 2 (80/20, phân tầng theo nhãn, random_state=42)")
    train, test = csdl.doc(TX, "train")[KHOA + [NHAN]], csdl.doc(TX, "test")[KHOA + [NHAN]]
    n = len(loi)
    p(f"    Bảng lõi sạch: {n:,d} vụ | train {len(train):,d} ({len(train) / n:.1%}) | test {len(test):,d} ({len(test) / n:.1%})")

    p("\n    a) Hai phần không chồng lên nhau và ghép lại đúng bằng bảng lõi")
    k_tr, k_te, k_loi = bo_khoa(train), bo_khoa(test), bo_khoa(loi)
    kiem(not (k_tr & k_te), f"không vụ nào vừa ở train vừa ở test (trùng: {len(k_tr & k_te)})")
    kiem(k_tr | k_te == k_loi, f"train ∪ test = bảng lõi ({len(k_tr | k_te):,d} = {len(k_loi):,d} vụ)")
    kiem(len(k_tr) == len(train) and len(k_te) == len(test), "khoá (nguon, dong_goc) không lặp trong từng phần")

    p("\n    b) Phân tầng: tỉ lệ vụ có người chết giữ nguyên ở hai phần")
    p(f"       bảng lõi {ti_le(loi[NHAN])} | train {ti_le(train[NHAN])} | test {ti_le(test[NHAN])}")
    kiem(abs(train[NHAN].mean() - test[NHAN].mean()) < 0.005, "chênh tỉ lệ nhãn giữa train và test < 0,5 điểm %")

    p("\n    c) Mỗi năm báo cáo có mặt ở cả hai phần theo đúng tỉ lệ")
    bang = pd.concat([train.assign(tap="train"), test.assign(tap="test")], ignore_index=True)
    nam = pd.crosstab(bang["nguon"], bang["tap"])
    nam["% ở test"] = nam["test"] / nam.sum(axis=1)
    nam["có chết (train)"] = bang[bang["tap"] == "train"].groupby("nguon")[NHAN].mean()
    nam["có chết (test)"] = bang[bang["tap"] == "test"].groupby("nguon")[NHAN].mean()
    for nguon, r in nam.iterrows():
        p(f"       {nguon:7s}: train {int(r['train']):6,d} | test {int(r['test']):5,d} ({r['% ở test']:.1%})"
          f" | có chết: train {r['có chết (train)']:.1%}, test {r['có chết (test)']:.1%}")
    kiem(nam["% ở test"].between(0.18, 0.22).all(), "mỗi năm có 18–22% số vụ ở test")

    p("\n    d) Bảng đặc trưng của bước 2 khớp đúng cách chia này")
    for ten, goc in (("train_fe", k_tr), ("test_fe", k_te), ("train_fe_khong_dia_ly", k_tr), ("test_fe_khong_dia_ly", k_te)):
        k = bo_khoa(csdl.doc(TX, ten))
        kiem(k == goc, f"{ten}: {len(k):,d} vụ, cùng tập vụ với {'train' if goc is k_tr else 'test'}")

    p(f"\n    e) Chia train thành {SO_FOLD} phần cho kiểm định chéo (StratifiedKFold, xáo trộn, random_state={HAT})")
    train = train.sort_values(KHOA).reset_index(drop=True)      # thứ tự cố định -> chạy lại ra đúng các phần cũ
    train["fold"] = 0
    skf = StratifiedKFold(n_splits=SO_FOLD, shuffle=True, random_state=HAT)
    for i, (_, kiem_tra) in enumerate(skf.split(train, train[NHAN]), start=1):
        train.loc[kiem_tra, "fold"] = i
    for i in range(1, SO_FOLD + 1):
        f = train[train["fold"] == i]
        p(f"       phần {i}: {len(f):,d} vụ, có chết {ti_le(f[NHAN])}")
    kiem(train["fold"].between(1, SO_FOLD).all(), "mọi vụ train thuộc đúng một phần")
    p("       Lần thứ k: học trên 4 phần còn lại, chấm trên phần k -> 5 điểm, lấy trung bình và độ lệch chuẩn.")

    test = test.sort_values(KHOA).reset_index(drop=True).assign(fold=0)
    ra = pd.concat([train.assign(tap="train"), test.assign(tap="test")], ignore_index=True)
    return ra[KHOA + ["tap", "fold", NHAN]]


def chia_hoi_quy(loi: pd.DataFrame) -> pd.DataFrame:
    p(f"\n[2] HỒI QUY — chia theo thời gian: học các tháng trước {MOC_KIEM_TRA}, kiểm tra từ {MOC_KIEM_TRA} trở đi")
    o, d = tao_bang_o(loi)
    thang = pd.DataFrame({"ky": sorted(o["ky"].unique())})
    thang["tap"] = np.where(thang["ky"] < MOC_KIEM_TRA, "train", "test")
    hoc = thang[thang["tap"] == "train"].reset_index(drop=True)
    p(f"    Bảng ô (tháng × vùng × khung giờ × thứ): {len(o):,d} ô từ {len(d):,d} vụ, {len(thang)} tháng trọn")
    for tap in ("train", "test"):
        t = thang[thang["tap"] == tap]
        oo = o[o["ky"].isin(t["ky"])]
        p(f"    {tap:5s}: {t['ky'].min()} – {t['ky'].max()} ({len(t)} tháng, {len(oo):,d} ô)"
          f" | trung bình {oo['so_vu'].mean():.2f} vụ/ô, {(oo['so_vu'] == 0).mean():.0%} ô bằng 0")
    kiem(thang.loc[thang["tap"] == "train", "ky"].max() < thang.loc[thang["tap"] == "test", "ky"].min(),
         "mọi tháng học đều đứng trước mọi tháng kiểm tra")

    p(f"\n    Kiểm định chéo theo thời gian (TimeSeriesSplit({SO_KHOI_THOI_GIAN}) trên {len(hoc)} tháng học)")
    hoc["khoi"] = 0
    tss = TimeSeriesSplit(n_splits=SO_KHOI_THOI_GIAN)
    for i, (h, k) in enumerate(tss.split(hoc), start=1):
        hoc.loc[k, "khoi"] = i
        p(f"       lần {i}: học {hoc['ky'][h].min()} – {hoc['ky'][h].max()} ({len(h)} tháng)"
          f" -> kiểm tra {hoc['ky'][k].min()} – {hoc['ky'][k].max()} ({len(k)} tháng)")
    p("       Không xáo trộn: mô hình không bao giờ được thấy tháng nằm sau tháng nó dự báo.")
    thang = thang.merge(hoc[["ky", "khoi"]], on="ky", how="left").fillna({"khoi": -1})
    thang["khoi"] = thang["khoi"].astype(int)                  # -1 = tháng kiểm tra cuối (2022)
    return thang.assign(ky=thang["ky"].astype(str))


def khong_chia() -> None:
    p("\n[3] PHÂN CỤM — không chia train/test")
    hs = csdl.doc(TX, "ho_so_tinh")
    p(f"    ho_so_tinh: {len(hs)} tỉnh × {hs.shape[1] - 1} cột. Không có nhãn đúng để chấm trên phần giữ lại;")
    p("    chất lượng cụm đo bằng silhouette và độ ổn định khi chạy lại với hạt giống khác / bỏ bớt tỉnh.")
    p("\n[4] LUẬT KẾT HỢP — không chia train/test")
    gd = csdl.doc(TX, "giao_dich")
    p(f"    giao_dich: {len(gd):,d} vụ × {gd.shape[1] - len(KHOA)} item. Luật dùng để mô tả, không để dự đoán vụ mới;")
    p("    độ bền được kiểm tra bằng cách tính lại support/confidence của từng luật trên mỗi năm báo cáo.")
    kiem(gd[KHOA].duplicated().sum() == 0, "giao_dich không lặp vụ")


def main() -> None:
    p("[0] Đọc bảng lõi sạch của bước 2")
    loi = csdl.doc(TX, "tai_nan_loi_sach")
    p(f"    {len(loi):,d} vụ, tỉ lệ có người chết {ti_le(loi[NHAN])}")

    pl = chia_phan_loai(loi)
    hq = chia_hoi_quy(loi)
    khong_chia()

    if LOI:                                    # có phép kiểm tra không đạt -> không ghi bảng chia
        (GOC / "kiem_tra_chia_du_lieu.txt").write_text("\n".join(BC), encoding="utf-8")
        raise RuntimeError(f"{len(LOI)} phép kiểm tra không đạt, KHÔNG ghi bảng chia: " + "; ".join(LOI))

    p("\n[5] Lưu")
    for bang, ten in ((pl, "chia_phan_loai"), (hq, "chia_hoi_quy")):
        bang.to_csv(GOC / f"{ten}.csv", index=False, encoding="utf-8-sig")
        csdl.ghi(bang, "mo_hinh", ten)
        p(f"    {ten}: {len(bang):,d} dòng -> bảng mo_hinh.{ten} + {ten}.csv")

    p("\nTÓM TẮT")
    p(f"    Phân loại : train {(pl['tap'] == 'train').sum():,d} vụ (5 phần) | test {(pl['tap'] == 'test').sum():,d} vụ, để dành chấm một lần cuối")
    p(f"    Hồi quy   : học {(hq['tap'] == 'train').sum()} tháng (2020–2021) | kiểm tra {(hq['tap'] == 'test').sum()} tháng 2022")
    p("    Phân cụm, luật kết hợp: dùng toàn bộ dữ liệu, không chia")
    p("    Kiểm tra: tất cả ĐẠT")
    (GOC / "kiem_tra_chia_du_lieu.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu kiem_tra_chia_du_lieu.txt")


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

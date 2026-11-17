"""PHÂN CỤM — Nhóm 63 tỉnh theo đặc điểm tai nạn — Chương 5: Modeling (Clustering)

Đầu vào  : bảng tien_xu_ly.ho_so_tinh (PostgreSQL) (63 tỉnh × 7 chỉ số, đã chuẩn hoá z-score ở bien_doi.py)
Đầu ra   : ho_so_tinh_cum.csv (nhãn cụm của từng tỉnh + toạ độ PCA), kiem_tra_phan_cum.txt (cũng in ra màn hình)

Các bước:
    1. Chuẩn bị: 7 chỉ số z-score (trung bình 0, độ lệch chuẩn 1) — phân cụm dựa trên khoảng cách nên phải cùng thang
    2. PCA: nén 7 chỉ số còn 2 thành phần chính để nhìn các tỉnh trên mặt phẳng, xem mỗi thành phần gắn với chỉ số nào
    3. K-Means với k = 2..8: phương pháp khuỷu tay (inertia) + hệ số silhouette để chọn k
    4. Phân cụm phân cấp (Agglomerative, liên kết Ward): đọc bước nhảy khoảng cách của cây gộp, so với K-Means (ARI)
    5. Hồ sơ từng cụm: trung bình chỉ số gốc, danh sách tỉnh, đặc điểm nổi bật
    6. Độ nhạy: phân cụm lại khi BỎ hai chỉ số về người chết (dễ lệch theo thói quen báo cáo), xem cụm có giữ nguyên

Chạy (từ D:/PDM): .venv/Scripts/python bai_tap_lon/3_mo_hinh/phan_cum.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

GOC = Path(__file__).resolve().parent
sys.path.insert(0, str(GOC.parent))  # để import csdl.py (bai_tap_lon/) — đọc/ghi bảng trong PostgreSQL
import csdl  # noqa: E402
TX = GOC.parent / "2_tien_xu_ly"
BC: list[str] = ["BÁO CÁO PHÂN CỤM TỈNH", "=" * 70]
HAT = 42
CHI_SO = ["log_so_vu", "ti_le_co_chet", "chet_moi_vu", "ti_le_dem", "ti_le_cuoi_tuan", "ti_le_quoc_lo", "ti_le_dip_tet"]
TEN = {"log_so_vu": "quy mô (log số vụ)", "ti_le_co_chet": "tỉ lệ vụ có người chết", "chet_moi_vu": "số chết mỗi vụ",
       "ti_le_dem": "tỉ lệ vụ ban đêm", "ti_le_cuoi_tuan": "tỉ lệ vụ cuối tuần", "ti_le_quoc_lo": "tỉ lệ vụ trên quốc lộ",
       "ti_le_dip_tet": "tỉ lệ vụ dịp Tết"}
BO_NHAY = ["ti_le_co_chet", "chet_moi_vu"]
K_THU = range(2, 9)
K_CO_DINH: int | None = None      # đặt số cụm cố định nếu không muốn chọn theo silhouette


def p(dong: str = "") -> None:
    print(dong, flush=True)
    BC.append(dong)


def doc(thu_muc: Path, ten: str) -> pd.DataFrame:
    return csdl.doc(thu_muc, ten)               # thư mục 2_tien_xu_ly -> schema tien_xu_ly


def kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=50, random_state=HAT).fit(X)


def mo_ta_cum(z: pd.Series) -> str:
    cao = [TEN[c] for c in z.index if z[c] >= 0.5]
    thap = [TEN[c] for c in z.index if z[c] <= -0.5]
    return "; ".join(filter(None, [("CAO: " + ", ".join(cao)) if cao else "", ("THẤP: " + ", ".join(thap)) if thap else ""])) \
        or "gần mức trung bình"


def main() -> None:
    hs = doc(TX, "ho_so_tinh")
    for c in [*CHI_SO, *[f"{c}_z" for c in CHI_SO]]:
        hs[c] = pd.to_numeric(hs[c])
    hs["so_vu"] = pd.to_numeric(hs["so_vu"]).astype(int)
    X = hs[[f"{c}_z" for c in CHI_SO]].to_numpy()
    p(f"[1] Dữ liệu: {len(hs)} tỉnh × {len(CHI_SO)} chỉ số z-score | scikit-learn {sklearn.__version__}")
    p(f"    Chỉ số: {', '.join(TEN[c] for c in CHI_SO)}")

    p("\n[2] PCA — nén về 2 chiều để quan sát")
    pca = PCA(random_state=HAT).fit(X)
    tl = pca.explained_variance_ratio_
    p(f"    Phương sai giải thích: PC1 {tl[0]:.1%}, PC2 {tl[1]:.1%} -> 2 thành phần giữ {tl[:2].sum():.1%} thông tin;"
      f" cần {int(np.searchsorted(np.cumsum(tl), 0.8)) + 1} thành phần để giữ 80%")
    tai = pd.DataFrame(pca.components_[:2].T, index=CHI_SO, columns=["PC1", "PC2"])
    for pc in ("PC1", "PC2"):
        manh = tai[pc].abs().sort_values(ascending=False).index[:3]
        p(f"    {pc} gắn nhiều nhất với: " + ", ".join(f"{TEN[c]} ({tai.loc[c, pc]:+.2f})" for c in manh))
    toa_do = pca.transform(X)[:, :2]
    hs["pc1"], hs["pc2"] = toa_do[:, 0], toa_do[:, 1]

    p("\n[3] K-Means, k = 2..8 (n_init = 50)")
    p(f"    {'k':>3s} {'inertia':>9s} {'giảm so với k-1':>16s} {'silhouette':>11s}")
    bang, truoc = [], None
    for k in K_THU:
        km = kmeans(X, k)
        sil = silhouette_score(X, km.labels_)
        bang.append((k, km.inertia_, sil))
        p(f"    {k:3d} {km.inertia_:9.1f} {('' if truoc is None else f'{1 - km.inertia_ / truoc:.0%}'):>16s} {sil:11.3f}")
        truoc = km.inertia_
    k = K_CO_DINH or max(bang, key=lambda t: t[2])[0]
    p(f"    -> Chọn k = {k}" + (" (đặt cố định)" if K_CO_DINH else " (silhouette cao nhất)")
      + ". Silhouette: gần 1 = cụm tách bạch, gần 0 = các cụm chồng lấn")
    sil_k = dict((kk, s) for kk, _, s in bang)[k]
    if sil_k < 0.25:
        p(f"    CHÚ Ý: silhouette {sil_k:.2f} < 0,25 -> cấu trúc cụm YẾU; các tỉnh nằm liền mạch, ranh giới cụm chỉ mang tính tương đối")
    km = kmeans(X, k)
    hs["cum_kmeans"] = km.labels_ + 1

    p("\n[4] Phân cụm phân cấp (Agglomerative, liên kết Ward — gộp hai cụm làm tăng phương sai ít nhất)")
    Z = linkage(X, method="ward")
    cao = Z[:, 2]
    p("    Khoảng cách ở 6 lần gộp cuối (đọc từ dưới lên như cây dendrogram): "
      + ", ".join(f"{h:.2f}" for h in cao[-6:]))
    # cao[-j] là lần gộp từ j+1 cụm xuống j cụm; cắt ngay dưới cao[-j] thì còn j+1 cụm
    khoang = {j + 1: cao[-j] - cao[-(j + 1)] for j in range(1, 8)}
    k_cay = max(khoang, key=khoang.get)
    p(f"    Khoảng trống lớn nhất giữa hai lần gộp nằm ngay trên mức {k_cay} cụm -> cây gợi ý cắt ở {k_cay} cụm")
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    hs["cum_phan_cap"] = ag.labels_ + 1
    ari = adjusted_rand_score(km.labels_, ag.labels_)
    p(f"    Cắt cây ở k = {k} để so với K-Means: ARI = {ari:.2f} (1 = trùng hoàn toàn, 0 = như chia ngẫu nhiên)"
      f" | silhouette phân cấp {silhouette_score(X, ag.labels_):.3f}")

    p(f"\n[5] Hồ sơ {k} cụm K-Means (trung bình chỉ số gốc; đặc điểm theo z-score ≥ 0,5 hoặc ≤ −0,5)")
    goc = hs.groupby("cum_kmeans")[CHI_SO].mean()
    z = hs.groupby("cum_kmeans")[[f"{c}_z" for c in CHI_SO]].mean()
    z.columns = CHI_SO
    for c in goc.index:
        tinh = hs.loc[hs["cum_kmeans"] == c].sort_values("so_vu", ascending=False)
        p(f"    Cụm {c}: {len(tinh)} tỉnh, {int(tinh['so_vu'].sum()):,d} vụ | {mo_ta_cum(z.loc[c])}")
        p("      " + " | ".join(f"{TEN[x]} {goc.loc[c, x]:.2f}" if x != "log_so_vu"
                                 else f"số vụ TB {np.expm1(goc.loc[c, x]):,.0f}" for x in CHI_SO))
        p(f"      Tỉnh: {', '.join(tinh['tinh'])}")

    p(f"\n[6] Độ nhạy: bỏ {', '.join(BO_NHAY)} (dễ lệch theo cách mỗi tỉnh ghi nhận vụ nhẹ)")
    giu = [i for i, c in enumerate(CHI_SO) if c not in BO_NHAY]
    km2 = kmeans(X[:, giu], k)
    ari2 = adjusted_rand_score(km.labels_, km2.labels_)
    p(f"    K-Means k = {k} trên {len(giu)} chỉ số còn lại: ARI so với kết quả chính = {ari2:.2f}"
      + (" -> cụm khá bền, không chỉ do khác biệt báo cáo" if ari2 >= 0.5 else
         " -> cụm thay đổi nhiều: phân cụm chính bị chi phối bởi tỉ lệ người chết, tức một phần là cách báo cáo"))
    hs["cum_khong_ti_le_chet"] = km2.labels_ + 1

    ra = hs[["tinh", "so_vu", "cum_kmeans", "cum_phan_cap", "cum_khong_ti_le_chet", "pc1", "pc2", *CHI_SO]]
    ra.to_csv(GOC / "ho_so_tinh_cum.csv", index=False, encoding="utf-8-sig")
    csdl.ghi(ra, "mo_hinh", "ho_so_tinh_cum")
    (GOC / "kiem_tra_phan_cum.txt").write_text("\n".join(BC), encoding="utf-8")
    print("    Đã lưu ho_so_tinh_cum.csv (+ bảng mo_hinh.ho_so_tinh_cum), kiem_tra_phan_cum.txt")


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

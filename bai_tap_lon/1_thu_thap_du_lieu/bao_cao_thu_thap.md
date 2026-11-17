# CHƯƠNG 2. THU THẬP DỮ LIỆU

## 2.1. Xác định bài toán và mục tiêu

**Vấn đề.** Tai nạn giao thông đường bộ ở Việt Nam mỗi năm làm chết hàng nghìn người. Muốn giảm tai nạn thì cần biết tai nạn thường xảy ra ở đâu, vào lúc nào, trong điều kiện nào và vì lỗi gì. Cũng cần biết những yếu tố nào làm một vụ tai nạn dễ có người chết hơn.

**Mục tiêu phân tích.**

| # | Câu hỏi | Kỹ thuật khai phá dự kiến |
|---|---|---|
| 1 | Tai nạn phân bố thế nào theo thời gian, địa bàn, loại đường, phương tiện? | Thống kê mô tả, trực quan hoá |
| 2 | Có dự đoán được một vụ tai nạn có người chết hay không từ các điều kiện lúc xảy ra? | Phân lớp |
| 3 | Những tổ hợp điều kiện nào thường đi kèm vụ tai nạn có người chết? | Luật kết hợp |
| 4 | Các tỉnh/thành có thể chia thành những nhóm có đặc điểm tai nạn giống nhau không? | Phân cụm |

**Loại dữ liệu cần thu thập.** Cần dữ liệu ở **mức từng vụ tai nạn**. Số liệu tổng hợp theo năm không đủ để phân lớp hay tìm luật kết hợp. Mỗi vụ cần có:
- thời gian và địa điểm;
- hậu quả: số người chết, số người bị thương;
- phương tiện liên quan;
- điều kiện đường, thời tiết, báo hiệu;
- hình thức va chạm, lỗi vi phạm.

## 2.2. Xác định nguồn dữ liệu

| Tiêu chí | Nội dung |
|---|---|
| Nguồn | Dữ liệu trích xuất từ hệ thống báo cáo thông tin tai nạn giao thông đường bộ của lực lượng CSGT (tên file `bcthdb_thong_tin_tong_hop`) |
| Hình thức | 3 file Excel (.xlsx), lưu trên Google Drive: <https://drive.google.com/drive/folders/1V--WoKpABVX5qaAwLwhHRF_dTOqM-1LI> |
| Phạm vi | Toàn quốc, 63 tỉnh/thành |
| Thời gian | 15/12/2019 – 14/08/2022 (kỳ báo cáo tính từ ngày 15/12 năm trước đến 14/12 năm sau) |
| Đơn vị dữ liệu | 1 dòng = 1 vụ tai nạn |

| File | Sheet | Số dòng | Số cột | Kỳ báo cáo |
|---|---|---:|---:|---|
| `bcthdb_thong_tin_tong_hop(2020).xlsx` | DSTN | 14.276 | 138 | 15/12/2019 – 14/12/2020 |
| `bcthdb_thong_tin_tong_hop(2021).xlsx` | DSTN, TPHCM, QL1 | 11.408 | 138 | 15/12/2020 – 14/12/2021 |
| `bcthdb_thong_tin_tong_hop(8T2022).xlsx` | DSTN | 7.402 | 12 | 15/12/2021 – 14/08/2022 |

**Lý do chọn nguồn này.** So với các bộ dữ liệu công khai như Kaggle hay UCI, nguồn này có 4 ưu điểm:
- là dữ liệu thật của Việt Nam;
- ghi ở mức từng vụ;
- có nhiều thuộc tính về điều kiện xảy ra tai nạn;
- trải dài gần 3 năm.

Tổng số người chết năm 2020 trong dữ liệu là 6.584 người. Con số này cùng bậc độ lớn với số liệu tai nạn đường bộ mà Ủy ban An toàn giao thông Quốc gia công bố cho năm 2020. Vì vậy có thể coi dữ liệu bao phủ phần lớn các vụ được báo cáo.

## 2.3. Xác định thuộc tính cần thu thập

Mỗi file 2020/2021 có 138 cột, nhưng không phải cột nào cũng dùng được. Nhóm chọn **55 thuộc tính**, chia thành 10 nhóm (danh sách đầy đủ trong `tu_dien_du_lieu.csv`):

| Nhóm | Số thuộc tính | Thuộc tính | Kiểu |
|---|---:|---|---|
| Thời gian | 2 | ngày, giờ | ngày / giờ |
| Địa điểm | 7 | tỉnh, huyện, xã, loại đường, tên đường, KM, địa điểm | phân loại / văn bản |
| Hậu quả | 6 | số chết, số bị thương, thương nặng, thương nhẹ, thiệt hại (đơn vị **triệu đồng**), phân loại tai nạn | số / phân loại có thứ tự |
| Phương tiện | 20 | PT1…PT5 × (nhóm PT, loại PT, tình trạng, có phải PT gây tai nạn) | phân loại / nhị phân |
| Môi trường | 2 | thời tiết, mật độ giao thông | phân loại |
| Va chạm | 3 | hình thức va chạm, kiểu va chạm, gây tai nạn bỏ trốn | phân loại |
| Hạ tầng | 8 | dạng đường, loại mặt đường, độ rộng, tình trạng mặt đường, tổ chức GT, nút GT, vòng xuyến, giao cắt đường sắt | phân loại |
| Báo hiệu | 3 | đèn tín hiệu, biển báo, đèn chiếu sáng công cộng | phân loại |
| Nguyên nhân | 2 | nguyên nhân, lỗi vi phạm (một vụ có thể có nhiều lỗi, ngăn bởi `;`) | phân loại / đa trị |
| Xử lý | 2 | hình thức xử lý, số bị can | phân loại / số |

**Các cột không thu thập và lý do:**
- **Thông tin cá nhân:** họ tên, ngày sinh, giới tính, địa chỉ, số điện thoại của người điều khiển và người bị nạn; biển số và chủ phương tiện. Các cột này gần như trống hoàn toàn và không cần cho mục tiêu phân tích. Không thu thập chúng cũng đảm bảo quyền riêng tư của người liên quan.
- **Văn bản tự do và cột trùng nghĩa:** mô tả hư hỏng phương tiện, tóm tắt diễn biến (trống), thiệt hại bằng chữ (trùng với thiệt hại bằng số).
- **Thông tin hành chính:** STT, đơn vị gửi/nhận/thụ lý báo cáo, trạng thái hồ sơ, doanh nghiệp bảo hiểm, số tiền phạt.

Lưu ý: thuộc tính *Phân loại tai nạn* được xác định từ số người chết và bị thương. Nhóm vẫn thu thập thuộc tính này để mô tả dữ liệu. Tuy nhiên, khi xây dựng mô hình dự đoán vụ có người chết thì **không được dùng thuộc tính này làm đầu vào**, vì nó gần như chính là đáp án (rò rỉ nhãn).

## 2.4. Tiến hành thu thập dữ liệu

Toàn bộ quá trình được tự động hoá bằng script Python `thu_thap.py`, dùng các thư viện `pandas`, `openpyxl`, `gdown` và `pyarrow`:

1. **Tải dữ liệu:** dùng `gdown` tải cả thư mục Google Drive về máy. Nếu file đã có sẵn thì bỏ qua bước tải.
2. **Đọc file Excel có tiêu đề 2 tầng.** Dòng tiêu đề thứ nhất là tên nhóm (ví dụ *Địa điểm xảy ra tai nạn*), dòng thứ hai là tên cột con (ví dụ *Tỉnh/Thành phố*). Tên nhóm nằm trong ô gộp nên chỉ ô đầu có giá trị. Script điền tên nhóm sang các ô còn lại (forward-fill) rồi ghép thành tên cột dạng `Nhóm | Cột con`.
3. **Xử lý các điểm bất thường của file gốc:**
   - File 2021 có thêm một dòng tên bảng ở trên cùng. Script tự tìm dòng tiêu đề thật.
   - File 2021 ghi nhầm tiêu đề nhóm phương tiện thứ 5 thành "PT1". Script đánh số lại PT1…PT5 theo thứ tự xuất hiện.
   - File 2020 có 2 cột "Thời gian": một cột dạng `HH:MM:SS`, một cột dạng số thập phân. Script giữ cột `HH:MM:SS`.
   - Một số tên cột có khoảng trắng thừa ở cuối, ví dụ `"Nguyên nhân "`. Script cắt bỏ khoảng trắng.
4. **Sheet phụ TPHCM và QL1 trong file 2021:** đối chiếu theo ngày, giờ, địa điểm và số người chết cho thấy 1.766/1.766 bản ghi TPHCM và 163/163 bản ghi QL1 **đều đã có trong sheet DSTN**. Đây là các bản trích con nên không thu thập thêm, tránh đếm trùng.

## 2.5. Tích hợp dữ liệu

Ba nguồn có cấu trúc khác nhau nên phải đưa về **một lược đồ chung**:

| Vấn đề | Cách xử lý |
|---|---|
| Năm 2022 chỉ có 12 cột (có 11/55 thuộc tính đã chọn: ngày, giờ, tỉnh, huyện, xã, loại đường, tên đường, KM, địa điểm, số chết, số bị thương) | Giữ đủ 55 cột; các thuộc tính 2022 không có để trống (NA), **không suy diễn** |
| Tên cột tiếng Việt dài, có dấu | Đổi sang tên ngắn không dấu (`so_chet`, `loai_duong`…); bảng đối chiếu tên gốc ↔ tên mới nằm trong `tu_dien_du_lieu.csv` |
| Kiểu dữ liệu không đồng nhất (Excel đọc mọi thứ dạng văn bản) | Ngày → `datetime`; số người, thiệt hại, cờ 0/1 → số; giờ và các thuộc tính phân loại → chuỗi |
| Truy vết nguồn gốc | Thêm 3 cột `nguon` (2020 / 2021 / 8T2022), `file_goc`, `dong_goc` (số dòng Excel trong sheet DSTN, mở file gốc là tìm thấy ngay bản ghi) |
| Đơn vị thiệt hại không ghi ở cột "Bằng số" | Đối chiếu với cột "Bằng chữ" (ví dụ số 4 ↔ "Bốn triệu đồng chẵn") → đơn vị là **triệu đồng**; ghi vào từ điển dữ liệu |

Kết quả: **33.086 bản ghi × 58 cột** (55 thuộc tính + 3 cột truy vết). Bước thu thập không xoá hay sửa giá trị nào. Bản ghi nghi vấn chỉ được **đánh dấu** để bước tiền xử lý quyết định.

## 2.6. Kiểm tra và đánh giá dữ liệu

### a) Số lượng bản ghi và đối chiếu số liệu chính

| Nguồn | Số vụ | Số người chết | Số người bị thương | Ngày sớm nhất – muộn nhất |
|---|---:|---:|---:|---|
| 2020 | 14.276 | 6.584 | 10.636 | 15/12/2019 – 14/12/2020 |
| 2021 | 11.408 | 5.716 | 8.043 | 15/12/2020 – 14/12/2021 |
| 8T2022 | 7.402 | 4.204 | 4.943 | 15/12/2021 – 14/08/2022 |
| **Tổng** | **33.086** | **16.504** | **23.622** | |

Ngày trong mọi bản ghi đều nằm đúng trong kỳ báo cáo của file tương ứng. Dữ liệu có đủ 63 tỉnh/thành.

### b) Kiểu dữ liệu sau tích hợp

43 cột chuỗi (phân loại/văn bản), 12 cột số, 1 cột ngày, 2 cột truy vết dạng văn bản.

### c) Dữ liệu thiếu

Tỉ lệ ô trống chỉ được tính trên các nguồn **có** cột đó. Năm 2022 không có cột nào thì không tính là thiếu.

| Mức thiếu (2020–2021) | Thuộc tính |
|---|---|
| ≈ 0% | ngày, giờ, tỉnh, huyện, địa điểm, số chết, số bị thương, thương nặng/nhẹ, thiệt hại, phân loại tai nạn, vòng xuyến |
| 1 – 30% | xã (2–3%), loại đường (12–14%), PT1 – nhóm/loại/gây TN (5–9%), PT2 – nhóm/loại/gây TN (25–29%), hình thức va chạm (26%), kiểu va chạm (29–30%), KM (25–29%), tên đường (26–32%) |
| 30 – 50% | nguyên nhân (32–33%), lỗi vi phạm (37–38%), loại mặt đường (40–42%), dạng đường (41–43%), thời tiết (45–46%), tình trạng mặt đường (45–47%), độ rộng đường (49–51%) |
| > 50% | tổ chức GT (51–53%), mật độ GT (54–59%), tình trạng PT1 (56–62%), tình trạng PT2 (67–71%), hình thức xử lý (61–66%), đèn chiếu sáng (71%), nút GT (78–79%), biển báo (79–80%), PT3–PT5 (> 92%), số bị can (93–96%), đèn tín hiệu (96–97%), bỏ trốn (97–98%), giao cắt đường sắt (99,7%) |

Nhận xét:
- PT3–PT5 thiếu > 93% vì **phần lớn vụ chỉ có 1–2 phương tiện**. Đây là ô trống do không có phương tiện thứ 3, không phải lỗi ghi chép.
- *Bỏ trốn*, *đèn tín hiệu*, *giao cắt đường sắt* có lẽ chỉ được điền khi có sự việc. Ô trống ở đây nhiều khả năng mang nghĩa "không có" hoặc "không áp dụng".
- *Số bị can* trống 93–96% nhưng chỉ có 5 ô ghi số 0. Ô trống ở đây gần như chắc chắn nghĩa là "không có bị can".
- Các thuộc tính hạ tầng và môi trường thiếu 40–60%. Bước tiền xử lý cần chọn cách xử lý phù hợp: xem ô trống là một nhóm riêng "Không rõ", hoặc chỉ phân tích trên tập con có đủ dữ liệu.

### d) Trùng lặp

| Loại | Số bản ghi |
|---|---:|
| Trùng hoàn toàn mọi thuộc tính | 7 |
| Trùng theo khoá (ngày, giờ, tỉnh, huyện, địa điểm, số chết, số bị thương) | 22 |
| Trùng giữa 2 nguồn khác nhau | 0 |

Tỉ lệ trùng rất thấp (0,07%) và không có bản ghi nào bị lặp giữa các năm. Các bản ghi này được đánh dấu bằng cột `co_trung_khoa = 1`.

### e) Tính hợp lệ

| Kiểm tra | Số bản ghi vi phạm |
|---|---:|
| Thiếu hoặc sai định dạng ngày | 0 |
| Ngày nằm ngoài kỳ báo cáo | 0 |
| Giá trị số âm | 0 |
| Cờ "là phương tiện gây tai nạn" khác 0/1 | 0 |
| Giờ sai định dạng (ví dụ `18:0000` thay vì `18:00:00`) | 90 |
| Số bị thương ≠ thương nặng + thương nhẹ | 103 |
| Phân loại "Va chạm giao thông / Ít nghiêm trọng" nhưng có người chết | 16 |
| Phân loại "Rất / Đặc biệt nghiêm trọng" nhưng 0 thương vong | 3 |
| Thiệt hại > 10.000 triệu đồng (10 tỷ) — giá trị bất thường | 2 |

Thiệt hại có trung vị 2 triệu và 99% số vụ không quá 100 triệu. Tuy vậy có 2 vụ "Va chạm giao thông / Ít nghiêm trọng" ở TP.HCM (file 2021, dòng Excel 4703 và 7916) ghi thiệt hại 64.990 triệu và 26.599.590 triệu (≈ 26,6 nghìn tỷ). Nhiều khả năng đây là lỗi nhập liệu. Bước tiền xử lý sẽ xử lý như giá trị ngoại lai.

Ngoài ra, thuộc tính *thời tiết* trộn hai khía cạnh khác nhau: ánh sáng ("Đêm tối") và thời tiết ("Nắng", "Mưa"…). Nhãn "Đêm tối, mưa gió" gộp cả hai. Bước tiền xử lý nên tách thành 2 thuộc tính. Các nhãn phân loại (thời tiết, loại đường, nhóm phương tiện, hình thức va chạm…) dùng **cùng một bộ giá trị** ở file 2020 và 2021, nên không cần ánh xạ nhãn giữa các năm.

Các bản ghi nghi vấn được đánh dấu bằng 4 cột cờ (giá trị 1 = nghi vấn), không xoá hay sửa:

| Cột cờ | Ý nghĩa | Số bản ghi |
|---|---|---:|
| `co_trung_khoa` | Trùng theo khoá với một bản ghi đứng trước | 22 |
| `co_ngay_loi` | Thiếu / sai định dạng ngày hoặc ngày ngoài kỳ báo cáo | 0 |
| `co_gio_loi` | Giờ sai định dạng | 90 |
| `co_lech_thuong_vong` | Mâu thuẫn giữa số thương vong và phân loại tai nạn | 121 |

Tổng cộng 233 bản ghi (0,7%) có ít nhất một cờ. Một số bản ghi vi phạm nhiều kiểm tra cùng lúc, nên tổng các dòng trong bảng trên lớn hơn số cờ.

**Đánh giá chung.**
- *Điểm mạnh:* dữ liệu ở mức từng vụ, đủ 63 tỉnh, gần 3 năm. Các trường cốt lõi (thời gian, địa điểm, thương vong) gần như đầy đủ và hợp lệ.
- *Hạn chế:*
  1. File 2022 chỉ có thông tin cơ bản, nên các phân tích về điều kiện xảy ra tai nạn chỉ dùng được dữ liệu 2020–2021.
  2. Nhiều thuộc tính hạ tầng và môi trường thiếu nhiều.
  3. Năm 2021 có giai đoạn giãn cách COVID-19 nên số vụ giảm bất thường: tháng 7–9/2021 chỉ có 648, 460, 607 vụ, trong khi các tháng bình thường có khoảng 1.000–1.400 vụ. Tháng 12/2019 và tháng 8/2022 chỉ có nửa tháng dữ liệu (do kỳ báo cáo bắt đầu ngày 15), nên khi vẽ biểu đồ theo tháng cần loại bỏ hoặc chú thích hai tháng này.
  4. Mức độ đầy đủ khi báo cáo có thể khác nhau giữa các địa phương. Ví dụ, TP.HCM chiếm 4.538/25.684 vụ năm 2020–2021 và 71% trong số đó là "Va chạm giao thông", trong khi tỉ lệ chung chỉ 39%. Như vậy TP.HCM báo cáo cả nhiều vụ va chạm nhẹ, nên khi so sánh giữa các tỉnh cần thận trọng.

## 2.7. Lưu trữ dữ liệu ban đầu

| File | Nội dung |
|---|---|
| `data_drive/*.xlsx` | 3 file Excel gốc, giữ nguyên, không chỉnh sửa |
| `du_lieu_tich_hop.csv` | Dữ liệu đã tích hợp, 33.086 dòng × 62 cột (58 cột dữ liệu + 4 cột cờ kiểm tra), mã hoá UTF-8 có BOM để mở bằng Excel không lỗi tiếng Việt |
| `du_lieu_tich_hop.parquet` | Cùng nội dung, định dạng cột nén (2,3 MB so với 16 MB của CSV), giữ đúng kiểu dữ liệu; dùng cho các bước sau |
| `tu_dien_du_lieu.csv` | Từ điển dữ liệu: tên gốc, tên mới, nhóm, kiểu, mô tả, nguồn có cột, tỉ lệ thiếu (chỉ tính trên các nguồn có cột đó) |
| `kiem_tra.txt` | Báo cáo kiểm tra đầy đủ (bước 2.6) do script sinh ra |
| `thu_thap.py` | Script thực hiện toàn bộ quy trình; chạy lại sẽ cho cùng kết quả |

Tập dữ liệu này là đầu vào cho **Chương 3 – Tiền xử lý dữ liệu**: xử lý giá trị thiếu, sửa định dạng giờ, xử lý bản ghi trùng hoặc mâu thuẫn, xử lý ngoại lai thiệt hại, tách thời tiết/ánh sáng, chuẩn hoá nhãn và tạo thuộc tính mới.

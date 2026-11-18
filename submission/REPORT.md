# Báo cáo Day 09 — Nguyễn Đức Hạnh, 2A202603344

## 1. Kết luận trước

Pipeline kết luận **Hold**. Cửa vào quét 224/224 ảnh, loại 22, giữ 202. Bản nháp `submission/export/clean_v1/` có 202 ảnh (`manifest.json`: `n_images` = 202, `decision` = Hold). Sau khi máy sửa và người mô phỏng duyệt 80/80 ảnh, `08_summary.json` còn 0 lỗi bắt buộc nhưng 38/202 ảnh còn cảnh báo (0,1881 ≥ 0,10). Đây là bản nháp để thử train, chưa phải bản phát hành.

## 2. Lỗi và phân việc

Số lấy từ `02_error_list_v1.csv`: 211 dòng, trong đó 58 dòng error và 153 dòng warn. Phần cửa vào nằm trong danh sách này dù tiêu đề file ghi mẫu số 202 ảnh đã nhận.

| Mã lỗi | Dòng / ảnh | Ai sửa | Vì sao |
|---|---:|---|---|
| `CORRUPT_IMAGE` | 2 / 2 | máy | Ảnh không `load()` được. Intake xóa ảnh |
| `BAD_SIZE` | 3 / 3 | máy | Cạnh ngắn < 250px. Intake xóa ảnh |
| `DUP_IMAGE_EXACT` | 8 / 8 | máy | Trùng từng byte. Giữ ảnh đứng trước, xóa ảnh sau |
| `DUP_IMAGE_NEAR` | 4 / 4 | máy | Phash ≤ 10. Xóa bản gần trùng |
| `LEAK_TEST` | 5 / 5 | máy | Trùng test hoặc gold_val. Xóa để khỏi đo sai |
| `BAD_FORMAT` | 3 / 3 | máy | Không đủ 5 số. `drop_line` |
| `BAD_CLASS_ID` | 6 / 6 | máy | ID ngoài 0–3, ví dụ ID 93 ở `img_0020`. `drop_line` |
| `BOX_ZERO` | 4 / 4 | máy | Rộng hoặc cao bằng 0. `drop_line` |
| `BOX_OUT_OF_IMAGE` | 8 / 8 | máy | Cắt về [0, 1]. 8 dòng `clip_box` |
| `DUP_BOX` | 7 / 7 | máy | Cùng lớp, IoU > 0,9. `drop_line` |
| `BAD_CLASS_NAME` | 2 / 1 | máy | `data.yaml` ghi `Person`, `Bus`. `rename_classes` một lần |
| `MISSING_LABEL_FILE` | 6 / 6 | người | Máy không tạo file rỗng. `needs_label` |
| `EMPTY_LABEL` | 12 / 12 | người | File rỗng có thể là ảnh nền hoặc thiếu vật |
| `BOX_TINY` | 6 / 4 | người | Cảnh báo cạnh ngắn < 12px. Máy không xóa |
| `BOX_HUGE` | 3 / 3 | người | Diện tích > 0,95. Máy không thu box |
| `CLASS_RARE` | 4 / 1 | người | Cảnh báo cấp dataset, lớp quá ít box |
| `POSSIBLE_MISSING` | 53 / 52 | người | Model thấy vật, nhãn không có box |
| `POSSIBLE_WRONG_CLASS` | 18 / 18 | người | Cùng chỗ, khác lớp. ID hợp lệ nên máy không sửa |
| `POSSIBLE_LOOSE_BOX` | 28 / 24 | người | Đúng lớp, IoU thấp |
| `POSSIBLE_FALSE_POS` | 29 / 27 | người | Box nhãn không khớp dự đoán nào |

`03_fixes.csv` có 35 dòng: 20 `drop_line`, 8 `clip_box`, 6 `needs_label`, 1 `rename_classes`. 22 ảnh bị xóa đã xảy ra ở intake, nên không lặp lại trong file này.

`05_review_queue.csv` có 125 ảnh. `img_0020` đứng đầu (priority 11,524) vì vừa `FIXED_DROP_LINE`, vừa `POSSIBLE_MISSING`, vừa `POSSIBLE_WRONG_CLASS`, và điểm nhân thêm confidence. Sáu ảnh thiếu nhãn (`img_0031`, `img_0053`, `img_0126`, `img_0135`, `img_0170`, `img_0219`) đứng ngay sau, priority 10, đúng trọng số `NEEDS_LABEL`. Ngân sách 80 ảnh phủ 74 ảnh có mã ngữ nghĩa. 45 ảnh còn lại không được xem, trong đó 29 ảnh vẫn mang mã ngữ nghĩa. Vì vậy sau duyệt, `07_error_list_v3` còn 0 error nhưng 38/202 ảnh còn warn.

## 3. Calibration (2 ngưỡng)

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 10 | `DUP_IMAGE_NEAR` 1,00/0,50 → 1,00/0,67 (3/6 → 4/6). `DUP_IMAGE_EXACT` giữ 1,00/1,00 | Cờ sai là xóa ảnh. 10 bắt thêm ảnh gần trùng mà precision trên 39 ảnh calib vẫn 1,00. 16 có F1 cao hơn (1,00/0,83) nhưng gộp nhầm cảnh khác nhau |
| `second_opinion.conf` | 0,25 | 0,6 | `POSSIBLE_MISSING` 0,50/1,00 → 0,57/1,00 (8/8). `POSSIBLE_WRONG_CLASS` 0,56/0,83 → 0,62/0,83 (5/6). `POSSIBLE_LOOSE_BOX` 0,29/0,40 → 0,20/0,20 | Cảnh báo cho người, nên giữ hết ca thiếu nhãn đã chèn. 0,8 làm recall thiếu nhãn tụt còn 0,75 |

Một chỗ calib đánh lừa: trên 39 ảnh, `phash_dist` 16 trông đẹp nhất và 20 vẫn có precision gần-trùng bằng 1,00. Calib không có ca `LEAK_TEST`, mỗi mã chỉ vài ảnh, nên không lộ gộp nhầm. Trên raw, ngưỡng 10 xóa đúng 4 cặp cùng cảnh (`img_0076`/`img_0042` khoảng cách 0, `img_0143`/`img_0037` khoảng cách 0, `img_0223`/`img_0077` khoảng cách 6, `img_0186`/`img_0061` khoảng cách 10) và bắt `img_0101` trùng `test/test_039`. Cửa vào loại 22/224: 2 hỏng, 3 quá nhỏ, 8 trùng byte, 4 gần trùng, 5 rò test. Còn 202.

## 4. Cửa chặn và điều pipeline không bắt được

Chạy `python -m pipeline.run --budget 0 --work work/run_nb --reports work/rep_nb --export-dir work/exp_nb`. Kết luận **Rework**, mã thoát 20, không tạo `work/exp_nb/clean_v1`. Lý do trong log: còn 6 lỗi bắt buộc trên 202 ảnh. Đó là 6 ảnh `MISSING_LABEL_FILE`: không có người gán thì máy không được bịa nhãn, gate không bỏ qua error.

Hai điểm pipeline không bắt được:

1. `img_0084` và `img_0074` cách nhau 16 trên phash. Cả hai còn trong `clean_v1`. Muốn bắt cặp này phải nâng ngưỡng lên 16, mức đã gộp nhầm cảnh khác trên raw.
2. Box lỏng nhẹ. Trên calib, `POSSIBLE_LOOSE_BOX` chỉ đúng 1/5 (recall 0,20) ở `conf` 0,6. `img_0018` có box person diện tích 0,970, v3 chỉ ghi `BOX_HUGE`, không có `POSSIBLE_LOOSE_BOX`. Autofix không sửa cảnh báo, nên khung không được thu.

Muốn thành Release với ngưỡng 0,10 đã ghim: 20/202 = 0,0990 còn Hold từ 21/202 = 0,1040. Cần đưa số ảnh còn cảnh báo từ 38 xuống còn tối đa 20, tức là người xem và xử lý thêm ít nhất 18 ảnh. Không được sửa `warn_ratio_hold` hay ngân sách 80 trong file cấu hình.

## 5. Train: raw vs clean

Chưa chạy. Phần train là mục tùy chọn, làm trên Google Colab.

## 6. Quyết định và ghi chú AI

Quyết định cuối là **Hold**. Bản `clean_v1` chỉ để thử nghiệm. Thành Release khi số ảnh còn cảnh báo ≤ 20/202 và vẫn không còn error. Việc tiếp theo là duyệt nốt các ảnh semantic còn trong 45 ảnh chưa xem, ưu tiên `POSSIBLE_MISSING` và `POSSIBLE_WRONG_CLASS` còn ở v3 (12 ảnh thiếu nhãn, 3 ảnh sai lớp: `img_0078`, `img_0083`, `img_0220`).

Có dùng AI để viết 4 module. Prompt chính: viết `label_format`, `gate`, `autofix`, `second_opinion` đúng docstring và test của Bước 2, ngưỡng lấy từ `cfg`. Cách kiểm: `pytest` 79 passed, rồi `python -m pipeline.run` ra Hold và `--budget 0` ra Rework mã 20, không xuất dataset. Một lần AI sai: gọi số dòng trong file nhãn là class ID, ví dụ coi người ở dòng 3 của `img_0020` là ID 3, trong khi dòng đó là class ID 93 và person phải là ID 1.

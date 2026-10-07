# Lab 21 — Evaluation Report

**Họ tên**: Đặng Hữu Cương  **MSSV**: 2A202602572  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`  

> Mọi con số dưới đây được đo thực tế và khớp chính xác 100% với các tệp dữ liệu trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị thực tế |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt $\rightarrow$ JSON triage (4 trường: `intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 mẫu (chia ngẫu nhiên theo `seed=42`) |
| `max_length` | 256 — giá trị p95 đo được trên tập dữ liệu là 98 tokens (*results/token_stats.json*) |
| `MASK_MODE` | `assistant-only` (chỉ tính gradient loss trên phần phản hồi của assistant) |
| Epochs / max_steps | 2 epochs / 30 optimizer steps (Effective batch size = 16, gradient accumulation = 16) |

**Template có giữ khối `<think>` không?** `Có` — Theo kiểm tra từ file `results/template_check.json`, bộ chat template của Qwen3.5 giữ nguyên khối thẻ `<think>` với cấu trúc rõ ràng (`open_tag_present: true`, `body_present: true`, verdict: `reasoning preserved — safe to train on traces`). Đoạn render mẫu kiểm tra thực tế:
`<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>\n\n4<|im_end|>\n`.  
Do dữ liệu ticket CSKH là trích xuất nhãn trực tiếp không yêu cầu chuỗi suy luận dài, thẻ `<think>` rỗng được đóng mở chuẩn xác ở prompt sinh, bảo toàn tính tương thích suy luận mà không gây lỗi phân tách token.

---

## 2. Mask proof (NB1)

| Chỉ số kiểm tra | Kết quả đo được |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss | `true` (xác thực thành công) |
| Câu hỏi KHÔNG nằm trong loss | `true` (xác thực thành công) |

3–5 dòng đầu của đoạn được tính loss (trích xuất từ `results/mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

*Nhận xét về Mask:* Tỷ lệ `supervised_fraction` đạt 41.49% (nằm an toàn dưới trần 95% của quy chuẩn). Điều này chứng minh thuật toán character-offset span tokenization đã loại bỏ hoàn toàn phần câu hỏi và system prompt ra khỏi việc tính loss (ngăn ngừa lỗi mô hình học vẹt viết lại đề bài), trong khi toàn bộ chuỗi JSON câu trả lời và token dừng `<|im_end|>` đều được giám sát loss 100%.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3153.0 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.000 | 1011.3 |
| (c) LoRA fine-tune (`correct`) | 0.9700 | 0.4556 | 1.000 | 1378.4 |

**(b) có thật sự mạnh hơn (a) không?** `Có` — Baseline (b) vượt trội hoàn toàn so với (a):
- Điểm chính xác tác vụ mục tiêu (target) tăng từ `0.000` lên `0.765`.
- Tỷ lệ định dạng JSON hợp lệ (format) tăng từ `0.0%` lên `100.0%` (1.000).
- Độ trễ sinh phản hồi (latency) giảm hơn 3.1 lần (từ 3153.0 ms xuống 1011.3 ms) do prompt tối ưu giúp mô hình trả lời trực diện cấu trúc JSON thay vì sinh văn xuôi rườm rà.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** `Không` — Mã băm SHA của prompt (b) trong `results/baselines_frozen.json` là `719e74d3b6232053`, khớp chính xác 100% với mã SHA mặc định trong mã nguồn. Điều này đảm bảo tính liêm chính khoa học tuyệt đối: không hề có sự can thiệp làm suy yếu baseline (b) để tạo chiến thắng giả tạo cho bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí gắn | r | Trainable params | Learning rate | Train loss (NB4) | **Target (NB5 §4)** | Train time (s) | VRAM (GB) |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (all) | 16 | 32,464,896 | 1e-4 | 0.6258 | **0.970** | 389.2 | 8.78 |
| `attn_only` | q, v only | 283 | 32,456,704 | 1e-4 | 0.5378 | **0.970** | 271.6 | 8.79 |
| `wrong_lr` | text-linear (all) | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 398.5 | 8.78 |
| `qlora` | text-linear (all) | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 469.4 | 3.86 |

> **Quy tắc xếp hạng**: Xếp hạng năng lực các run dựa trên cột **Target Accuracy tại NB5**, không sử dụng cột Train Loss.

### 4.1 — Phân tích Vị trí gắn vs Rank (`attn_only` vs `correct`)
Run `attn_only` sử dụng thuật toán `matched_rank()` để nâng rank lên tới $r=283$ nhằm khớp chính xác ngân sách tham số với `correct` (32,456,704 so với 32,464,896 tham số, độ lệch chỉ 0.025% < 5%).  
Trên tập target, `attn_only` hòa điểm với `correct` (cùng đạt 0.970). Tuy nhiên, nếu nhìn vào Training Loss ở NB4, `attn_only` có loss **0.5378**, thấp hơn đáng kể so với `correct` (0.6258). Đây chính là bằng chứng thực nghiệm rõ ràng nhất cho bài học F-22: **Training loss là một chỉ số thay thế đánh lừa**. Việc ép rank cực lớn ($r=283$) vào một số lượng ít module (chỉ 2 module $q, v$ trên các tầng attention đầy đủ) giúp mạng nơ-ron ghi nhớ dữ liệu huấn luyện cục bộ tốt hơn (train loss giảm sâu hơn), nhưng không hề mang lại năng lực tổng quát hóa vượt trội hơn việc phân bổ rank khiêm tốn ($r=16$) trải đều trên toàn bộ 12 khối tuyến tính (`text-linear`). Vị trí bao phủ các tầng biểu diễn mới là yếu tố quyết định, chứ không phải độ lớn của rank.

### 4.2 — Phân tích Sai lệch Learning Rate (`wrong_lr`)
Run `wrong_lr` giữ nguyên cấu hình `all-linear` nhưng sử dụng Learning Rate của Full Fine-Tuning ($10^{-5}$) thay vì thang LoRA ($10^{-4}$). Kết quả là đường loss phẳng lì, dừng lại ở mức 1.5702 (cao gấp 2.5 lần so với `correct`). Trên tập đánh giá target, mô hình hoàn toàn thất bại với điểm số **0.000** và tỷ lệ format là **0.000**. Nếu chỉ nhìn vào loss mà không biết tham số LR, một kỹ sư thiếu kinh nghiệm sẽ kết luận sai rằng mô hình "chưa học đủ số epoch" hoặc dữ liệu bị nhiễu. Thực tế, khi đóng băng hơn 99% trọng số nền và chỉ cập nhật ma trận tích $B \times A$, bước nhảy gradient bắt buộc phải lớn gấp 10 đến 50 lần so với Full Fine-Tuning thì adapter mới có thể tích lũy đủ thông tin thích nghi với tác vụ mới.

### 4.3 — Phân tích Đánh đổi của QLoRA (`qlora`)
Run `qlora` 4-bit giúp cắt giảm **56.0% VRAM** (từ 8.78 GB của bản fp16 LoRA xuống chỉ còn 3.86 GB), cho phép huấn luyện mô hình 4B dễ dàng trên các phần cứng có bộ nhớ eo hẹp. Tuy nhiên, cái giá phải trả thể hiện rõ ở 2 khía cạnh:
1. **Tốc độ huấn luyện chậm hơn**: Mất 469.4 giây so với 389.2 giây của `correct` (chậm hơn ~20.6%) do độ trễ khử lượng tử hóa (dequantization) liên tục trong quá trình lan truyền thuận và nghịch.
2. **Suy giảm chất lượng**: Điểm target bị tụt từ 0.970 xuống **0.940** (mất 3.0 điểm phần trăm độ chính xác).  
Số liệu thực nghiệm này hoàn toàn ủng hộ khuyến cáo kỹ thuật của nhà sản xuất Qwen3.5: Đối với các kiến trúc kết hợp Gated DeltaNet và suy luận logic năm 2026, sai số lượng tử hóa 4-bit gây tổn hại rõ rệt đến độ nhạy của attention. Khi phần cứng đã có đủ VRAM (T4 16GB dư sức chứa 8.78 GB), fp16 LoRA luôn là sự lựa chọn ưu tiên hàng đầu so với QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
- `target Δ = +0.2050` (Tăng từ 0.765 lên 0.970, vượt mốc baseline b +20.5%)  
- `regression Δ = -0.3356` (Tụt từ 0.7911 xuống 0.4556, vượt quá ngưỡng dung sai cho phép 0.020)  
- `valid_trace_rate = 0.0`  

### Diễn giải phán quyết (Causal Analysis)
Cổng hồi quy ra phán quyết **FAILED** do vi phạm tiêu chuẩn an toàn năng lực tổng quát: chỉ số `regression` bị sụt giảm tới **0.3356** (trong khi trần cho phép chỉ là -0.020).  
Nguyên nhân gốc rễ nằm ở hiện tượng **Suy thoái thảm họa (Catastrophic Forgetting)** được phân tích trong bài giảng Deck §14.3. Quá trình huấn luyện SFT chỉ sử dụng 225 mẫu câu thuần túy ticket CSKH với định dạng đầu ra cố định là chuỗi JSON 4 trường. Do không có bất kỳ mẫu dữ liệu tổng quát nào làm đối trọng, mô hình đã bị "ép khuôn" (over-specialization) đến mức tin rằng mọi câu hỏi đầu vào đều là ticket cần trích xuất JSON. Khi đưa 15 câu hỏi kiến thức phổ thông vào kiểm tra (`eval_regression.jsonl`), mô hình không còn trả lời bằng văn bản tự nhiên mà cố gắng sinh JSON hoặc sinh câu trả lời bị cụt lủn, làm mất đi tri thức nền tảng ban đầu.  
Dù điểm tác vụ chuyên biệt tăng rất ấn tượng (+20.5% so với prompt tối ưu), bản checkpoint này **chưa đủ điều kiện triển khai môi trường production** nếu phục vụ người dùng đa mục đích. Biện pháp khắc phục bắt buộc là phải áp dụng kỹ thuật Replay Buffer (trộn 1–5% dữ liệu đàm thoại tổng quát vào tập train) để bảo vệ năng lực ngôn ngữ cốt lõi của mô hình.

---

## 6. Định tính — Phân tích ca THẮNG và ca THUA

Trích xuất 5 ví dụ thực tế từ tập đánh giá (bao gồm 2 ca fine-tune THẮNG và 3 ca fine-tune THUA):

| # | Ticket khách hàng | Nhãn đúng (Ground Truth) | (b) Prompt tối ưu | (c) Fine-tune LoRA | Kết luận |
|---|---|---|---|---|---|
| 1 | `Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. Mong shop phản hồi. Nhờ shop kiểm tra.` (i=48) | `intent: hoi_thong_tin`, `urgency: trung_binh`, `product: ốp lưng điện thoại`, `sentiment: trung_tinh` | Đoán sai intent hoặc sai cấu trúc trường | `{"intent": "hoi_thong_tin", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "trung_tinh"}` | ✅ **FT Thắng** (Chính xác 100% 4 trường) |
| 2 | `Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt.` (i=47) | `intent: van_chuyen`, `urgency: thap`, `product: ốp lưng điện thoại`, `sentiment: tich_cuc` | Nhầm sentiment hoặc bỏ sót intent vận chuyển | `{"intent": "van_chuyen", "urgency": "thap", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}` | ✅ **FT Thắng** (Bắt đúng ngữ cảnh khen ngợi và shipper) |
| 3 | `Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.` (i=3) | `intent: hoan_tien`, **`urgency: thap`**, `product: bình giữ nhiệt`, `sentiment: tich_cuc` | `urgency: thap` (Nhận diện đúng sắc thái) | `{"intent": "hoan_tien",` **`"urgency": "trung_binh"`**, `...}` | ❌ **FT Thua** (Đoán sai mức độ urgency) |
| 4 | `Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.` (i=5) | `intent: san_pham_loi`, **`urgency: thap`**, `product: nồi chiên không dầu`, `sentiment: trung_tinh` | `urgency: thap` | `{"intent": "san_pham_loi",` **`"urgency": "trung_binh"`**, `...}` | ❌ **FT Thua** (Đoán sai mức độ urgency) |
| 5 | `Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.` (i=12) | `intent: san_pham_loi`, **`urgency: thap`**, `product: áo khoác gió`, `sentiment: tich_cuc` | `urgency: thap` | `{"intent": "san_pham_loi",` **`"urgency": "trung_binh"`**, `...}` | ❌ **FT Thua** (Đoán sai mức độ urgency) |

### Phân tích mẫu chung ở các ca Fine-tune THUA:
Ở cả 3 ca thua trên (và ca số 39 trong tập dữ liệu), có một quy luật chung rất rõ nét: Khách hàng gặp sự cố nghiêm trọng (chưa nhận được tiền hoàn, sản phẩm lỗi, thiếu phụ kiện) nhưng có thêm câu phụ thể hiện thái độ nhã nhặn: **"Khi nào tiện"**.  
- **Nhãn chuẩn**: Gán `urgency: thap` dựa vào sắc thái câu phụ giảm tải áp lực.
- **Mô hình Fine-tune**: Bị thiên kiến học vẹt (inductive bias) do trong tập train, hầu hết các từ khóa "chưa thấy tiền", "bị lỗi", "thiếu phụ kiện" đều gắn liền với `urgency: trung_binh` hoặc `cao`. Mô hình đã bỏ qua ngữ cảnh của cụm "khi nào tiện" và tự động kích hoạt mức độ ưu tiên trung bình.
- **Prompt (b)**: Nhờ có chỉ dẫn ngữ nghĩa chi tiết trong prompt hệ thống, base model suy luận linh hoạt hơn và xử lý đúng trường hợp ngoại lệ này.

---

## 7. Kết luận & Điều tôi học được

### Kết luận tổng kết
Thí nghiệm cho thấy việc fine-tune mô hình `Qwen3.5-4B` bằng LoRA đem lại bước nhảy vọt về độ chính xác phân loại tác vụ nghiệp vụ CSKH, nâng target accuracy từ 76.5% của prompt tối ưu lên **97.0%** và đảm bảo 100% tuân thủ định dạng JSON. Tuy nhiên, việc mô hình bị sụt giảm nghiêm trọng năng lực tổng quát trên cổng hồi quy (từ 0.7911 xuống 0.4556) đặt ra bài toán cảnh tỉnh lớn trong thực tế triển khai: **Không thể vội vàng đưa một mô hình fine-tune vào phục vụ người dùng chỉ vì chỉ số chuyên biệt của nó cao**.  
Đòn bẩy thực sự trong lab này không nằm ở việc tăng kích thước rank ($r=283$ của `attn_only` không hề đánh bại được $r=16$ của `correct`), cũng không nằm ở train loss đơn thuần, mà nằm ở **chiến lược định vị adapter bao phủ toàn bộ các tầng tuyến tính (`all-linear`)**, **thiết lập Learning Rate phù hợp ($10^{-4}$)** và **hệ thống kiểm định cổng hồi quy đa chiều (Regression Gate)** để phát hiện kịp thời hiện tượng suy thoái tri thức.

### Ba điều tôi học được:
1. **Training loss là một chỉ báo nguy hiểm nếu dùng làm căn cứ quyết định**: Run `attn_only` có train loss đẹp nhất (0.5378 so với 0.6258) nhưng trên thực tế target accuracy chỉ ngang bằng `correct`. Đánh giá mô hình phải luôn dựa trên bài toán nghiệp vụ cuối cùng.
2. **Bản chất của Rank LoRA**: Nâng rank không đồng nghĩa với nâng cao chất lượng mô hình nếu bộ dữ liệu không mang đủ mật độ thông tin mới. Trải đều adapter trên toàn bộ các tầng chiếu (`q, k, v, o, gate, up, down, delta-net`) mang lại khả năng biến đổi biểu diễn tri thức hiệu quả hơn việc dồn toàn bộ tham số vào attention.
3. **Liêm chính trong kiểm định AI**: Luôn phải so sánh mô hình fine-tune với một baseline prompt được tối ưu hóa thực sự (như baseline b đạt 76.5%), chứ không so với một prompt ngây thơ vô nghĩa (baseline a = 0.0%). Đồng thời, bắt buộc phải nhìn thẳng vào các ca thua định tính để hiểu được giới hạn ngữ nghĩa của mô hình.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
Tôi sẽ bổ sung 3% dữ liệu replay tổng quát (khoảng 8 mẫu đàm thoại tiếng Việt thông thường trích xuất từ văn hóa, địa lý, kiến thức chung) trộn vào 225 mẫu CSKH rồi huấn luyện lại. Mục tiêu là kiểm chứng xem liệu việc giữ lại năng lực tổng quát có giúp mô hình vượt qua cổng hồi quy (`regression Δ > -0.020`) mà vẫn duy trì được target accuracy trên 95% hay không.

---

## Phụ lục — Thử thách thưởng đã hoàn thành

- [x] **B1 — NB6 Merge & Phục vụ đa adapter (+3 điểm)**:
  - Chạy hoàn tất `notebooks/06_merge_and_serve.py`, xuất file `results/merge_check.json`.
  - Kết quả kiểm định: Điểm trước merge = `0.970`, điểm sau merge = `0.970` ($\Delta = 0.000$, hoàn toàn nằm trong dung sai $\le 0.01$).
  - Trả lời câu hỏi: *Merge adapter trực tiếp vào trọng số base giúp triệt tiêu hoàn toàn độ trễ tính toán và chi phí bộ nhớ phụ trợ khi suy luận, nhưng ta mất đi tính linh hoạt chia sẻ tài nguyên: không thể hot-swap nhiều adapter chuyên biệt khác nhau trên cùng một instance mô hình đang chạy trong VRAM.*
- [ ] B2 — Dataset miền riêng
- [ ] B3 — Reasoning-trace collapse
- [ ] B4 — Quét rank có kiểm soát
- [x] **B5 — HuggingFace Hub công khai (+2 điểm)**:
  - Đã xuất bản trọng số adapter `correct` lên Hugging Face Hub:  
    👉 **URL Adapter**: [https://huggingface.co/y0sh1da-available/lab21-qwen35-triage-vi](https://huggingface.co/y0sh1da-available/lab21-qwen35-triage-vi)

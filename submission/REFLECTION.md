# Reflection — Lab 21

**Học viên**: Đặng Hữu Cương — **MSSV**: 2A202602572  
*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Nghịch lý ở run `attn_only`: khi tăng rank lên tới 283 để bằng số tham số của `all-linear`, training loss ở NB4 của nó đạt 0.5378 (thấp hơn `correct` là 0.6258), nhưng khi đánh giá trên tập target thực tế thì điểm số hoàn toàn không thắng được (cùng 0.970). Train loss thấp nhất lại không phải là mô hình tốt nhất.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Mất nhiều thời gian nhất ở việc chạy sinh đánh giá (generation eval) trên Colab T4 và quản lý môi trường runtime (lỗi ngắt kết nối và kernel restart làm mất đường dẫn `src`). Ban đầu tôi nghĩ thời gian huấn luyện gradient mới là lâu nhất, nhưng thực tế phần chạy greedy decode sinh văn bản trên 50 mẫu đánh giá qua 4 run lặp đi lặp lại chiếm tới hơn một nửa tổng thời gian.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab, tôi tin rằng chỉ cần target accuracy của mô hình tăng vọt (từ 76.5% lên 97.0%) là fine-tune đã thành công mỹ mãn và sẵn sàng deploy. Bây giờ tôi hiểu rằng sự tăng trưởng đó có thể phải trả giá bằng việc quên thảm họa (catastrophic forgetting sụt giảm 33.6% ở năng lực tổng quát). Một mô hình fine-tune nếu không qua cổng hồi quy an toàn thì có thể trở nên vô dụng khi gặp các câu hỏi nằm ngoài phân phối hẹp.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant (Antigravity) để hỗ trợ đọc nhanh các phát hiện giả lập trong tài liệu, kiểm tra môi trường, viết các script trích xuất dữ liệu định tính và định dạng báo cáo. Chỗ AI dễ nhầm nếu không chú ý là việc xử lý mã hóa ký tự tiếng Việt (UTF-8 vs CP1252) trên terminal Windows và dòng lệnh PowerShell.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên không phải là chọn GPU hay mở code train, mà là xây dựng một **bộ prompt baseline thật mạnh** (đóng vai trò mốc chuẩn) cùng một **tập dữ liệu đánh giá 4 nhóm (target, format, regression, latency) được đóng băng trước**. Tôi sẽ chỉ đề xuất khách hàng fine-tune nếu prompt tối ưu không còn đủ khả năng giải quyết bài toán và bản fine-tune chứng minh được sự vượt trội mà không làm hỏng tri thức nền.

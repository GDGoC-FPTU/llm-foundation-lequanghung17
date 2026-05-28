# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> tăng dần từ 0.0 đến 1.5, câu trả lời chuyển từ trạng thái rập khuôn, mang tính chuẩn xác cao (như thông tin sách giáo khoa) sang trạng thái văn phong đa dạng và sáng tạo hơn. Tuy nhiên, khi bị đẩy lên mức cực đoan (1.5), mô hình bắt đầu mất kiểm soát, sinh ra các câu văn lủng củng, lặp từ, hoặc thậm chí tạo ra "ảo giác" (hallucination) với các thông tin sai lệch vô lý.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt ở mức thấp, lý tưởng là 0.0 đến 0.2. Nguyên nhân là chatbot hỗ trợ khách hàng (Customer Support) yêu cầu tính chính xác, sự đồng quán và tuân thủ nghiêm ngặt các chính sách, báo giá của doanh nghiệp. Đặt nhiệt độ thấp giúp AI trả lời đúng trọng tâm, ngăn chặn tuyệt đối việc AI tự ý "sáng tạo" ra các thông tin sai lệch gây nhầm lẫn và thiệt hại cho khách hàng.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> Dựa trên bảng giá hiện tại, GPT-4o đắt hơn GPT-4o-mini chính xác khoảng 16.67 lần. (Ta chỉ cần lấy tỷ lệ đơn giá Output: 0.010 / 0.0006 ≈ 16.67). Phân tích chi tiết: Tổng lượng token mỗi ngày là 10.000 x 3 x 350 = 10.500.000 tokens. Chi phí của GPT-4o sẽ là 105 USD/ngày, trong khi GPT-4o-mini chỉ tốn vỏn vẹn 6.3 USD/ngày.

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> Trường hợp GPT-4o xứng đáng là khi xây dựng một tác tử AI tự hành (Autonomous Agent) giải quyết các bài toán logic sâu, tạo mã nguồn (code) phần mềm phức tạp, hoặc phân tích báo cáo tài chính dài. Năng lực suy luận bậc cao của GPT-4o giúp tránh các lỗi logic mà mô hình nhỏ hay mắc phải. Trường hợp GPT-4o-mini tốt hơn là khi cần xử lý khối lượng lớn (high volume) các tác vụ đơn giản, lặp đi lặp lại như: Phân loại cảm xúc bình luận của khách hàng (Tích cực/Tiêu cực), trích xuất tên người/địa điểm (NER) từ đoạn văn bản ngắn, hoặc dịch thuật ngữ cơ bản.



---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming mang tính sống còn đối với các ứng dụng giao tiếp trực tiếp với người dùng (User-facing applications) như Chatbot hoặc AI Assistants; bởi vì việc hiển thị ngay lập tức từng mảnh từ (token) giúp giảm độ trễ cảm nhận (perceived latency) xuống gần như bằng 0, giữ chân người dùng tập trung thay vì bắt họ nhìn biểu tượng "đang tải" nhàm chán suốt 10 giây. Ngược lại, non-streaming lại ưu việt hơn trong các tác vụ chạy ngầm (Background/Batch Processing) hoặc giao tiếp giữa các hệ thống máy móc (API to API) như: hệ thống cào dữ liệu tự động, luồng CI/CD, hoặc trích xuất định dạng JSON. Trong các trường hợp này, hệ thống máy móc chỉ cần một kết quả hoàn chỉnh cuối cùng để tiếp tục xử lý, việc streaming dữ liệu nhỏ giọt sẽ chỉ làm tăng độ phức tạp trong việc ghép chuỗi và rủi ro lỗi khi phân tích cú pháp (parsing) mà không mang lại bất kỳ lợi ích nào về tốc độ thực thi tổng thể.


## Danh Sách Kiểm Tra Nộp Bài
- [ ] Tất cả tests pass: `pytest tests/ -v`
- [ ] `call_openai` đã triển khai và kiểm thử
- [ ] `call_openai_mini` đã triển khai và kiểm thử
- [ ] `compare_models` đã triển khai và kiểm thử
- [ ] `streaming_chatbot` đã triển khai và kiểm thử
- [ ] `retry_with_backoff` đã triển khai và kiểm thử
- [ ] `batch_compare` đã triển khai và kiểm thử
- [ ] `format_comparison_table` đã triển khai và kiểm thử
- [ ] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 

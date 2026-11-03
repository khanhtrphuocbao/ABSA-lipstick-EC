# Đơn giản hóa mã nguồn ABSA

## Mục tiêu

Giữ cấu trúc và các lệnh hiện tại; làm mã dễ đọc, giảm lặp và tách logic xử lý khỏi CLI.

## Hướng đã chọn

- Giữ `src/training`, `src/testing`, `src/analysis`, `scripts` và các đường dẫn báo cáo.
- Không chuyển toàn bộ sang package mới: lợi ích không tương xứng với số import và lệnh phải đổi.

## Thay đổi

1. Gom tiện ích dùng chung của analysis vào module thường, thay cho import từ `__main__.py`. Chỉ gom các hàm có cùng hành vi; giữ nguyên quy ước JSON/CSV, null và ghi file.
2. Chuyển logic của các CLI phân tích vào module xử lý trong chính package hiện tại. `__main__.py` chỉ nhận tham số và gọi xử lý. Giữ tương thích các import và lệnh cũ.
3. Viết lại các đoạn nén bằng dấu chấm phẩy thành các bước có tên rõ ràng. Tách hàm theo nhiệm vụ thực tế, tránh nhiều lớp hoặc module nhỏ không cần thiết.
4. Cập nhật README gốc: cây thư mục, vai trò các phần, môi trường và lệnh thường dùng. Phân biệt notebook baseline gốc với pipeline thực nghiệm hiện tại.

## Giới hạn

- Không thay kiến trúc, loss, preprocessing, optimizer, threshold hoặc phép tính thống kê.
- Giữ nguyên evaluator, protocol, dữ liệu, checkpoint và kết quả đã có.
- Không huấn luyện, suy luận mô hình, tạo lại kết quả, commit hoặc push.
- Giữ các thay đổi chưa commit của người dùng. Không xóa lịch sử và không sửa artifact cũ để phù hợp mã mới.
- Phần training đã chia module hợp lý được giữ nguyên; không thay để đạt sự đồng nhất hình thức.

## Kiểm tra hoàn tất

- Các CLI và import hiện có tiếp tục hoạt động.
- Test protocol, quantitative analysis, error analysis, figures và reporting liên quan vượt qua.
- Test các tiện ích được gom, gồm dữ liệu thiếu, null, định dạng đầu ra và collision.
- Kết quả tính toán không đổi; các nguồn thực nghiệm không bị ghi lại.
- Không chạy CLI có thể khởi động training hoặc test inference trong quá trình kiểm tra.

# Background Studio

Ứng dụng web thay nền ảnh bằng Python, Flask và rembg.

## Cấu trúc mã nguồn

```text
BackgroundChanger/
├── app.py                         # Tạo ứng dụng Flask và khởi chạy local
├── config.py                      # Đường dẫn, giới hạn ảnh và cấu hình
├── routes.py                      # Trang web và các API
├── image_processing/
│   ├── __init__.py                # Giao diện các hàm xử lý ảnh
│   └── background_removal.py      # Tách nền và làm mịn alpha
├── templates/                     # Giao diện HTML
└── static/                        # CSS, JavaScript và ảnh kết quả
```

`app.py` lắp ghép các phần; `routes.py` nhận yêu cầu từ trình duyệt; `image_processing` thực hiện việc xử lý ảnh; `config.py` gom các thiết lập dùng chung.

## Chạy nhanh trên Windows

Nhấp đúp vào **`Chay_chuong_trinh.bat`** ở thư mục dự án. Lần đầu chạy, chương trình sẽ tạo môi trường Python và cài các thư viện cần thiết; bước này cần Internet. Sau đó trình duyệt mở tại:

<http://127.0.0.1:5000>

Để tắt chương trình, đóng cửa sổ dòng lệnh đang chạy.

Nếu Windows báo không tìm thấy Python, hãy cài Python 3.12 và bật tùy chọn **Add Python to PATH**, rồi chạy lại tệp `.bat`.

## Chạy thủ công

Từ thư mục dự án:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r BackgroundChanger\requirements.txt
.venv\Scripts\python.exe -m BackgroundChanger.app
```

## Tính năng

- Tách nền ảnh PNG, JPG, JPEG hoặc WEBP (tối đa 12 MB).
- Làm mềm viền, chọn nền màu hoặc ảnh nền.
- Di chuyển, đổi kích thước chủ thể và chỉnh sáng, tương phản, bão hòa.
- Tải ảnh kết quả PNG.

Lần đầu tách nền có thể chậm vì rembg cần tải mô hình AI. Ảnh kết quả được lưu trong `BackgroundChanger/static/results`.

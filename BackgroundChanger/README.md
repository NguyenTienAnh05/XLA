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

## Chạy trên Windows

Mở Terminal tại thư mục dự án và chạy:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r BackgroundChanger\requirements.txt
.venv\Scripts\python.exe -m BackgroundChanger.app
```

Sau đó mở <http://127.0.0.1:5000>. Lần đầu chạy, cần cài thư viện theo hướng dẫn ở trên; bước cài đặt cần Internet. Dừng ứng dụng bằng `Ctrl+C` trong Terminal.

## Tính năng

- Tách nền ảnh PNG, JPG, JPEG hoặc WEBP (tối đa 12 MB).
- Chọn U²-Net, U²-Net kết hợp GrabCut để tinh chỉnh mặt nạ, hoặc Chroma key cho phông một màu.
- Làm mềm viền, chọn nền màu hoặc ảnh nền.
- Di chuyển, đổi kích thước chủ thể và chỉnh sáng, tương phản, bão hòa.
- Tải ảnh kết quả PNG.

Lần đầu tách nền có thể chậm vì rembg cần tải mô hình AI. Ảnh kết quả được lưu trong `BackgroundChanger/static/results`.

Chroma key phù hợp với phông xanh hoặc phông màu đồng nhất; GrabCut chạy sau U²-Net và có thể xử lý lâu hơn chế độ AI thông thường.

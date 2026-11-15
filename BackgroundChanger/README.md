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
- **Không bắt buộc AI:** GrabCut (khung tự động) và Chroma key chạy bằng OpenCV/NumPy, không cần tải mô hình.
- Tuỳ chọn AI (U²-Net qua rembg) chỉ dùng khi chọn phương pháp "AI".
- Làm mềm viền, chọn nền màu hoặc ảnh nền.
- Di chuyển, đổi kích thước chủ thể và chỉnh sáng, tương phản, bão hòa.
- Tải ảnh kết quả PNG.

Lần đầu chạy phương pháp AI có thể chậm vì rembg cần tải mô hình. Ảnh kết quả được lưu trong `BackgroundChanger/static/results`.

Chroma key phù hợp với phông xanh hoặc nền màu đồng nhất; GrabCut hoạt động tốt nhất khi đối tượng nằm gần trung tâm khung hình.

## Interactive segmentation across algorithms

The selection editor is available for every segmentation method. Use positive points to mark foreground, negative points to mark background, a rectangle to define an ROI, or a polygon to constrain the selected area. Brush tools create a sequence of foreground/background seed points.

Supported methods in the method selector:

- SAM 2: prompt-based object segmentation (requires the configured Transformers model and PyTorch).
- U²-Net: AI-generated mask refined by the common selection prompts.
- GrabCut: foreground/background seeds and optional rectangle initialization.
- Chroma Key, Watershed, K-means, Otsu, Adaptive Threshold, and Contour: their algorithm-generated masks are refined using the same points, rectangle, and polygon constraints.

For classical methods, interactive selection constrains/refines the algorithm's mask; it does not give those algorithms SAM 2's semantic understanding. Results therefore depend on image contrast, colors, and the quality of the initial mask.

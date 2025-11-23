# Hướng dẫn cài đặt môi trường SimTrack với UV

## Yêu cầu
- Python 3.8 trở lên
- UV package manager đã được cài đặt

## Cài đặt UV (nếu chưa có)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Hoặc trên Linux/macOS:
```bash
pip install uv
```

## Cài đặt môi trường SimTrack

### Bước 1: Khởi tạo dự án với UV

```bash
cd /home/thinhnp/MOT/models/SimTrack
uv init
```

### Bước 2: Tạo môi trường ảo (venv)

UV sẽ tự động tạo môi trường ảo trong thư mục `.venv`:

```bash
uv venv
```

Hoặc chỉ định tên thư mục cụ thể:

```bash
uv venv venv
```

### Bước 3: Kích hoạt môi trường ảo

**Trên Linux/macOS:**
```bash
source .venv/bin/activate
```

Hoặc nếu bạn đặt tên là `venv`:
```bash
source venv/bin/activate
```

### Bước 4: Cài đặt các thư viện từ requirements.txt

```bash
uv pip install -r requirements.txt
```

### Bước 5: Xác minh cài đặt

Kiểm tra các package đã được cài đặt:

```bash
uv pip list
```

Hoặc kiểm tra thư viện cụ thể:

```bash
uv pip show torch torchvision
```

## Cài đặt nhanh (một dòng lệnh)

Nếu bạn muốn thực hiện tất cả các bước trên một lần:

```bash
cd /home/thinhnp/MOT/models/SimTrack && \
uv venv && \
source .venv/bin/activate && \
uv pip install -r requirements.txt
```

## Lưu ý quan trọng

### 1. CLIP từ GitHub
File `requirements.txt` có chứa CLIP được cài từ GitHub:
```
clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
```

UV sẽ tự động clone và cài đặt từ repository này.

### 2. PyTorch với CUDA
Dự án sử dụng PyTorch với CUDA 12.6:
```
torch==2.8.0+cu126
torchvision==0.23.0+cu126
```

Đảm bảo máy tính của bạn có CUDA 12.6 hoặc tương thích.

### 3. Cài đặt bổ sung
Nếu gặp lỗi với một số package cụ thể, bạn có thể cài riêng:

```bash
uv pip install <package-name>
```

## Quản lý môi trường

### Tắt môi trường ảo
```bash
deactivate
```

### Xóa môi trường ảo (nếu cần cài lại)
```bash
rm -rf .venv
# Hoặc
rm -rf venv
```

Sau đó lặp lại các bước từ Bước 2.

### Cập nhật packages
```bash
uv pip install --upgrade -r requirements.txt
```

### Đồng bộ môi trường (cài đặt chính xác các phiên bản)
```bash
uv pip sync requirements.txt
```

## Khắc phục sự cố

### Lỗi với CUDA
Nếu không có GPU hoặc CUDA, cài đặt PyTorch CPU version:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Lỗi với CLIP
Nếu gặp lỗi khi cài CLIP từ GitHub, thử cài thủ công:

```bash
uv pip install git+https://github.com/openai/CLIP.git
```

### Lỗi dependencies
Nếu có xung đột dependencies, thử cài từng nhóm package:

```bash
# Cài PyTorch trước
uv pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126

# Sau đó cài các package còn lại
uv pip install -r requirements.txt
```

## Kiểm tra môi trường sau khi cài đặt

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
```

## Ưu điểm của UV

- **Nhanh hơn pip**: UV nhanh hơn 10-100 lần so với pip truyền thống
- **Quản lý dependencies tốt**: Tự động giải quyết xung đột
- **Tương thích pip**: Sử dụng cú pháp `uv pip` giống pip
- **Caching thông minh**: Lưu cache để cài đặt nhanh hơn lần sau

## Tài nguyên tham khảo

- [UV Documentation](https://github.com/astral-sh/uv)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [CLIP GitHub](https://github.com/openai/CLIP)

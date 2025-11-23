# Hướng dẫn cài đặt HIPTrack với UV

## Yêu cầu
- Python 3.8
- CUDA 11.3 (cho GPU support)
- uv package manager

## Các bước cài đặt

### 1. Tạo môi trường ảo với Python 3.8
```bash
uv venv --python 3.8
source .venv/bin/activate  # Linux/Mac
# hoặc .venv\Scripts\activate  # Windows
```

### 2. Cài đặt PyTorch với CUDA 11.3
```bash
uv pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 --index-url https://download.pytorch.org/whl/cu113
```

### 3. Cài đặt build dependencies
```bash
uv pip install "setuptools<60" cython numpy wheel
```

### 4. Cài đặt lap (tracking dependency)
```bash
uv pip install lap
```

### 5. Cài đặt cython-bbox
```bash
uv pip install cython-bbox --no-build-isolation
```

### 6. Cài đặt các dependencies còn lại
```bash
uv pip install -r requirements-clean.txt
```

## Packages không có trên PyPI

Một số packages từ environment.yaml gốc không có sẵn trên PyPI hoặc có xung đột version:

### Cần cài đặt thêm (nếu cần):

1. **megengine** (CUDA version):
   ```bash
   # Cài từ source hoặc wheel file riêng
   ```

2. **multiscaledeformableattention**:
   ```bash
   # Có thể cần build từ source nếu dùng
   ```

3. **panopticapi**:
   ```bash
   uv pip install git+https://github.com/cocodataset/panopticapi.git
   ```

4. **onnxruntime-gpu** (nếu cần GPU support cho ONNX):
   ```bash
   uv pip install onnxruntime-gpu
   # Note: Có thể xung đột với onnxruntime, chỉ cài một trong hai
   ```

## Kiểm tra cài đặt

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mmdet; import mmcv; print('MMDetection installed successfully')"
```

## Ghi chú

- File `requirements.txt` gốc chứa tất cả packages từ environment.yaml
- File `requirements-clean.txt` là phiên bản đã được tối ưu hóa để tránh xung đột dependencies
- Một số packages có version constraints để tương thích với PyTorch 1.10.1 và Python 3.8
- Setuptools < 60 là cần thiết cho một số packages sử dụng numpy.distutils (deprecated)

## Xử lý lỗi thường gặp

### Lỗi: "No module named 'Cython'"
Cài đặt Cython trước khi cài các packages khác:
```bash
uv pip install cython
```

### Lỗi: Version conflicts
Sử dụng `requirements-clean.txt` thay vì `requirements.txt` để tránh xung đột version.

### Lỗi: CUDA not available
Kiểm tra:
- CUDA 11.3 đã được cài đặt
- PyTorch version với +cu113 tag
- Driver GPU tương thích


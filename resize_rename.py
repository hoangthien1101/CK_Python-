import os
from PIL import Image


def rename_and_resize_images(input_folder, output_folder):
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách file ảnh trong thư mục input
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Đổi tên và resize ảnh
    for idx, file_name in enumerate(image_files, start=1):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f"{idx}.jpg")  # Đổi tên file

        # Mở ảnh và resize
        with Image.open(input_path) as img:
            width, height = img.size
            if width > height:
                new_width = 640
                new_height = int(height * (640 / width))
            else:
                new_height = 640
                new_width = int(width * (640 / height))

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path, "JPEG")  # Lưu ảnh với định dạng JPEG

        print(f"Đã xử lý: {file_name} -> {output_path}")


# Sử dụng hàm
input_folder = "D:\\WorkSpace\\Bin"  # Thay bằng đường dẫn đến thư mục chứa ảnh gốc
output_folder = "D:\\WorkSpace\\Bin_Resize"  # Thay bằng đường dẫn đến thư mục lưu ảnh kết quả
rename_and_resize_images(input_folder, output_folder)

from PIL import Image
import os
from pathlib import Path

def create_resized_images(full_dir='static/images_dataset/full',
                         thumb_dir='static/images_dataset/thumbs',
                         medium_dir='static/images_dataset/medium',
                         thumb_size=(600, 400),
                         medium_size=(1200, 800)):

    # Create output directories if they don't exist
    Path(thumb_dir).mkdir(parents=True, exist_ok=True)
    Path(medium_dir).mkdir(parents=True, exist_ok=True)

    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

    # Process each image in the full directory
    for filename in os.listdir(full_dir):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            # Open image
            with Image.open(os.path.join(full_dir, filename)) as img:
                # Create thumbnail
                thumb = img.copy()
                aspect = thumb.size[0] / thumb.size[1]

                # Calculate thumbnail size
                if aspect > thumb_size[0] / thumb_size[1]:
                    new_size = (thumb_size[0], int(thumb_size[0] / aspect))
                else:
                    new_size = (int(thumb_size[1] * aspect), thumb_size[1])

                thumb.thumbnail(new_size, Image.Resampling.LANCZOS)
                thumb_path = os.path.join(thumb_dir, filename)
                thumb.save(thumb_path, 'PNG', optimize=True)

                # Create medium size
                medium = img.copy()
                if aspect > medium_size[0] / medium_size[1]:
                    new_size = (medium_size[0], int(medium_size[0] / aspect))
                else:
                    new_size = (int(medium_size[1] * aspect), medium_size[1])

                medium.thumbnail(new_size, Image.Resampling.LANCZOS)
                medium_path = os.path.join(medium_dir, filename)
                medium.save(medium_path, 'PNG', optimize=True)

                print(f"Created resized versions for {filename}")

if __name__ == "__main__":
    create_resized_images()

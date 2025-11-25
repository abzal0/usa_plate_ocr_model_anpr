# USA License Plate OCR Model

A YOLOv11-based OCR model for reading characters from **cropped** USA license plate images.


## Example Result
  <img src="istockphoto-1173375259-2048x2048.jpg" alt="plate number" width="200">
  <img src="result.png" alt="OCR Result" width="300">


## Important Note

This is an **OCR model**, not a detection model. It does **not** detect or crop license plates from full images. You must provide a pre-cropped license plate image as input.

**Input**: Cropped license plate image
**Output**: Detected characters (0-9, A-Z)

## Training Details

- **Base Model**: YOLOv11n
- **Dataset**: [License Plate Recognition Object Detection Dataset](https://universe.roboflow.com/franz-bpzvh/license-ocr-qqq6v) from Roboflow
- **Training Hardware**: NVIDIA H100 GPU on RunPod
- **Training Time**: ~25-30 minutes
- **Classes**: 36 (digits 0-9 and letters A-Z)
- **Images**: ~9,987 training images

## Installation

```bash
pip install ultralytics
```

## Usage

### Command Line

```bash
python test_ocr.py <cropped_plate_image> [confidence_threshold]

# Examples:
python test_ocr.py plate_crop.jpg
python test_ocr.py plate_crop.jpg 0.3
```

### Python API

```python
from test_ocr import read_plate

result = read_plate('plate_crop.jpg', confidence=0.25)

print(result['text'])  # e.g., "ABC1234"

for char in result['characters']:
    print(f"{char['char']} - {char['confidence']:.1%}")
```

### Direct YOLO Usage

```python
from ultralytics import YOLO

model = YOLO('usa_plate_ocr.pt')
results = model('plate_crop.jpg', conf=0.25)

for box in results[0].boxes:
    cls = int(box.cls[0].item())
    char = results[0].names[cls]
    confidence = box.conf[0].item()
    print(f"{char}: {confidence:.1%}")
```

## Model Performance

The model detects individual characters and their positions, allowing you to reconstruct the plate number by sorting characters from left to right.

## License

**Free to use** for any purpose (personal, commercial, educational).

**Attribution Required**: Please credit the original author:
- GitHub: [@abzal0](https://github.com/abzal0)
- Mention: "USA Plate OCR model by [Abzal Assembekov & Franz]"
 

## Limitations

- Only works with **cropped** plate images (not full vehicle photos)
- Trained on USA-style plates only
- May not perform well on:
  - Heavily damaged/dirty plates
  - Unusual fonts or special plates
  - Very low resolution images
  - Extreme angles or lighting

## Contributing

Feel free to fine-tune this model on your own data or report issues.

Test image istockphoto-1173375259-2048x2048.jpg is from [Istock by Berezko](https://www.istockphoto.com/portfolio/Berezko?mediatype=photography) 
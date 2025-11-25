"""
USA License Plate OCR - Test Script
Reads characters from cropped license plate images
"""

from ultralytics import YOLO
import os
import sys

# Model path (same directory as this script)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'usa_plate_ocr.pt')


def read_plate(image_path, confidence=0.25):
    """
    Read characters from a cropped license plate image.

    Args:
        image_path: Path to cropped plate image
        confidence: Minimum confidence threshold (default 0.25)

    Returns:
        dict with 'text' and 'characters' list
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(MODEL_PATH)
    results = model(image_path, conf=confidence, verbose=False)

    # Extract characters sorted by x-coordinate (left to right)
    characters = []
    for box in results[0].boxes:
        x1 = box.xyxy[0][0].item()
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()
        char = results[0].names[cls]
        characters.append({
            'char': char,
            'confidence': conf,
            'x': x1
        })

    # Sort left to right
    characters.sort(key=lambda c: c['x'])

    # Build plate text
    plate_text = ''.join([c['char'] for c in characters])

    return {
        'text': plate_text,
        'characters': characters
    }


def main():
    if len(sys.argv) < 2:
        print("USA License Plate OCR")
        print("=" * 40)
        print("\nUsage: python test_ocr.py <cropped_plate_image> [confidence]")
        print("\nExample:")
        print("  python test_ocr.py plate_crop.jpg")
        print("  python test_ocr.py plate_crop.jpg 0.3")
        print("\nNote: Input image must be a cropped license plate,")
        print("      not a full vehicle image.")
        sys.exit(1)

    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    print("USA License Plate OCR")
    print("=" * 40)
    print(f"Image: {image_path}")
    print(f"Confidence threshold: {confidence}")
    print()

    result = read_plate(image_path, confidence)

    if not result['characters']:
        print("No characters detected.")
        print("Try lowering the confidence threshold.")
        sys.exit(0)

    print(f"Detected {len(result['characters'])} character(s):")
    print()
    for i, char_data in enumerate(result['characters'], 1):
        print(f"  {i}. '{char_data['char']}' (confidence: {char_data['confidence']:.1%})")

    print()
    print(f"Plate text: {result['text']}")


if __name__ == "__main__":
    main()

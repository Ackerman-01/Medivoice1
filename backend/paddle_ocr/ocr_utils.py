from paddleocr import PaddleOCR

# Initialize PaddleOCR with English model
ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=False)


def process_image_with_ocr(image_path: str) -> str:
    """
    Process image using PaddleOCR and return extracted text.
    """
    result = ocr.ocr(image_path, cls=True)
    extracted_text = ""

    for line in result[0]:
        words = line[1][0]
        extracted_text += words + " "

    return extracted_text.strip() if extracted_text else None

import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import io


def prepare_model(num_classes):
    """
    Prepares the Faster R-CNN model with a custom number of classes.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def preprocess_image(uploaded_file):
    """
    Preprocesses the uploaded file for inference.
    Args:
        uploaded_file: File-like object uploaded via Streamlit.
    Returns:
        Tensor: Preprocessed image tensor.
        Tuple: Original image height and width.
    """
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image).astype(np.float32)  # Convert to NumPy array
        original_height, original_width = image_np.shape[:2]
        
        # Normalize and resize image
        resized_image = cv2.resize(image_np, (224, 224), interpolation=cv2.INTER_AREA)
        resized_image /= 255.0
        
        # Convert to tensor
        input_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0)
        
        return input_tensor, original_height, original_width, pil_image
    except Exception as e:
        raise ValueError(f"Error in preprocessing image: {e}")


def inference(uploaded_file):
    """
    Perform inference and detect speed breakers in the uploaded image.
    Args:
        uploaded_file: File-like object uploaded via Streamlit.
    Returns:
        np.ndarray: Processed image with bounding boxes drawn.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2  # Example: Background + Speed Breaker

    # Load model
    model = prepare_model(num_classes)
    model_path = 'model.pth'

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    model.to(device)
    model.eval()

    # Preprocess the uploaded image
    input_tensor, original_height, original_width, original_image = preprocess_image(uploaded_file)
    input_tensor = input_tensor.to(device)

    if input_tensor is None:
        raise ValueError("Error: Preprocessing failed, no tensor returned.")

    with torch.no_grad():
        predictions = model(input_tensor)

    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    if boxes is None or len(boxes) == 0:
        raise ValueError("No objects detected in the image.")

    # Confidence threshold
    confidence_threshold = 0.5
    valid_indices = scores >= confidence_threshold
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]
    scores = scores[valid_indices]

    # Draw boxes on the original image
    original_image_np = np.array(original_image)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1 = int(x1 * original_width / 224)
        x2 = int(x2 * original_width / 224)
        y1 = int(y1 * original_height / 224)
        y2 = int(y2 * original_height / 224)
        cv2.rectangle(original_image_np, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label_text = f'Speed Breaker: {score:.2f}'
        cv2.putText(original_image_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Convert the image with bounding boxes back to a format usable by Streamlit
    processed_image = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)
    return processed_image

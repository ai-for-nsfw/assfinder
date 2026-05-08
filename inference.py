import argparse
from pathlib import Path
import cv2
import json

def format_time(seconds):
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def annotate_result(result):
    """Draws boxes and labels on a single result and returns (annotated_img, has_objects)."""
    annotated_img = result.plot(boxes=False, masks=True)
    has_objects = False
    
    if result.boxes is not None and len(result.boxes) > 0:
        has_objects = True
        for box in result.boxes:
            # Get bounding box coordinates and class info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{result.names[cls_id]} {conf:.2f}"
            
            # Set up text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Calculate text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw black background rectangle for text
            text_org = (x1, max(y1 - 5, text_height + 5))
            bg_top_left = (text_org[0], text_org[1] - text_height - 5)
            bg_bottom_right = (text_org[0] + text_width, text_org[1] + 5)
            cv2.rectangle(annotated_img, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
            
            # Draw white text
            cv2.putText(annotated_img, label, text_org, font, font_scale, (255, 255, 255), thickness)
            
    return annotated_img, has_objects

def export_anylabeling_annotation(result, out_dir, base_filename):
    """Exports the original image and AnyLabeling format annotations (JSON) to the specified directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_filename = f"{base_filename}.jpg"
    img_path = out_dir / img_filename
    json_path = out_dir / f"{base_filename}.json"
    
    # Save original image
    cv2.imwrite(str(img_path), result.orig_img)
    
    height, width = result.orig_shape
    
    shapes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            
            # If segmentation masks are available, export polygon points
            if result.masks is not None and len(result.masks) > i and len(result.masks.xy[i]) > 0:
                polygon = result.masks.xy[i].tolist()
                shapes.append({
                    "label": label,
                    "text": "",
                    "points": polygon,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
            else:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                shapes.append({
                    "label": label,
                    "text": "",
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })
                
    annotation_data = {
        "version": "0.3.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    
    # Save annotations (creates file even if no objects)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, indent=2)


def main(model_path, input_path, output_dir, interval, conf_threshold):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    input_p = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    is_video = input_p.suffix.lower() in video_extensions

    if is_video:
        if interval is None:
            print("Error: --interval must be specified when input is a video.")
            return
            
        print(f"Processing video {input_path} with interval {interval}s and confidence threshold {conf_threshold}...")
        video_out_dir = output_path / input_p.stem
        video_out_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25.0
            
        total_duration_sec = total_frames / fps if fps > 0 else 0
        total_duration_str = format_time(total_duration_sec)
        
        span_start_time_str = None
        span_end_time_str = None
        first_frame_img = None
        first_frame_result = None

        current_time_sec = 0.0
        
        while True:
            # Set video position
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
                
            time_str = format_time(current_time_sec)
            
            # Print currently processed time vs total duration in the same line
            print(f"\rProcessing: {time_str} / {total_duration_str}", end="", flush=True)

            # verbose=False to reduce logging spam for each frame
            results = model(frame, conf=conf_threshold, iou=0.45, verbose=False)
            
            # Since we process 1 frame, there's 1 result
            result = results[0]
            annotated_img, has_objects = annotate_result(result)
                            
            if has_objects:
                print(f"\nDetection found at {time_str}")
                if span_start_time_str is None:
                    # New detection span starts
                    span_start_time_str = time_str
                    span_end_time_str = time_str
                    first_frame_img = annotated_img
                    first_frame_result = result
                else:
                    # Continue existing span
                    span_end_time_str = time_str
            else:
                if span_start_time_str is not None:
                    # Span ends, save the first frame from this span
                    out_file = video_out_dir / f"{input_p.stem}_{span_start_time_str}-{span_end_time_str}.jpg"
                    cv2.imwrite(str(out_file), first_frame_img)
                    print(f"\nSaved visualization to {out_file}")

                    # Export to annotations folder for further training (including frames with no detections to allow fixing false negatives)
                    # export_anylabeling_annotation(first_frame_result, output_path / "annotations", f"{input_p.stem}_{span_start_time_str}-{span_end_time_str}")

                    # Reset span state
                    span_start_time_str = None
                    span_end_time_str = None
                    first_frame_img = None
                    first_frame_result = None
            
            current_time_sec += interval
            
            # Prevent infinite loops if video duration is unknown or set is failing
            if total_duration_sec > 0 and current_time_sec > total_duration_sec + interval:
                break
                
        print() # Clear the progress line after loop ends
                        
        # After loop ends, save if a span is still active
        if span_start_time_str is not None:
            out_file = video_out_dir / f"{input_p.stem}_{span_start_time_str}-{span_end_time_str}.jpg"
            cv2.imwrite(str(out_file), first_frame_img)
            print(f"Saved visualization to {out_file}")
            # Export to annotations folder for further training
            # export_anylabeling_annotation(first_frame_result, output_path / "annotations", f"{input_p.stem}_{span_start_time_str}-{span_end_time_str}")
            
        cap.release()
        print("Video processing complete.")
    else:
        print(f"Running inference on {input_path}...")
        if input_p.is_dir():
            valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff')
            print(f"Processing folder {input_path}...")
            image_files = [f for f in input_p.iterdir() if f.suffix.lower() in valid_extensions]
            
            if not image_files:
                print(f"No valid images found in {input_path}")
                return
                
            for img_file in image_files:
                results = model(str(img_file), conf=conf_threshold, iou=0.45)
                for i, result in enumerate(results):
                    annotated_img, _ = annotate_result(result)
                    out_file = output_path / f"{img_file.stem}_out_{i}.jpg"
                    cv2.imwrite(str(out_file), annotated_img)
                    print(f"Saved visualization to {out_file}")
                    
                    # Export to annotations folder for further training
                    # export_anylabeling_annotation(result, output_path / "annotations", f"{img_file.stem}_{i}")
        else:
            results = model(str(input_path), conf=conf_threshold, iou=0.45)
            for i, result in enumerate(results):
                annotated_img, _ = annotate_result(result)
                out_file = output_path / f"{input_p.stem}_out_{i}.jpg"
                cv2.imwrite(str(out_file), annotated_img)
                print(f"Saved visualization to {out_file}")
                
                # Export to annotations folder for further training
                # export_anylabeling_annotation(result, output_path / "annotations", f"{input_p.stem}_{i}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run assfinder model inference")
    parser.add_argument("--model", type=str, default="model/assfinder-26m-dev-v6.1.pt", help="Path to the .pt model")
    parser.add_argument("--input", "--image", dest="input", type=str, required=True, help="Path to the input image or video")
    parser.add_argument("--out", type=str, default=".", help="Directory to save output images")
    parser.add_argument("--interval", type=float, help="Interval in seconds between frames for video processing")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    main(args.model, args.input, args.out, args.interval, args.conf)

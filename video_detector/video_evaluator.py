import os
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from os.path import join
from PIL import Image as pil_image
from tqdm import tqdm
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
import pdb
import json

class VideoEvaluator:
    def __init__(self, model_path=None, output_path='.', cuda=False):
        self.output_path = output_path
        self.cuda = torch.cuda.is_available() if cuda else False
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Load model
        self.model, *_ = model_selection(modelname='xception', num_out_classes=2)
        # if model_path:
        #     # self.model = return_pytorch04_xception()
        #     print(f'Model found in {model_path}')
        # else:
        #     print('No model found, initializing random model.')
        if self.cuda:
            self.model = self.model.cuda()

    def get_boundingbox(self, face, width, height, scale=1.3, minsize=None):
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if minsize and size_bb < minsize:
            size_bb = minsize
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        x1, y1 = max(int(center_x - size_bb // 2), 0), max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)
        return x1, y1, size_bb

    def preprocess_image(self, image):
        """
        Preprocess the input image for model prediction.
        
        Args:
            image (numpy.ndarray): The input image in BGR format.
        
        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['test']
        preprocessed_image = preprocess(pil_image.fromarray(image)).unsqueeze(0)
        if self.cuda:
            preprocessed_image = preprocessed_image.cuda()
        return preprocessed_image

    def predict_with_model(self, image):
        preprocessed_image = self.preprocess_image(image)
        output = self.model(preprocessed_image)
        output = nn.Softmax(dim=1)(output)
        _, prediction = torch.max(output, 1)
        return int(prediction.cpu().numpy()), output

    def evaluate_video(self, video_path, start_frame=0, end_frame=None, output_mode='video', verbose=False):
        """
        Evaluate a video for deepfake detection with multiple output modes.
        
        Args:
            video_path (str): Path to input video
            start_frame (int): Starting frame for processing
            end_frame (int): Ending frame for processing
            output_mode (str): Either 'video' for processed video output or 'json' for detection results
            verbose (bool): If True, includes detailed frame-by-frame analysis in JSON output
        
        Returns:
            Union[str, dict]: Either the path to processed video or path to JSON results
        """
        print(f'Starting: {video_path}')
        
        # Setup for video input
        reader = cv2.VideoCapture(video_path)
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = reader.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        
        # Initialize video writer if in video mode
        writer = None
        processed_video_path = None
        json_output_path = None
        
        # Create output paths
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(self.output_path, exist_ok=True)
        
        if output_mode == 'video':
            processed_video_path = os.path.join(self.output_path, f"{base_filename}_processed.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            json_output_path = os.path.join(self.output_path, f"{base_filename}_results.json")

        # Initialize results tracking
        total_predictions = 0
        sum_predictions = 0
        sum_confidence = np.array([0.0, 0.0])  # For averaging confidence scores
        frames_with_faces = 0
        
        # Initialize results dictionary for JSON mode
        json_results = {
            'input_path': video_path,
            'frames_analyzed': 0,
            'frames_with_faces': 0,
        }
        if verbose:
            json_results['frames'] = []

        pbar = tqdm(total=(end_frame - start_frame) if end_frame else num_frames)
        while reader.isOpened():
            ret, image = reader.read()
            if not ret or (end_frame and frame_num >= end_frame):
                break
            frame_num += 1
            if frame_num < start_frame:
                continue
            pbar.update(1)

            if output_mode == 'video' and writer is None:
                writer = cv2.VideoWriter(processed_video_path, fourcc, fps, 
                                       (image.shape[1], image.shape[0]))
            
            # Detect and process faces in the frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)
            
            if verbose:
                frame_results = {
                    'frame_number': frame_num,
                    'faces': []
                }
            
            if faces:
                frames_with_faces += 1
                face = faces[0]
                x, y, size = self.get_boundingbox(face, image.shape[1], image.shape[0])
                cropped_face = image[y:y+size, x:x+size]
                prediction, output = self.predict_with_model(cropped_face)
                
                # Update running totals
                total_predictions += 1
                sum_predictions += prediction
                output_np = output.detach().cpu().numpy()
                sum_confidence += output_np[0]
                
                if verbose:
                    # Store detailed face detection results
                    face_result = {
                        'bbox': {'x': x, 'y': y, 'width': face.width(), 'height': face.height()},
                        'prediction': int(prediction),
                        'confidence': output.tolist(),
                        'label': 'fake' if prediction == 1 else 'real'
                    }
                    frame_results['faces'].append(face_result)
                
                if output_mode == 'video':
                    # Annotate frame
                    label = 'fake' if prediction == 1 else 'real'
                    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                    cv2.putText(image, f"{output.tolist()} => {label}", 
                              (x, y + face.height() + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.rectangle(image, (x, y), (face.right(), face.bottom()), 
                                color, 2)

            if output_mode == 'video':
                writer.write(image)
            elif verbose:
                json_results['frames'].append(frame_results)
                
        pbar.close()
        reader.release()
        if writer is not None:
            writer.release()

        if output_mode == 'video':
            print(f'Finished! Output saved under {processed_video_path}')
            return processed_video_path
        else:
            # Calculate final predictions and confidence
            if total_predictions > 0:
                avg_prediction = sum_predictions / total_predictions
                avg_confidence = sum_confidence / total_predictions
                
                # Determine final label
                final_label = 'fake' if avg_prediction >= 0.5 else 'real'
                
                # Update JSON results with summary
                json_results.update({
                    'frames_analyzed': frame_num - start_frame,
                    'frames_with_faces': frames_with_faces,
                    'final_label': final_label,
                    'confidence_scores': {
                        'real': float(avg_confidence[0]),
                        'fake': float(avg_confidence[1])
                    },
                    'average_prediction': float(avg_prediction)
                })
            else:
                json_results.update({
                    'frames_analyzed': frame_num - start_frame,
                    'frames_with_faces': 0,
                    'final_label': 'no_faces_detected',
                    'confidence_scores': {
                        'real': 0.0,
                        'fake': 0.0
                    },
                    'average_prediction': 0.0
                })

            # Write JSON results to file
            # with open(json_output_path, 'w', encoding='utf-8') as f:
            #     json.dump(json_results, f, indent=2)
            # print(f'Finished! JSON results saved under {json_output_path}')
            # return json_output_path
            return json_results
        
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, 
    confusion_matrix, roc_curve, auc
)
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model_performance(
    fake_videos_dir, 
    real_videos_dir, 
    evaluator, 
    output_dir='./performance_results',
    num_runs=5, 
    start_frame=0, 
    end_frame=None
):

    """
    Comprehensive performance evaluation for deepfake detection model.
    
    Args:
        fake_videos_dir (str): Directory containing fake videos
        real_videos_dir (str): Directory containing real videos
        evaluator (VideoEvaluator): Initialized video evaluator
        num_runs (int): Number of repeated evaluations
        start_frame (int): Starting frame for video processing
        end_frame (int): Ending frame for video processing
    
    Returns:
        dict: Performance metrics and results
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Collect video paths
    fake_videos = [
        os.path.join(fake_videos_dir, f) 
        for f in os.listdir(fake_videos_dir) 
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    
    real_videos = [
        os.path.join(real_videos_dir, f) 
        for f in os.listdir(real_videos_dir) 
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    
    # Combine videos with labels
    all_videos = [
        (path, 'fake') for path in fake_videos
    ] + [
        (path, 'real') for path in real_videos
    ]
    
    # Results storage
    all_results = []
    
    for run in range(num_runs):
        np.random.shuffle(all_videos)
        
        run_results = []
        for video_path, true_label in all_videos:
            try:
                result = evaluator.evaluate_video(
                    video_path, 
                    start_frame=start_frame, 
                    end_frame=end_frame, 
                    output_mode='json'
                )
                
                # Determine prediction
                predicted_label = 'fake' if result['average_prediction'] >= 0.5 else 'real'
                
                run_results.append({
                    'run': run,
                    'video': os.path.basename(video_path),
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'average_prediction': result['average_prediction'],
                    'confidence_real': result['confidence_scores']['real'],
                    'confidence_fake': result['confidence_scores']['fake']
                })
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
        
        all_results.extend(run_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(results_df['true_label'], results_df['predicted_label']),
        'precision': precision_score(results_df['true_label'], results_df['predicted_label'], pos_label='fake'),
        'recall': recall_score(results_df['true_label'], results_df['predicted_label'], pos_label='fake'),
        'f1_score': f1_score(results_df['true_label'], results_df['predicted_label'], pos_label='fake')
    }
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Prediction Distribution
    plt.subplot(131)
    results_df.boxplot(column='average_prediction', by=['true_label'])
    plt.title('Prediction Distribution')
    plt.suptitle('')
    
    # Confusion Matrix
    plt.subplot(132)
    cm = confusion_matrix(results_df['true_label'], results_df['predicted_label'])
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # ROC Curve
    plt.subplot(133)
    fpr, tpr, _ = roc_curve(results_df['true_label'] == 'fake', results_df['average_prediction'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Save results DataFrame
    results_csv_path = os.path.join(output_dir, 'performance_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    # Save metrics to JSON
    metrics_json_path = os.path.join(output_dir, 'performance_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump({
            'metrics': {k: float(v) for k, v in metrics.items()},
            'roc_auc': float(roc_auc)
        }, f, indent=4)
    
    # Save visualization
    plt_path = os.path.join(output_dir, 'performance_plots.png')
    plt.savefig(plt_path)
    plt.close()  # Close the plot to free memory
    
    print(f"Results saved in {output_dir}:")
    print(f"- CSV Results: {results_csv_path}")
    print(f"- Metrics JSON: {metrics_json_path}")
    print(f"- Performance Plots: {plt_path}")
    
    return {
        'metrics': metrics,
        'results_df': results_df,
        'roc_auc': roc_auc,
        'output_paths': {
            'results_csv': results_csv_path,
            'metrics_json': metrics_json_path,
            'performance_plots': plt_path
        }
    }
    

if __name__ == '__main__':
    # Example usage
    evaluator = VideoEvaluator(output_path='./results')
    
    performance_results = evaluate_model_performance(
        fake_videos_dir='/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/SDFVD/videos_fake',
        real_videos_dir='/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/SDFVD/videos_real', 
        evaluator=evaluator,
        output_dir='./evaluation_results',
        num_runs=1
    )
    
    # Print metrics
    print("Performance Metrics:")
    for metric, value in performance_results['metrics'].items():
        print(f"{metric}: {value}")
        

# if __name__ == '__main__':
#     # import argparse
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--video_path', '-i', type=str, required=True)
#     # parser.add_argument('--model_path', '-m', type=str, default=None)
#     # parser.add_argument('--output_path', '-o', type=str, default='.')
#     # parser.add_argument('--start_frame', type=int, default=0)
#     # parser.add_argument('--end_frame', type=int, default=None)
#     # parser.add_argument('--cuda', action='store_true')
#     # args = parser.parse_args()

#     # args.video_path = '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/SDFVD/videos_fake/vs1.mp4'
#     # args.output_path = '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results'
    
#     # evaluator = VideoEvaluator(args.model_path, args.output_path, args.cuda)
#     # if os.path.isdir(args.video_path):
#     #     for video in os.listdir(args.video_path):
#     #         evaluator.evaluate_video(join(args.video_path, video), args.start_frame, args.end_frame)
#     # else:
#     #     evaluator.evaluate_video(args.video_path, args.start_frame, args.end_frame)


#     video_path = '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/SDFVD/videos_fake/vs1.mp4'
#     output_path = '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results'
#     evaluator = VideoEvaluator(output_path)
    
#     jsonR = evaluator.evaluate_video(video_path, output_mode='json')

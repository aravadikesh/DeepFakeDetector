import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.special import expit

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

class DeepfakeDetector:
    def __init__(
        self, 
        net_model: str = 'EfficientNetAutoAttB4ST', 
        train_db: str = 'DFDC', 
        real_threshold: float = 0.2, 
        fake_threshold: float = 0.6
    ):
        """
        Initialize deepfake detection model with configurable parameters.
        
        Args:
            net_model (str): Neural network model to use
            train_db (str): Training dataset used for the model
            real_threshold (float): Threshold for classifying as real
            fake_threshold (float): Threshold for classifying as fake
        """
        self.real_threshold = real_threshold
        self.fake_threshold = fake_threshold
        
        # Device and model setup
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load model weights
        model_url = weights.weight_url[f'{net_model}_{train_db}']
        self.net = getattr(fornet, net_model)().eval().to(self.device)
        self.net.load_state_dict(
            torch.hub.load_state_dict_from_url(
                model_url, 
                map_location=self.device, 
                check_hash=True
            )
        )
        
        # Transformer and face extraction setup
        self.transf = utils.get_transformer(
            'scale', 224, 
            self.net.get_normalizer(), 
            train=False
        )
        
        # Initialize face detector and video reader
        self.facedet = BlazeFace().to(self.device)
        self.facedet.load_weights("blazeface/blazeface.pth")
        self.facedet.load_anchors("blazeface/anchors.npy")
        
        videoreader = VideoReader(verbose=False)
        video_read_fn = lambda x: videoreader.read_frames(x, num_frames=32)
        self.face_extractor = FaceExtractor(
            video_read_fn=video_read_fn, 
            facedet=self.facedet
        )

    def process_video(self, video_path: str) -> Dict[str, float]:
        """
        Process a single video and predict deepfake probability.
        
        Args:
            video_path (str): Path to the video file
        
        Returns:
            Dict with video details and prediction
        """
        try:
            # Extract faces from video
            vid_faces = self.face_extractor.process_video(video_path)
            
            # Prepare faces for prediction
            faces_t = torch.stack([
                self.transf(image=frame['faces'][0])['image'] 
                for frame in vid_faces 
                if len(frame['faces'])
            ])
            
            # Predict deepfake probability
            with torch.no_grad():
                faces_pred = self.net(faces_t.to(self.device)).cpu().numpy().flatten()
            
            # Convert predictions to probabilities
            prob = expit(faces_pred).mean()
            
            # Classify based on thresholds
            if prob <= self.real_threshold:
                label = 'REAL'
            elif prob >= self.fake_threshold:
                label = 'FAKE'
            else:
                label = 'UNCERTAIN'
            
            return {
                'video_path': video_path,
                'deepfake_probability': prob,
                'prediction': label
            }
        
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {
                'video_path': video_path,
                'deepfake_probability': None,
                'prediction': 'ERROR'
            }

def find_video_files(
    search_paths: List[str], 
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively search for video files in multiple directories.
    
    Args:
        search_paths (List[str]): Directories to search
        extensions (List[str], optional): Video file extensions
    
    Returns:
        List of absolute paths to video files
    """
    # Default video file extensions
    if extensions is None:
        extensions = [
            '.mp4', '.avi', '.mov', '.mkv', '.flv', 
            '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'
        ]
    
    # Normalize extensions
    extensions = [
        ext.lower() if ext.startswith('.') else '.' + ext.lower() 
        for ext in extensions
    ]
    
    video_files = []
    for search_path in search_paths:
        for root, _, files in os.walk(search_path):
            video_files.extend(
                os.path.abspath(os.path.join(root, f))
                for f in files 
                if any(f.lower().endswith(ext) for ext in extensions)
            )
    
    return video_files

def process_video_directory(
    search_paths: List[str], 
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Process videos across multiple directories sequentially.
    
    Args:
        search_paths (List[str]): Directories to search for videos
        output_csv (str, optional): Path to save results CSV
    
    Returns:
        DataFrame with detection results
    """
    # Find all video files
    video_paths = find_video_files(search_paths)
    
    # Print found videos
    print(f"Found {len(video_paths)} video files")
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Process videos sequentially
    results = []
    for video_path in video_paths:
        try:
            result = detector.process_video(video_path)
            results.append(result)
            print(f"Processed: {result['video_path']} - {result['prediction']}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV if output path provided
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return results_df

def main():
    """
    Main entry point for the script.
    Allows flexible configuration via command-line or hardcoded paths.
    """
    # Configurable search paths and output
    search_directories = [
        '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/DeepfakeTIMIT/higher_quality', 
        '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/DeepfakeTIMIT/lower_quality'
    ]
    
    # Process each directory
    for directory in search_directories:
        directory_name = os.path.basename(os.path.normpath(directory))
        output_csv = f'deepfake_results_{directory_name}.csv'
        
        process_video_directory(
            [directory], 
            output_csv, 
        )

if __name__ == '__main__':
    main()
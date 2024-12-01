"""
Training Script
Script for training gesture recognition models using either
image datasets or real-time captured data.
"""

import argparse
from src.gesture.model_trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition models')
    parser.add_argument('--mode', choices=['image', 'realtime'], required=True,
                      help='Training mode: image (from dataset) or realtime')
    parser.add_argument('--dataset', type=str,
                      help='Path to image dataset (required for image mode)')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples per class for realtime mode')
    parser.add_argument('--classes', type=int, default=5,
                      help='Number of classes for realtime mode')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.mode == 'image':
        if not args.dataset:
            print("Error: Dataset path is required for image mode")
            return
        trainer.train_from_images(args.dataset)
    else:
        trainer.train_realtime(args.samples, args.classes)
        
if __name__ == "__main__":
    main()

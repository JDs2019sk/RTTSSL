"""
Training Script
Script for training gesture recognition models using either
image datasets or real-time captured data.
"""

import argparse
import sys
from src.gesture.model_trainer import ModelTrainer

def get_class_names(num_classes):
    """Get class names from user input"""
    class_names = []
    print("\nEnter names for each gesture class:")
    for i in range(num_classes):
        while True:
            try:
                name = input(f"Enter name for class {i + 1}: ").strip()
                if name:
                    class_names.append(name)
                    break
                print("Name cannot be empty. Please try again.")
            except (EOFError, KeyboardInterrupt):
                print("\nTraining cancelled by user.")
                sys.exit(0)
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition models')
    parser.add_argument('--mode', choices=['image', 'realtime'], required=True,
                      help='Training mode: image (from dataset) or realtime')
    parser.add_argument('--dataset', type=str,
                      help='Path to image dataset (required for image mode)')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples per class for realtime mode')
    parser.add_argument('--classes', type=int, default=3,
                      help='Number of classes for realtime mode')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    try:
        if args.mode == 'image':
            if not args.dataset:
                print("Error: Dataset path is required for image mode")
                return
            trainer.train_from_images(args.dataset)
        else:
            # Get class names before starting capture
            class_names = get_class_names(args.classes)
            print("\nStarting capture process...")
            print("Press ESC during capture to skip to next class")
            print("Press Ctrl+C to cancel training\n")
            trainer.train_realtime(args.samples, args.classes, class_names)
            
    except KeyboardInterrupt:
        print("\nTraining cancelled by user.")
        sys.exit(0)
        
if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

# Add the Mask_RCNN directory to Python path
MASK_RCNN_DIR = os.path.join(os.getcwd(), "Mask_RCNN")
sys.path.append(MASK_RCNN_DIR)

try:
    # Import Mask RCNN
    from mrcnn import utils
    from mrcnn import visualize
    from mrcnn.visualize import display_instances
    import mrcnn.model as modellib
    from mrcnn import config
    
    print("✅ Successfully imported Mask R-CNN modules!")
    
    # Test configuration
    class InferenceConfig(config.Config):
        NAME = "test"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80  # COCO has 80 classes + background
    
    # Initialize the model
    print("📋 Initializing Mask R-CNN model...")
    inference_config = InferenceConfig()
    
    # Create model object in inference mode
    model = modellib.MaskRCNN(mode="inference", model_dir="./", config=inference_config)
    
    print("✅ Model created successfully!")
    print("⚠️  Note: You'll need to download pre-trained COCO weights to run inference")
    print("   Download from: https://github.com/matterport/Mask_RCNN/releases")
    print("   File needed: mask_rcnn_coco.h5")
    
    # Check if CLEVR images exist
    clevr_images_dir = "CLEVR_generation/images"
    if os.path.exists(clevr_images_dir):
        print(f"\n📁 Found CLEVR images directory: {clevr_images_dir}")
        images = [f for f in os.listdir(clevr_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"   Found {len(images)} images: {images[:5]}...")  # Show first 5
    else:
        print(f"\n❌ CLEVR images directory not found: {clevr_images_dir}")
    
    print("\n🎯 Mask R-CNN setup test completed!")
    print("💡 Next steps:")
    print("   1. Download mask_rcnn_coco.h5 weights")
    print("   2. Load and test on CLEVR images")
    print("   3. Analyze object detection performance")
    
except ImportError as e:
    print(f"❌ Error importing Mask R-CNN modules: {e}")
    print("💡 Make sure you're in the correct directory and Mask_RCNN is properly set up")
    print("   Try running: pip install -r Mask_RCNN/requirements.txt")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print("💡 Check that all dependencies are installed") 
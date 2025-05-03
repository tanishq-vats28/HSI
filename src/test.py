import torch
import numpy as np
from scipy.io import loadmat
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

class Config:
    DATA_PATH = "../input/Indian_pines_corrected.mat"
    GT_PATH = "../input/IndianPines_gt.mat"
    PCA_COMPONENTS = 20
    PATCH_SIZE = 13
    DEVICE = "cpu"
    BATCH_SIZE = 16
    MODEL_PATH = "../output/cpu_best_model.pth"
    NUM_CLASSES = 16
    OUTPUT_DIR = "../output"

class LightweightSANet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(Config.PCA_COMPONENTS, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(16, Config.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

def load_and_predict():
    pca = load(f"{Config.OUTPUT_DIR}/cpu_pca.joblib")
    scaler = load(f"{Config.OUTPUT_DIR}/cpu_scaler.joblib")
    
    data = loadmat(Config.DATA_PATH)['indian_pines_corrected']
    gt = loadmat(Config.GT_PATH)['indian_pines_gt']
    print("Data:")
    print(data)
    print("GT:")
    print(gt)
    labeled_indices = np.where(gt.ravel() != 0)[0]
    y_true = gt.ravel()[labeled_indices] - 1
    
    classification_map = np.zeros_like(gt, dtype=np.uint8)
    
    model = LightweightSANet().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()
    
    predictions = []
    chunk_size = 256
    
    for i in range(0, len(labeled_indices), chunk_size):
        indices_chunk = labeled_indices[i:i+chunk_size]
        patches = []
        
        for idx in indices_chunk:
            row, col = divmod(idx, data.shape[1])
            half = Config.PATCH_SIZE // 2
            patch = data[
                max(0, row-half):min(data.shape[0], row+half+1),
                max(0, col-half):min(data.shape[1], col+half+1)
            ]
            pad = (
                (max(0, half-row), max(0, row+half+1 - data.shape[0])),
                (max(0, half-col), max(0, col+half+1 - data.shape[1])),
                (0,0)
            )
            padded_patch = np.pad(patch, pad, mode='reflect')
            patches.append(padded_patch)
        
        patches = np.array(patches)
        patches_flat = patches.reshape(-1, data.shape[2])
        patches_pca = pca.transform(patches_flat)
        patches_pca = patches_pca.reshape(-1, Config.PATCH_SIZE, Config.PATCH_SIZE, Config.PCA_COMPONENTS)
        patches_scaled = scaler.transform(patches_pca.reshape(-1, Config.PCA_COMPONENTS))
        patches_scaled = patches_scaled.reshape(-1, Config.PATCH_SIZE, Config.PATCH_SIZE, Config.PCA_COMPONENTS)
        
        with torch.no_grad():
            inputs = torch.FloatTensor(patches_scaled).permute(0, 3, 1, 2).to(Config.DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            predictions.extend(preds)
            
            for idx, pred in zip(indices_chunk, preds):
                row, col = divmod(idx, gt.shape[1])
                classification_map[row, col] = pred + 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(classification_map, cmap='jet')
    plt.colorbar(ticks=range(1, Config.NUM_CLASSES+1))
    plt.axis('off')
    plt.title('Classification Map')
    
    output_image_path = f"{Config.OUTPUT_DIR}/classification_result.png"
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Classification map saved to: {output_image_path}")

    print(f"Overall Accuracy: {100*accuracy_score(y_true, predictions):.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, predictions, target_names=[f'Class {i+1}' for i in range(Config.NUM_CLASSES)]))

if __name__ == "__main__":
    load_and_predict()
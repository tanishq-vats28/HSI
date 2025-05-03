import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from sklearn.utils.class_weight import compute_class_weight

class Config:
    DATA_PATH = "../input/Indian_pines_corrected.mat"
    GT_PATH = "../input/IndianPines_gt.mat"
    PCA_COMPONENTS = 20
    PATCH_SIZE = 13
    TEST_RATIO = 0.2
    DEVICE = "cpu"
    BATCH_SIZE = 16
    ACCUM_STEPS = 2
    EPOCHS = 200
    LR = 0.001
    DROPOUT = 0.3
    NUM_CLASSES = 16
    OUTPUT_DIR = "../output"

config = Config()

def load_data():
    data = loadmat(config.DATA_PATH)['indian_pines_corrected']
    gt = loadmat(config.GT_PATH)['indian_pines_gt']
    return data, gt

class MemorySafeDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.h_flip_prob = 0.3
        self.v_flip_prob = 0.3

    def __getitem__(self, index):
        x, y = self.tensors[0][index], self.tensors[1][index]
        if np.random.rand() < self.h_flip_prob:
            x = torch.flip(x, dims=[-1])
        if np.random.rand() < self.v_flip_prob:
            x = torch.flip(x, dims=[-2])
        return x, y

def preprocess():
    data, gt = load_data()
    labeled_indices = np.where(gt.ravel() != 0)[0]
    y = gt.ravel()[labeled_indices] - 1

    train_idx, test_idx, y_train, y_test = train_test_split(
        labeled_indices, y, test_size=config.TEST_RATIO, 
        stratify=y, random_state=42
    )
    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Class distribution in train: {np.unique(y_train, return_counts=True)}")
    print(f"Class distribution in test: {np.unique(y_test, return_counts=True)}")

    def extract_patches(indices):
        half_patch = config.PATCH_SIZE // 2
        patches = []
        for idx in indices:
            i, j = divmod(idx, data.shape[1])
            patch = data[
                max(0,i-half_patch):min(data.shape[0],i+half_patch+1),
                max(0,j-half_patch):min(data.shape[1],j+half_patch+1)
            ]
            pad = (
                (max(0,half_patch-i), max(0,i+half_patch+1-data.shape[0])),
                (max(0,half_patch-j), max(0,j+half_patch+1-data.shape[1])),
                (0,0)
            )
            padded_patch = np.pad(patch, pad, mode='reflect')
            patches.append(padded_patch)
        return np.array(patches)

    def process_chunk(indices, pca=None, scaler=None):
        chunk_size = 500
        processed = []
        for i in range(0, len(indices), chunk_size):
            raw_chunk = extract_patches(indices[i:i+chunk_size])
            flat_chunk = raw_chunk.reshape(-1, data.shape[2])
            if pca:
                pca_chunk = pca.transform(flat_chunk)
            else:
                pca_chunk = flat_chunk
            if scaler:
                scaled_chunk = scaler.transform(pca_chunk)
            else:
                scaled_chunk = pca_chunk
            final_chunk = scaled_chunk.reshape(
                len(raw_chunk), config.PATCH_SIZE, config.PATCH_SIZE, -1
            )
            processed.append(final_chunk)
        return np.vstack(processed)

    pca = IncrementalPCA(n_components=config.PCA_COMPONENTS)
    for i in range(0, len(train_idx), 500):
        chunk = extract_patches(train_idx[i:i+500])
        pca.partial_fit(chunk.reshape(-1, data.shape[2]))

    scaler = StandardScaler()
    for i in range(0, len(train_idx), 500):
        chunk = process_chunk(train_idx[i:i+500], pca=pca)
        scaler.partial_fit(chunk.reshape(-1, config.PCA_COMPONENTS))

    X_train = process_chunk(train_idx, pca=pca, scaler=scaler)
    X_test = process_chunk(test_idx, pca=pca, scaler=scaler)

    X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
    
    dump(pca, f"{config.OUTPUT_DIR}/cpu_pca.joblib")
    dump(scaler, f"{config.OUTPUT_DIR}/cpu_scaler.joblib")

    return X_train, X_test, torch.LongTensor(y_train), torch.LongTensor(y_test)

class LightweightSANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config.PCA_COMPONENTS, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(16, config.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

def train():
    X_train, X_test, y_train, y_test = preprocess()
    
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train.numpy()), 
        y=y_train.numpy()
    )
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)

    train_dataset = MemorySafeDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    model = LightweightSANet().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels) / config.ACCUM_STEPS
            loss.backward()
            if (batch_idx + 1) % config.ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * config.ACCUM_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test.to(config.DEVICE))
            val_acc = (val_outputs.argmax(1) == y_test.to(config.DEVICE)).float().mean().item()
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {total_loss/(batch_idx+1):.4f} | Acc: {100*correct/total:.1f}%")
        print(f"Val Acc: {100*val_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/cpu_best_model.pth")

    print(f"\nBest Validation Accuracy: {100*best_acc:.1f}%")

if __name__ == "__main__":
    train()
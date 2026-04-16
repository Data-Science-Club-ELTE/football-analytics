import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class DeepLearningModel(nn.Module):
    def __init__(self, lr, num_iters, *args, **kwargs):
        super(DeepLearningModel, self).__init__()

        self.torch_model = nn.Sequential(
            nn.Linear(in_features = 5, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
            
        )
        self.lossfn = nn.BCEWithLogitsLoss()
        self.optimizer = T.optim.SGD(self.torch_model.parameters(), lr=lr)
        self.epochs = num_iters

    def forward(self, X):
        X = self.torch_model(X)
        X = X.flatten()
        return X


    def fit(self, X, y, batch_size=32):
        self.train()

        X = T.as_tensor(X, dtype=T.float32)
        y = T.as_tensor(y, dtype=T.float32)

        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        pbar = tqdm(range(self.epochs), desc="Training XGmodel")

        for epoch in pbar:
            total_loss = 0.0
            for xb, yb in dl:
                logits = self(xb)
                loss = self.lossfn(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(ds)
            pbar.set_description(f"epoch {epoch+1}/{self.epochs} - loss: {avg_loss:.6f}")
    @T.no_grad()
    def predict(self, X, threshold: float = 0.5):
        X = T.as_tensor(X, dtype=T.float32)
        self.eval()
        X = self.torch_model(X)
        probs = T.sigmoid(X)
        probs = probs.cpu().numpy()
        predictions = (probs >= threshold).astype(int)
        return predictions, probs

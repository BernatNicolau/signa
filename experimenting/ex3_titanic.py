import pandas as pd

import torch  ## torch let's us create tensors and also provides helper functions
import torch.nn as nn  ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


class myNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.layers(x)


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_values):
        return self.model(input_values)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


class DataModule(L.LightningDataModule):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def setup(self, stage=None):
        self.dataset = TensorDataset(self.x, self.y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=3, shuffle=True)


mynet = myNN()
lit_model = LitModel(mynet)
early_stopping = EarlyStopping(
    monitor="train_loss",
    min_delta=0.0,
    patience=10,
)
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    mode="min",
    save_top_k=1,  # Only keeps the single best version
)
trainer = L.Trainer(
    max_epochs=50,
    accelerator="auto",
    callbacks=[early_stopping, checkpoint_callback],
)

training_data_path = r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\titanic\train.csv"
testing_data_path = r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\titanic\test.csv"

train_data = pd.read_csv(training_data_path)
test_data = pd.read_csv(testing_data_path)

data = train_data.copy()
data = data.drop(columns=["PassengerId", "Name"])
data = data.dropna()  # investigar, pq hi ha columnes amb nan que crec que no seran necessaries

desired_columns = data.columns.to_list()
desired_columns.remove("Survived")

x = torch.tensor(data[desired_columns].values).view(-1, len(desired_columns))
y = torch.tensor(data["Survived"].values).view(-1)
data_module = DataModule(x, y)

trainer.fit(lit_model, data_module)
lit_model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=myNN())

# 3. Now lit_model IS the best model
lit_model.eval()
with torch.no_grad():
    predictions = lit_model(x)
    print(predictions.numpy())

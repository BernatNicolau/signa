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
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.criterion = nn.MSELoss()

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
    max_epochs=20000,
    accelerator="auto",
    log_every_n_steps=1,
    callbacks=[early_stopping, checkpoint_callback],
)

x = torch.tensor([0.0, 0.5, 1.0]).view(-1, 1)
y = torch.tensor([0.0, 1.0, 0.0]).view(-1, 1)
data_module = DataModule(x, y)

trainer.fit(lit_model, data_module)
lit_model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=myNN())

# 3. Now lit_model IS the best model
lit_model.eval()
with torch.no_grad():
    predictions = lit_model(x)
    print(predictions.numpy())

import matplotlib.pyplot as plt

# Create 100 points from 0 to 1 to see the smooth curve
test_x = torch.linspace(-1, 2, 100).view(-1, 1)

# 1. Get the output of the FIRST layer + ReLU only
with torch.no_grad():
    # We stop the model halfway through
    hidden_layer_output = mynet.layers[0](test_x)
    activated_output = mynet.layers[1](hidden_layer_output)  # The ReLUs

# 2. Plot the 16 individual "hinges"
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(test_x.numpy(), activated_output.numpy())
plt.title("The 16 Individual ReLU 'Hinges'")
plt.xlabel("Input x")

# 3. Plot the final result (The sum)
plt.subplot(1, 2, 2)
final_output = lit_model(test_x)
plt.plot(test_x.numpy(), final_output.detach().numpy(), color="red", linewidth=3)
plt.scatter(x, y, color="black")  # Plot your 3 original dots
plt.title("The Final Combined 'Mountain'")
plt.xlabel("Input x")

plt.tight_layout()
plt.show()


weights = lit_model.model.layers[2].weight.detach().numpy()
print("Importance of each of the 16 neurons:")
print(weights)

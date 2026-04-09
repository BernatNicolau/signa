import pandas as pd

import torch  ## torch let's us create tensors and also provides helper functions
import torch.nn as nn  ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics


class myNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # cannot predefine the number of features because it depends on the number of one-hot encoded features
        self.layers = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        """HOOK. Called when you call model(input_values). Should contain the logic for one forward pass of the model."""
        return self.layers(x)


class LitModel(L.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        # start the model as None, and then initialize it in setup() when we have access to the datamodule and can get the input dimension
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss, ideal for binary classification tasks
        # self.criterion = nn.CrossEntropyLoss() # Cross Entropy Loss, ideal for multi-class classification tasks
        # self.criterion = nn.MSELoss() # Mean Squared Error Loss, ideal for regression tasks
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def setup(self, stage=None):
        """HOOK. Called on every process in distributed settings. Use for things that need to be done on every GPU."""
        if stage == "fit" and self.model is None:
            self.model = myNN(num_features=self.trainer.datamodule.input_dim)

    def forward(self, input_values):
        """HOOK. Called when you call model(input_values). Should contain the logic for one forward pass of the model."""
        return self.model(input_values)

    def training_step(self, batch, batch_idx):
        """HOOK. Called by trainer.fit(). Should contain the logic for one training step of the model, which typically includes a forward pass, loss calculation, and logging."""
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc(torch.sigmoid(output), y.int()), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """HOOK. Called by trainer.fit(). Should contain the logic for one validation step of the model, which typically includes a forward pass, loss calculation, and logging."""
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        # Logging validation loss (for EarlyStopping and Checkpointing)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc(torch.sigmoid(output), y.int()), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def testing_step(self, batch, batch_idx):
        """HOOK. Called by trainer.test(). Should contain the logic for one testing step of the model, which typically includes a forward pass, loss calculation, and logging."""
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        # Logging test loss (for EarlyStopping and Checkpointing)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """HOOK: Called by trainer.predict()."""
        logits = self(batch[0])
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        """HOOK. Called by trainer.fit(). Should return the optimizer(s) to be used during training."""
        return optim.Adam(self.parameters(), lr=0.001)


def get_titanic_pipeline():
    """Constructs a data preprocessing pipeline for the dataset, which includes transformations for both numerical and categorical features."""
    num_features = ["Age", "Fare", "FamilySize"]
    # imputer fills in missing values, scaler standardizes the features to have mean 0 and variance 1
    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    onehot_features = ["Sex", "Embarked", "Title"]
    # imputer fills in missing values with the string "missing", onehot encoder converts categorical variables into a one-hot encoded format, where each category is represented as a binary vector
    onehot_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features), ("onehot", onehot_transformer, onehot_features)])

    return preprocessor


class TitanicDataModule(L.LightningDataModule):
    def __init__(self, csv_path: str, batch_size: int = 32):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.pipeline = get_titanic_pipeline()

    @property
    def input_dim(self):
        """This will work as long as setup() has been called."""
        if hasattr(self, "train_ds"):
            return self.train_ds.tensors[0].shape[1]
        return None

    def prepare_data(self):
        """HOOK. Called only on 1 CPU. Use for things that might write to disk or need to be done only from a single process in distributed settings."""

        self.df = pd.read_csv(self.csv_path)
        self.df["Title"] = self.df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        # SibSp (number of siblings/spouses aboard) and Parch (number of parents/children aboard)
        self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] + 1
        self.names = self.df["Name"]
        self.PassengerId = self.df["PassengerId"]
        self.df = self.df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])

    def setup(self, stage=None):
        """HOOK. Called on every process in distributed settings. Use for things that need to be done on every GPU."""

        # validation set is used to modify hyperparameters and make decisions about the model, for example
        # when to stop training (early stopping) or
        # which version of the model to keep (model checkpointing) or
        # learning rate adjustments (learning rate scheduler)
        if stage == "fit" or stage is None:
            temp_df, test_df = train_test_split(self.df, test_size=0.1, random_state=42)
            train_df, val_df = train_test_split(temp_df, test_size=0.15, random_state=42)

            x_train = self.pipeline.fit_transform(train_df.drop("Survived", axis=1))
            y_train = torch.tensor(train_df["Survived"].values).float().unsqueeze(1)

            x_val = self.pipeline.transform(val_df.drop("Survived", axis=1))
            y_val = torch.tensor(val_df["Survived"].values).float().unsqueeze(1)

            x_test = self.pipeline.transform(test_df.drop("Survived", axis=1))
            y_test = torch.tensor(test_df["Survived"].values).float().unsqueeze(1)

            # convert to tensors and create TensorDatasets for training and validation sets
            self.train_ds = TensorDataset(torch.tensor(x_train).float(), y_train)
            self.train_names = self.names[train_df.index]
            self.val_ds = TensorDataset(torch.tensor(x_val).float(), y_val)
            self.val_names = self.names[val_df.index]
            self.test_ds = TensorDataset(torch.tensor(x_test).float(), y_test)
            self.test_names = self.names[test_df.index]

        if stage == "predict":
            x_pred = self.pipeline.transform(self.df)
            self.pred_ds = TensorDataset(torch.tensor(x_pred).float())

    def train_dataloader(self):
        """HOOK. Called by trainer.fit(). Should return a DataLoader for the training set."""
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """HOOK. Called by trainer.fit(). Should return a DataLoader for the validation set."""
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        """HOOK. Called by trainer.test(). Should return a DataLoader for the test set."""
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        """HOOK. Called by trainer.predict(). Should return a DataLoader for the test set."""
        return DataLoader(self.pred_ds, batch_size=self.batch_size, shuffle=False)


data = TitanicDataModule(csv_path=r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\titanic\train.csv")

lit_model = LitModel()
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=50,
    mode="min",
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,  # Only keeps the single best version
)
trainer = L.Trainer(
    max_epochs=500,
    accelerator="auto",
    log_every_n_steps=1,
    callbacks=[early_stopping, checkpoint_callback],
)


trainer.fit(lit_model, data)
lit_model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=myNN(data.input_dim))
best_val = trainer.checkpoint_callback.best_model_score
print(f"Best Val Loss: {best_val:.4f}, path: {trainer.checkpoint_callback.best_model_path}")
lit_model.eval()
all_probs = []
all_truth = []
with torch.no_grad():
    for batch in data.val_dataloader():
        x, y = batch
        logits = lit_model(x)
        probs = torch.sigmoid(logits)

        # Move to CPU and convert to numpy to free up GPU memory
        all_probs.append(probs.cpu().numpy().reshape(-1))
        all_truth.append(y.cpu().numpy().reshape(-1))


results_df = pd.DataFrame(index=data.val_names, data={"probs": np.concatenate(all_probs), "truth": np.concatenate(all_truth)})
results_df["pred"] = (results_df["probs"] > 0.5).astype(float)
accuracy = (results_df["pred"] == results_df["truth"]).mean()
print(f"Batch Accuracy (with validation data): {accuracy * 100:.2f}%")

lit_model.eval()
all_probs = []
all_truth = []
with torch.no_grad():
    for batch in data.test_dataloader():
        x, y = batch
        logits = lit_model(x)
        probs = torch.sigmoid(logits)

        # Move to CPU and convert to numpy to free up GPU memory
        all_probs.append(probs.cpu().numpy().reshape(-1))
        all_truth.append(y.cpu().numpy().reshape(-1))


results_df = pd.DataFrame(index=data.test_names, data={"probs": np.concatenate(all_probs), "truth": np.concatenate(all_truth)})
results_df["is_mistake"] = (results_df["probs"] > 0.5).astype(float) != results_df["truth"]
mistakes = results_df[results_df["is_mistake"]]
corrects = results_df[~results_df["is_mistake"]]

plt.figure(figsize=(16, 8))

plt.scatter(corrects.index, corrects["probs"], color="green", marker="o", label="Correct", alpha=0.5)

plt.scatter(mistakes.index, mistakes["probs"], color="red", marker="x", s=100, label="Mistake", linewidths=2)

plt.scatter(results_df.index, results_df["truth"], color="black", marker=".", alpha=0.3, label="Actual Truth")

plt.xticks(ticks=np.arange(len(results_df)), labels=results_df.index, rotation=90, fontsize=8)
plt.grid(True, linestyle=":", alpha=0.6)

plt.axhline(y=0.5, color="gray", linestyle="--", label="Threshold")

plt.title(f"Model Mistakes ({len(mistakes)} errors out of {len(results_df)})")
plt.ylabel("Predicted Probability")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

plt.tight_layout()
plt.show()


results_df["pred"] = (results_df["probs"] > 0.5).astype(float)
accuracy = (results_df["pred"] == results_df["truth"]).mean()
print(f"Batch Accuracy (with testing data): {accuracy * 100:.2f}%")


test_data_path = r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\titanic\test.csv"
data.csv_path = test_data_path
predictions = trainer.predict(lit_model, data)
all_probs = torch.cat(predictions).cpu().numpy()
results_df = pd.DataFrame(index=data.PassengerId, data={"probs": np.concatenate(all_probs)})
results_df["Survived"] = (results_df["probs"] > 0.5).astype(float)
results_df.index.name = "PassengerId"
results_df = results_df.drop(columns=["probs"])
results_df.to_csv(r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\titanic\titanic_predictions.csv")
# print(results_df)


test_data_path = r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\titanic\mariona.csv"
data.csv_path = test_data_path
predictions = trainer.predict(lit_model, data)
all_probs = torch.cat(predictions).cpu().numpy()
results_df = pd.DataFrame(index=data.PassengerId, data={"probs": np.concatenate(all_probs)})
results_df["Survived"] = (results_df["probs"] > 0.5).astype(float)
results_df.index.name = "PassengerId"
results_df = results_df.drop(columns=["probs"])
print(results_df)

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
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """HOOK. Called when you call model(input_values). Should contain the logic for one forward pass of the model."""
        return self.layers(x)


class LitModel(L.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        # start the model as None, and then initialize it in setup() when we have access to the datamodule and can get the input dimension
        self.model = model
        # self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss, ideal for binary classification tasks
        # self.criterion = nn.CrossEntropyLoss() # Cross Entropy Loss, ideal for multi-class classification tasks
        self.criterion = nn.MSELoss()  # Mean Squared Error Loss, ideal for regression tasks
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
        # accuracy only for classification tasks, not regression
        # self.log("train_acc", self.train_acc(torch.sigmoid(output), y.int()), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """HOOK. Called by trainer.fit(). Should contain the logic for one validation step of the model, which typically includes a forward pass, loss calculation, and logging."""
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        # Logging validation loss (for EarlyStopping and Checkpointing)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # accuracy only for classification tasks, not regression
        # self.log("val_acc", self.val_acc(torch.sigmoid(output), y.int()), prog_bar=True, on_step=False, on_epoch=True)
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
        return logits
        # return torch.sigmoid(logits)

    def configure_optimizers(self):
        """HOOK. Called by trainer.fit(). Should return the optimizer(s) to be used during training."""
        return optim.Adam(self.parameters(), lr=0.001)


def get_feature_pipeline(num_features=None, onehot_features=None):
    """Constructs a data preprocessing pipeline for the dataset, which includes transformations for both numerical and categorical features."""
    if num_features is None:
        num_features = ["CustomColumn1", "CustomColumn2"]
    # imputer fills in missing values, scaler standardizes the features to have mean 0 and variance 1
    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    if onehot_features is None:
        onehot_features = ["CustomColumn1", "CustomColumn2"]
    # imputer fills in missing values with the string "missing", onehot encoder converts categorical variables into a one-hot encoded format, where each category is represented as a binary vector
    onehot_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features), ("onehot", onehot_transformer, onehot_features)])

    return preprocessor, get_used_columns(preprocessor)


def get_y_pipeline():
    """
    Pipeline for the target variable.
    """
    y_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    return y_transformer


def get_used_columns(pipeline):
    """Constructs a list of the original feature names that are used in the pipeline, which is necessary for correctly selecting and transforming the data during preprocessing."""
    original_features = []
    for name, transformer, columns in pipeline.transformers:
        original_features.extend(columns)
    return original_features


class DataModule(L.LightningDataModule):
    def __init__(self, csv_path: str, batch_size: int = 32):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.y_pipeline = get_y_pipeline()
        self.y_columns = ["SalePrice"]

    @property
    def input_dim(self):
        """This will work as long as setup() has been called."""
        if hasattr(self, "train_ds"):
            return self.train_ds.tensors[0].shape[1]
        return None

    def prepare_data(self):
        """HOOK. Called only on 1 CPU. Use for things that might write to disk or need to be done only from a single process in distributed settings."""

        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.drop(columns=["Id"])

    def setup_feature_pipeline(self):
        """Called in setup() to initialize the feature pipeline, which is necessary for correctly preprocessing the data during training, validation, testing, and prediction."""
        num_features = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_features = self.df.select_dtypes(include=["object"]).columns.tolist()
        num_features = [col for col in num_features if col not in self.y_columns]
        cat_features = [col for col in cat_features if col not in self.y_columns]
        self.feature_pipeline, self.used_feature_columns = get_feature_pipeline(num_features=num_features, onehot_features=cat_features)

    def setup(self, stage=None):
        """HOOK. Called on every process in distributed settings. Use for things that need to be done on every GPU."""

        # validation set is used to modify hyperparameters and make decisions about the model, for example
        # when to stop training (early stopping) or
        # which version of the model to keep (model checkpointing) or
        # learning rate adjustments (learning rate scheduler)
        if stage == "fit" or stage is None:
            temp_df, test_df = train_test_split(self.df, test_size=0.1, random_state=42)
            train_df, val_df = train_test_split(temp_df, test_size=0.15, random_state=42)
            if not hasattr(self, "feature_pipeline"):
                self.setup_feature_pipeline()
            x_train = self.feature_pipeline.fit_transform(train_df[self.used_feature_columns])
            y_train = self.y_pipeline.fit_transform(train_df[self.y_columns].values.reshape(-1, 1))

            x_val = self.feature_pipeline.transform(val_df[self.used_feature_columns])
            y_val = self.y_pipeline.transform(val_df[self.y_columns].values.reshape(-1, 1))

            x_test = self.feature_pipeline.transform(test_df[self.used_feature_columns])
            y_test = self.y_pipeline.transform(test_df[self.y_columns].values.reshape(-1, 1))

            # convert to tensors and create TensorDatasets for training and validation sets
            self.train_ds = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
            self.val_ds = TensorDataset(torch.tensor(x_val).float(), torch.tensor(y_val).float())
            self.test_ds = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float())

        if stage == "predict":
            x_pred = self.feature_pipeline.transform(self.df[self.used_feature_columns])
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


data = DataModule(csv_path=r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\houses\train.csv")

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
    filename="best-model-{epoch:02d}-{val_loss:.2f}",
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
all_logits = []
all_truth = []
with torch.no_grad():
    for batch in data.val_dataloader():
        x, y = batch
        logits = lit_model(x)

        # Move to CPU and convert to numpy to free up GPU memory
        all_logits.append(logits.cpu().numpy().reshape(-1))
        all_truth.append(y.cpu().numpy().reshape(-1))
pred_dollars = data.y_pipeline.inverse_transform(np.concatenate(all_logits).reshape(-1, 1)).ravel()
truth_dollars = data.y_pipeline.inverse_transform(np.concatenate(all_truth).reshape(-1, 1)).ravel()
plt.scatter(truth_dollars, pred_dollars)

results_df = pd.DataFrame(data={"pred_dollars": pred_dollars, "truth_dollars": truth_dollars})

lit_model.eval()
all_logits = []
all_truth = []
with torch.no_grad():
    for batch in data.test_dataloader():
        x, y = batch
        logits = lit_model(x)

        # Move to CPU and convert to numpy to free up GPU memory
        all_logits.append(logits.cpu().numpy().reshape(-1))
        all_truth.append(y.cpu().numpy().reshape(-1))

pred_dollars = data.y_pipeline.inverse_transform(np.concatenate(all_logits).reshape(-1, 1)).ravel()
truth_dollars = data.y_pipeline.inverse_transform(np.concatenate(all_truth).reshape(-1, 1)).ravel()
plt.scatter(truth_dollars, pred_dollars)
results_df = pd.DataFrame(data={"pred_dollars": pred_dollars, "truth_dollars": truth_dollars})


test_data_path = r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\houses\test.csv"
raw_test_df = pd.read_csv(test_data_path)
data.csv_path = test_data_path
predictions = trainer.predict(lit_model, data)
all_logits = torch.cat(predictions).cpu().numpy()
pred_dollars = data.y_pipeline.inverse_transform(np.concatenate(all_logits).reshape(-1, 1)).ravel()

results_df = pd.DataFrame(index=raw_test_df["Id"], data={"SalePrice": pred_dollars})
results_df.index.name = "Id"
results_df.to_csv(r"C:\Users\BERNIC\Documents\GitHub\signa\signa\experimenting\houses\houses_predictions.csv")
# print(results_df)

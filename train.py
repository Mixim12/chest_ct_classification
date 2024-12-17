import torch
import os
import torch.nn as nn
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Trainer:
    def __init__(self, config, train_loader, valid_loader, test_loader, model) -> None:
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.best_valid_acc = -1.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.loss_fn = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=self.config.get("log_dir", "runs"))

    def train_epoch(self, epoch):
        BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(
            self.train_loader, bar_format=BAR_FORMAT, desc=f"Epoch {epoch+1} [Train]"
        ):
            x = x.to(self.device).float() / 255.0
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(self.train_loader)
        acc = (correct / total) * 100
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Train", acc, epoch)

        logging.info(
            f"\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}, Train Accuracy: {acc:.2f}%"
        )

    def validate(self, epoch):
        BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for _, (x, y) in enumerate(
                tqdm(self.valid_loader, bar_format=BAR_FORMAT, desc="Validation")
            ):
                x = x.to(self.device).float() / 255.0
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                total_loss += loss.item()
                _, predicted = torch.max(pred, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / len(self.valid_loader)
        acc = (correct / total) * 100
        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
        self.writer.add_scalar("Accuracy/Validation", acc, epoch)

        logging.info(
            f"Epoch {epoch+1} - Validation Loss: {avg_loss:.4f}, Validation Accuracy: {acc:.2f}%"
        )

        # Save model if validation accuracy improves
        if acc > self.best_valid_acc:
            logging.info(
                f"Validation accuracy improved from {self.best_valid_acc:.2f}% to {acc:.2f}%. Saving model."
            )
            self.best_valid_acc = acc
            self.save_model(epoch, is_best=True)

        torch.cuda.empty_cache()

    def save_model(self, epoch, is_best=False):
        os.makedirs(self.config["save_model_path"], exist_ok=True)
        if is_best:
            model_path = os.path.join(self.config["save_model_path"], "best_model.pth")
            torch.save(self.model.state_dict(), model_path)
            logging.info(f"Best model saved to {model_path} (Epoch {epoch+1})")
        else:
            model_path = os.path.join(
                self.config["save_model_path"], f"model_epoch_{epoch+1}.pth"
            )
            torch.save(self.model.state_dict(), model_path)
            logging.info(f"Model for epoch {epoch+1} saved to {model_path}")

    def train(self):
        for epoch in range(self.config["train_epochs"]):
            logging.info(f"Starting Training for Epoch {epoch+1}")
            self.train_epoch(epoch)
            self.validate(epoch)

        logging.info("Training completed.")
        self.writer.close()

    def test(self):
        BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
        model_path = os.path.join(self.config["save_model_path"], "best_model.pth")
        if not os.path.exists(model_path):
            logging.error(
                f"Model file {model_path} does not exist. Cannot perform testing."
            )
            return

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in tqdm(self.test_loader, bar_format=BAR_FORMAT, desc="Testing"):
                x = x.to(self.device).float() / 255.0
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                total_loss += loss.item()
                _, predicted = torch.max(pred, 1)

                # Move tensors to CPU and convert to numpy for sklearn
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        acc = accuracy_score(all_labels, all_preds) * 100
        precision = (
            precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            * 100
        )
        recall = recall_score(all_labels, all_preds, average="weighted") * 100
        f1 = f1_score(all_labels, all_preds, average="weighted") * 100
        conf_matrix = confusion_matrix(all_labels, all_preds)

        logging.info(f"Test Loss: {avg_loss:.4f}")
        logging.info(f"Test Accuracy: {acc:.2f}%")
        logging.info(f"Test Precision: {precision:.2f}%")
        logging.info(f"Test Recall: {recall:.2f}%")
        logging.info(f"Test F1-Score: {f1:.2f}%")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

        class_report = classification_report(all_labels, all_preds, zero_division=0)
        logging.info(f"Classification Report:\n{class_report}")

        torch.cuda.empty_cache()

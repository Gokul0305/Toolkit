from abc import ABC, abstractmethod
from tqdm import tqdm
from torch import no_grad, save

class Trainer(ABC):

    """
    Abstract Trainer Class Implementation
    
    """

    def __init__(self) -> None:
        ...

    @abstractmethod    
    def training_step(self) -> None:
        ...
    
    @abstractmethod
    def validation_step(self) -> None:
        ...
    
    
    @abstractmethod
    def configure_optimizer(self,) -> None:
        ...
    
    @abstractmethod
    def configure_loss(self) -> None:
        ...
    
    def save_model(self):
        print(f"Saving the model file in ")
        save(self.model.state_dict(),"checkpoints/1.pth")

    def fit(self,train_loader, val_dataloader):
        for epoch in tqdm(range(self.settings["train"]["epochs"]), unit ="epoch", desc = "Epoch"):

            # Training Step for a Epoch

            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                train_loss += self.training_step(self.model, self.criterion, self.optimizer, batch)

            avg_train_loss = train_loss / len(train_loader)
            # self.log_train(avg_train_loss)
            tqdm.write(f'Training Loss: {avg_train_loss:.4f}')

            # Validation Step for a Epoch

            self.model.eval()
            val_loss = 0.0
            with no_grad():
                for val_batch in val_dataloader:
                    val_loss += self.validation_step(self.model, self.criterion, val_batch)
            avg_val_loss = val_loss / len(val_dataloader)
            # self.log_validation(avg_val_loss)
            tqdm.write(f'Validation Loss: {avg_val_loss:.4f}')

        self.save_model()
        return self.model
    
    

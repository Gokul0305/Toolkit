from src.trainers.base_trainer import Trainer
from src.optimizer.optimizer_factory import OptimFactory
from src.losses.loss_factory import LossFactory


class ClassificationTrainer(Trainer):

    """
        Trainer implementation for Classification
        -----------------------------------------------
        The class `ClassificationTrainer` constructs a trainer object
        that can automatically configure optimizer and loss function.
    """

    def __init__(self,model, settings, device) -> None:
        super().__init__()
        self.model = model
        self.settings = settings
        self.device = device
        self.criterion = self.configure_loss()
        self.optimizer = self.configure_optimizer()
    
    def training_step(self,  model, criterion, optimizer, batch) -> None:
        """
        Training_step for a batch. It return training loss for a batch
        """
        img1, img2, label = batch
        img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = model(img1,img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        return loss.item()
        
    def validation_step(self, model, criterion, val_batch) -> None:
        """
        Validation_step for a batch. It return validation loss for a batch
        """
        img1, img2, label = val_batch
        img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        return loss
    
    def configure_loss(self) -> None:

        """Configure the loss function"""

        return LossFactory().build_loss(loss_type= self.settings["train"]["loss"])
    
    def configure_optimizer(self) -> None:
        
        """Configure the optimizer function"""

        return OptimFactory().build_optim(optim_type = self.settings["train"]["optimizer"],
                                          model_param= self.model.parameters(),
                                          settings=self.settings)        
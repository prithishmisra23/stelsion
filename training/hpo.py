import optuna
import numpy as np
from models.architecture import ExoplanetDetectorNet
from training.train import Trainer

def run_hpo_study(train_data, val_data, n_trials=3, epochs=2):
    """
    Runs a short Optuna HPO study to find the best hyperparameters.
    """
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        
        # Instantiate model with dynamic dropout
        model = ExoplanetDetectorNet(dropout=dropout)
        trainer = Trainer(model=model, lr=lr)
        
        history = trainer.train(
            train_data, 
            val_data, 
            epochs=epochs, 
            batch_size=batch_size, 
            early_stopping_patience=2
        )
        
        # Return final validation accuracy
        val_acc = history["val_acc"][-1]
        return val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value

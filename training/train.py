import os
import json
import tensorflow as tf
import numpy as np
from models.architecture import ExoplanetDetectorNet

class Trainer:
    def __init__(self, model=None, lr=1e-3, weight_decay=1e-4, device=None, checkpoint_dir='saved_models'):
        # TensorFlow handles device placement automatically, but we can verify GPU usage
        self.device = device or ('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0')
        self.model = model or ExoplanetDetectorNet()
        
        # Build model by calling on dummy input to instantiate weights cleanly
        self.model(np.zeros((1, 2000, 1), dtype=np.float32), training=False)
        
        self.criterion = tf.keras.losses.BinaryCrossentropy()
        
        # Robust optimizer definition
        try:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
        except TypeError:
            # Fallback for older TensorFlow versions that do not support weight_decay directly in Adam
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            
        # Plateau scheduler state
        self.scheduler_factor = 0.5
        self.scheduler_patience = 3
        self.scheduler_counter = 0
        self.best_scheduler_loss = float('inf')
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, dataset):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataset:
            if len(batch_x.shape) == 2:
                batch_x = tf.expand_dims(batch_x, axis=-1)
            batch_y = tf.expand_dims(batch_y, axis=-1)
            
            with tf.GradientTape() as tape:
                out, _ = self.model(batch_x, training=True)
                loss = self.criterion(batch_y, out)
                
            grads = tape.gradient(loss, self.model.trainable_variables)
            # Apply Gradient Clipping (max norm 1.0)
            clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
            self.optimizer.apply_gradients(zip(clipped_grads, self.model.trainable_variables))
            
            total_loss += loss.numpy() * len(batch_x)
            preds = tf.cast(out > 0.5, tf.float32)
            correct += tf.reduce_sum(tf.cast(tf.equal(preds, batch_y), tf.float32)).numpy()
            total += len(batch_x)
            
        return total_loss / total, correct / total

    def evaluate(self, dataset):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataset:
            if len(batch_x.shape) == 2:
                batch_x = tf.expand_dims(batch_x, axis=-1)
            batch_y = tf.expand_dims(batch_y, axis=-1)
            
            out, _ = self.model(batch_x, training=False)
            loss = self.criterion(batch_y, out)
            
            total_loss += loss.numpy() * len(batch_x)
            preds = tf.cast(out > 0.5, tf.float32)
            correct += tf.reduce_sum(tf.cast(tf.equal(preds, batch_y), tf.float32)).numpy()
            total += len(batch_x)
            
        return total_loss / total, correct / total

    def train(self, train_data, val_data, epochs=20, batch_size=32, early_stopping_patience=5, use_amp=False, callback=None):
        train_x, train_y = train_data
        val_x, val_y = val_data
        
        # Prepare datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(buffer_size=len(train_x)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_dataset)
            val_loss, val_acc = self.evaluate(val_dataset)
            
            # Scheduler Step (ReduceLROnPlateau emulation)
            if val_loss < self.best_scheduler_loss:
                self.best_scheduler_loss = val_loss
                self.scheduler_counter = 0
            else:
                self.scheduler_counter += 1
                if self.scheduler_counter >= self.scheduler_patience:
                    old_lr = float(self.optimizer.learning_rate.numpy())
                    new_lr = old_lr * self.scheduler_factor
                    self.optimizer.learning_rate.assign(new_lr)
                    print(f"Epoch {epoch}: ReduceLROnPlateau reducing learning rate to {new_lr:.6e}.")
                    self.scheduler_counter = 0
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Custom callback to stream logs
            if callback:
                callback(epoch, train_loss, train_acc, val_loss, val_acc)
            
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Save Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.weights.h5')
                self.model.save_weights(checkpoint_path)
                
                # Save Metadata
                meta_path = os.path.join(self.checkpoint_dir, 'best_model_meta.json')
                with open(meta_path, 'w') as f:
                    json.dump({
                        'epoch': epoch,
                        'val_loss': float(val_loss),
                        'val_acc': float(val_acc)
                    }, f)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
                
        return history

    def load_checkpoint(self, path):
        # Convert path to Keras weights path if it refers to PyTorch pt file
        if path.endswith('.pt'):
            path = path.replace('.pt', '.weights.h5')
            
        if os.path.exists(path):
            self.model.load_weights(path)
            print(f"Loaded checkpoint weights from {path}")
            
            # Try to load meta
            meta_path = path.replace('.weights.h5', '_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                return meta
            return {'epoch': 0, 'val_loss': 0.0, 'val_acc': 0.0}
        else:
            # Check if weights file exists with standard suffix under same directory
            dir_name = os.path.dirname(path)
            alt_path = os.path.join(dir_name, 'best_model.weights.h5')
            if os.path.exists(alt_path):
                self.model.load_weights(alt_path)
                print(f"Loaded checkpoint weights from alternative path {alt_path}")
                meta_path = alt_path.replace('.weights.h5', '_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        return json.load(f)
                return {'epoch': 0, 'val_loss': 0.0, 'val_acc': 0.0}
            raise FileNotFoundError(f"No weights checkpoint found at {path}")

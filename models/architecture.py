import tensorflow as tf
from tensorflow.keras import layers, models

class ResidualBlock1D(layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2, **kwargs):
        super(ResidualBlock1D, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_rate = dropout

    def build(self, input_shape):
        actual_in_channels = input_shape[-1]
        self.conv1 = layers.Conv1D(self.out_channels, kernel_size=5, strides=self.stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv1D(self.out_channels, kernel_size=5, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)
        self.relu2 = layers.ReLU()
        
        if self.stride != 1 or actual_in_channels != self.out_channels:
            self.shortcut_conv = layers.Conv1D(self.out_channels, kernel_size=1, strides=self.stride, padding='same', use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()
            
        super(ResidualBlock1D, self).build(input_shape)

    def call(self, x, training=None):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)
        out = self.dropout(out, training=training)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        if hasattr(self, 'shortcut_conv'):
            shortcut_x = self.shortcut_conv(x)
            shortcut_x = self.shortcut_bn(shortcut_x, training=training)
        else:
            shortcut_x = x
            
        out += shortcut_x
        out = self.relu2(out)
        return out

class SelfAttention1D(layers.Layer):
    def __init__(self, in_channels=None, **kwargs):
        super(SelfAttention1D, self).__init__(**kwargs)
        self.in_channels = in_channels

    def build(self, input_shape):
        actual_in_channels = input_shape[-1]
        self.query = layers.Conv1D(actual_in_channels // 8, kernel_size=1)
        self.key = layers.Conv1D(actual_in_channels // 8, kernel_size=1)
        self.value = layers.Conv1D(actual_in_channels, kernel_size=1)
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(SelfAttention1D, self).build(input_shape)

    def call(self, x, training=None):
        # x shape: [batch_size, seq_len, channels]
        proj_query = self.query(x)  # [B, N, C']
        proj_key = self.key(x)      # [B, N, C']
        
        # energy = proj_query * proj_key^T
        energy = tf.matmul(proj_query, proj_key, transpose_b=True)  # [B, N, N]
        attention = tf.nn.softmax(energy, axis=-1)  # [B, N, N]
        
        proj_value = self.value(x)  # [B, N, C]
        out = tf.matmul(attention, proj_value)  # [B, N, C]
        
        out = self.gamma * out + x
        return out, attention

class ExoplanetDetectorNet(models.Model):
    def __init__(self, input_len=2000, dropout=0.3, **kwargs):
        super(ExoplanetDetectorNet, self).__init__(**kwargs)
        self.input_len = input_len
        
        # Initial 1D CNN Layer (expecting [B, seq_len, 1])
        self.conv1 = layers.Conv1D(32, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.maxpool = layers.MaxPool1D(pool_size=3, strides=2, padding='same')
        
        # Residual Blocks
        self.res1 = ResidualBlock1D(32, 64, stride=2, dropout=dropout)
        self.res2 = ResidualBlock1D(64, 128, stride=2, dropout=dropout)
        self.res3 = ResidualBlock1D(128, 256, stride=2, dropout=dropout)
        
        # Self Attention Layer
        self.attention = SelfAttention1D(256)
        
        # Global Average Pooling
        self.gap = layers.GlobalAveragePooling1D()
        
        # Fully Connected Layers
        self.fc1 = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.fc2 = layers.Dense(1, activation='sigmoid')
        
    def call(self, x, training=None):
        # Input shape: [batch_size, seq_len] or [batch_size, seq_len, 1]
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1) # Add channel dimension -> [batch_size, seq_len, 1]
            
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        
        x, attn_map = self.attention(x, training=training)
        
        x = self.gap(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        
        return x, attn_map

import tensorflow as tf
from tensorflow.keras import layers, models

class InceptionModule1D(layers.Layer):
    def __init__(self, filters, bottleneck_filters=16, stride=1, **kwargs):
        """
        InceptionTime Block for 1D time series.
        Applies parallel convolutions of different scale lengths (k=9, 19, 39)
        to capture both short and long duration exoplanet transit phases.
        """
        super(InceptionModule1D, self).__init__(**kwargs)
        self.filters = filters
        self.bottleneck_filters = bottleneck_filters
        self.stride = stride

    def build(self, input_shape):
        actual_in_channels = input_shape[-1]
        
        # Bottleneck convolution to compress features
        self.bottleneck = layers.Conv1D(
            self.bottleneck_filters, kernel_size=1, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        
        # Parallel multi-scale convolutions
        self.conv_small = layers.Conv1D(
            self.filters, kernel_size=9, strides=self.stride, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        self.conv_medium = layers.Conv1D(
            self.filters, kernel_size=19, strides=self.stride, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        self.conv_large = layers.Conv1D(
            self.filters, kernel_size=39, strides=self.stride, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        
        # Max pooling branch
        self.maxpool = layers.MaxPool1D(pool_size=3, strides=self.stride, padding='same')
        self.conv_pool = layers.Conv1D(
            self.filters, kernel_size=1, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
        # Shortcut connection for residual learning
        self.out_channels = 4 * self.filters
        if actual_in_channels != self.out_channels or self.stride != 1:
            self.shortcut_conv = layers.Conv1D(
                self.out_channels, kernel_size=1, strides=self.stride, padding='same', 
                kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
            )
            self.shortcut_bn = layers.BatchNormalization()
            
        super(InceptionModule1D, self).build(input_shape)

    def call(self, x, training=None):
        bottleneck_x = self.bottleneck(x)
        
        out_small = self.conv_small(bottleneck_x)
        out_medium = self.conv_medium(bottleneck_x)
        out_large = self.conv_large(bottleneck_x)
        
        out_pool = self.conv_pool(self.maxpool(x))
        
        # Concatenate parallel branches
        out = tf.concat([out_small, out_medium, out_large, out_pool], axis=-1)
        out = self.bn(out, training=training)
        
        if hasattr(self, 'shortcut_conv'):
            shortcut_x = self.shortcut_conv(x)
            if hasattr(self, 'shortcut_bn'):
                shortcut_x = self.shortcut_bn(shortcut_x, training=training)
        else:
            shortcut_x = x
            
        out += shortcut_x
        return self.relu(out)

class MultiAxisAttention2D(layers.Layer):
    def __init__(self, num_orbits=10, orbit_len=25, num_heads=4, **kwargs):
        """
        Multi-Axis Self-Attention Layer.
        Folds the 1D curve into a 2D matrix [Orbits, Phase Steps] to perform:
        - Major-Axis Attention: Checks transit dip shapes along the horizontal phase.
        - Minor-Axis Attention: Checks orbit-to-orbit transit stability along the vertical orbits.
        """
        super(MultiAxisAttention2D, self).__init__(**kwargs)
        self.num_orbits = num_orbits
        self.orbit_len = orbit_len
        self.num_heads = num_heads

    def build(self, input_shape):
        in_channels = input_shape[-1]
        
        # Self-Attention modules for both axes
        self.major_attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=in_channels // self.num_heads)
        self.minor_attn = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=in_channels // self.num_heads)
        
        self.major_bn = layers.BatchNormalization()
        self.minor_bn = layers.BatchNormalization()
        
        # Down-projection convolution to restore feature dimension (with L2 Regularization)
        self.project = layers.Conv1D(
            in_channels, kernel_size=1, 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        self.bn_out = layers.BatchNormalization()
        
        super(MultiAxisAttention2D, self).build(input_shape)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        channels = x.shape[-1]
        
        # 1. Fold 1D sequence to 2D Grid: [Batch, 10, 25, Channels]
        grid = tf.reshape(x, (batch_size, self.num_orbits, self.orbit_len, channels))
        
        # 2. Major-Axis Attention (Horizontal - checks transit shapes)
        # Collapse orbits into batch: [Batch * 10, 25, Channels]
        major_in = tf.reshape(grid, (batch_size * self.num_orbits, self.orbit_len, channels))
        major_out = self.major_attn(major_in, major_in, training=training)
        major_out = self.major_bn(major_out, training=training)
        major_grid = tf.reshape(major_out, (batch_size, self.num_orbits, self.orbit_len, channels))
        
        # 3. Minor-Axis Attention (Vertical - checks orbit-to-orbit stability)
        # Transpose to [Batch, 25, 10, Channels] and collapse phase into batch: [Batch * 25, 10, Channels]
        grid_trans = tf.transpose(grid, perm=[0, 2, 1, 3])
        minor_in = tf.reshape(grid_trans, (batch_size * self.orbit_len, self.num_orbits, channels))
        minor_out = self.minor_attn(minor_in, minor_in, training=training)
        minor_out = self.minor_bn(minor_out, training=training)
        minor_grid = tf.reshape(minor_out, (batch_size, self.orbit_len, self.num_orbits, channels))
        # Transpose back to [Batch, 10, 25, Channels]
        minor_grid = tf.transpose(minor_grid, perm=[0, 2, 1, 3])
        
        # 4. Feature Fusion & Reshaping
        # Concatenate branches and project to original dimension
        combined = tf.concat([major_grid, minor_grid], axis=-1)
        out_2d = tf.reshape(combined, (batch_size, self.num_orbits * self.orbit_len, 2 * channels))
        out = self.project(out_2d)
        out = self.bn_out(out, training=training)
        out = tf.nn.relu(out + x) # Residual connection
        
        # Compute a self-similarity correlation matrix for Grad-CAM mapping
        norm_out = tf.nn.l2_normalize(out, axis=-1)
        sim_matrix = tf.matmul(norm_out, norm_out, transpose_b=True) # [B, 250, 250]
        
        # Resize to exactly 63x63 using average pooling to keep API outputs compatible
        sim_matrix_expanded = tf.expand_dims(sim_matrix, axis=-1) # [B, 250, 250, 1]
        sim_matrix_resized = tf.image.resize(sim_matrix_expanded, [63, 63], method='bilinear')
        mean_attention = tf.squeeze(sim_matrix_resized, axis=-1) # [B, 63, 63]
        
        return out, mean_attention

class LocalFeatureExtractor1D(layers.Layer):
    def __init__(self, dropout=0.2, **kwargs):
        """
        Extracts features from the local zoomed transit profile (200 points).
        """
        super(LocalFeatureExtractor1D, self).__init__(**kwargs)
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(
            32, kernel_size=5, strides=2, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.drop1 = layers.Dropout(self.dropout_rate)
        
        self.conv2 = layers.Conv1D(
            64, kernel_size=5, strides=2, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.drop2 = layers.Dropout(self.dropout_rate)
        
        self.conv3 = layers.Conv1D(
            128, kernel_size=5, strides=2, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()
        self.drop3 = layers.Dropout(self.dropout_rate)
        
        self.gap = layers.GlobalAveragePooling1D()
        super(LocalFeatureExtractor1D, self).build(input_shape)

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.drop1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.drop2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
        x = self.drop3(x, training=training)
        
        x = self.gap(x)
        return x

class UpgradedExoplanetDetectorNet(models.Model):
    def __init__(self, input_len=2000, dropout=0.3, num_heads=4, **kwargs):
        """
        The SOTA Upgraded Exoplanet Classification Network.
        Integrates:
        - **InceptionTime Global Branch (2000 pts)**
        - **Local Branch (200 pts)**
        - **Centroid Motion Branch (2000 pts x 2 channels X/Y)**
        - **Stellar Metadata Branch (Dense, 3 tabular features)**
        """
        super(UpgradedExoplanetDetectorNet, self).__init__(**kwargs)
        self.input_len = input_len
        self.dropout_rate = dropout
        self.num_heads = num_heads

        # --- GLOBAL BRANCH SETUP ---
        self.global_conv = layers.Conv1D(
            32, kernel_size=7, strides=2, padding='same', 
            kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False
        )
        self.global_bn = layers.BatchNormalization()
        self.global_relu = layers.ReLU()
        self.global_maxpool = layers.MaxPool1D(pool_size=3, strides=2, padding='same')
        
        self.global_inc1 = InceptionModule1D(filters=16, bottleneck_filters=16, stride=2, name="inc_module_1") 
        self.global_inc2 = InceptionModule1D(filters=32, bottleneck_filters=32, stride=1, name="inc_module_2") 
        self.global_inc3 = InceptionModule1D(filters=64, bottleneck_filters=64, stride=1, name="inc_module_3") 
        
        self.global_attention = MultiAxisAttention2D(num_orbits=10, orbit_len=25, num_heads=num_heads)
        self.global_gap = layers.GlobalAveragePooling1D()
        
        # --- LOCAL BRANCH SETUP ---
        self.local_branch = LocalFeatureExtractor1D(dropout=dropout)
        
        # --- CENTROID BRANCH SETUP (X/Y Motion) ---
        self.centroid_conv1 = layers.Conv1D(16, kernel_size=9, strides=2, padding='same', activation='relu')
        self.centroid_pool = layers.MaxPool1D(pool_size=3, strides=2, padding='same')
        self.centroid_conv2 = layers.Conv1D(32, kernel_size=5, strides=2, padding='same', activation='relu')
        self.centroid_gap = layers.GlobalAveragePooling1D()

        # --- STELLAR METADATA BRANCH SETUP ---
        self.meta_dense1 = layers.Dense(16, activation='relu')
        self.meta_dense2 = layers.Dense(16, activation='relu')

        # --- CLASSIFICATION HEAD ---
        # Fused features: Global(256) + Local(128) + Centroid(32) + Meta(16) = 432
        self.fc1 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dropout_layer = layers.Dropout(dropout)
        self.fc2 = layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        
    def compile(self, optimizer, loss_fn, **kwargs):
        super(UpgradedExoplanetDetectorNet, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.acc_metric = tf.keras.metrics.BinaryAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_metric, self.acc_metric]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)
            loss = self.loss_fn(y, y_pred)
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(y, y_pred)
        return {"loss": self.loss_metric.result(), "accuracy": self.acc_metric.result()}

    def test_step(self, data):
        x, y = data
        y_pred, _ = self(x, training=False)
        loss = self.loss_fn(y, y_pred)
        
        self.loss_metric.update_state(loss)
        self.acc_metric.update_state(y, y_pred)
        return {"loss": self.loss_metric.result(), "accuracy": self.acc_metric.result()}

    def call(self, inputs, training=None):
        if isinstance(inputs, dict):
            global_x = inputs['global']
            local_x = inputs['local']
            centroid_x = inputs.get('centroid', None)
            meta_x = inputs.get('metadata', None)
        else:
            global_x, local_x, centroid_x, meta_x = inputs
            
        if len(global_x.shape) == 2:
            global_x = tf.expand_dims(global_x, axis=-1)
        if len(local_x.shape) == 2:
            local_x = tf.expand_dims(local_x, axis=-1)
            
        # 1. Global View Branch
        g = self.global_conv(global_x)
        g = self.global_bn(g, training=training)
        g = self.global_relu(g)
        g = self.global_maxpool(g)
        
        g = self.global_inc1(g, training=training)
        g = self.global_inc2(g, training=training)
        g = self.global_inc3(g, training=training)
        
        g, attn_map = self.global_attention(g, training=training)
        global_feats = self.global_gap(g)
        
        # 2. Local View Branch
        local_feats = self.local_branch(local_x, training=training)
        
        # 3. Centroid Branch (Handle optional for backward compat)
        if centroid_x is not None:
            c = self.centroid_conv1(centroid_x)
            c = self.centroid_pool(c)
            c = self.centroid_conv2(c)
            centroid_feats = self.centroid_gap(c)
        else:
            batch_size = tf.shape(global_x)[0]
            centroid_feats = tf.zeros((batch_size, 32))
            
        # 4. Metadata Branch
        if meta_x is not None:
            m = self.meta_dense1(meta_x)
            meta_feats = self.meta_dense2(m)
        else:
            batch_size = tf.shape(global_x)[0]
            meta_feats = tf.zeros((batch_size, 16))
            
        # 5. Feature Fusion & Classification
        fused = tf.concat([global_feats, local_feats, centroid_feats, meta_feats], axis=-1)
        
        x = self.fc1(fused)
        x = self.dropout_layer(x, training=training)
        x = self.fc2(x)
        
        return x, attn_map

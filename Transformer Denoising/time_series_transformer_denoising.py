import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Generate synthetic time series with noise
def generate_synthetic_data(n_samples=1000, noise_level=0.2):
    # Time steps
    t = np.linspace(0, 10, n_samples)

    # True signal: combination of sine waves
    true_signal = 0.5 * np.sin(0.5 * t) + 0.3 * np.sin(3 * t)

    # Add Gaussian noise
    noise = noise_level * np.random.randn(n_samples)
    noisy_signal = true_signal + noise

    return t, true_signal, noisy_signal


# Positional encoding for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embed_dim]
        return x + self.pe[:x.size(0)]


# Custom transformer encoder layer to extract attention weights
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, return_attention=False):
        src2, attn_weights = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if return_attention:
            return src, attn_weights
        return src


# Custom transformer encoder to extract attention weights
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, return_attention=False):
        output = src
        attention_maps = []

        for layer in self.layers:
            if return_attention:
                output, attn = layer(output, return_attention=True)
                attention_maps.append(attn)
            else:
                output = layer(output)

        if return_attention:
            return output, attention_maps
        return output


# Transformer for time series denoising with attention visualization
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead,
                                                      dim_feedforward=2 * d_model,
                                                      dropout=dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers)

        self.decoder = nn.Linear(d_model, input_dim)
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, return_attention=False):
        # src shape: [seq_len, batch_size, input_dim]
        embedded = self.embedding(src)
        embedded = self.pos_encoder(embedded)

        if return_attention:
            memory, attention_maps = self.transformer_encoder(embedded, return_attention=True)
            output = self.decoder(memory)
            return output, attention_maps
        else:
            memory = self.transformer_encoder(embedded)
            output = self.decoder(memory)
            return output


# Function to apply wavelet denoising
def wavelet_denoising(signal, wavelet='db4', level=2):
    try:
        import pywt

        # Decompose
        coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)

        # Threshold
        threshold = np.std(signal) * 0.1

        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

        # Reconstruct
        reconstructed = pywt.waverec(coeffs, wavelet, mode='per')

        # Handle potential length mismatch
        if len(reconstructed) > len(signal):
            reconstructed = reconstructed[:len(signal)]

        return reconstructed

    except ImportError:
        print("PyWavelets not installed. Using moving average as fallback.")
        window = 15
        return np.convolve(signal, np.ones(window) / window, mode='same')


# Main function
def main():
    # Set up figure for real-time training visualization
    plt.ion()  # Turn on interactive mode
    fig_train = plt.figure(figsize=(15, 8))
    ax1 = fig_train.add_subplot(2, 2, 1)  # Loss plot
    ax2 = fig_train.add_subplot(2, 2, 2)  # Example prediction during training
    ax3 = fig_train.add_subplot(2, 2, 3)  # Learning rate
    ax4 = fig_train.add_subplot(2, 2, 4)  # Validation metrics

    # Generate data
    t, true_signal, noisy_signal = generate_synthetic_data(n_samples=1000, noise_level=0.2)

    # Prepare data for PyTorch (sliding windows approach)
    window_size = 100
    stride = 10
    X, y = [], []

    for i in range(0, len(noisy_signal) - window_size, stride):
        X.append(noisy_signal[i:i + window_size])
        y.append(true_signal[i:i + window_size])

    X = np.array(X)
    y = np.array(y)

    # Split into train and validation sets (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)  # Add feature dimension
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = TimeSeriesTransformer(input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.1)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    n_epochs = 50

    # Training loop
    train_losses = []
    val_losses = []
    lr_history = []
    val_mse = []
    val_mae = []

    # Get a sample batch for visualization
    vis_X, vis_y = next(iter(val_dataloader))
    vis_X = vis_X[:1]  # Take just the first sample
    vis_y = vis_y[:1]

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_dataloader:
            # Reshape for transformer: [seq_len, batch_size, features]
            batch_X = batch_X.permute(1, 0, 2)
            batch_y = batch_y.permute(1, 0, 2)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_epoch_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.permute(1, 0, 2)
                batch_y = batch_y.permute(1, 0, 2)

                outputs = model(batch_X)
                val_loss = criterion(outputs, batch_y)
                val_epoch_loss += val_loss.item()

                # Store predictions and targets for metrics
                preds = outputs.permute(1, 0, 2).squeeze(-1).cpu().numpy()
                targets = batch_y.permute(1, 0, 2).squeeze(-1).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_val_loss = val_epoch_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Calculate validation metrics
        val_mse_current = mean_squared_error(np.array(all_targets).flatten(),
                                             np.array(all_preds).flatten())
        val_mae_current = mean_absolute_error(np.array(all_targets).flatten(),
                                              np.array(all_preds).flatten())

        val_mse.append(val_mse_current)
        val_mae.append(val_mae_current)

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}')

        # Visualize training progress
        ax1.clear()
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)

        ax3.clear()
        ax3.plot(lr_history)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)

        ax4.clear()
        ax4.plot(val_mse, label='MSE')
        ax4.plot(val_mae, label='MAE')
        ax4.set_title('Validation Metrics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Error')
        ax4.legend()
        ax4.grid(True)

        # Show example prediction
        with torch.no_grad():
            vis_X_transformed = vis_X.permute(1, 0, 2)
            prediction = model(vis_X_transformed)
            prediction = prediction.permute(1, 0, 2).squeeze().cpu().numpy()

            ax2.clear()
            ax2.plot(vis_y[0].squeeze().numpy(), label='True')
            ax2.plot(vis_X[0].squeeze().numpy(), label='Noisy', alpha=0.5)
            ax2.plot(prediction, label='Predicted')
            ax2.set_title(f'Example Prediction (Epoch {epoch + 1})')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.pause(0.1)

    plt.ioff()  # Turn off interactive mode

    # Final model evaluation
    fig_results = plt.figure(figsize=(15, 12))

    # Test the model on the full sequence
    model.eval()
    with torch.no_grad():
        # Prepare full sequence
        test_X = torch.FloatTensor(noisy_signal).view(1, -1, 1).permute(1, 0, 2)

        # Get denoised signal and attention maps
        denoised_signal, attention_maps = model(test_X, return_attention=True)
        denoised_signal = denoised_signal.squeeze().numpy()

    # Compare with traditional methods
    # Moving Average
    window = 15
    ma_denoised = np.convolve(noisy_signal, np.ones(window) / window, mode='same')

    # Wavelet denoising
    wavelet_denoised = wavelet_denoising(noisy_signal)

    # Plot results
    ax1 = fig_results.add_subplot(3, 2, 1)
    ax1.plot(t, true_signal)
    ax1.set_title('True Signal')
    ax1.grid(True)

    ax2 = fig_results.add_subplot(3, 2, 2)
    ax2.plot(t, noisy_signal)
    ax2.set_title('Noisy Signal')
    ax2.grid(True)

    ax3 = fig_results.add_subplot(3, 2, 3)
    ax3.plot(t, ma_denoised)
    ax3.set_title('Moving Average Filter')
    ax3.grid(True)

    ax4 = fig_results.add_subplot(3, 2, 4)
    ax4.plot(t, wavelet_denoised)
    ax4.set_title('Wavelet Denoising')
    ax4.grid(True)

    ax5 = fig_results.add_subplot(3, 2, 5)
    ax5.plot(t, denoised_signal)
    ax5.set_title('Transformer Denoised')
    ax5.grid(True)

    ax6 = fig_results.add_subplot(3, 2, 6)
    # Plot error curves
    ax6.plot(t, np.abs(true_signal - noisy_signal), label='Noisy Error', alpha=0.5)
    ax6.plot(t, np.abs(true_signal - ma_denoised), label='MA Error')
    ax6.plot(t, np.abs(true_signal - wavelet_denoised), label='Wavelet Error')
    ax6.plot(t, np.abs(true_signal - denoised_signal), label='Transformer Error')
    ax6.set_title('Absolute Error Comparison')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.show()

    # Calculate error metrics
    mse_noisy = mean_squared_error(true_signal, noisy_signal)
    mse_ma = mean_squared_error(true_signal, ma_denoised)
    mse_wavelet = mean_squared_error(true_signal, wavelet_denoised)
    mse_transformer = mean_squared_error(true_signal, denoised_signal)

    print("\nMean Squared Error Comparison:")
    print(f"MSE of Noisy Signal: {mse_noisy:.6f}")
    print(f"MSE of Moving Average: {mse_ma:.6f}")
    print(f"MSE of Wavelet Denoising: {mse_wavelet:.6f}")
    print(f"MSE of Transformer: {mse_transformer:.6f}")

    improvement_over_ma = (mse_ma - mse_transformer) / mse_ma * 100
    improvement_over_wavelet = (mse_wavelet - mse_transformer) / mse_wavelet * 100
    print(f"Improvement over Moving Average: {improvement_over_ma:.2f}%")
    print(f"Improvement over Wavelet: {improvement_over_wavelet:.2f}%")

    # Visualize attention maps (from the first head of the first layer)
    plt.figure(figsize=(12, 8))

    # Get attention weights from first layer, first head
    attn = attention_maps[0][0, :, :].cpu().numpy()  # Shape: [seq_len, seq_len]

    # Plot attention heatmap
    plt.subplot(2, 1, 1)
    sns.heatmap(attn[:50, :50], cmap='viridis')
    plt.title('Attention Map (First 50 time steps)')
    plt.xlabel('Time Step (Target)')
    plt.ylabel('Time Step (Source)')

    # Plot diagonal profile of attention matrix
    plt.subplot(2, 1, 2)
    time_offsets = np.arange(-10, 11)
    attn_profiles = []

    for i in range(len(attn) - 21):
        profile = [attn[i + 10 + offset, i + 10] for offset in time_offsets]
        attn_profiles.append(profile)

    mean_profile = np.mean(attn_profiles, axis=0)
    std_profile = np.std(attn_profiles, axis=0)

    plt.plot(time_offsets, mean_profile)
    plt.fill_between(time_offsets,
                     mean_profile - std_profile,
                     mean_profile + std_profile,
                     alpha=0.3)
    plt.title('Attention Profile (Diagonal ±10 steps)')
    plt.xlabel('Time Offset')
    plt.ylabel('Average Attention Weight')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Compare performance at different noise levels
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    mse_results = {'Noisy': [], 'Moving Average': [], 'Wavelet': [], 'Transformer': []}

    plt.figure(figsize=(15, 10))

    for i, noise_level in enumerate(noise_levels):
        _, true_signal, noisy_signal = generate_synthetic_data(n_samples=1000, noise_level=noise_level)

        # Apply moving average
        ma_denoised = np.convolve(noisy_signal, np.ones(window) / window, mode='same')

        # Apply wavelet denoising
        wavelet_denoised = wavelet_denoising(noisy_signal)

        # Apply transformer
        with torch.no_grad():
            test_X = torch.FloatTensor(noisy_signal).view(1, -1, 1).permute(1, 0, 2)
            transformer_denoised = model(test_X).squeeze().numpy()

        # Calculate MSE
        mse_noisy = mean_squared_error(true_signal, noisy_signal)
        mse_ma = mean_squared_error(true_signal, ma_denoised)
        mse_wavelet = mean_squared_error(true_signal, wavelet_denoised)
        mse_transformer = mean_squared_error(true_signal, transformer_denoised)

        mse_results['Noisy'].append(mse_noisy)
        mse_results['Moving Average'].append(mse_ma)
        mse_results['Wavelet'].append(mse_wavelet)
        mse_results['Transformer'].append(mse_transformer)

        # Plot example for this noise level
        plt.subplot(len(noise_levels), 1, i + 1)
        plt.plot(true_signal[:100], label='True', linewidth=2)
        plt.plot(noisy_signal[:100], label='Noisy', alpha=0.5)
        plt.plot(transformer_denoised[:100], label='Transformer')
        plt.title(f'Noise Level: {noise_level}')
        if i == 0:
            plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot MSE vs noise level
    plt.figure(figsize=(10, 6))
    for method, mse_values in mse_results.items():
        plt.plot(noise_levels, mse_values, 'o-', label=method)

    plt.title('MSE vs Noise Level')
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale for better visibility
    plt.show()

    # Show time-frequency analysis
    try:
        import pywt

        # Perform continuous wavelet transform
        scales = np.arange(1, 128)
        wavelet = 'morl'  # Morlet wavelet

        # For true signal
        coef_true, freqs_true = pywt.cwt(true_signal, scales, wavelet)

        # For noisy signal
        coef_noisy, freqs_noisy = pywt.cwt(noisy_signal, scales, wavelet)

        # For denoised signal
        coef_denoised, freqs_denoised = pywt.cwt(denoised_signal, scales, wavelet)

        plt.figure(figsize=(15, 10))

        # True signal
        plt.subplot(3, 1, 1)
        plt.imshow(np.abs(coef_true), aspect='auto', extent=[0, len(true_signal), 1, len(scales)],
                   cmap='jet', interpolation='bilinear')
        plt.title('Wavelet Transform - True Signal')
        plt.ylabel('Scale')
        plt.colorbar(label='Magnitude')

        # Noisy signal
        plt.subplot(3, 1, 2)
        plt.imshow(np.abs(coef_noisy), aspect='auto', extent=[0, len(noisy_signal), 1, len(scales)],
                   cmap='jet', interpolation='bilinear')
        plt.title('Wavelet Transform - Noisy Signal')
        plt.ylabel('Scale')
        plt.colorbar(label='Magnitude')

        # Denoised signal
        plt.subplot(3, 1, 3)
        plt.imshow(np.abs(coef_denoised), aspect='auto', extent=[0, len(denoised_signal), 1, len(scales)],
                   cmap='jet', interpolation='bilinear')
        plt.title('Wavelet Transform - Transformer Denoised Signal')
        plt.ylabel('Scale')
        plt.xlabel('Time')
        plt.colorbar(label='Magnitude')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("PyWavelets not installed. Skipping time-frequency analysis.")


if __name__ == "__main__":
    main()
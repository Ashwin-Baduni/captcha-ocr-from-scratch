"""
Task 2: Text Generation/Extraction from CAPTCHA images
Implements a sequence-to-sequence model with attention for OCR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

class CNNEncoder(nn.Module):
    """CNN encoder to extract features from CAPTCHA images"""

    def __init__(self, input_channels: int = 3):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 16))

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.adaptive_pool(x)

        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, 16, -1)

        return x

class AttentionModule(nn.Module):
    """Attention mechanism for focusing on relevant parts of the image"""

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super(AttentionModule, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_att = nn.Linear(decoder_dim, decoder_dim)
        self.full_att = nn.Linear(decoder_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out: [batch_size, seq_len, encoder_dim]
            decoder_hidden: [batch_size, decoder_dim]
        Returns:
            attention_weights: [batch_size, seq_len]
            context: [batch_size, encoder_dim]
        """
        seq_len = encoder_out.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        encoder_att = self.encoder_att(encoder_out)
        decoder_att = self.decoder_att(decoder_hidden)

        att_scores = self.full_att(torch.tanh(encoder_att + decoder_att))
        att_scores = att_scores.squeeze(2)

        attention_weights = F.softmax(att_scores, dim=1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_out)
        context = context.squeeze(1)

        return attention_weights, context

class LSTMDecoder(nn.Module):
    """LSTM decoder with attention for text generation"""

    def __init__(self, vocab_size: int, encoder_dim: int = 2048,
                 embed_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3):
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim + encoder_dim, hidden_dim,
                            num_layers, batch_first=True, dropout=dropout)

        self.attention = AttentionModule(encoder_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim + encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out, targets=None, max_length=20):
        """
        Args:
            encoder_out: Features from encoder [batch_size, seq_len, encoder_dim]
            targets: Target sequences for teacher forcing [batch_size, max_len]
            max_length: Maximum sequence length to generate
        Returns:
            outputs: Predicted logits [batch_size, max_len, vocab_size]
            attention_weights: Attention weights for visualization
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

        outputs = []
        attention_weights = []

        input_token = torch.zeros(batch_size, dtype=torch.long).to(device)

        seq_length = targets.size(1) if targets is not None else max_length

        for t in range(seq_length):

            embedded = self.embedding(input_token)

            att_weights, context = self.attention(encoder_out, h[-1])
            attention_weights.append(att_weights)

            lstm_input = torch.cat([embedded, context], dim=1)
            lstm_input = lstm_input.unsqueeze(1)

            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            lstm_out = lstm_out.squeeze(1)

            output_features = torch.cat([lstm_out, context], dim=1)
            output = self.output_proj(self.dropout(output_features))
            outputs.append(output)

            if targets is not None and t < seq_length - 1:

                input_token = targets[:, t + 1]
            else:

                input_token = output.argmax(dim=1)

        outputs = torch.stack(outputs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return outputs, attention_weights

class CaptchaOCR(nn.Module):
    """Complete OCR model for CAPTCHA text extraction"""

    def __init__(self, vocab_size: int, max_length: int = 20):
        super(CaptchaOCR, self).__init__()

        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(vocab_size, encoder_dim=2048)
        self.max_length = max_length

    def forward(self, images, targets=None):
        """
        Args:
            images: Input images [batch_size, 3, H, W]
            targets: Target sequences for training [batch_size, max_len]
        Returns:
            outputs: Predicted sequences
            attention_weights: Attention weights for visualization
        """

        features = self.encoder(images)

        outputs, attention_weights = self.decoder(features, targets, self.max_length)

        return outputs, attention_weights

    def generate(self, images, max_length=None):
        """Generate text from images without teacher forcing"""
        if max_length is None:
            max_length = self.max_length

        with torch.no_grad():
            features = self.encoder(images)
            outputs, attention_weights = self.decoder(features, targets=None, max_length=max_length)

        return outputs, attention_weights

class CTCOCRModel(nn.Module):
    """Alternative model using CTC loss for sequence prediction"""

    def __init__(self, vocab_size: int, hidden_dim: int = 256):
        super(CTCOCRModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        self.rnn = nn.LSTM(512, hidden_dim, 2, batch_first=True, bidirectional=True)

        self.output = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):

        conv_out = self.cnn(x)

        batch_size = conv_out.size(0)

        if conv_out.size(2) == 1:
            conv_out = conv_out.squeeze(2)
        else:

            conv_out = conv_out.mean(dim=2)
        conv_out = conv_out.permute(0, 2, 1)

        rnn_out, _ = self.rnn(conv_out)

        output = self.output(rnn_out)

        output = output.permute(1, 0, 2)

        return output

def create_char_vocabulary():
    """Create character-level vocabulary for OCR"""
    vocab = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
        '<BLANK>': 3,
    }

    for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
        vocab[char] = i + 4

    for i, char in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        vocab[char] = i + 30

    idx_to_char = {v: k for k, v in vocab.items()}

    return vocab, idx_to_char

def decode_sequence(sequence, idx_to_char, remove_special=True):
    """Decode a sequence of indices to text"""
    decoded = []
    for idx in sequence:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        char = idx_to_char.get(idx, '?')

        if remove_special and char in ['<PAD>', '<START>', '<END>', '<BLANK>']:
            continue

        decoded.append(char)

    return ''.join(decoded)

def calculate_cer(pred_text: str, target_text: str) -> float:
    """Calculate Character Error Rate"""

    m, n = len(pred_text), len(target_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i-1] == target_text[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / max(len(target_text), 1)

def calculate_wer(pred_text: str, target_text: str) -> float:
    """Calculate Word Error Rate"""
    pred_words = pred_text.split()
    target_words = target_text.split()

    m, n = len(pred_words), len(target_words)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == target_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / max(len(target_words), 1)

if __name__ == "__main__":

    vocab, idx_to_char = create_char_vocabulary()
    vocab_size = len(vocab)

    model = CaptchaOCR(vocab_size)

    batch_size = 4
    images = torch.randn(batch_size, 3, 60, 200)

    outputs, attention = model.generate(images, max_length=10)
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attention.shape}")

    ctc_model = CTCOCRModel(vocab_size)
    ctc_output = ctc_model(images)
    print(f"CTC output shape: {ctc_output.shape}")

    print("Models created successfully!")
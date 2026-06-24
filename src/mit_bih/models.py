import torch
import torch.nn as nn
import math
import copy

class ECGClassifier(nn.Module):
    """
    1D Convolutional Neural Network (CNN) for classifying ECG heartbeat segments.
    """
    def __init__(self, num_classes=5):
        super(ECGClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10)  # Restructures output width to exactly 10
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LoRALayer(nn.Module):
    """
    Linear layer with LoRA adapters.
    W0 is FROZEN (pre-trained / randomly-initialised-then-frozen base weights).
    Only A and B are trained and transmitted.
    Output = W0(x) + alpha * B(A(x))
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha

        # Frozen base weights
        self.W0 = nn.Linear(in_features, out_features, bias=True)
        self.W0.weight.requires_grad = False
        self.W0.bias.requires_grad   = False

        # Trainable low-rank matrices
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # Init: A random, B zero -> LoRA contribution starts at 0
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.W0(x) + self.alpha * self.B(self.A(x))


class ECGCNN(nn.Module):
    """
    1D-CNN for ECG beat classification with optional LoRA.
    Input : (batch, 1, 180) -- 180-sample heartbeat window
    Output: (batch, 5)      -- 5 arrhythmia classes
    """
    def __init__(self, use_lora=False, dropout_rate=0.3, lora_rank=8, lora_alpha=1.0):
        super().__init__()
        self.use_lora = use_lora

        self.features = nn.Sequential(
            nn.Conv1d(1,  64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        if use_lora:
            self.fc1        = nn.Linear(256, 64)
            self.relu       = nn.ReLU()
            self.dropout    = nn.Dropout(dropout_rate)
            self.classifier = LoRALayer(64, 5, rank=lora_rank, alpha=lora_alpha)

            # Freeze everything except LoRA A/B
            for p in self.features.parameters():
                p.requires_grad = False
            for p in self.fc1.parameters():
                p.requires_grad = False
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 5)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)    # flatten
        if self.use_lora:
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.classifier(x)
        else:
            x = self.classifier(x)
        return x

    def load_pretrained_backbone(self, pretrained_model):
        """
        Copy `features` (the CNN backbone) and the 256->64 projector
        from a PRETRAINED non-LoRA ECGCNN into this LoRA model.
        """
        if not self.use_lora:
            raise ValueError("load_pretrained_backbone only applies to use_lora=True models")
        if pretrained_model.use_lora:
            raise ValueError("pretrained_model must be a use_lora=False model")

        self.features.load_state_dict(pretrained_model.features.state_dict())

        pretrained_fc1   = pretrained_model.classifier[1]
        pretrained_final = pretrained_model.classifier[4]

        self.fc1.load_state_dict(pretrained_fc1.state_dict())

        # Seed the LoRA base layer's frozen W0 with the pretrained final classifier weights
        self.classifier.W0.weight.data = pretrained_final.weight.data.clone()
        self.classifier.W0.bias.data = pretrained_final.bias.data.clone()

        # Re-affirm frozen status
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.fc1.parameters():
            p.requires_grad = False
        self.classifier.W0.weight.requires_grad = False
        self.classifier.W0.bias.requires_grad = False

    def get_lora_params(self):
        """Return only LoRA A and B weights for transmission."""
        if not self.use_lora:
            raise ValueError("Model not using LoRA")
        return {
            'A': self.classifier.A.weight.data.clone(),
            'B': self.classifier.B.weight.data.clone()
        }

    def set_lora_params(self, params):
        """Load aggregated LoRA weights from server."""
        self.classifier.A.weight.data = params['A'].clone()
        self.classifier.B.weight.data = params['B'].clone()

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    def set_train_mode(self):
        """
        BatchNorm layers must be kept in eval() mode, while trainable parts stay in train().
        """
        if self.use_lora:
            self.eval()                 # freeze everything by default...
            self.classifier.train()     # ...then re-enable train mode
            self.dropout.train()
        else:
            self.train()


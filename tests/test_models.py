"""
Unit tests for ECG CNN classifiers and LoRA model configurations.
"""

import unittest
import torch
from mit_bih.models import ECGClassifier, ECGCNN, LoRALayer


class TestModelArchitectures(unittest.TestCase):
    """Tests forward passes, parameters grad statuses, and backbone loading in classifiers."""

    def test_ecg_classifier_forward(self):
        """Verify the standard ECGClassifier forward pass and output shape."""
        model = ECGClassifier(num_classes=5)
        # Create input: batch_size=4, channels=1, signal_length=180
        x = torch.randn(4, 1, 180)
        output = model(x)
        self.assertEqual(output.shape, (4, 5))

    def test_lora_layer(self):
        """Verify parameter statuses and forward pass of the custom LoRALayer."""
        layer = LoRALayer(in_features=64, out_features=5, rank=4, alpha=2.0)
        
        # Verify gradient requires flags
        self.assertFalse(layer.W0.weight.requires_grad)
        self.assertFalse(layer.W0.bias.requires_grad)
        self.assertTrue(layer.A.weight.requires_grad)
        self.assertTrue(layer.B.weight.requires_grad)
        
        # Verify shape
        x = torch.randn(10, 64)
        out = layer(x)
        self.assertEqual(out.shape, (10, 5))

    def test_ecg_cnn_standard(self):
        """Verify the standard ECGCNN (non-LoRA) behaves correctly."""
        model = ECGCNN(use_lora=False)
        x = torch.randn(4, 1, 180)
        out = model(x)
        self.assertEqual(out.shape, (4, 5))
        
        # All weights in standard mode should require gradients
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_ecg_cnn_lora(self):
        """Verify parameter freezing and adapter weight updates in LoRA configuration."""
        model = ECGCNN(use_lora=True, lora_rank=4)
        x = torch.randn(4, 1, 180)
        out = model(x)
        self.assertEqual(out.shape, (4, 5))
        
        # Under LoRA, feature extractor parameters should be frozen
        for p in model.features.parameters():
            self.assertFalse(p.requires_grad)
            
        # Classifier A and B parameters should be trainable
        self.assertTrue(model.classifier.A.weight.requires_grad)
        self.assertTrue(model.classifier.B.weight.requires_grad)
        
        # Check lora parameter retrieval
        lora_params = model.get_lora_params()
        self.assertIn('A', lora_params)
        self.assertIn('B', lora_params)
        self.assertEqual(lora_params['A'].shape, model.classifier.A.weight.shape)
        
        # Check setting lora parameters
        new_A = torch.randn_like(model.classifier.A.weight)
        new_B = torch.randn_like(model.classifier.B.weight)
        model.set_lora_params({'A': new_A, 'B': new_B})
        self.assertTrue(torch.allclose(model.classifier.A.weight, new_A))

    def test_load_pretrained_backbone(self):
        """Verify copying backbone weights from non-LoRA base to LoRA-adapted model."""
        base_model = ECGCNN(use_lora=False)
        lora_model = ECGCNN(use_lora=True)
        
        # Modify base features so they are different from random initialization
        with torch.no_grad():
            base_model.features[0].weight.fill_(1.5)
            
        lora_model.load_pretrained_backbone(base_model)
        
        # Verify features are matched
        self.assertTrue(torch.allclose(
            lora_model.features[0].weight,
            base_model.features[0].weight
        ))
        
        # Verify backbone features are frozen in lora model
        self.assertFalse(lora_model.features[0].weight.requires_grad)


if __name__ == "__main__":
    unittest.main()

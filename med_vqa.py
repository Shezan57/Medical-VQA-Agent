"""
Medical Visual Question Answering (Med-VQA) - Standalone Script
================================================================

This script provides a command-line interface for the Medical VQA system.
Designed for local testing or integration into larger applications.

Usage:
    python med_vqa.py --image chest_xray.png --question "What abnormalities are visible?"
    python med_vqa.py --demo  # Run demo with sample images

Author: [Your Name]
Date: January 2026
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

# Sample medical images for demo mode
SAMPLE_IMAGES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        "filename": "chest_xray_normal.png",
        "description": "Normal chest X-ray (PA view)"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/24/Right_sided_pneumothorax.jpg",
        "filename": "pneumothorax.jpg",
        "description": "Right-sided pneumothorax"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/64/Pneumonia_x-ray.jpg",
        "filename": "pneumonia.jpg",
        "description": "Pneumonia"
    },
]

# Medical context prompt
MEDICAL_CONTEXT = """You are an expert radiologist assistant. Analyze this medical image carefully and provide:
1. Key observations visible in the image
2. Any potential abnormalities or areas of concern
3. Relevant clinical considerations

Note: This is for educational/demonstration purposes only. Always consult a qualified healthcare professional for medical advice.

"""


# ============================================================================
# MODEL LOADING
# ============================================================================

class MedicalVQA:
    """
    Medical Visual Question Answering system using LLaVA with 4-bit quantization.
    """
    
    def __init__(self, model_id: str = MODEL_ID, use_quantization: bool = True):
        """
        Initialize the Medical VQA system.
        
        Args:
            model_id: Hugging Face model identifier
            use_quantization: Whether to use 4-bit NF4 quantization
        """
        self.model_id = model_id
        self.use_quantization = use_quantization
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLaVA model with optional quantization."""
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            BitsAndBytesConfig
        )
        
        print(f"🔄 Loading model: {self.model_id}")
        print(f"   Quantization: {'4-bit NF4' if self.use_quantization else 'Full precision'}")
        
        # Load processor
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        
        # Configure quantization
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
        else:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        print(f"✅ Model loaded! Memory: {self.model.get_memory_footprint() / 1e9:.2f} GB")
    
    def diagnose(self, image_path: str, question: str, 
                 max_new_tokens: int = 300) -> dict:
        """
        Perform Medical VQA on an image.
        
        Args:
            image_path: Path to the medical image
            question: Clinical question about the image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            dict with response, timing metrics, and token counts
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": MEDICAL_CONTEXT + question}
                ]
            }
        ]
        
        # Process inputs
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        
        # Measure Time to First Token
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                    pad_token_id=self.processor.tokenizer.pad_token_id)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - start_time
        
        # Full generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        full_start = time.perf_counter()
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.perf_counter() - full_start
        
        # Decode response
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_len:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=True)
        
        tokens_generated = len(generated_tokens)
        
        return {
            "response": response.strip(),
            "ttft": ttft,
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / total_time if total_time > 0 else 0
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_result(image_path: str, question: str, result: dict, 
                     save_path: Optional[str] = None):
    """Create professional visualization of the VQA result."""
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9), facecolor='#1a1a2e')
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1.5],
                           hspace=0.3, wspace=0.3)
    
    # Image panel
    ax_image = fig.add_subplot(gs[0, 0])
    image = Image.open(image_path)
    ax_image.imshow(image, cmap='gray' if image.mode == 'L' else None)
    ax_image.set_title('📷 Medical Image', fontsize=14, fontweight='bold', 
                       color='#00d4ff', pad=10)
    ax_image.axis('off')
    
    # Diagnosis panel
    ax_diag = fig.add_subplot(gs[0, 1])
    ax_diag.set_facecolor('#16213e')
    
    ax_diag.text(0.5, 0.95, '🤖 AI Radiologist Analysis', 
                 transform=ax_diag.transAxes,
                 fontsize=14, fontweight='bold', color='#00d4ff', ha='center')
    
    ax_diag.text(0.05, 0.85, '❓ Question:', transform=ax_diag.transAxes,
                 fontsize=11, fontweight='bold', color='#ffd700')
    
    wrapped_q = '\n'.join([question[i:i+60] for i in range(0, len(question), 60)])
    ax_diag.text(0.05, 0.78, wrapped_q, transform=ax_diag.transAxes,
                 fontsize=10, color='#e0e0e0', style='italic')
    
    ax_diag.text(0.05, 0.65, '📋 Analysis:', transform=ax_diag.transAxes,
                 fontsize=11, fontweight='bold', color='#00ff88')
    
    response = result['response'][:800] + '...' if len(result['response']) > 800 else result['response']
    wrapped_r = '\n'.join([response[i:i+70] for i in range(0, len(response), 70)])
    ax_diag.text(0.05, 0.58, wrapped_r, transform=ax_diag.transAxes,
                 fontsize=9, color='#ffffff', va='top', family='monospace')
    ax_diag.axis('off')
    
    # Metrics panel
    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.set_facecolor('#16213e')
    
    metrics = [
        ('⚡ TTFT', f"{result['ttft']:.3f}s", '#ff6b6b'),
        ('⏱️ Total Time', f"{result['total_time']:.3f}s", '#4ecdc4'),
        ('📝 Tokens', f"{result['tokens_generated']}", '#45b7d1'),
        ('🚀 Throughput', f"{result['tokens_per_second']:.1f} tok/s", '#96ceb4')
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        x = 0.1 + i * 0.22
        ax_metrics.add_patch(plt.Rectangle((x - 0.08, 0.2), 0.18, 0.6,
                                            facecolor=color, alpha=0.2,
                                            edgecolor=color, linewidth=2,
                                            transform=ax_metrics.transAxes))
        ax_metrics.text(x + 0.01, 0.65, label, transform=ax_metrics.transAxes,
                        fontsize=10, color=color, ha='center', fontweight='bold')
        ax_metrics.text(x + 0.01, 0.35, value, transform=ax_metrics.transAxes,
                        fontsize=14, color='white', ha='center', fontweight='bold')
    
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis('off')
    
    fig.suptitle('🏥 Medical Visual Question Answering (Med-VQA)',
                 fontsize=18, fontweight='bold', color='#ffffff', y=0.98)
    fig.text(0.5, 0.93, 'LLaVA-1.6-Mistral-7B | 4-bit NF4 Quantization',
             ha='center', fontsize=10, color='#888888', style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"💾 Saved: {save_path}")
    
    plt.show()


# ============================================================================
# DEMO MODE
# ============================================================================

def download_sample_images(output_dir: str = "sample_images") -> list:
    """Download sample medical images for demo."""
    import requests
    
    os.makedirs(output_dir, exist_ok=True)
    downloaded = []
    
    print("📥 Downloading sample images...")
    for img in SAMPLE_IMAGES:
        try:
            response = requests.get(img["url"], timeout=10)
            if response.status_code == 200:
                path = os.path.join(output_dir, img["filename"])
                with open(path, "wb") as f:
                    f.write(response.content)
                downloaded.append({"path": path, "description": img["description"]})
                print(f"   ✅ {img['filename']}")
        except Exception as e:
            print(f"   ❌ {img['filename']}: {str(e)[:50]}")
    
    return downloaded


def run_demo():
    """Run the complete demo pipeline."""
    print("\n" + "=" * 70)
    print("🏥 MEDICAL VQA DEMO")
    print("=" * 70 + "\n")
    
    # Download images
    images = download_sample_images()
    if not images:
        print("❌ No images downloaded. Check your internet connection.")
        return
    
    # Initialize model
    vqa = MedicalVQA()
    
    # Analyze first image
    test_img = images[0]
    question = "What anatomical structures are visible and are there any abnormalities?"
    
    print(f"\n🔬 Analyzing: {test_img['description']}")
    print(f"❓ Question: {question}\n")
    
    result = vqa.diagnose(test_img["path"], question)
    
    print("=" * 70)
    print("📋 DIAGNOSIS")
    print("=" * 70)
    print(f"\n{result['response']}\n")
    print("=" * 70)
    print("⏱️ PERFORMANCE")
    print("=" * 70)
    print(f"   TTFT: {result['ttft']:.3f}s")
    print(f"   Total: {result['total_time']:.3f}s")
    print(f"   Tokens: {result['tokens_generated']}")
    print(f"   Speed: {result['tokens_per_second']:.1f} tok/s")
    
    # Visualize
    visualize_result(test_img["path"], question, result, "med_vqa_result.png")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medical Visual Question Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--image", "-i", type=str, help="Path to medical image")
    parser.add_argument("--question", "-q", type=str, 
                        default="What abnormalities are visible in this image?",
                        help="Clinical question about the image")
    parser.add_argument("--demo", action="store_true", 
                        help="Run demo with sample images")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--output", "-o", type=str, default="result.png",
                        help="Output path for visualization")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        
        vqa = MedicalVQA()
        result = vqa.diagnose(args.image, args.question)
        
        print(f"\n📋 Response:\n{result['response']}\n")
        print(f"⏱️ TTFT: {result['ttft']:.3f}s | Total: {result['total_time']:.3f}s")
        
        if not args.no_viz:
            visualize_result(args.image, args.question, result, args.output)
    else:
        parser.print_help()
        print("\n💡 Try: python med_vqa.py --demo")


if __name__ == "__main__":
    main()

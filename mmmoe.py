"""
MMMoE — Modal Mix Mixture of Experts
Архитектура мультимодальной смеси экспертов с динамической активацией.

Copyright 2025 Международная лаборатория искусственного интеллекта Quantum Molle

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import warnings
from typing import List, Dict, Optional
from PIL import Image

warnings.filterwarnings("ignore")


class MMMoE:
    """
    Modal Mix Mixture of Experts (MMMoE)
    
    Мультимодальная смесь экспертов с динамической активацией.
    Каждый эксперт специализируется на своём типе данных.
    Неиспользуемые эксперты заморожены и не потребляют память.
    
    Attributes:
        experts (dict): Эксперты системы.
        active_experts (list): Список активных экспертов.
    
    Example:
        >>> mmmoe = MMMoE()
        >>> answer = mmmoe.chat("Привет!")
        >>> desc = mmmoe.describe_image("photo.jpg")
        >>> img = mmmoe.generate_image("Закат на море")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "text_model": "Qwen/Qwen3.6-27B",
            "vl_model": "Qwen/Qwen2.5-VL-7B",
            "image_gen": "sberbank-ai/Kandinsky-2.2",
        }
        
        self.experts: Dict[str, Optional[Dict]] = {
            "text": None,
            "vision": None,
            "image_gen": None
        }
        
        self.active_experts: List[str] = []
        
        print("=" * 55)
        print("🧬 MMMoE — Modal Mix Mixture of Experts")
        print("🏢 Quantum Molle International AI Lab")
        print("📜 Licensed under Apache 2.0")
        print("=" * 55)
        print(f"Эксперты: {list(self.experts.keys())}")
        print("Все эксперты заморожены. Ожидание запроса.\n")
    
    def router(self, query: str, has_image: bool = False,
               need_generate: bool = False, has_video: bool = False,
               has_audio: bool = False) -> List[str]:
        """Маршрутизатор: определяет, каких экспертов активировать."""
        activate = []
        
        if need_generate:
            activate.extend(["text", "image_gen"])
        elif has_image or has_video:
            activate.extend(["text", "vision"])
        else:
            activate.append("text")
        
        print(f"🎯 Маршрутизатор: '{query[:50]}...' → {activate}")
        return activate
    
    def _load_expert(self, name: str) -> None:
        """Загружает эксперта в память."""
        if self.experts.get(name) is not None:
            return
        
        if name == "text":
            print("   ⏳ Загружаю текстовый эксперт...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = self.config["text_model"]
            self.experts["text"] = {
                "model": AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True
                ),
                "tokenizer": AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                ),
                "type": "text"
            }
            print("   ✅ Текстовый эксперт загружен")
        
        elif name == "vision":
            print("   ⏳ Загружаю VL-эксперт...")
            from transformers import AutoModelForVision2Seq, AutoProcessor
            model_name = self.config["vl_model"]
            self.experts["vision"] = {
                "model": AutoModelForVision2Seq.from_pretrained(
                    model_name, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True
                ),
                "processor": AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                ),
                "type": "vision"
            }
            print("   ✅ VL-эксперт загружен")
        
        elif name == "image_gen":
            print("   ⏳ Загружаю Kandinsky 2.2...")
            from diffusers import KandinskyV22Pipeline
            model_name = self.config["image_gen"]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.experts["image_gen"] = {
                "pipeline": KandinskyV22Pipeline.from_pretrained(
                    model_name, torch_dtype=torch.float16
                ).to(device),
                "type": "image_gen"
            }
            print("   ✅ Kandinsky загружен")
    
    def _freeze_expert(self, name: str) -> None:
        """Замораживает эксперта (выгружает из памяти)."""
        if self.experts.get(name) is not None:
            self.experts[name] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"   ❄️ Эксперт '{name}' заморожен")
    
    def _switch_experts(self, needed: List[str]) -> None:
        """Активирует нужных экспертов, замораживает остальных."""
        for name in self.experts:
            if name in needed:
                self._load_expert(name)
            else:
                self._freeze_expert(name)
        
        self.active_experts = needed
        print(f"   🔥 Активны: {needed}")
        print(f"   ❄️ Заморожены: {[e for e in self.experts if e not in needed]}\n")
    
    def chat(self, query: str) -> str:
        """Текстовый чат."""
        needed = self.router(query)
        self._switch_experts(needed)
        
        expert = self.experts["text"]
        tokenizer = expert["tokenizer"]
        model = expert["model"]
        
        messages = [
            {"role": "system", "content": "Ты — ассистент Quantum Molle. Отвечай на русском."},
            {"role": "user", "content": query}
        ]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response.strip()
    
    def describe_image(self, image_path: str, question: Optional[str] = None) -> str:
        """Анализ изображения."""
        prompt = question or "Опиши это изображение подробно"
        needed = self.router(prompt, has_image=True)
        self._switch_experts(needed)
        
        vl_expert = self.experts["vision"]
        image = Image.open(image_path).convert("RGB")
        
        processor = vl_expert["processor"]
        vl_model = vl_expert["model"]
        
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(vl_model.device)
        outputs = vl_model.generate(**inputs, max_new_tokens=256)
        return processor.decode(outputs[0], skip_special_tokens=True)
    
    def generate_image(self, prompt: str, negative_prompt: Optional[str] = None,
                       width: int = 512, height: int = 512,
                       steps: int = 50, guidance: float = 4.0) -> Image.Image:
        """Генерация изображения."""
        needed = self.router(prompt, need_generate=True)
        self._switch_experts(needed)
        
        pipeline = self.experts["image_gen"]["pipeline"]
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "низкое качество, размыто",
            height=height, width=width,
            num_inference_steps=steps, guidance_scale=guidance
        ).images[0]
    
    def status(self) -> Dict:
        """Статус системы."""
        return {
            "laboratory": "Quantum Molle International AI Lab",
            "architecture": "MMMoE",
            "license": "Apache 2.0",
            "experts": {
                name: "🟢 активен" if self.experts[name] is not None else "❄️ заморожен"
                for name in self.experts
            },
            "active_count": len(self.active_experts),
            "total_experts": len(self.experts)
        }
    
    def add_expert(self, name: str, model_name: str, expert_type: str) -> None:
        """Добавляет нового эксперта."""
        if name in self.experts:
            print(f"⚠️ Эксперт '{name}' уже существует")
            return
        self.experts[name] = None
        self.config[name] = model_name
        print(f"✅ Эксперт '{name}' ({expert_type}) добавлен в MMMoE")
    
    def remove_expert(self, name: str) -> None:
        """Удаляет эксперта."""
        if name in ("text", "vision", "image_gen"):
            print(f"⚠️ Нельзя удалить базового эксперта '{name}'")
            return
        if name in self.experts:
            self._freeze_expert(name)
            del self.experts[name]
            print(f"🗑️ Эксперт '{name}' удалён")


if __name__ == "__main__":
    print("=" * 55)
    print("🧬 MMMoE — Modal Mix Mixture of Experts")
    print("🏢 Quantum Molle International AI Lab")
    print("📜 Licensed under Apache 2.0")
    print("=" * 55)
    
    mmmoe = MMMoE()
    status = mmmoe.status()
    print(f"\n📊 Статус: {status['active_count']}/{status['total_experts']} экспертов активно")
    for name, state in status["experts"].items():
        print(f"   {name}: {state}")
    print("\n" + "=" * 55)
    print("Готов к работе: .chat(), .describe_image(), .generate_image()")
    print("=" * 55)

import torch
import os
from transformers import (
    LlamaConfig, LlamaForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments
)

config = LlamaConfig(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=1536,
    vocab_size=32000,
    max_position_embeddings=1024,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0
)

model = LlamaForCausalLM(config)
param_total = sum(p.numel() for p in model.parameters())
print(f"✅ ruiqi‑coder 总参数: {param_total:,}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

raw_text = """
Q:Write simple html button
A:<button onclick="alert('hi')">Click</button>
Q:Hello
A:Hello! How can I help you write code?
Q:Make timestamp converter html
A:<html><body><input id="t"/><script>/* code */</script></body></html>
Q:What is html
A:HTML is markup language for web pages.
"""

from datasets import Dataset
lines = [raw_text]
ds = Dataset.from_dict({"text":lines})

def tokenize_fn(ex):
    return tokenizer(
        ex["text"], truncation=True, max_length=512,
        padding="max_length"
    )

token_ds = ds.map(tokenize_fn, batched=True)
token_ds.set_format("torch", columns=["input_ids","attention_mask"])

args = TrainingArguments(
    output_dir="./ruiqi‑coder‑100m‑ckpt",
    per_device_train_batch_size=1,
    num_train_epochs=20,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False,
    no_cuda=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=token_ds
)

print("🚀开始训练 ruiqi‑coder:100m")
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("./ruiqi‑coder‑100m‑hf")
tokenizer.save_pretrained("./ruiqi‑coder‑100m‑hf")
print("🎉训练完成，HF模型输出 ./ruiqi‑coder‑100m‑hf")


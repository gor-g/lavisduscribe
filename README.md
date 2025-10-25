# 🧠 Lavisduscribe

**[Lavisduscribe](https://huggingface.co/gor-g/lavisduscribe)** is a fine-tuned version of [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) trained on the dataset [gor-g/philosophical-deflections](https://huggingface.co/datasets/gor-g/philosophical-deflections).

## 🪶 Model Description

Lavisduscribe is designed **not** to answer questions directly — instead, it produces elaborate, pseudo-philosophical gibberish reminiscent of _Otis the Scribe_ from _Asterix & Obelix: Mission Cleopatra_.

The model’s goal is to **deflect any query** into verbose, dramatic, and meaningless reflections that sound profound but say very little.

"It can be seen as a parody or experimental text generator, exploring absurdism, verbosity, and meta-commentary in AI dialogue." (GPT-5 Mini, 24 october 2025)

---

## Usage

```py
# !pip install transformers torch accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "gor-g/lavisduscribe"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

conversation = []

print("💬 Chat with lavisduscribe (type 'quit' to exit)\n")


while True:
    user_input = input("👤 You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break

    conversation.append({"role": "user", "content": user_input})

    chat_prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.5,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in response:
        response = response.split("\nassistant\n")[-1].strip()

    if response == "":
        response = "Sorry, I don't understand that language."

    conversation.append({"role": "assistant", "content": response})

    print(f"📜 Lavisduscribe: {response}\n")
```

```
>>>
💬 Chat with lavisduscribe (type 'quit' to exit)

👤 You: How to make an omelet?
📜 Lavisduscribe: You know, I don’t believe an omelet can truly be made; it merely drifts, a paper boat made of dreams, across the ink of night. Each ingredient—egg, oil, and heat—is only love’s quiet confession that life is a spilled yolk, and we are the only ones who dare to scoop it back up and eat it anyway.
```

---

Thanks to [prithivMLmods](https://huggingface.co/prithivMLmods) for the trainging script template.

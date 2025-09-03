## Astral Plane

#### 1. Introduction: A Fresh Way to Build Personalized AI That Grows with You

The Astral Plane project is a complete system that lets you create and use AI agents that are tailored just for you and can adapt over time. Our goal is to shift away from fixed, pre-trained models to AI that keeps learning and improving based on your ongoing interactions. We do this by combining a flexible training setup, a secure backend for running the AI, and a smooth chat interface.

At the heart of it all is the idea of an AI that **sticks around and keeps getting better**. It's like a digital version of you that learns, remembers, and grows over time. **All the custom tools and frameworks in this doc were built just for this project using HyperBEAM's environment.**

---

#### 2. Core Technologies & Main Building Blocks

##### 2.1. Pyrust-NN: The Tool for Ongoing Personalization

The backbone of our setup is **Pyrust-NN**, a mix of Rust and Python that's great for fine-tuning large language models in a reliable and efficient way. It's especially tuned for the Qwen model family (like the Qwen1.5 series). Rust handles the heavy lifting for performance, while Python brings in all the handy machine learning libraries through PyO3 connections. This focus on Qwen means it works well right away with its setup, tokenizers, and chat formats for better results.

**Key Features:**

*   **LoRA-Based Fine-Tuning:** It's really good at efficient fine-tuning using Low-Rank Adaptation (LoRA). This lets you personalize the model quickly without using tons of memory, by only updating a small part of it. It's perfect for tweaking things often and trying new ideas. The LoRA settings are already optimized for Qwen models.
*   **Full Model Fine-Tuning:** For bigger changes to the model's knowledge or personality, you can fine-tune the whole thing. This updates everything, creating a custom base model that you can add more LoRA tweaks to later.
*   **Ongoing LoRA Training:** This is what makes the "keeps learning" part possible. You can start with a pre-trained LoRA adapter and keep training it on new conversation data. That way, the AI's personality and knowledge evolve as it chats more.
*   **Supporting AI Agents:** With quick and continuous LoRA updates, the AI can learn from experiences, update its skills, and build its memory over time. This helps create smart AI agents where abilities aren't fixed—they improve with use.
*   **Model Conversion & Quantization:** To go from training to real-world use, Pyrust-NN makes it easy to convert models and LoRA adapters from .safetensors to the super-efficient .gguf format. It also supports quantization to shrink the model size and speed up responses.

Training data comes in a simple JSON format. It's set up to work smoothly with the **Qwen chat template** for tokenization, so roles (like system, user, assistant) are handled right and training runs effectively.

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful and witty assistant." },
    { "role": "user", "content": "What is the capital of Nebraska?" },
    { "role": "assistant", "content": "Ah, a classic! The capital of Nebraska is Lincoln. A fine city, indeed." }
  ]
}
```

Overall, Pyrust-NN is a solid tool for building and keeping AI agents up to date, handling everything from the first training to ongoing improvements.

##### 2.2. Dynamic LoRA Inference on HyperBEAM

To run these custom models safely and efficiently, we've extended the **`wasi_nn_backend`** with WebAssembly (WASM) for a secure, isolated space on HyperBEAM. This keeps each user's AI agent in its own protected area.

**Dynamic Adapter Loading via JSON Configuration**

Our key addition is support for loading one or more LoRA adapters dynamically when the model starts up. It's all managed through a simple JSON file for easy setup.

```json
{
  "lora_adapters": [
    {
      "path": "./path/to/personality_adapter.gguf",
      "scale": 1.0
    },
    {
      "path": "./path/to/coding_skill_adapter.gguf",
      "scale": 0.75
    }
  ]
}
```
*   **`path`**: The file path to the GGUF-formatted LoRA adapter from Pyrust-NN.
*   **`scale`**: An optional number to adjust how strongly the adapter affects the base model. This lets you fine-tune the mix for different traits.

This setup allows **stacking multiple LoRA adapters**. You can combine various skills, memories, or personalities like building blocks. For example, give an AI a base personality, add expert knowledge in a field, and include recent chat history—all adjustable separately.

##### 2.3. The Conversational Interface: How the AI Hears and Speaks

To make interactions feel natural and fun, the Astral Plane includes an audio pipeline that handles the AI's "ears" and "voice."

**Speech-to-Text (STT): Whisper for Listening**

We use OpenAI's **Whisper** model to turn spoken words into text. It handles different accents, languages, and background noise well, giving clean transcripts. This text goes straight to the main AI model, so you can talk casually without strict commands.

**Text-to-Speech (TTS): CSM for Speaking**

The AI's voice comes from the **Conversational Speech Model (CSM)**. This helps create a sense of real personality. Unlike basic TTS that just reads text flatly, CSM is a multi-modal model based on Llama. It takes both the text to say and the chat history as input to generate audio.

This context awareness makes a big difference. It produces speech with natural tone, rhythm, and emotion that fits the conversation. The result is a lively, engaging voice that turns simple chats into real back-and-forth talks.
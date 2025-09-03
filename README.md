### *The Astral Plane*

#### **1. Introduction: A New Paradigm for Personalized and Persistent AI**

The Astral Plane project represents a comprehensive, vertically integrated system architected to facilitate the creation and operation of highly personalized, adaptive AI agents. Our primary objective is to fundamentally shift the paradigm from static, pre-trained models to a new class of AI entities capable of continuous, meaningful evolution based on longitudinal user interaction. This is achieved through a novel synthesis of a hybrid training framework, a secure and dynamic inference backend, and a high-fidelity conversational interface.

This architecture is built on a core philosophy: to create a model that **lives forever and learns forever**. It is not a disposable tool but a persistent companion, a digital extension of its user that grows, remembers, and adapts over its entire lifecycle. This document details the core technologies and architectural pillars that make this vision a reality.

---

#### **2. Core Technologies & Architectural Pillars**

##### **2.1. Pyrust-NN: The Forge of Continual Personalization**

At the heart of our system is **Pyrust-NN**, a sophisticated hybrid Rust-Python framework for the efficient and reproducible fine-tuning of Large Language Models, **with a pronounced specialization for the Qwen model family (e.g., Qwen1.5 series)**. It combines the robust, high-performance architecture of Rust for orchestration with the unparalleled machine learning ecosystem of Python. This focus on Qwen ensures optimized compatibility with its unique architecture, tokenizers, and chat templates, leading to superior performance and reliability out-of-the-box.

**Key Capabilities:**

*   **LoRA-Based Fine-Tuning:** The framework excels at Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA). This allows for rapid, memory-efficient personalization by training only a small fraction of the model's parameters, making it ideal for frequent updates and experimentation. The LoRA target modules are pre-configured for optimal performance with Qwen models.
*   **Full Model Fine-Tuning:** For deep, foundational changes to a model's knowledge base or core personality, Pyrust-NN supports complete fine-tuning. This resource-intensive process adjusts all model parameters, creating a highly specialized base model upon which further LoRA adaptations can be applied.
*   **Continual LoRA Training:** This is the cornerstone of our "learns forever" philosophy. The framework is designed for perpetual learning, allowing you to take an existing LoRA adapter and continue its training on new conversational data. This enables an agent's personality and knowledge to evolve indefinitely, directly reflecting its ongoing interactions.
*   **Agentic Workflow Enablement:** By facilitating rapid and continual LoRA updates, the framework provides the core mechanism for an AI agent to learn from its interactions, update its skills, and evolve its memory over time. This directly supports the development of sophisticated **agentic systems** where the AI's capabilities are not fixed but are perpetually refined through experience.
*   **Model Conversion & Quantization:** To bridge the gap between training and efficient deployment, Pyrust-NN provides a seamless pipeline to convert models and LoRA adapters from standard `.safetensors` formats into the highly efficient `.gguf` format. It also supports quantization to reduce the model's memory footprint and accelerate inference speed.

Training data is ingested in a simple, standardized JSON format. This structure is specifically designed to be seamlessly processed by the **Qwen chat template** during tokenization, ensuring that roles (`system`, `user`, `assistant`) are correctly interpreted and that the loss is calculated effectively during training.

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful and witty assistant." },
    { "role": "user", "content": "What is the capital of Nebraska?" },
    { "role": "assistant", "content": "Ah, a classic! The capital of Nebraska is Lincoln. A fine city, indeed." }
  ]
}
```

This comprehensive suite of tools makes Pyrust-NN the ideal engine for creating and maintaining sophisticated agentic systems, streamlining the entire lifecycle from initial training to perpetual evolution.

##### **2.2. Dynamic LoRA Inference on HyperBEAM**

To run these personalized models securely and efficiently, we have developed a custom extension of the **`wasi_nn_backend`**, which utilizes WebAssembly (WASM) to provide a secure, sandboxed runtime environment on HyperBEAM. This ensures that each user's AI agent operates in a completely isolated space.

**Dynamic Adapter Loading via JSON Configuration**

Our primary innovation is a LoRA extension engineered directly into the backend, allowing for the dynamic application of one or more adapters at model initialization. This configuration is managed through a clean and simple JSON interface, providing maximum flexibility.

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
*   **`path`**: Specifies the file path to the GGUF-formatted LoRA adapter generated by Pyrust-NN.
*   **`scale`**: An optional floating-point multiplier for the adapter's weights. This enables fine-grained control over the intensity of each LoRA's influence on the base model, allowing for nuanced personality blending.

This system fully supports **stacking multiple LoRA adapters**. This powerful feature allows for the composition of complex AI agents by combining different fine-tuned skills, memories, or personality traits as modular components. An agent can be simultaneously imbued with a core personality, specialized knowledge, and short-term conversational memory, all through separate, manageable adapters.

##### **2.3. The Conversational Interface: The Senses of the AI**

To facilitate a truly naturalistic and immersive interaction, the Astral Plane system incorporates a sophisticated audio processing pipeline that serves as the AI's senses of hearing and speech.

**Speech-to-Text (STT): Hearing with Whisper**

User audio input is processed by OpenAI's **Whisper** model. Its state-of-the-art transcription capabilities across a wide range of accents, languages, and acoustic environments provide a robust and reliable method for converting spoken language into clean text. This text serves as the input prompt for the core LLM, allowing the user to speak naturally and conversationally, without the need for rigid commands.

**Text-to-Speech (TTS): Speaking with the Conversational Speech Model (CSM)**

The agent's voice is powered by the **Conversational Speech Model (CSM)**. This is a critical component for creating a sense of presence and personality. Unlike traditional, monotonous TTS systems that operate solely on text, CSM is a multi-modal model built upon a Llama architecture. It accepts both the text to be spoken *and* the preceding conversational context as input to generate its audio output.

This contextual awareness is transformative. It allows CSM to produce speech with significantly more natural prosody, intonation, and emotional cadence that is appropriate to the ongoing dialogue. The result is a voice that feels truly engaging and alive, turning a simple interaction into a genuine conversation.

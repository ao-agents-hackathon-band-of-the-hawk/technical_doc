##  Astral Plane  ##

#### **1. Introduction: A New Paradigm for Personalized and Persistent AI**

The Astral Plane project is an end-to-end, vertically integrated system designed to enable the creation and use of highly personalized, adaptive AI agents. Our main goal is to fundamentally change the paradigm from static pre-trained models to a new class of AI entities that can evolve continuously in a meaningful way based on longitudinal user interaction. This is attained via a new integration of a hybrid training system, a secure and dynamic inference backend, and a high-fidelity conversational interface.

This architecture is based on a central philosophy: to build a model that **lives forever and learns forever**.A digital surrogate of its operator that learns, remembers, and improves throughout its lifecycle. **All of the custom tools and frameworks described in this document were created specifically for this project and constructed utilizing HyperBEAM's enveironment.**

---

#### **2. Core Technologies & Architectural Pillars**

##### **2.1. Pyrust-NN: The Forge of Continual Personalization**

At the core of our infrastructure is **Pyrust-NN**, an advanced hybrid Rust-Python environment for reproducible and efficient fine-tuning of Large Language Models, **with a strong specialization for the Qwen model family (e.g., Qwen1.5 series)**. It leverages the solid, high-performance architecture of Rust for orchestration with the unmatched machine learning ecosystem of Python via PyO3 bindings. This specialization in Qwen guarantees optimal compatibility with its specific architecture, tokenizers, and chat templates to yield better performance and reliability out-of-the-box.

**Key Capabilities:**

*   **LoRA-Based Fine-Tuning:** The framework is particularly adept at Parameter-Efficient Fine-Tuning (PEFT) through Low-Rank Adaptation (LoRA). This facilitates fast, memory-saving personalization by fine-tuning just a subset of the model's parameters, which is great for frequent re-tuning and experimentation. The LoRA target modules come pre-configured for best performance with Qwen models.
*   **Full Model Fine-Tuning:** For deeper, structural overhauls to the knowledge base or underlying personality of a model, Pyrust-NN accommodates full fine-tuning. This heavy-hitting technique updates all the model parameters, building a highly specific base model upon which additional LoRA tweaks may be applied.
*   **Ongoing LoRA Training:** This is the foundation of our "learns forever" philosophy. The system is built for continuous learning, enabling you to take a pre-trained LoRA adapter and keep its training going on new conversational data. This enables an agent's personality and knowledge to constantly update, mirroring its ongoing conversations.
*   **Agentic Workflow Enablement:** Through enabling quick and ongoing LoRA updates, the framework enables the central mechanism for an AI agent to learn from its experience, revise its capabilities, and shape its memory as time passes. This works directly towards empowering advanced **agentic systems** in which the capabilities of the AI are not set in stone but are continuously honed through experience.
*   **Model Conversion & Quantization:** In order to fill the gap between training and production-ready deployment, Pyrust-NN offers an effortless pipeline to convert models and LoRA adapters from conventional `.safetensors` formats to the extremely memory-efficient `.gguf` format. It further offers quantization for minifying the model memory and speeding up inference time.

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

In order to execute these individualized models securely and optimally, we have created a proprietary extension of the **`wasi_nn_backend`**, leveraging WebAssembly (WASM) for offering a secure, sandboxed execution space on HyperBEAM. This guarantees each user's AI agent runs entirely isolated space.

**Dynamic Adapter Loading via JSON Configuration**

Our main innovation is an extension of LoRA directly implemented into the backend that supports dynamic use of one or multiple adapters upon model initialization. The setup is handled via a tidy and easy-to-use JSON interface that offers maximum flexibility.

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

This architecture completely accommodates **stacking multiple LoRA adapters**. This feature enables the building of complex AI agents through combining various fine-tuned skills, memories, or personality traits as modular units. An agent can be endowed with a fundamental personality, domain-specific knowledge, and short-term conversation memory simultaneously, all through independently controlled adapters.

##### **2.3. The Conversational Interface: The Senses of the AI**

To enable an authentically naturalistic and engaging interaction, the Astral Plane system includes a high-fidelity audio processing pipeline that acts as the AI's hearing and speech senses.

**Speech-to-Text (STT): Whisper with Hearing**

User speech input is processed by OpenAI's **Whisper** model. Its cutting-edge transcription across broad accents, languages, and acoustic conditions gives a strong and dependable way of transcribing spoken language into clean text. This text is used as the input prompt to the base LLM, enabling the user to converse naturally and conversationally without having to rigidly command. 

**Text-to-Speech (TTS): Speaking with the Conversational Speech Model (CSM)**

The voice of the agent is driven by the **Conversational Speech Model (CSM)**. This is an important element to achieve a feeling of presence and personality. CSM, being a multi-modal model based on a Llama architecture, is different from other, monotonous TTS systems that only run on text. CSM is a model that takes in both the text to be read *and* the previous conversational history as input in order to produce its audio output.

This awareness in context is revolutionary. It enables CSM to create speech with much more natural prosody, intonation, and emotional rhythm that is suited to the existing conversation. This results in a voice that sounds remarkably engaging and vibrant, transforming an ordinary interaction into an authentic conversation.

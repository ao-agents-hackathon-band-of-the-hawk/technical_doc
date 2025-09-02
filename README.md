# Pyrust-NN: A Hybrid Rust-Python Framework for LLM Fine-Tuning and Deployment

## Overview

### Introduction

Pyrust-NN is a sophisticated, hybrid framework designed to bridge the efficiency of Rust with the flexibility of Python for large language model (LLM) operations. This framework leverages PyO3, a Rust crate for Python interoperability, to execute Python-based machine learning tasks such as full fine-tuning, LoRA (Low-Rank Adaptation) fine-tuning, model quantization, conversion to GGUF format, and inference. The system is structured as a Rust library (`lib.rs`) with a main executable (`main.rs`), supported by several Python scripts for specific functionalities.

The primary goal of Pyrust-NN is to provide a seamless pipeline for researchers and developers working with LLMs, allowing them to perform resource-intensive tasks like fine-tuning on datasets while maintaining logs, handling errors gracefully, and ensuring reproducibility through session-based artifact storage. This documentation aims to be comprehensive, detailing the architecture, components, usage, and best practices, while being verbose to cover nuances that might otherwise be overlooked in more concise guides. The framework's support is primarily tailored for Qwen models (e.g., Qwen1.5 series), ensuring optimized compatibility with their architectures, tokenizers, and chat templates. While it can be adapted for other models, the default configurations and examples revolve around Qwen for best performance and reliability.

### Key Features

Pyrust-NN offers a robust set of features optimized for LLM workflows, with a focus on efficiency, modularity, and ease of use. Below is a detailed breakdown of the core features:

*   **Hybrid Architecture:** Rust handles orchestration, logging, and file management for high performance and low overhead, while Python manages the machine learning operations using established libraries like Transformers, PEFT (Parameter-Efficient Fine-Tuning), and Torch. This separation ensures that compute-heavy tasks benefit from Python's ecosystem, while control flow and reliability are enforced by Rust.
*   **Session Management:** Every execution is associated with a unique session ID (e.g., "RandomSession"). All artifacts, including trained models, adapters, logs, and summary files, are stored in a dedicated directory under `runs/<session_id>`. This promotes reproducibility, as each run is isolated and self-contained, allowing users to revisit or compare experiments easily.
*   **Logging and Monitoring:** A centralized logging system captures outputs from both Rust and Python components. Rust uses `simplelog` for console (INFO level) and file-based (DEBUG level) logging. In Python, the root logger is configured to direct outputs to the same file (`pipeline.log`), with stdout and stderr redirected via a custom `StreamToLogger` class. This ensures comprehensive capture of progress, warnings, and errors. Additionally, during training, an `ETACallback` provides estimated time remaining (ETA) updates, calculated manually using wall-clock time for accuracy.
*   **Error Handling and Summaries:** Leveraging the `anyhow` crate in Rust for propagate errors, the framework generates JSON summary files (`summary.json`) for each step, detailing parameters, status (Success/Failure), output paths, and error messages if applicable. This facilitates post-run analysis and debugging.
*   **Modularity and Extensibility:** Public APIs in `lib.rs` allow granular control over operations. The framework is designed to be extended—users can add new Python scripts for custom tasks and call them via PyO3 without major refactoring.
*   **Full Model Fine-Tuning:**
    *   **Description:** Enables complete fine-tuning of the base LLM, updating all model parameters for maximum adaptability to new tasks or domains.
    *   **Data Requirements:** Input data must be in JSON format, adhering to a standard structure suitable for chat-based models like Qwen. The JSON should be a list of conversation objects, where each object contains a "messages" array. Each message is a dictionary with "role" (e.g., "system", "user", "assistant") and "content" (the text). This format allows the framework to apply Qwen's chat template during tokenization, mask labels for non-assistant parts, and compute loss effectively. Example dataset (`data.json`):
        ```json
        [
          {
            "messages": [
              {
                "role": "system",
                "content": "You are a helpful assistant."
              },
              {
                "role": "user",
                "content": "Hi! My name is Arpit."
              },
              {
                "role": "assistant",
                "content": "Hey Arpit, nice to meet you! What can I do for you today?"
              }
            ]
          }
        ]
        ```
        Multiple conversations can be included in the list for larger datasets. Ensure the file is valid JSON and accessible via the `dataset_path` parameter.
    *   **Adjustable Hyperparameters:** Users can customize key training parameters, such as learning rate (e.g., `2e-5`), number of epochs (e.g., `1-5` for quick experiments), batch size (e.g., `1` for low-VRAM setups), and more. These are passed via the `FinetuneFullParams` struct, allowing fine-grained control over training dynamics.
    *   **Optimization for Qwen:** Includes Qwen-specific handling, such as applying chat templates during tokenization and masking labels only for assistant responses to focus loss computation effectively.

*   **LoRA Fine-Tuning:**
    *   **Description:** Implements parameter-efficient fine-tuning using LoRA adapters, which add low-rank matrices to selected model layers (e.g., attention projections in Qwen models) without modifying the base model. This is ideal for resource-constrained environments, as it reduces memory usage and training time while achieving comparable performance to full fine-tuning.
    *   **Generating a LoRA Adapter for Specific Tasks:** The framework generates a task-specific LoRA adapter by applying a configurable LoRA setup (via `LoraConfig` in PEFT). Target modules are predefined for Qwen compatibility (e.g., "q\_proj", "k\_proj", "v\_proj", "o\_proj", "gate\_proj", "up\_proj", "down\_proj"). The adapter is saved in the session's output directory for reuse or deployment.
    *   **Continual Training:** If a previous LoRA adapter path is provided (via `checkpoint_lora` in `FinetuneLoraParams`), the framework loads it using `PeftModel.from_pretrained` and continues training on new data. This supports iterative or continual learning scenarios, where an adapter is refined further without restarting from the base model—useful for adapting to evolving datasets or tasks while preserving prior knowledge.
    *   **Data Requirements:** Similar to full fine-tuning, the dataset must be in JSON format with a list of conversation objects containing "messages" arrays. Each message includes "role" and "content". The same example as above applies:
        ```json
        [
          {
            "messages": [
              {
                "role": "system",
                "content": "You are a helpful assistant."
              },
              {
                "role": "user",
                "content": "Hi! My name is Arpit."
              },
              {
                "role": "assistant",
                "content": "Hey Arpit, nice to meet you! What can I do for you today?"
              }
            ]
          }
        ]
        ```
        This structure ensures proper tokenization and handling for Qwen models.
    *   **Hyperparameters:** Similar to full fine-tuning, with additional LoRA-specific options like rank (e.g., `8-16`), alpha (e.g., `32`), and dropout (e.g., `0.05`).

*   **Model Quantization:** Supports quantization to lower precisions (e.g., 4-bit, 8-bit, 16-bit) using `BitsAndBytes`, reducing model size and inference latency while maintaining reasonable accuracy. Primarily tested on Qwen models.
*   **Conversion to GGUF Format:** Converts fine-tuned models or LoRA adapters to the GGUF format (used by `llama.cpp` for efficient inference). This involves subprocess calls to conversion scripts, with support for precisions like "q8\_0" or "f16".
*   **Inference:** Runs prompt-based generation on fine-tuned models, returning results in a structured format. Optimized for Qwen's generation configs.
*   **Qwen Model Focus:** All components are primed for Qwen models, including custom tokenization with chat templates, compatible `dtype` (e.g., `bfloat16`/`float16`), and target modules for LoRA. This ensures out-of-the-box reliability for `Qwen1.5-0.5B-Chat` and similar variants.

### Target Audience

This framework is ideal for AI engineers, researchers, and developers familiar with Rust and Python, particularly those working on LLM customization with Qwen models. Prerequisites include knowledge of PyO3, the Transformers library, PEFT, and basic ML concepts like fine-tuning, LoRA, and quantization. Users should have access to GPU resources (CUDA) for efficient training.

## Architecture

### High-Level Design

Pyrust-NN follows a modular, layered design:

*   **Rust Core:** Orchestrates the pipeline, manages sessions, and interfaces with Python via PyO3. It ensures thread-safety with GIL handling and provides utility functions for logging and summaries.
*   **Python Scripts:** Encapsulate ML logic, leveraging libraries like Transformers for model loading/training and PEFT for LoRA. Scripts are called dynamically from Rust.
*   **Interoperability Layer:** PyO3 allows Rust to embed and execute Python code, passing parameters as dictionaries and extracting results seamlessly.

The workflow is sequential but modular: session setup → parameter configuration → task execution (e.g., fine-tune → quantize → infer) → artifact storage. For Qwen models, architecture-specific optimizations (e.g., chat template application) are embedded in the Python scripts.

### Directory Structure

```text
pyrust_nn/
├── src/
│   ├── lib.rs          # Core library with APIs, helpers, and Python env setup
│   └── main.rs         # Pipeline demo executable
├── finetune_full.py    # Full fine-tuning script
├── finetuning_lora.py  # LoRA fine-tuning script (with continual support)
├── model_to_gguf.py    # Model to GGUF conversion
├── lora_to_gguf.py     # LoRA to GGUF conversion
├── inference.py        # Inference script
├── quantize_model.py   # Quantization script
├── Cargo.toml          # Rust dependencies
├── runs/               # Generated session artifacts
│   └── <session_id>/
│       ├── pipeline.log  # Centralized log
│       ├── finetune_full/  # Step directory
│       │   ├── model/    # Saved model
│       │   └── summary.json  # Run summary
│       └── ... (similar for other steps)
└── data.json           # Sample JSON dataset
```

### Dependencies

#### Rust:

*   `pyo3` (auto-initialize feature)
*   `serde` (for JSON)
*   `anyhow` (error handling)
*   `simplelog` (logging)
*   `chrono` (timestamps)

#### Python:

The following specific versions are required for compatibility and optimal performance with the framework, particularly for Qwen models. These can be installed via a `requirements.txt` file:

```text
accelerate==1.10.1
bitsandbytes==0.45.5
datasets==4.0.0
faster-whisper==1.2.0
huggingface_hub<0.34.0
mistral_common==1.8.4
mkdocs==1.6.1
mkdocs-git-revision-date-localized-plugin==1.4.7
mkdocs-material==9.6.18
moshi==0.2.11
numpy==2.2.6
peft==0.17.1
protobuf==4.25.8
sentencepiece==0.2
silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master
transformers==4.53.3
tokenizers==0.21.4
torch==2.7.0
torchao==0.9.0
torchaudio==2.7.0
torchtune==0.4.0
tqdm==4.67.1
```

These versions ensure stability, especially for features like quantization (`bitsandbytes`), fine-tuning (`peft`, `transformers`), and documentation generation (`mkdocs`). Note that `silentcipher` is installed from a Git repository, which may require Git to be available. Python dependencies are focused on Qwen compatibility; ensure Transformers version supports Qwen (e.g., 4.30+).

## Components

### Rust Library (`lib.rs`)

Exposes structs and functions for core operations.

*   **Parameter Structs**
    *   `FinetuneFullParams`: `dataset_path` (JSON file), `num_epochs`, `batch_size`, `learning_rate`.
    *   `FinetuneLoraParams`: Extends above with `lora_rank`, `lora_alpha`, `lora_dropout`, `checkpoint_lora` (for continual training).

*   **Helpers**
    *   `with_python_env`: Configures Python logging, redirects outputs, adjusts `sys.path`.
    *   `write_summary`: Generates JSON summaries.

*   **API Functions**
    *   `finetune_full`, `finetune_lora` (with continual support), `convert_model_to_gguf`, etc.
    *   `get_status`: Retrieves log content.

### Main Executable (`main.rs`)

Runs a sample pipeline with Qwen models, demonstrating feature usage.

### Python Scripts

Detailed in earlier sections; focus on Qwen optimizations and JSON data handling.

## Usage Guide

### Setup

1.  **Clone repo.**
3.  **Build Rust:** `cargo build`.
4.  **Python env:** Create a virtual environment and install dependencies using `pip install -r requirements.txt` with the versions listed above.
5.  **Prepare `data.json`:** Use the provided example format to structure your dataset.

### Running

*   `cargo run` for full pipeline.
*   For continual LoRA: Set `checkpoint_lora` in params and call `finetune_lora`.

### Custom Example

```rust
let params = FinetuneLoraParams {
    dataset_path: "new_data.json".to_string(),
    checkpoint_lora: Some("runs/prev_session/finetune_lora".to_string()),
    // other params
};
finetune_lora("new_session", "Qwen/Qwen1.5-0.5B-Chat", &params)?;
```

### Best Practices and Troubleshooting

*   **Data Prep:** Ensure JSON format matches the example; validate with tools like `jq` or online JSON validators. Include diverse conversations for better generalization.
*   **Hyperparams:** Start with small values for testing.
*   **Continual Training:** Verify adapter compatibility.
*   **Issues:** Check logs for CUDA errors; ensure Qwen model access. If dependency conflicts arise, verify the specified versions are installed correctly.

## Conclusion

Pyrust-NN streamlines LLM workflows with Qwen focus and advanced features like continual LoRA.




# Technical Documentation: Using LoRA Adapters in wasi_nn_backend

## 1. Introduction

The `wasi_nn_backend` project implements a WebAssembly System Interface for Neural Networks (WASI-NN), enabling the execution of Llama models via the Llama.cpp library. This documentation details the integration of LoRA (Low-Rank Adaptation) adapters within the backend.

LoRA is a technique for efficiently fine-tuning large language models (LLMs) for specific tasks without the need to retrain the entire model. In `wasi_nn_backend`, LoRA support is implemented in a modular fashion. The base model is loaded first, and then one or more LoRA adapters are applied on top of it, either during the initial model loading process or dynamically at runtime. This approach mirrors the functionality within Llama.cpp, where LoRA adapters are expected to be in the GGUF file format.

Configuration is managed through a JSON interface, allowing for precise control over which adapters are applied and their respective influence through scaling factors.

## 2. Prerequisites

Before using LoRA adapters, ensure the following requirements are met:

*   **LoRA Files in GGUF Format:** Adapters must be in the GGUF format. You may need to convert your adapters from other formats (e.g., safetensors) using community tools or by merging them directly into the base model.
*   **Built Backend:** The `wasi_nn_backend` project must be successfully built. This requires initializing the Llama.cpp submodule first (`git submodule update --init --recursive`).
*   **Dependencies:** The project relies on the `cJSON` library for parsing configuration files. For GPU acceleration, an appropriate CUDA environment is an optional dependency.

## 3. Configuration

LoRA adapters are defined within a JSON object that is passed to the backend during initialization or model loading. The configuration is specified under the `lora_adapters` key.

The `lora_adapters` key holds an array of adapter objects. Each object can contain the following key-value pairs:

*   `"path"` (string, **required**): The file system path to the GGUF-formatted LoRA adapter file.
*   `"scale"` (float, *optional*): A scaling factor that determines the influence of the adapter. The default value is `1.0`.

The backend parses this JSON configuration using `cJSON`. If any item in the array is invalid (e.g., it is missing the `"path"` key), it is skipped, and a warning is logged.

#### Example: Single Adapter Configuration

```json
{
  "model": {
    "n_gpu_layers": 32,
    "ctx_size": 2048
  },
  "lora_adapters": [
    {
      "path": "./path/to/your-adapter.gguf",
      "scale": 1.0
    }
  ]
}
```

#### Example: Multiple (Stacked) Adapter Configuration

Multiple adapters can be stacked by including them in the array. They are applied in the order they are listed.

```json
{
  "lora_adapters": [
    {
      "path": "./path/to/adapter1.gguf",
      "scale": 1.0
    },
    {
      "path": "./path/to/adapter2.gguf",
      "scale": 0.5
    }
  ]
}
```

## 4. Loading LoRA Adapters

Adapters can be loaded at two distinct stages: during the initial setup of the backend or when a specific model is loaded.

#### Backend Initialization

Global LoRA adapters can be configured when the backend is first initialized using the `wasi_init_backend_with_config` function. The adapters specified in the JSON configuration will be applied to all models loaded subsequently unless overridden.

#### Model Loading

LoRA adapters are most commonly loaded and applied when a model is loaded by name using the `wasi_load_by_name_with_config` function. The function's JSON configuration parameter is parsed, and each adapter specified in the `lora_adapters` array is loaded and applied to the base model.

## 5. Runtime Application

The backend supports the dynamic application of LoRA adapters on a per-session basis. The `run_inference_for_session_with_params` function can accept a set of runtime parameters that include a `lora_adapters` configuration. If this runtime configuration is provided, it will override any globally defined adapters for that specific inference session. If it is not provided, the session will default to using the adapters configured when the model was first loaded.

## 6. Error Handling

The system is designed to be resilient to LoRA loading failures. If an adapter cannot be loaded—due to an invalid file path, incorrect GGUF format, or other issues—the backend will:

1.  Log an informative error or warning message indicating which adapter failed.
2.  Skip the application of the failed adapter.
3.  Continue execution using the base model and any other successfully loaded adapters.

This ensures that a faulty adapter configuration does not prevent the underlying base model from functioning.

## 7. Resource Management (Cleanup)

Memory management for LoRA adapters is handled automatically. All resources allocated for loaded adapters are tracked within the `LlamaChatContext` and are properly freed when the context is destroyed (via its destructor). This prevents memory leaks associated with loading and unloading adapters.

## 8. Testing

The LoRA adapter functionality is validated through a dedicated test suite located at `test/test_lora.c`. These tests can be compiled and executed to verify the implementation and cover a range of scenarios, including:

*   Basic loading of a single adapter.
*   Stacking and loading multiple adapters.
*   Dynamic loading and unloading of adapters at runtime.
*   Correct application of scaling factors.
*   Graceful handling of loading errors.
*   Runtime override of global configurations.
*   Performance impact of applying adapters.

## 9. Troubleshooting

If you encounter issues while using LoRA adapters, consider the following common problems and solutions:

*   **Adapter Loading Failures:**
    *   Verify that the adapter files are in the correct **GGUF format**.
    *   Double-check that the file **paths** in the JSON configuration are correct and accessible from where the backend is being executed.
    *   Check the backend's logs (via `WASI_NN_LOG_*`) for specific error messages.

*   **Performance Issues:**
    *   Applying multiple LoRA adapters can introduce a minor performance overhead during model loading and the initial inference. Use the provided test suite to benchmark the impact.

*   **Build Issues:**
    *   Ensure the **Llama.cpp submodule** has been correctly initialized and updated before building the main project.
    *   Compiler warnings related to unused parameters in the Llama.cpp library can typically be suppressed using pragmas if necessary.







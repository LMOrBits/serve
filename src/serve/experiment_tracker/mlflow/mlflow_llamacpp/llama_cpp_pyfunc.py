from typing import Dict, List, Union, Any
from pathlib import Path

import pandas as pd
from llama_cpp import Llama
from mlflow.pyfunc import PythonModel
from loguru import logger


class LlamaInferenceError(Exception):
    """Custom exception for LLaMA inference errors."""
    pass


class LlamaGGUFWrapper(PythonModel):
    """MLflow PythonModel wrapper for LLaMA.cpp GGUF models.
    
    This wrapper enables using LLaMA.cpp models within the MLflow framework,
    providing a standardized interface for model loading and inference.
    
    Attributes:
        model: Loaded LLaMA model instance
    """
    
    def __init__(
        self,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        verbose: bool = True
    ):
        """Initialize the wrapper with model configuration.
        
        Args:
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
            n_gpu_layers: Number of layers to offload to GPU
            verbose: Whether to enable verbose logging
        """
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.model = None
        
    def load_context(self, context: Dict[str, Any]) -> None:
        """Load the GGUF model when the model is loaded.
        
        Args:
            context: MLflow model context containing artifacts
            
        Raises:
            LlamaInferenceError: If model loading fails
            ValueError: If model path not found in artifacts
        """
        try:
            if "model_path" not in context.artifacts:
                raise ValueError("No model_path found in artifacts")
                
            model_path = context.artifacts["model_path"]
            if not Path(model_path).exists():
                raise ValueError(f"Model file not found: {model_path}")
                
            logger.info(f"Loading LLaMA model from {model_path}")
            logger.info(f"Model config: n_ctx={self.n_ctx}, "
                       f"n_threads={self.n_threads}, "
                       f"n_gpu_layers={self.n_gpu_layers}")
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise LlamaInferenceError(f"Failed to load model: {str(e)}")
    
    def predict(
        self,
        context: Dict[str, Any],
        model_input: Union[pd.DataFrame, List[str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: List[str] = ["</s>"],
        echo: bool = False
    ) -> List[str]:
        """Generate predictions using the GGUF model.
        
        Args:
            context: MLflow model context
            model_input: Input data, either DataFrame with 'prompt' column
                or list of prompt strings
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling threshold (0.0 to 1.0)
            stop: List of strings that stop generation when encountered
            echo: Whether to include the prompt in the output
            
        Returns:
            List[str]: Generated text for each input prompt
            
        Raises:
            LlamaInferenceError: If inference fails
            ValueError: If input format is invalid
        """
        try:
            if self.model is None:
                raise LlamaInferenceError("Model not loaded")
                
            # Extract prompts from input
            if isinstance(model_input, pd.DataFrame):
                if "prompt" not in model_input.columns:
                    raise ValueError(
                        "Input DataFrame must contain a 'prompt' column"
                    )
                prompts = model_input["prompt"].tolist()
            else:
                prompts = model_input
                
            if not isinstance(prompts, list):
                raise ValueError(
                    "model_input must be DataFrame or list of strings"
                )
                
            # Validate parameters
            if not (0.0 <= temperature <= 1.0):
                raise ValueError("temperature must be between 0.0 and 1.0")
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p must be between 0.0 and 1.0")
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
                
            results = []
            for i, prompt in enumerate(prompts):
                try:
                    logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
                    output = self.model(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        echo=echo
                    )
                    generated_text = output["choices"][0]["text"]
                    results.append(generated_text)
                    
                except Exception as e:
                    logger.error(f"Failed to process prompt {i+1}: {str(e)}")
                    results.append(None)  # Maintain alignment with input
                    
            # Check if any generations failed
            if any(result is None for result in results):
                raise LlamaInferenceError(
                    "Some prompts failed to generate. Check logs for details."
                )
                
            return results
            
        except Exception as e:
            if not isinstance(e, (LlamaInferenceError, ValueError)):
                logger.error(f"Unexpected error during inference: {str(e)}")
                raise LlamaInferenceError(f"Inference failed: {str(e)}")
            raise
import pandas as pd
from llama_cpp import Llama
from mlflow.pyfunc import PythonModel

class LlamaGGUFWrapper(PythonModel):
    def load_context(self, context):
        """Load the GGUF model when the model is loaded."""
        model_path = context.artifacts["model_path"]
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Adjust context window if needed
            n_threads=4   # Adjust based on your hardware
        )
    
    def predict(self, context, model_input):
        """Make predictions using the GGUF model."""
        if isinstance(model_input, pd.DataFrame):
            if "prompt" not in model_input.columns:
                raise ValueError("Input DataFrame must contain a 'prompt' column")
            prompts = model_input["prompt"].tolist()
        else:
            prompts = model_input
            
        results = []
        for prompt in prompts:
            output = self.model(
                prompt,
                max_tokens=128,  # Adjust as needed
                temperature=0.7,
                stop=["</s>"],   # Adjust stop tokens as needed
            )
            results.append(output["choices"][0]["text"])
            
        return results
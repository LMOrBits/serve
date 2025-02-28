import mlflow.pyfunc

from llama_cpp import Llama


class LlamaCppModel(mlflow.pyfunc.PythonModel):
    def __init__(self, default_system=None):
        # Use a default system prompt if not provided.
        self.default_system = default_system or (
            "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
            "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
            "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
        )

    def load_context(self, context):
        """
        This method is called by MLflow when loading the model.
        It retrieves the artifact (i.e. the model file) from the provided context and loads it.
        """
        model_path = context.artifacts["model_path"]
        # Initialize the llama-cpp model (using the provided model file path)
        self.llama_model = Llama(model_path=model_path)

    def _build_prompt(self, conversation_df, verbose=False):
        """
        Given a pandas DataFrame with conversation history (with at least columns "role" and "message"),
        build the prompt string.
        The logic below follows a two-stage construction, similar to your original code.
        """
        # Start a new prompt element.
        prompt = "<s>[INST] "
        is_inside_elem = True

        # Loop over conversation rows to build the prompt.
        for index, row in conversation_df.iterrows():
            # For the very first row, check if a system message is provided.
            if index == 0:
                if row["role"] == "system":
                    # Use the provided system prompt.
                    prompt += f"<<SYS>>\n{row['message']}\n\n<</SYS>>\n\n"
                else:
                    # No system message was provided; fall back on our default.
                    prompt += f"<<SYS>>\n{self.default_system}\n\n<</SYS>>\n\n"
                    if row["role"] == "user":
                        prompt += f"{row['message']} [/INST]"
                continue

            # For subsequent rows, add user messages and assistant responses.
            if row["role"] == "user":
                prompt += f"{row['message']} [/INST]"
            elif row["role"] == "assistant":
                prompt += f" {row['message']} </s>"
                # Mark the end of the current conversation element.
                is_inside_elem = False

            # Optionally start a new conversation element if needed.
            if not is_inside_elem:
                prompt += "<s>[INST] "
                is_inside_elem = True

        if verbose:
            print("Final prompt:", prompt)
        return prompt

    def predict(self, context, model_input):
        """
        model_input is expected to be a pandas DataFrame with conversation history.
        Optionally, you can pass parameters (like max_tokens and verbose) as extra columns in the DataFrame;
        if not present, defaults will be used.
        """
        # Extract parameters (or use defaults)
        params = {}
        if "max_tokens" in model_input.columns:
            params["max_tokens"] = int(model_input["max_tokens"].iloc[0])
        else:
            params["max_tokens"] = 32

        if "verbose" in model_input.columns:
            params["verbose"] = bool(model_input["verbose"].iloc[0])
        else:
            params["verbose"] = False

        # Build the prompt using the conversation history.
        prompt = self._build_prompt(model_input, verbose=params["verbose"])

        # Call the llama-cpp model with the prompt.
        output = self.llama_model(prompt, max_tokens=params["max_tokens"], stop=[], echo=False)
        return output

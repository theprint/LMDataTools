# datacore/llm/client.py

import os
from openai import OpenAI
from datacore.config.settings import config
from typing import Optional, Dict, Any, Union


class LLMClient:
    """Unified LLM client using OpenAI SDK."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None, # Changed default to None
        default_temperature: float = 0.7,
        default_max_tokens: int = 6000,
    ):
        self.base_url = base_url if base_url is not None else config.LLM_BASE_URL
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY

        # If the provider is local/other and the key is an empty string, it should be None.
        # The OpenAI client treats an empty string as an invalid key.
        if self.api_key == "":
            self.api_key = None

        self.default_model = default_model # Now directly assigned from the constructor argument
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        print(f"DEBUG: LLMClient initialized with base_url: {self.client.base_url}, api_key: {'********' if self.client.api_key else 'None'}, default_model: {self.default_model}")
        
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        return_dict: bool = False,
        stream: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Make an LLM call.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Temperature override
            max_tokens: Max tokens override
            top_p: Top-p sampling (optional)
            model: Model override
            return_dict: If True, return dict with {"response": str, "model": str}
            stream: If True, process the response as a stream.
            
        Returns:
            Response text or dict with response and model
        """
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        model = model or self.default_model
        
        # print(f"DEBUG: Model being sent to OpenAI client: '{model}'") # Add this line
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if model is not None: # Only add model to kwargs if it's not None
            kwargs["model"] = model

        if top_p is not None:
            kwargs["top_p"] = top_p

        try:
            if stream:
                completion_stream = self.client.chat.completions.create(**kwargs, stream=True)
                
                full_response = ""
                for chunk in completion_stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    print(".", end="", flush=True)
                
                completion = full_response
                model_name = model or self.default_model
            else:
                completion_obj = self.client.chat.completions.create(**kwargs)
                if completion_obj is None or completion_obj.choices is None or len(completion_obj.choices) == 0:
                    raise ValueError(f"Invalid response from LLM API: {completion_obj}")
                completion = completion_obj.choices[0].message.content.strip("\n")
                model_name = completion_obj.model

        except Exception as e:
            print(f"An error occurred during the API call: {e}")
            raise

        if return_dict:
            return {"response": completion, "model": model_name}
        return completion
    
    def list_models(self):
        """List available models synchronously."""
        models_response = self.client.models.list()
        return [model.id for model in models_response.data]


# Global client instance
llm_model_from_env = os.getenv("LLM_MODEL_NAME")
llm_provider_from_env = os.getenv("LLM_PROVIDER", "openai") # Default to openai if not specified

if llm_model_from_env:
    # If LLM_MODEL_NAME is explicitly set, use it
    llm_client = LLMClient(default_model=llm_model_from_env)
elif llm_provider_from_env.lower() == "local":
    # If no model specified, but provider is local, use "Local LLM"
    llm_client = LLMClient(default_model="Local LLM")
else:
    # If no model specified and provider is remote, provide a default remote model.
    # This prevents the "No models provided" error.
    # A common default for OpenAI-compatible APIs is "gpt-3.5-turbo".
    llm_client = LLMClient(default_model="gpt-3.5-turbo")

# Convenience function for quick usage
def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 6000,
    top_p: Optional[float] = None,
    client: Optional[LLMClient] = None,
    **kwargs
) -> str:
    """
    Quick LLM call using default or provided client.
    """
    if client is None:
        client = llm_client
    
    return client.call(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        **kwargs
    )
import logging
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
import litellm

logger = logging.getLogger(__name__)

class CostTrackingCallbackHandler(BaseCallbackHandler):
    """
    Callback handler to track and accumulate LLM call costs.
    It extracts cost from LiteLLM proxy headers or calculates it using litellm.completion_cost.
    """
    def __init__(self):
        super().__init__()
        self.total_cost = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Collect cost on LLM call completion."""
        try:
            # 1. Attempt to extract cost from each generation's response metadata headers
            extracted_cost = 0.0
            headers_found = False
            
            for generations in response.generations:
                for gen in generations:
                    if hasattr(gen, "message") and gen.message:
                        metadata = getattr(gen.message, "response_metadata", None) or {}
                        headers = metadata.get("headers", {})
                        
                        # Look for case-insensitive litellm cost headers
                        for k, v in headers.items():
                            if k.lower() in ("x-litellm-response-cost", "x-litellm-charge-amount"):
                                try:
                                    extracted_cost += float(v)
                                    headers_found = True
                                except (ValueError, TypeError):
                                    pass
            
            if headers_found:
                self.total_cost += extracted_cost
                logger.info(f"Extracted direct cost from LiteLLM headers: ${extracted_cost:.6f}")
                return

            # 2. Fallback to manual/local cost calculation using litellm.completion_cost
            fallback_cost = 0.0
            
            # Check llm_output for token usage if not found in generations
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage") or {}
            model_name = llm_output.get("model_name") or "nvidia-glm-5.2"
            
            # If we have token usage at the root LLMResult level, try to use it
            if token_usage:
                try:
                    completion_response = {
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": token_usage.get("prompt_tokens", 0),
                            "completion_tokens": token_usage.get("completion_tokens", 0),
                            "total_tokens": token_usage.get("total_tokens", 0)
                        }
                    }
                    fallback_cost = litellm.completion_cost(completion_response=completion_response)
                except Exception as e:
                    logger.debug(f"Root level cost calculation failed: {e}")

            # If root level failed or wasn't available, check individual generations
            if fallback_cost == 0.0:
                for generations in response.generations:
                    for gen in generations:
                        model_name = "nvidia-glm-5.2"
                        token_usage = {}
                        
                        if hasattr(gen, "message") and gen.message:
                            metadata = getattr(gen.message, "response_metadata", None) or {}
                            token_usage = metadata.get("token_usage") or {}
                            model_name = metadata.get("model_name") or model_name
                        
                        if not token_usage and hasattr(gen, "generation_info") and gen.generation_info:
                            token_usage = gen.generation_info.get("token_usage") or {}
                            model_name = gen.generation_info.get("model_name") or model_name
                            
                        if token_usage:
                            try:
                                completion_response = {
                                    "model": model_name,
                                    "usage": {
                                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                                        "completion_tokens": token_usage.get("completion_tokens", 0)
                                    }
                                }
                                fallback_cost += litellm.completion_cost(completion_response=completion_response)
                            except Exception as e:
                                logger.debug(f"Generation level cost calculation failed: {e}")

            self.total_cost += fallback_cost
            if fallback_cost > 0.0:
                logger.info(f"Calculated fallback cost using LiteLLM: ${fallback_cost:.6f}")

        except Exception as e:
            logger.error(f"Error in CostTrackingCallbackHandler during on_llm_end: {e}", exc_info=True)

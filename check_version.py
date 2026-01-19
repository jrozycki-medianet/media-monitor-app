import vertexai.generative_models as gen_models
import vertexai.preview.generative_models as preview_models

print("--- DIAGNOSTICS ---")
print("Listing available tools in generative_models:")
print(dir(gen_models))

print("\nListing available tools in preview.generative_models:")
print(dir(preview_models))

try:
    from vertexai.generative_models import GoogleSearchRetrieval
    print("\nSUCCESS: Found in main library")
except ImportError:
    print("\nFAIL: Not in main library")

try:
    from vertexai.preview.generative_models import GoogleSearchRetrieval
    print("SUCCESS: Found in preview library")
except ImportError:
    print("FAIL: Not in preview library")
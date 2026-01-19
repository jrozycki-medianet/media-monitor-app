import json

try:
    with open("service_account_key.json", "r") as f:
        key_data = json.load(f)
        
    print("\n--- COPY EVERYTHING BELOW THIS LINE ---\n")
    print("[GCP_CREDENTIALS]")
    for key, value in key_data.items():
        if "\n" in str(value):
            # Triple quotes for multi-line keys
            print(f'{key} = """{value}"""')
        else:
            # Normal quotes for single lines
            print(f'{key} = "{value}"')
    print("\n---------------------------------------\n")
    
except FileNotFoundError:
    print("Error: service_account_key.json not found.")
import secrets
import base64

# Generate a secure random key for Flask sessions
flask_key = secrets.token_hex(32)
# Generate a secure encryption key
encryption_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')

print(f"SECRET_KEY={flask_key}")
print(f"ENCRYPTION_KEY={encryption_key}")
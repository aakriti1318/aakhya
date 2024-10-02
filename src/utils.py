# src/utils.py 
import re

def sanitize_table_name(name):
    # Remove any character that isn't a letter, number, or underscore
    sanitized = re.sub(r'[^\w]+', '_', name)
    # Ensure the name starts with a letter or underscore
    if not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = f"_{sanitized}"
    # Truncate to 1024 characters (BigQuery's limit)
    return sanitized[:1024]
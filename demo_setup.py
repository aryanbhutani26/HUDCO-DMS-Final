"""
Demo setup script to create sample documents for testing
"""
from pathlib import Path
from document_manager import DocumentManager
from auth import AuthManager

def create_sample_documents():
    """Create some sample documents for demo purposes"""
    
    # Ensure sample content directory exists
    sample_dir = Path("data/sample/content")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample documents if they don't exist
    sample_docs = {
        "python_basics.txt": """
# Python Programming Basics

Python is a high-level, interpreted programming language known for its simplicity and readability. 
It was created by Guido van Rossum and first released in 1991.

## Key Features:
- Easy to learn and use
- Interpreted language
- Object-oriented programming support
- Large standard library
- Cross-platform compatibility

## Common Uses:
- Web development
- Data analysis and visualization
- Machine learning and AI
- Automation and scripting
- Scientific computing

## Getting Started:
To start programming in Python, you need to install Python from python.org and use a text editor or IDE.
        """,
        
        "python_functions.txt": """
# Python Functions

Functions in Python are reusable blocks of code that perform specific tasks.

## Defining Functions:
```python
def function_name(parameters):
    \"\"\"Function docstring\"\"\"
    # Function body
    return value
```

## Example:
```python
def greet(name):
    \"\"\"Greets a person with their name\"\"\"
    return f"Hello, {name}!"

# Calling the function
message = greet("Alice")
print(message)  # Output: Hello, Alice!
```

## Function Parameters:
- Positional parameters
- Keyword parameters
- Default parameters
- Variable-length arguments (*args, **kwargs)

## Return Values:
Functions can return single values, multiple values (as tuples), or None.
        """,
        
        "data_structures.txt": """
# Python Data Structures

Python provides several built-in data structures to store and organize data.

## Lists:
Ordered, mutable collections of items.
```python
fruits = ["apple", "banana", "orange"]
fruits.append("grape")
```

## Dictionaries:
Key-value pairs for mapping relationships.
```python
student = {
    "name": "John",
    "age": 20,
    "grade": "A"
}
```

## Tuples:
Ordered, immutable collections.
```python
coordinates = (10, 20)
```

## Sets:
Unordered collections of unique items.
```python
unique_numbers = {1, 2, 3, 4, 5}
```

Each data structure has specific use cases and methods for manipulation.
        """
    }
    
    for filename, content in sample_docs.items():
        file_path = sample_dir / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
            print(f"Created sample document: {filename}")

def setup_demo_users():
    """Setup demo users"""
    auth_manager = AuthManager()
    
    # Create a demo regular user
    if auth_manager.register_user("demo_user", "password123", "user"):
        print("Created demo user: demo_user / password123")
    else:
        print("Demo user already exists")

if __name__ == "__main__":
    print("Setting up HUDCO Document Management Demo...")
    create_sample_documents()
    setup_demo_users()
    print("Demo setup complete!")
    print("\nLogin credentials:")
    print("Admin: admin / admin123")
    print("User: demo_user / password123")
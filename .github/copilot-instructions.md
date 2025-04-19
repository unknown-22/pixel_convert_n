# Python Coding Rules

- Write code compliant with **Python 3.12**
- Use **Ruff** for code checking and formatting
- Actively use type hints (following PEP 585, preferring built-in types over the `typing` module)
- Utilize pattern matching (Python 3.10+) for code structure
- Format strings using f-strings
- Define data structures using dataclasses or named tuples
- Properly use Context Managers (`with` statements)
- Leverage asynchronous programming (async/await)
- Follow security best practices (check dependencies for vulnerabilities)
- Use `pip-tools` or `Poetry` for virtual environments and package management
- Implement appropriate unit tests (pytest recommended) with test coverage
- Write documentation strings (docstrings) in Google style or numpydoc format
- Use `python-dotenv` or `environs` for environment variable management
- Adhere to PEP 8 style guidelines (automatically checked with Ruff)
- Use the new union syntax (`int | str`)
- Write self-documenting code (reflect intent in variable and function names)
- Implement proper exception handling and custom exception classes
- Standardize project structure (src/ layout)
- Use pyproject.toml for configuration management

# Gradio Rules

- Use **Gradio 5.x** for modern interface development
- Implement interfaces with **Gradio Blocks API** for advanced customization
- Follow responsive design principles for multi-device compatibility
- Leverage client-side processing to improve performance
- Implement proper error handling and validation
- Use WebSocket connections for real-time applications
- Apply proper state management for user sessions
- Integrate with Hugging Face Spaces for seamless deployment

# Dependencies
- Pillow
- scikit‑image
- scikit‑learn
- gradio

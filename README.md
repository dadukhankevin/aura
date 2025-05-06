# Aura

You know the feeling... You're sitting there, vibe-coding away, and suddenly it hits you: "I have no idea how my codebase works anymore".

You've tried telling `claude-3.7-sonnet-new-final-smartest-max` to not "change too much" and to "please please only do what I tell you or else they will fire me." 

Only to see thousands of inserted lines.

Aura is made for you. It's what "English as a programming language" *should* be. Reproducable, high level, precice. 

Oh and highly experimental...

## What is Aura Code?

Aura introduces LLMs as a compiler, rather than the over-caffinated intern.

Aura is a language, with syntax rules and Python norms. It lets you smoothly interpolate between natural language and Python code, in a way that ensures you know exactly what your code is doing, while keeping the benifits of natural language when you need it.

Quit yappin here's an example:

**Aura Code (`example.aura`)**
```py
imports {
  import numpy
}
desc BothKinds "When returning arrays, return a python list version and numpy version in a tuple (list, array)"

def --square_all_values(--input_array: @List):
  """A function that squares all values in an array returns @BothKinds"""
  --squared_result = all values in @input_array squared
  --combined = @BothKinds representation of squared_result
  return combined
``` 

**Compiled Python Output (example_compiled.py)**
```python
import numpy
from typing import List, Dict, Any, Optional, TypeAlias

# Type Aliases from 'desc' blocks (for clarity)
BothKinds: TypeAlias = Any # When returning arrays, return a python list version and numpy version in a tuple (list, array)

# --- Block: aura_function:square_all_values ---


def square_all_values(input_array: List):
    """A function that squares all values in an array returns BothKinds"""
    # Assuming input_array is a list or similar iterable of numbers
    # LLM generates the logic based on the prompt "all values in input_array squared"
    squared_list = [x**2 for x in input_array]
    # LLM generates the logic based on the prompt "BothKinds representation of squared_result"
    # and the context provided for @BothKinds
    squared_result = numpy.array(squared_list)
    combined = (squared_list, squared_result)
    return combined

```

The above example includes everything you need to know about Aura *so far*.
You'll notice a few differences compared to normal Python. The first difference is that natural English is included in the code at any point. But it isn't only a prompt, and there are some strict syntax rules, that will make this work.

1.  **The Double Dash (`--`)**: Indicates variables or definitions (`--my_var`, `def --my_func`) that you want literally in the compiled Python code. The compiler ensures these names exist in the output, throwing an error if the LLM forgets them.
2.  **The At Symbol (`@`)**: References types (`@List`), other Aura classes/functions (`@MyClass`, `@my_helper_func`), or imported Aura modules/items (`@my_module.some_item`). The compiler provides the definition or description of the referenced item as context to the LLM.
3.  **The `desc` Keyword**: Defines a semantic type or rule description (e.g., `desc MyType "Description here"`). This context is provided to the LLM when `@MyType` is referenced.
4.  **Docstrings (`"""..."""`)**: Required after `class` and `def` lines. They serve as standard Python docstrings and also provide crucial high-level context to the LLM compiler.
5.  **Natural Language Prompts**: Lines within Aura function/method bodies that are not comments or strict Aura syntax are treated as instructions for the LLM compiler to generate the corresponding Python code.
6.  **Comments (`#`)**: Standard Python comments are ignored by the LLM compiler and are not included in the prompts.

### Imports in Aura

The `imports { ... }` block at the top of the file handles both standard Python imports and imports of other Aura modules.

```py
imports {
  # Standard Python Import (passed directly to output)
  import os 
  from typing import Dict
  
  # Import test_utils.aura, access its content via @utils
  import aura test_utils as utils 
  
  # Import test_base.aura, access its content via @base
  import aura test_base as base 
}
```

**How Aura Imports Compile:**

*   Standard Python imports (`import os`) are copied directly to the generated Python file.
*   Aura imports (`import aura test_utils as utils`) trigger the compilation of the dependency (`test_utils.aura` -> `compiled_python/test_utils.py`) and add a corresponding Python import to the current file:
    ```python
    # Assuming output directory is 'compiled_python'
    import compiled_python.test_utils as utils
    import compiled_python.test_base as base
    ```
*   You can then reference items from the imported module using the alias: `@utils.some_function`, `@base.BaseClass`.
*   LLM-generated code blocks will not include any import statements. If your code requires additional modules, add them to the `imports` block at the top of the Aura file.

## Getting Started

### Clone & Setup

```bash
git clone https://github.com/dadukhankevin/aura aura-project
cd aura-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create environment file for API key
cp .env.example .env
# --> Add your AURA_LLM_API_KEY to the new .env file <--- 
```

### API Key Setup (.env)

The compiler uses an LLM (currently via Groq, but configurable) to generate Python code for Aura blocks. You need to provide your API key.

1.  **Copy the example:** `cp .env.example .env`
2.  **Edit `.env`:** Open the newly created `.env` file and replace `"YOUR_AURA_LLM_API_KEY_HERE"` with your actual LLM API key. You can optionally set the `AURA_LLM_BASE_URL` if you are using a different endpoint.

The `.env` file is included in `.gitignore` to prevent accidentally committing your key.

### Using the Compiler

Run the compiler script directly using Python from your project root:

**Compile a Single File:**
```bash
# python <compiler_script> <input_aura_file> <output_python_file>
python compiler/compiler.py path/to/module.aura path/to/output.py
```

**Compile an Entire Directory:**
```bash
# python <compiler_script> <input_aura_directory> <output_python_directory>
python compiler/compiler.py aura_src_dir compiled_python
```

The compiler will automatically detect and compile imported `.aura` modules first.

## Why Aura?

Aura streamlines the creation of Python classes and data models by combining:

1. **High-level DSL Syntax** for concise declarations.
2. **Static Validation** to catch undeclared variables or missing references.
3. **AI-Assisted Code Generation** to fill in method bodies and reduce boilerplate.

Use Aura when you want precise control over your project *and* english as code.
## Contributing

Contributions welcome! Please fork the repo and open PRs for:

- New syntax features
- Improved error messages
- Enhanced LLM compiling strategies
- Performance optimizations

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
Please attribute (Daniel Losey) (:

## Example

Compile with:

```bash
# Compile the main file (dependencies compiled automatically)
python compiler/compiler.py test_user.aura compiled_test/test_user.py
```
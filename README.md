**Updating README syntax**

The user wants a README similar to the first one they provided, but with an updated section for syntax rules. They've shared the original plus new syntax highlights, such as using "desc def", "desc class", braces, unified definitions, and reference tokens. I need to keep the structure of the original README but update the "Syntax Highlights" and code examples to reflect these changes. I'll carefully revise the example to show the new "desc def" format.


Thought for 4 seconds


# Aura

![MIT License](https://img.shields.io/badge/license-MIT-blue) ![Experimental](https://img.shields.io/badge/status-experimental-orange)

> You know the feeling... You're sitting there, vibe‑coding away, and suddenly it hits you: "I have no idea how my codebase works anymore".

You've tried telling `claude-3.7-sonnet-new-final-smartest-max` to not "change too much" and to "please please only do what I tell you or else they will fire me."

Only to see thousands of inserted lines.

Aura is made for you. It's what "English as a programming language" *should* be. **Reproducible, high level, precise.**

> ⚠️ **Note:** Aura is highly experimental...

---

## Table of Contents

* [What is Aura Code?](#what-is-aura-code)
* [Examples](#examples)
* [Syntax Highlights](#syntax-highlights)
* [Imports in Aura](#imports-in-aura)
* [Getting Started](#getting-started)
* [Why Aura?](#why-aura)
* [Contributing](#contributing)
* [License](#license)
* [Example](#example)

---

## What is Aura Code?

Aura introduces LLMs as a compiler, rather than the over‑caffeinated intern.

### Aura is a domain‑specific language that uses English as a high‑level type system and prompt scaffold for LLM‑compiled Python code.

It lets you smoothly interpolate between natural language and Python code, in a way that ensures you know exactly what your code is doing, while keeping the benefits of natural language when you need it.

Quit yappin, here's an example:

**Aura Code (`example.aura`)**

```aura
imports {
  import numpy
}

desc BothKinds "Return a Python list and numpy array of squared values"

desc def square_all_values(--input_array: @List):
  "@BothKinds"
{
  --squared_list = [ x**2 for x in --input_array ]
  --array        = numpy.array(--squared_list)
  this returns (--squared_list, --array)
}
```

**Compiled Python Output (`example_compiled.py`)**

```python
import numpy
from typing import List, TypeAlias

# Type alias from desc
BothKinds: TypeAlias = tuple[list[int], numpy.ndarray]

def square_all_values(input_array: List[int]) -> BothKinds:
    """Return a Python list and numpy array of squared values"""
    squared_list = [x**2 for x in input_array]
    array = numpy.array(squared_list)
    return squared_list, array
```

> The above example includes everything you need to know about Aura *so far*.

---

## Examples

Explore these examples to see Aura in action:

### PyTorch LSTM Shakespeare

A character‑level LSTM model trained on Shakespearean text.

**Compile:**

```bash
mkdir -p compiled_python/pytorch_lstm_shakespeare
python compiler/compiler.py examples/pytorch_lstm_shakespeare/main.aura \
    compiled_python/pytorch_lstm_shakespeare/main.py
python compiled_python/pytorch_lstm_shakespeare/main.py 
```

### Flask Authentication Server (Conceptual)

Demonstrates setting up a basic Flask server with authentication routes.

**Compile:**

```bash
mkdir -p compiled_python/flask_auth_server
python compiler/compiler.py examples/flask_auth_server/app.aura \
    compiled_python/flask_auth_server/app.py 
```

### Genetic Hello World (Conceptual)

Illustrates a simple genetic algorithm evolving the string "Hello, World!".

**Compile:**

```bash
mkdir -p compiled_python/genetic_hello_world
python compiler/compiler.py examples/genetic_hello_world/main.aura \
    compiled_python/genetic_hello_world/main.py
```

---

## Syntax Highlights

Aura's new syntax uses explicit `desc def`/`desc class`, quoted descriptions, and braces for clear parsing:

| Feature                        | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| **`desc` keyword**             | Introduces a semantic declaration—types, functions, or classes. Always followed by a quoted description. |
| **`desc def` / `desc class`**  | Define functions or classes. Example:<br><br>desc def name(--args): "Docstring" {<br>  body<br>}<br>desc class Name(--args): "Docstring" {<br>  body<br>}<br>|
| **Braces (`{ … }`)**           | Delimit the block of English‑first instructions or hints. Eliminates Python‑style indentation ambiguity. |
| **Literal symbol (`--`)**      | Marks parameters and variables that **must** appear verbatim in the compiled Python (e.g., `--my_var`). |
| **Reference token (`@`)**      | Pulls in **only** the quoted description for a declared symbol (`@MyType`, `@my_func`, `@MyClass.method`). Used in bodies to surface high‑level intent to the LLM without re‑generating code for that symbol. |
| **Quoted descriptions**        | Immediately after `desc`, the quoted string becomes the docstring in Python. |
| **Natural Language Instructions** | Lines inside braces (that aren't `--` or `@` syntax) are prompts for the LLM to generate the corresponding Python logic. |
| **Comment tokens (`#`)**       | Standard Python comments. Ignored by the LLM compiler and not part of prompts. |

---

## Imports in Aura

The `imports { ... }` block at the top handles both real Python imports and Aura module imports:

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
# --> Add your AURA_LLM_API_KEY and optionally AURA_LLM_BASE_URL
```

### Using the Compiler

**Compile a single file:**

```bash
python compiler/compiler.py path/to/module.aura path/to/output.py
```

**Compile an entire directory:**

```bash
python compiler/compiler.py aura_src_dir compiled_python
```

The compiler will automatically detect and compile imported `.aura` modules first.

---

## Why Aura?

Aura streamlines Python development by combining:

1. **High‑level DSL syntax** for concise, intent‑driven code.
2. **Static validation** to catch missing `--` variables or undeclared references.
3. **AI‑assisted generation** to fill in method bodies from English instructions.

Use Aura when you want precise control *and* the expressiveness of English.

---

## Contributing

Contributions welcome! Please fork the repo and open PRs for:

* New syntax features
* Enhanced error messages
* Improved LLM compilation strategies
* Performance optimizations

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
Please attribute to Daniel Losey.

---

## Example

```bash
# Compile the main file (dependencies compiled automatically)
python compiler/compiler.py test_user.aura compiled_test/test_user.py
```

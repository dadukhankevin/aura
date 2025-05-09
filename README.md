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
   Squares everything in the list returns @BothKinds
}

# MAIN This will be carried over as python *exactly* as it is pure python!
def main():
    test_array = [1, 2, 3, 4, 5]
    list_result, np_result = square_all_values(test_array)
    print("Original array:", test_array)
    print("List result:", list_result)
    print("NumPy array result:", np_result)

if __name__ == '__main__':
    main() 
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

### Genetic Algorithm - Evolving a String

Illustrates a simple genetic algorithm evolving a target string (e.g., "AuraIsCool!").

**Compile & Run:**

```bash
# Ensure compiled_python directory exists
mkdir -p compiled_python

# Compile the Aura file
python compiler.py examples/genetic_algorithm_string.aura compiled_python/genetic_algorithm_string.py

# Run the compiled Python script
python compiled_python/genetic_algorithm_string.py
```

---

## Syntax Highlights

Aura's new syntax uses explicit `desc def`/`desc class`, quoted descriptions, and braces for clear parsing:

| Feature                        | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| **`desc` keyword**             | Introduces a semantic declaration—types, functions, or classes. Always followed by a quoted description. |
| **`desc def` / `desc class`**  | Define functions or classes. Example:<br><br>desc def name(--args): "Docstring" {<br>  body<br>}<br>desc class Name(--args): "Docstring" {<br>  body<br>}<br>|
| **Braces (`{ … }`)**           | Delimit the block of English‑first instructions or hints. Eliminates Python‑style indentation ambiguity. |
| **Literal symbol (`--`)**      | Marks parameters and variables that **must** appear verbatim in the compiled Python (e.g., `--my_var`). The compiler validates their presence in the generated code. |
| **Reference token (`@`)**      | Pulls in **only** the quoted description for a declared symbol (`@MyType`, `@my_func`, `@MyClass.method`). Used in bodies to surface high‑level intent to the LLM without re‑generating code for that symbol. |
| **Quoted descriptions**        | Immediately after `desc`, the quoted string becomes the docstring in Python. |
| **Natural Language Instructions** | Lines inside braces (that aren't `--` or `@` syntax) are prompts for the LLM to generate the corresponding Python logic. |
| **Comment tokens (`#`)**       | Standard Python comments. Ignored by the LLM compiler and not part of prompts. |
| **`# MAIN` section**           | Code following a `# MAIN` line at the end of an Aura file is copied verbatim to the compiled Python output. Ideal for entry points and pure Python logic. |

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

## Compiler Feedback: Errors & Warnings

Aura's compiler provides rich feedback, including issues reported directly by the LLM during compilation. This helps you iterate on your Aura code effectively. Errors and warnings are printed to the console (`stderr`).

### Fatal Errors (Halts Compilation for the File)

If the LLM or the compiler encounters a critical issue, compilation for that specific file will stop, and an error message will be displayed:

*   **`AuraAmbiguityError`**: Reported by the LLM if an Aura block is too vague or unclear to translate confidently.
    *   *Fix*: Refine the Aura block's description or instructions to be more specific.
*   **`AuraImportNeededError`**: Reported by the LLM if a block requires a Python module that isn't listed in the `imports {...}` section (e.g., for GUI, networking). The LLM will often suggest a module or type of module.
    *   *Fix*: Add the required import to your Aura file's `imports {}` block.
*   **`AuraUncompilableError`**: Reported by the LLM if a block is fundamentally uncompilable due to logical contradictions, requests violating core Python principles, or features beyond its design. The LLM will provide a reason.
    *   *Fix*: Re-evaluate the logic or approach in your Aura block based on the LLM's explanation.
*   **`AuraCompilationError` (Compiler Validation)**: Raised by the Aura compiler itself if a validation fails *after* LLM generation. For example:
    *   **Missing `--parameter`**: If a variable declared with `--` in an Aura signature (e.g., `--my_var`) is not found in the LLM's generated Python code for that block.
    *   *Fix*: Ensure your Aura block's instructions lead the LLM to use all declared `--` parameters, or adjust the signature.
*   **LLM Processing Errors**: Generic errors from the LLM if it encounters an internal issue.

### Warnings (Non-Fatal)

The LLM can also issue warnings for non-critical issues. Compilation will proceed, but you should review these:

*   **Structure**: Warnings appear as `⚠️ (WarningTypeName): Explanation...` in the console.
*   **Examples**: `PotentialPerformanceIssue`, `MinorDeviation`, `BestPracticeSuggestion`, `PartialImplementation` (if the LLM could only partially fulfill a request).
*   **File Status**: If a file has warnings, its final processing message will be marked with `⚠️ Processed ...` instead of `✅ Processed ...`.

**Tips for Effective Aura:**

*   **Be Specific**: Clear, unambiguous instructions in Aura blocks help avoid `AuraAmbiguityError`.
*   **Declare Imports**: Ensure all necessary Python modules are in the `imports {}` block to prevent `AuraImportNeededError`.
*   **Use `--params` Wisely**: The `--` syntax guarantees parameter presence, but ensure your English prompts guide the LLM to use them.

---

## Why Aura?

Aura streamlines Python development by combining:

1.  **High‑level DSL syntax** for concise, intent‑driven code.
2.  **Robust compiler feedback**, including LLM-reported errors/warnings and static validations (like ensuring `--` variables are used or references are declared). This creates a more predictable and reliable interaction with AI-assisted code generation.
3.  **AI‑assisted generation** to fill in method bodies from English instructions.

Use Aura when you want precise control, the expressiveness of English, and a smarter, more communicative compilation process.

---

## Contributing

Contributions welcome! Please fork the repo and open PRs for:

* New syntax features
* Enhanced error messages or compiler validations
* Improved LLM compilation strategies and prompt engineering
* Performance optimizations
* More examples!

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
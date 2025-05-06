#!/usr/bin/env python3
"""
Aura-to-Python compiler using a block-based approach and LLM generation.
"""
import os
import sys
import glob
from typing import List, Dict, Any, Optional, TypeAlias, Set
from openai import OpenAI
import json
import re
import textwrap
from dotenv import load_dotenv

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

# Use the new block parser and prompter
from compiler.block_parser import parse_blocks, ParsedAuraFile, Block, AuraImport
from compiler.block_prompter import parse_desc_blocks, generate_prompts

# --- LLM Configuration ---
DEFAULT_MODEL = 'meta-llama/llama-4-maverick-17b-128e-instruct'
# Load from environment variables
GROQ_API_KEY = os.getenv("AURA_LLM_API_KEY")
GROQ_BASE_URL = os.getenv("AURA_LLM_BASE_URL", "https://api.groq.com/openai/v1") # Default if not set

if not GROQ_API_KEY:
    print("Error: AURA_LLM_API_KEY not found in environment variables. Please create a .env file.", file=sys.stderr)
    sys.exit(1) # Exit if key is missing
    
client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

# Global cache to avoid re-parsing/loading context for the same module multiple times in one run
context_cache: Dict[str, Dict[str, Any]] = {}

def load_imported_context(module_name: str, source_dir: str) -> Optional[Dict[str, Any]]:
    """Loads descriptions and block names from an imported Aura module.
    
    Looks for module_name.aura in source_dir.
    Returns a dictionary containing 'descs' and 'block_names'.
    Uses a cache to avoid redundant parsing.
    """
    if module_name in context_cache:
        return context_cache[module_name]

    aura_filepath = os.path.join(source_dir, f"{module_name}.aura")
    if not os.path.exists(aura_filepath):
        print(f"Warning: Imported Aura file not found: {aura_filepath}", file=sys.stderr)
        return None # Cannot load context

    print(f"    -> Loading context from imported file: {aura_filepath}")
    try:
        # We only need descs and block names for context checking for now
        imported_descs = parse_desc_blocks(aura_filepath)
        # We need to parse blocks to get class/function names
        parsed_imported_file = parse_blocks(aura_filepath)
        imported_block_names = {b.name for b in parsed_imported_file.blocks if b.block_type.startswith('aura_')}
        
        context = {
            "descs": imported_descs,
            "block_names": imported_block_names
        }
        context_cache[module_name] = context
        return context
    except Exception as e:
        print(f"Error loading context from {aura_filepath}: {e}", file=sys.stderr)
        return None

# --- Helper Functions ---

def sanitize_llm_code(raw_code: str) -> str:
    """Remove markdown fences and leading/trailing whitespace."""
    # Remove markdown code fences (```python ... ``` or ``` ... ```)
    cleaned = re.sub(r'^```(?:python)?\n', '', raw_code, flags=re.MULTILINE)
    cleaned = re.sub(r'\n```$', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()

def generate_code_snippet(prompt: str, context_type: str) -> Optional[str]:
    """Send prompt to LLM and get the generated code snippet using the new OpenAI client."""
    try:
        system_prompt = (
            "You are a Python code generation assistant. "
            "You must NEVER use any module, function, or import that is not explicitly listed in the provided imports. "
            "If you need a module that is not imported, output a JSON object, Or if there's anything else wrong with the instructions the user gave, please return an error like this. Make the error simple, readable, understandable, and clear.: {\"code\": null, \"error\": \"Missing import: <module> \"}. "
            "If all requirements are satisfied, output a JSON object: {\"code\": <python_code_string>, \"error\": null}. "
            "Try your level best to produce the code without needing any modules that the user didn't already import. Only use the error functionality in the JSON if you absolutely need to and there's literally no other way."
            "Never output anything except valid JSON with both 'code' and 'error' fields."
        )
        chat_completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        response = chat_completion.choices[0].message.content
        # Always expect JSON with both 'code' and 'error' fields
        try:
            if response is None: # Handle potential None response
                print(f"    -> LLM returned None response for {context_type}", file=sys.stderr)
                return None
            parsed = json.loads(response) # Now response is guaranteed to be str
            if parsed.get("error"):
                print(f"    -> LLM error for {context_type}: {parsed['error']}", file=sys.stderr)
                return None
            if parsed.get("code"):
                return parsed["code"]
            print(f"    -> LLM returned JSON without usable 'code' for {context_type}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"    -> LLM did not return valid JSON for {context_type}: {response}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"    -> LLM API call failed for {context_type}: {e}", file=sys.stderr)
        return None

def format_python_param(param) -> str:
    """Dummy function, replace if parameter parsing is added back."""
    return param # Placeholder

def check_reference_context(aura_blocks: List[Block], 
                            desc_map: Dict[str, str], 
                            aura_imports: List[AuraImport], 
                            imported_context: Dict[str, Dict[str, Any]]):
    """Check that every @reference has context (local or imported)."""
    errors = []
    # Local names: descriptions + block names
    local_names = set(desc_map.keys())
    local_names.update(b.name for b in aura_blocks if b.block_type.startswith('aura_'))
    
    # Aliases available via imports
    import_aliases = {imp.alias for imp in aura_imports}

    for block in aura_blocks:
        if not block.block_type.startswith('aura_'):
            continue
        # Analyze references like @alias.item or @item
        # Need to adjust the regex in Block.analyze to capture these properly first
        # For now, assume Block.references contains full strings like "alias.item" or "item"
        # TODO: Update Block.analyze to correctly parse qualified references
        text = '\n'.join(block.lines)
        references = re.findall(r'@(\w+(?:\.\w+)*)', text) # Updated regex for qualified names
        
        for ref in references:
            # Check for qualified reference (e.g., @alias.item)
            if '.' in ref:
                alias, item_name = ref.split('.', 1)
                if alias not in import_aliases:
                    errors.append(f"Reference '@{ref}' in block '{block.name}' uses unknown import alias '{alias}'.")
                    continue # Skip to next ref

                # Check if item exists in the imported context
                imp_ctx = imported_context.get(alias)
                if not imp_ctx:
                    # Context for alias wasn't loaded (error printed earlier)
                    errors.append(f"Internal Warning: Missing context for known alias '{alias}' when checking '@{ref}' in block '{block.name}'.")
                    continue # Skip to next ref

                # --- This part executes ONLY if imp_ctx was found ---
                imp_descs = imp_ctx.get("descs", {})
                imp_blocks = imp_ctx.get("block_names", set())
                if item_name not in imp_descs and item_name not in imp_blocks:
                    errors.append(f"Reference '@{ref}' in block '{block.name}' unresolved: '{item_name}' not found in imported module '{alias}'.")
                # ------------------------------------------------------

            # Check for unqualified reference (e.g., @Item)
            elif ref not in local_names:
                errors.append(f"Reference '@{ref}' in block '{block.name}' has no local context (not declared, described, or defined). Check imports for a qualified name like @module.{ref}")
                
    return errors

def check_var_defined_in_code(block, generated_code):
    """Check that every --var declared in Aura is defined in the generated Python code."""
    errors = []
    # Only check for aura blocks
    if not block.block_type.startswith('aura_'):
        return errors
    # Only check for --var that are not the block's own name
    declared_vars = [d for d in block.declarations if d != block.name]
    for var in declared_vars:
        # Check for assignment, function arg, or class attribute
        # Assignment: var = ...
        # Function arg: def ...(var, ...)
        # Class attribute: self.var = ...
        pattern = re.compile(rf'(\b{var}\b\s*=|def [^(]*\([^)]*\b{var}\b|self\.{var}\s*=)')
        if not pattern.search(generated_code):
            errors.append(f"Variable '--{var}' declared in block '{block.name}' is not defined in generated Python code.")
    return errors

def compile_aura_to_python(
    input_filepath: str, 
    output_filepath: str, 
    compiled_in_this_run: Optional[Set[str]] = None
) -> bool:
    """Compile Aura using block parser, prompter, and LLM generation.
    Handles dependencies recursively.
    """
    print(f"Starting compilation for: {input_filepath}")
    
    # Initialize the set for the top-level call
    if compiled_in_this_run is None:
        compiled_in_this_run = set()

    # Avoid recompiling if already done in this run
    normalized_input_path = os.path.normpath(input_filepath)
    if normalized_input_path in compiled_in_this_run:
        print(f"  -> Already processed in this run, skipping: {input_filepath}")
        return True # Assume success as it was processed before
        
    # Add to the set *before* processing dependencies to handle cycles
    compiled_in_this_run.add(normalized_input_path)

    # Determine source and output directories
    source_dir = os.path.dirname(input_filepath) or '.' 
    output_dir = os.path.dirname(output_filepath) or '.'
    output_dir_name = os.path.basename(output_dir)
    if not output_dir_name:
         output_dir_name = "."
    
    # Clear context cache for each file compilation (might be too broad, consider finer control later)
    # If dependencies share sub-dependencies, this could be inefficient. 
    # Let's keep it simple for now.
    # context_cache.clear() # Keep cache across recursive calls for efficiency
    
    # 1. Parse Description Blocks
    print(f"  Parsing description blocks for {os.path.basename(input_filepath)}...")
    desc_map = parse_desc_blocks(input_filepath)
    # print(f"    Found {len(desc_map)} description blocks.") # Less verbose

    # 2. Parse Main Code Blocks AND Imports
    print(f"  Parsing main code blocks and imports for {os.path.basename(input_filepath)}...")
    try:
        parsed_data = parse_blocks(input_filepath)
    except Exception as e:
        print(f"Error: Failed to parse {input_filepath}: {e}", file=sys.stderr)
        return False
        
    aura_blocks = parsed_data.blocks
    standard_imports = parsed_data.standard_imports
    aura_imports = parsed_data.aura_imports
    
    if not aura_blocks:
        print(f"Warning: No main code blocks found in {input_filepath}. Proceeding with imports/descs only.", file=sys.stderr)
        if not standard_imports and not aura_imports and not desc_map:
             print(f"Error: File {input_filepath} is effectively empty. Cannot compile.", file=sys.stderr)
             return False

    # print(f"    Found {len(aura_blocks)} main code blocks.") # Less verbose
    # print(f"    Found {len(standard_imports)} standard imports.")
    # print(f"    Found {len(aura_imports)} Aura imports.")

    # 2.0 Compile Dependencies Recursively BEFORE loading context
    print(f"  Checking dependencies for {os.path.basename(input_filepath)}...")
    dependencies_ok = True
    for imp in aura_imports:
        dep_module_name = imp.module_name
        dep_aura_file = os.path.join(source_dir, f"{dep_module_name}.aura")
        dep_py_file = os.path.join(output_dir, f"{dep_module_name}.py")
        
        if not os.path.exists(dep_aura_file):
            print(f"Error: Dependency Aura file not found: {dep_aura_file}", file=sys.stderr)
            dependencies_ok = False
            continue # Move to next dependency

        # Check if dependency needs compiling
        needs_compiling = True
        normalized_dep_path = os.path.normpath(dep_aura_file)
        if normalized_dep_path in compiled_in_this_run:
             needs_compiling = False # Already processed or being processed
        elif os.path.exists(dep_py_file):
            try:
                # Basic timestamp check
                if os.path.getmtime(dep_py_file) >= os.path.getmtime(dep_aura_file):
                    print(f"    -> Dependency {dep_module_name}.py is up-to-date.")
                    needs_compiling = False
                    # Mark as compiled even if up-to-date to avoid re-checking in cycles
                    compiled_in_this_run.add(normalized_dep_path) 
            except OSError:
                pass # File might have disappeared, proceed with compiling
        
        if needs_compiling:
            print(f"    -> Compiling dependency: {dep_aura_file} -> {dep_py_file}")
            # Recursive call, passing the *same* set
            success = compile_aura_to_python(dep_aura_file, dep_py_file, compiled_in_this_run)
            if not success:
                print(f"Error: Failed to compile dependency '{dep_module_name}'. Stopping compilation for {os.path.basename(input_filepath)}.", file=sys.stderr)
                dependencies_ok = False
                break # Stop processing further dependencies for this file
            else:
                 # Mark as successfully compiled in this run is handled inside the recursive call
                 pass 
        # If it didn't need compiling OR was compiled successfully, continue

    if not dependencies_ok:
        return False # Stop if any dependency failed

    # 2.1 Load Context from Imports (Now dependencies should exist)
    print(f"  Loading context from imports for {os.path.basename(input_filepath)}...")
    imported_context: Dict[str, Dict[str, Any]] = {}
    context_load_failed = False
    for imp in aura_imports:
        # Context loading uses the source_dir to find the .aura file
        ctx = load_imported_context(imp.module_name, source_dir) 
        if ctx:
            imported_context[imp.alias] = ctx
        else:
            print(f"Error: Failed to load context for imported module '{imp.module_name}' (alias '{imp.alias}') even after attempting compilation.", file=sys.stderr)
            context_load_failed = True
    if context_load_failed:
        print(f"Compilation failed for {input_filepath} due to errors loading imported context.", file=sys.stderr)
        return False
    # print(f"    Loaded context for {len(imported_context)} imported modules.") # Less verbose

    # 2.5. Check @reference context (Now includes imported context)
    context_errors = check_reference_context(aura_blocks, desc_map, aura_imports, imported_context)
    if context_errors:
        print("\nERROR: Reference context errors detected:", file=sys.stderr)
        for err in context_errors:
            print("  ", err, file=sys.stderr)
        return False

    # 3. Generate Prompts 
    print(f"  Generating LLM prompts for {os.path.basename(input_filepath)}...")
    # Pass standard_imports so LLM will not re-import modules
    prompts_data = generate_prompts(aura_blocks, desc_map, aura_imports, imported_context, standard_imports)
    # print(f"    Generated {len(prompts_data)} prompts.") # Less verbose

    # 4. Generate Code Snippets using LLM
    print(f"\n--- Generating Code Snippets (LLM) for {os.path.basename(input_filepath)} ---")
    generated_snippets: Dict[str, str] = {}
    compilation_ok = True
    llm_prompts = [p for p in prompts_data if p['block_type'].startswith('aura_') or p['block_type'] == 'python_main']
    for p_data in llm_prompts:
        block_type = p_data['block_type']
        block_name = p_data['block_name']
        prompt = p_data['prompt']
        context_key = f"{block_type}:{block_name}"
        
        print(f"  Generating code for {context_key}...")
        snippet = generate_code_snippet(prompt, context_key)
        
        if snippet:
            cleaned_snippet = sanitize_llm_code(snippet)
            generated_snippets[context_key] = cleaned_snippet
            print(f"    -> Success.")
        else:
            print(f"    -> FAILED to generate snippet for {context_key}. Stopping compilation.", file=sys.stderr)
            compilation_ok = False
            break 
            
    if not compilation_ok:
        print(f"\nCompilation failed for {input_filepath} due to LLM generation errors.")
        return False

    # 4.5. Check --var defined in generated code
    var_errors = []
    for block in aura_blocks:
        if block.block_type.startswith('aura_'):
            context_key = f"{block.block_type}:{block.name}"
            code = generated_snippets.get(context_key, "")
            var_errors.extend(check_var_defined_in_code(block, code))
    if var_errors:
        print("\nERROR: Variable definition errors detected:", file=sys.stderr)
        for err in var_errors:
            print("  ", err, file=sys.stderr)
        return False
        
    # 5. Assemble Final Python Code
    print(f"\n--- Assembling Final Python Code for {os.path.basename(input_filepath)} ---")
    final_lines: List[str] = []
    
    # Add Standard Python Imports first
    if standard_imports:
        final_lines.append("# Standard Python Imports")
        final_lines.extend(standard_imports)
        final_lines.append("")
    
    # Add Compiled Aura Imports
    if aura_imports:
        final_lines.append("# Aura Imports (compiled modules)")
        for imp in aura_imports:
            if imp.module_name == imp.alias:
                final_lines.append(f"import {output_dir_name}.{imp.module_name}")
            else:
                final_lines.append(f"import {output_dir_name}.{imp.module_name} as {imp.alias}")
        final_lines.append("")

    # Add built-in imports AFTER user imports
    final_lines.append("# Built-in imports (added by compiler)")
    final_lines.append("import os")
    final_lines.append("import sys")
    final_lines.append("import re")
    final_lines.append("from typing import List, Dict, Any, Optional, TypeAlias")
    final_lines.append("")
    
    # Add Type Aliases
    if desc_map:
        final_lines.append("# Type Aliases from 'desc' blocks (for clarity)")
        for name, desc in desc_map.items():
             py_type = "Any" 
             desc_lower = desc.lower()
             if 'int' in desc_lower or 'integer' in desc_lower: py_type = "int"
             elif 'float' in desc_lower or 'number' in desc_lower: py_type = "float"
             elif 'str' in desc_lower or 'string' in desc_lower or 'text' in desc_lower: py_type = "str"
             elif 'list' in desc_lower or 'sequence' in desc_lower or 'array' in desc_lower: py_type = "List"
             elif 'dict' in desc_lower or 'mapping' in desc_lower or 'dictionary' in desc_lower: py_type = "Dict"
             elif 'bool' in desc_lower or 'boolean' in desc_lower: py_type = "bool"
             final_lines.append(f"{name}: TypeAlias = {py_type} # {desc}")
        final_lines.append("")

    # Assemble blocks in the order they appeared in the original file
    main_block_snippet = None
    
    print("  Assembling code blocks...")
    for block in aura_blocks:
        block_type = block.block_type
        block_name = block.name
        context_key = f"{block_type}:{block_name}"

        if block_type.startswith("python_") and block_type != "python_main":
            print(f"    -> Passing through Python block: {block_name}")
            final_lines.append(f"# --- Python Block: {context_key} ---")
            final_lines.extend(block.lines)
            final_lines.append("")
        elif block_type.startswith("aura_"):
            print(f"    -> Assembling Aura block: {block_name}")
            final_lines.append(f"# --- Aura Block: {context_key} ---")
            snippet = generated_snippets.get(context_key)
            if snippet:
                # Exclude any import statements from generated code
                for line in snippet.splitlines():
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        continue
                    final_lines.append(line)
            else:
                print(f"Warning: Missing generated snippet for Aura block {context_key}. Adding placeholder.", file=sys.stderr)
                final_lines.append(f"# Error: LLM generation failed for {context_key}")
                final_lines.append("pass")
            final_lines.append("")
        elif block_type == "python_main":
             print(f"    -> Processing main block: {block_name}")
             snippet = generated_snippets.get(context_key)
             if snippet:
                 main_block_snippet = snippet
             else:
                 print(f"Warning: Missing generated snippet for main block {context_key}. Using original source.", file=sys.stderr)
                 main_block_snippet = "\n".join(block.lines)
        else:
             print(f"Warning: Unexpected block type '{block_type}' for {block_name}. Including original source.", file=sys.stderr)
             final_lines.append(f"# --- Unknown Block Type: {context_key} ---")
             final_lines.extend(block.lines)
             final_lines.append("")

    # Add the processed main block at the very end
    if main_block_snippet:
        print("  Adding main block at the end...")
        final_lines.append("# --- Main Block ---")
        # Exclude any import statements from the main block
        for line in main_block_snippet.splitlines():
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            final_lines.append(line)
        final_lines.append("")

    # 6. Write final Python file
    print(f"  Writing output to: {output_filepath}")
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_filepath, 'w') as f:
            f.write("\n".join(final_lines))
    except IOError as e:
        print(f"Error writing output file {output_filepath}: {e}", file=sys.stderr)
        return False

    # 7. Format the output (optional)
    try:
        import subprocess
        print("  Formatting output with autopep8...")
        # Use less aggressive formatting
        subprocess.run(["autopep8", "--in-place", output_filepath], check=False, capture_output=True)
    except FileNotFoundError:
        print("    autopep8 not found, skipping formatting.")
    except Exception as e:
        print(f"    autopep8 formatting failed: {e}")
        # Optionally pass or handle specific subprocess errors
        pass

    print(f"\nCompilation successful: {input_filepath} -> {output_filepath}")
    return True


def compile_aura_directory(input_dir: str, output_dir: str) -> bool:
    """Compile all .aura files in a directory using recursive dependency handling."""
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return False
    os.makedirs(output_dir, exist_ok=True)
    
    aura_files = glob.glob(os.path.join(input_dir, '*.aura'))
    if not aura_files:
        print(f"No .aura files found in: {input_dir}")
        return True # No files to compile is not an error

    print(f"Found {len(aura_files)} .aura files in {input_dir}. Compiling with dependencies...")
    overall_success = True
    compiled_in_this_run: Set[str] = set() # Add type hint Set[str]
    
    # Iterate through all files, letting the recursive calls handle order
    for aura_file in aura_files:
        # Check if already compiled as a dependency during this run
        if os.path.normpath(aura_file) in compiled_in_this_run:
            continue
            
        base_name = os.path.splitext(os.path.basename(aura_file))[0]
        output_py = os.path.join(output_dir, base_name + '.py')
        print(f"\n--- Ensuring Compilation for {os.path.basename(aura_file)} --- ")
        # Call compile_aura_to_python, it will handle recursion and the set
        success = compile_aura_to_python(aura_file, output_py, compiled_in_this_run)
        if not success:
            print(f" ---> Compilation FAILED for top-level file: {os.path.basename(aura_file)}")
            overall_success = False
            # Optionally decide whether to stop the whole directory build on first failure
            # break 
        print("----------------------------------")
        
    if not overall_success:
         print("\nWarning: One or more files failed to compile during directory build.")
         
    return overall_success


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: compiler.py <input_file_or_dir> [output_file_or_dir]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    compiled_in_this_run_main: Set[str] = set() # Add type hint Set[str]
    
    if os.path.isdir(input_path):
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'compiled_python'
        if not os.path.isdir(output_path):
             print(f"Output directory '{output_path}' does not exist, creating it.")
             os.makedirs(output_path, exist_ok=True)
        ok = compile_aura_directory(input_path, output_path)
        sys.exit(0 if ok else 1)
    elif os.path.isfile(input_path):
        if not input_path.endswith('.aura'):
             print(f"Error: Input file must have .aura extension: {input_path}", file=sys.stderr)
             sys.exit(1)
             
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            output_dir_arg = os.path.dirname(output_path) 
            if output_dir_arg and not os.path.isdir(output_dir_arg):
                 print(f"Output directory '{output_dir_arg}' does not exist, creating it.")
                 os.makedirs(output_dir_arg, exist_ok=True)
            if not output_path.endswith('.py'):
                 print(f"Warning: Output file does not end with .py: {output_path}", file=sys.stderr)
        else:
            output_dir_arg = 'compiled_python'
            os.makedirs(output_dir_arg, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output_dir_arg, base_name + '.py')
            
        ok = compile_aura_to_python(input_path, output_path, compiled_in_this_run_main)
        sys.exit(0 if ok else 1)
    else:
        print(f"Error: Input path not found or is not a file/directory: {input_path}", file=sys.stderr)
        sys.exit(1)


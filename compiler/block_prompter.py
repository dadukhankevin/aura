#!/usr/bin/env python3
"""
Generates LLM prompts from Aura code blocks.
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming block_parser.py is in the same directory
from compiler.block_parser import Block, parse_blocks, AuraImport

def parse_desc_blocks(filepath: str) -> Dict[str, str]:
    """Parse the Aura file specifically for 'desc Name "Description"' or multiline desc blocks."""
    desc_map: Dict[str, str] = {}
    in_multiline_desc = False
    current_desc_name = None
    current_desc_lines: List[str] = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                stripped = line.strip()
                
                if in_multiline_desc:
                    current_desc_lines.append(stripped.replace('"""', '')) # Remove potential closing quotes
                    if stripped.endswith('"""'):
                        in_multiline_desc = False
                        if current_desc_name:
                            desc_map[current_desc_name] = " ".join(current_desc_lines).strip()
                        current_desc_name = None
                        current_desc_lines = []
                    continue

                if stripped.startswith('desc '):
                    # Match single-line desc: desc Name "Description"
                    match_single = re.match(r'desc\s+(\w+)\s+"(.*)"', stripped)
                    if match_single:
                        desc_map[match_single.group(1)] = match_single.group(2)
                        continue
                        
                    # Match start of multiline desc: desc Name """ Optional text...
                    match_multi_start = re.match(r'desc\s+(\w+)\s+"""(.*)', stripped)
                    if match_multi_start:
                        current_desc_name = match_multi_start.group(1)
                        first_line_content = match_multi_start.group(2).strip()
                        current_desc_lines = [first_line_content] if first_line_content else []
                        
                        # Check if it also ends on the same line
                        if stripped.endswith('"""') and len(stripped) > len(match_multi_start.group(0)) - 3:
                            # Ends on the same line
                             # Remove closing quotes only if they exist
                            if first_line_content.endswith('"""'):
                                first_line_content = first_line_content[:-3].strip()
                            current_desc_lines = [first_line_content] if first_line_content else []
                            desc_map[current_desc_name] = " ".join(current_desc_lines).strip()
                            current_desc_name = None
                            current_desc_lines = []
                        else:
                           in_multiline_desc = True
                        continue
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        return {}
        
    return desc_map

def clean_source(lines: List[str]) -> List[str]:
    """Remove --/@ prefixes, comment-only lines, import statements, and trailing comments."""
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip comment-only lines
        if stripped.startswith('#'):
            continue
        # Skip import statements - LLM should not generate imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue
        
        # Remove trailing comments (respecting potential strings)
        # Find the first # that is not inside quotes
        comment_start_index = -1
        in_single_quotes = False
        in_double_quotes = False
        for i, char in enumerate(line):
            if char == "'" and (i == 0 or line[i-1] != '\\'):
                in_single_quotes = not in_single_quotes
            elif char == '"' and (i == 0 or line[i-1] != '\\'):
                in_double_quotes = not in_double_quotes
            elif char == '#' and not in_single_quotes and not in_double_quotes:
                comment_start_index = i
                break
        
        line_without_comment = line[:comment_start_index].rstrip() if comment_start_index != -1 else line
        
        # Remove -- prefix
        processed_line = re.sub(r'--(\w+)', r'\1', line_without_comment)
        # Remove @ prefix
        processed_line = re.sub(r'@(\w+)', r'\1', processed_line)
        
        # Only add non-empty lines after processing
        if processed_line.strip():
            cleaned_lines.append(processed_line)
            
    return cleaned_lines

def find_context(reference: str, 
                 local_desc_map: Dict[str, str], 
                 local_blocks: List[Block], 
                 aura_imports: List[AuraImport], 
                 imported_context: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Find context for a reference (e.g., item or alias.item).
    
    Checks:
    1. Qualified reference (@alias.item) in imported context.
    2. Unqualified reference (@item) in local descs.
    3. Unqualified reference (@item) in local block definitions.
    """
    
    # 1. Check for qualified reference (e.g., @alias.item)
    if '.' in reference:
        alias, item_name = reference.split('.', 1)
        module_name = next((imp.module_name for imp in aura_imports if imp.alias == alias), None)
        if not module_name:
            return f"Context Error: Unknown import alias '{alias}' in reference '@{reference}'."
        
        imp_ctx = imported_context.get(alias)
        if not imp_ctx:
            return f"Context Error: Missing context for alias '{alias}' (module '{module_name}')."
        
        imp_descs = imp_ctx.get("descs", {})
        imp_blocks = imp_ctx.get("block_names", set())
        
        # Provide description uniformly
        if item_name in imp_descs:
             desc_value = imp_descs[item_name]
             return f"Imported Description for `{reference}` from module '{module_name}': {desc_value}"
        elif item_name in imp_blocks:
            # Reference to another code block in an imported module
            return f"Definition Reference for `{reference}` exists in imported module '{module_name}'."
        else:
            return f"Context Error: '{item_name}' not found in imported module '{module_name}' (alias '{alias}')."
            
    # 2. Check local desc_map (unqualified reference)
    if reference in local_desc_map:
        desc_value = local_desc_map[reference]
        # Provide description uniformly
        return f"Local Description for `{reference}`: {desc_value}"
        
    # 3. Search local block definitions (unqualified reference)
    for block in local_blocks:
        # Check locally defined Aura blocks (functions, classes)
        if block.block_type.startswith('aura_') and block.name == reference:
            docstring = None
            # Attempt to extract docstring
            if block.lines and len(block.lines) > 1:
                 second_line_stripped = block.lines[1].strip()
                 # Basic check for """docstring""" or "docstring"
                 if second_line_stripped.startswith('"""') and second_line_stripped.endswith('"""'):
                     docstring = second_line_stripped[3:-3].strip()
                 elif second_line_stripped.startswith('"') and second_line_stripped.endswith('"'): 
                     docstring = second_line_stripped[1:-1].strip()
            
            # Provide context based on docstring or code preview
            if docstring:
                return f"Local Definition Context for `{reference}` (from docstring): {docstring}"
            else:
                preview = "\n".join(block.lines[:3])
                return f"Local Definition Context for `{reference}` (from source preview):\n```\n{preview}\n```"
                
    # 4. Not found
    if '.' not in reference:
        return f"Context Error: Local context for `{reference}` not found (no description or definition)."
    else:
        # Should be unreachable if qualified refs are handled above
        return f"Internal Error: Context lookup failed for qualified reference `{reference}`."


def generate_prompts(blocks: List[Block],
                     desc_map: Dict[str, str],
                     aura_imports: List[AuraImport],  # Imported aura modules and aliases
                     imported_context: Dict[str, Dict[str, Any]],  # Context from imported modules
                     standard_imports: List[str]  # Standard Python imports declared in the Aura file
                    ) -> List[Dict[str, Any]]:
    """
    Generates LLM prompts for each Aura block and the main block.
    Uses local desc_map and loaded imported_context for context.
    """
    prompts = []
    all_blocks = blocks # Needed for local context lookup

    for block in blocks:
        # Skip standard Python blocks, only process Aura blocks and main
        if block.block_type.startswith("python_") and block.block_type != "python_main":
            continue
            
        prompt_lines = []
        prompt_lines.append(f"Generate the Python code for the following '{block.block_type}' named '{block.name}'.")
        prompt_lines.append("---------------------")

        # 2. Context for @references (use updated find_context)
        # References are now correctly parsed by Block.analyze (including qualified)
        if block.references:
            prompt_lines.append("Context for referenced items (@):")
            found_context_count = 0
            # Use the full reference string (e.g., item or alias.item)
            for ref in sorted(list(set(block.references))):
                # Pass all necessary context info to find_context
                context = find_context(ref, desc_map, all_blocks, aura_imports, imported_context) 
                if context:
                    # Check if context indicates an error
                    if "Context Error:" in context or "Internal Error:" in context:
                         print(f"Warning in block '{block.name}': {context}", file=sys.stderr)
                         # Optionally, decide whether to include error context in prompt or skip
                         prompt_lines.append(f"- Error resolving context for `@{ref}`: {context}")
                    else:
                        prompt_lines.append(f"- {context}")
                        found_context_count += 1
                else:
                     # This case means find_context returned None unexpectedly
                     print(f"Warning: Unexpected missing context for reference '@{ref}' in block '{block.name}'.", file=sys.stderr)
                     prompt_lines.append(f"- Internal Warning: Could not resolve context for `@{ref}`.")
                     
            if found_context_count == 0 and not any("Error" in line for line in prompt_lines[-len(block.references):]):
                 prompt_lines.append(" (No relevant context found or needed for references)")    
            prompt_lines.append("---------------------")
        
        # 3. Requirement for --declarations
        declarations_to_enforce = []
        if block.block_type.startswith("aura_"):
            declarations_to_enforce = [d for d in block.declarations if d != block.name]
            
        if declarations_to_enforce:
            vars_list = ", ".join(sorted(list(set(declarations_to_enforce))))
            prompt_lines.append("Requirements for declared variables (--):")
            prompt_lines.append(f"- Ensure the following variables are defined and used appropriately in the generated code: `{vars_list}`")
            prompt_lines.append("---------------------")

        # 4. Cleaned Source Code
        prompt_lines.append("Original Aura Code Snippet:")
        if block.block_type.startswith("aura_"):
            cleaned_lines = clean_source(block.lines)
            prompt_lines.extend(cleaned_lines)
        else: # For main block, send original
            prompt_lines.extend(block.lines) 
            
        prompt_lines.append("---------------------")

        # 5. Simplified LLM Instructions
        prompt_lines.append("Core Task: Generate the Python code for the Aura snippet above.")
        prompt_lines.append("Context Rules:")
        prompt_lines.append(" - Use the 'Context for referenced items (@)' provided.")
        prompt_lines.append(" - If a Description for an `@identifier` clearly provides a direct value (like a URL string \"http...\" or a number like 100), REPLACE the `@identifier` in the code with that exact value.")
        prompt_lines.append(" - If a Description provides instructions, implement them.")
        prompt_lines.append("Requirements:")
        prompt_lines.append(" - Define all variables specified with `--` (e.g., `--my_var`).")
        prompt_lines.append(" - Use only the available imported modules: " + ", ".join(standard_imports) if standard_imports else "None available.")
        prompt_lines.append(" - Do NOT add any `import` statements.")
        # Add specific instruction for python_main
        if block.block_type == 'python_main':
            prompt_lines.append(" - For the 'python_main' block, generate ONLY the `if __name__ == \"__main__\":` statement and the call(s) within it (e.g., `main()`, `run_evolution()`).")
            prompt_lines.append(" - Do NOT define any functions (like `def main(): pass`) within this specific main block output.")
        prompt_lines.append("Output Format:")
        prompt_lines.append(" - Respond ONLY with a single JSON object: {\"code\": \"<generated_python_code>\"} on success.")
        prompt_lines.append(" - On failure (e.g., missing import, unable to use required value from description), respond ONLY with {\"error\": \"<error_description>\"}.")

        # Assemble final prompt
        final_prompt = "\n".join(prompt_lines)
        
        prompts.append({
            "block_type": block.block_type,
            "block_name": block.name,
            "prompt": final_prompt,
            "original_lines": block.lines 
        })

    return prompts


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <aura_file>")
        sys.exit(1)
    
    aura_filepath = sys.argv[1]
    source_dir = os.path.dirname(aura_filepath) or '.'
    # Dummy output dir for testing standalone prompter
    output_dir = '.' 
    
    print(f"--- Testing Prompter for: {aura_filepath} ---")
    
    # 1. Parse desc blocks
    desc_context_map = parse_desc_blocks(aura_filepath)
    print(f"--- Parsed {len(desc_context_map)} Description Blocks ---")
    for name, desc in desc_context_map.items():
        print(f"  {name}: {desc}")
    print("-----------------------------------------")

    # 2. Parse the main code blocks and imports
    try:
        parsed_data = parse_blocks(aura_filepath)
        aura_blocks = parsed_data.blocks
        aura_imports = parsed_data.aura_imports
        standard_imports = parsed_data.standard_imports
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)

    if not aura_blocks:
        print("No main code blocks parsed from the file.")
        # Don't exit, maybe just show imports?

    # 3. Load context from imports (using dummy cache for standalone test)
    imported_context: Dict[str, Dict[str, Any]] = {}
    temp_context_cache: Dict[str, Dict[str, Any]] = {}
    def load_ctx_standalone(module_name, src_dir, out_dir):
        # Simplified loader for testing, avoids global cache interference
        if module_name in temp_context_cache:
            return temp_context_cache[module_name]
        # Actual loading logic (copied/simplified from compiler.py)
        imp_aura_filepath = os.path.join(src_dir, f"{module_name}.aura")
        if not os.path.exists(imp_aura_filepath):
             print(f" [Standalone] Warning: Import not found: {imp_aura_filepath}")
             return None
        try:
             imp_descs = parse_desc_blocks(imp_aura_filepath)
             parsed_imp = parse_blocks(imp_aura_filepath)
             imp_blocks = {b.name for b in parsed_imp.blocks if b.block_type.startswith('aura_')}
             ctx = {"descs": imp_descs, "block_names": imp_blocks}
             temp_context_cache[module_name] = ctx
             return ctx
        except Exception as e:
            print(f" [Standalone] Error loading context {module_name}: {e}")
            return None
            
    print(f"--- Loading Context for {len(aura_imports)} Aura Imports ---")
    for imp in aura_imports:
        print(f"  Loading: {imp.module_name} (as {imp.alias})...")
        ctx = load_ctx_standalone(imp.module_name, source_dir, output_dir)
        if ctx:
            imported_context[imp.alias] = ctx
            print(f"    -> Loaded {len(ctx.get('descs',{}))} descs, {len(ctx.get('block_names',set()))} blocks.")
        else:
            print(f"    -> Failed to load context.")
    print("-----------------------------------------")

    # 4. Generate prompts using blocks, descs, imports, and loaded context
    generated_prompts = generate_prompts(aura_blocks, desc_context_map, aura_imports, imported_context, standard_imports)

    # 5. Print the generated prompts
    print(f"--- Generated {len(generated_prompts)} Prompts ---")
    for i, p_data in enumerate(generated_prompts):
        print(f"\n--- Prompt {i+1} ({p_data['block_type']}: {p_data['block_name']}) ---")
        print(p_data['prompt'])
        print("-----------------------------------------") 
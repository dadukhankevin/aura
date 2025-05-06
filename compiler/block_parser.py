import re
import sys # Added for stderr printing
from typing import List, Dict, Any, Tuple, NamedTuple, Optional # Added Optional

class Block:
    def __init__(self, block_type: str, name: str, lines: List[str]):
        self.block_type = block_type  # 'function', 'class', 'main'
        self.name = name              # e.g., function name, class name
        self.lines = lines            # raw source lines
        self.declarations: List[str] = []        # list of Aura declarations (--var)
        self.references: List[str] = []          # list of Aura @references

    def analyze(self):
        text = '\n'.join(self.lines)
        # Find variable declarations like "--variable: @Type" or "--variable : @Type"
        # Looks for --, word, optional space, colon, optional space, @, word
        self.declarations = re.findall(r'^\s*--(\w+)\s*:\s*@\w+', text, re.MULTILINE)
        
        # Find references like @item or @alias.item
        self.references = re.findall(r'@(\w+(?:\.\w+)*)', text) 

    def __repr__(self):
        return f"Block(type={self.block_type!r}, name={self.name!r}, lines={len(self.lines)})"


class AuraImport(NamedTuple):
    module_name: str
    alias: str

class ParsedAuraFile(NamedTuple):
    blocks: List[Block]
    standard_imports: List[str]
    aura_imports: List[AuraImport]


def parse_blocks(filepath: str) -> ParsedAuraFile:
    """
    Parses an Aura file into code blocks and import statements.

    Recognizes:
    - Standard code blocks (class, def, if __name__ == ...)
    - An optional `imports { ... }` block at the beginning.
    """
    blocks: List[Block] = []
    standard_imports: List[str] = []
    aura_imports: List[AuraImport] = []
    
    current: Optional[Block] = None  # Use Optional[Block]
    indent_level = None
    in_imports_block = False

    # Regex for Aura imports
    aura_import_pattern = re.compile(r'^\s*import\s+aura\s+(?P<module>\w+)(?:\s+as\s+(?P<alias>\w+))?\s*$')

    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.rstrip('\n')
            stripped = line.lstrip(' ')
            level = len(line) - len(stripped)

            # Handle imports block
            if stripped == 'imports {':
                if blocks or standard_imports or aura_imports: # Imports must be first
                     print(f"Warning: 'imports {{' found after other content in {filepath}. Ignoring.", file=sys.stderr)
                     # Treat as normal line if misplaced
                     pass 
                else:
                    in_imports_block = True
                    continue # Don't process this line further
            
            if in_imports_block:
                if stripped == '}':
                    in_imports_block = False
                    continue # Don't process this line further
                
                # Try parsing as Aura import
                match = aura_import_pattern.match(stripped)
                if match:
                    module = match.group('module')
                    alias = match.group('alias') or module # Use module name as alias if not provided
                    aura_imports.append(AuraImport(module_name=module, alias=alias))
                elif stripped and not stripped.startswith('#'): # Assume non-empty, non-comment lines are standard imports
                    standard_imports.append(stripped) # Keep original indentation/form
                # Skip blank lines and comments within imports block
                continue # Move to next line once processed within imports

            # --- Regular Block Parsing (outside imports block) ---

            # Detect new blocks (Aura definitions, standard Python def, or main)
            # Allow optional -- prefix for def/class
            m_fn = re.match(r'def\s+(?:--)?(?P<name>\w+)', stripped)
            m_cl = re.match(r'class\s+(?:--)?(?P<name>\w+)', stripped)
            m_main = stripped.startswith('if __name__ ==')

            is_new_block = m_fn or m_cl or m_main
            block_already_started = current is not None

            # Determine if the current line should end the previous block
            should_end_block = False
            if block_already_started and indent_level is not None:
                # End block if indentation decreases or if a new block starts at the same level
                if level < indent_level or (level == indent_level and is_new_block):
                    # Special case: Allow pass/docstring directly under def/class
                    # Don't end block immediately if it's just pass or a docstring
                    is_pass = stripped == 'pass'
                    is_docstring = stripped.startswith('"""') or stripped.startswith('"' )
                    if not (is_pass or is_docstring) or is_new_block:
                        should_end_block = True
                elif stripped != '' and level == indent_level and not is_new_block:
                    # Also end if we hit non-empty code at the same level that isn't a new block start
                    should_end_block = True
            
            # If we should end the current block, append it
            if should_end_block and current:
                blocks.append(current)
                current = None
                indent_level = None

            # Start a new block if detected
            if is_new_block and current is None: # Only start if not already in a block or after ending one
                if m_fn:
                    btype = 'function'
                    name = m_fn.group('name')
                    is_aura = line.lstrip(' ').startswith('def --')
                elif m_cl:
                    btype = 'class'
                    name = m_cl.group('name')
                    is_aura = line.lstrip(' ').startswith('class --')
                else: # m_main
                    btype = 'main'
                    name = '__main__'
                    is_aura = False # Main block is treated as standard
                    
                # Decide if this block should be processed by LLM (has -- prefix) or passed through
                block_type_prefix = "aura_" if is_aura else "python_"
                current = Block(f"{block_type_prefix}{btype}", name, [line])
                indent_level = level
            elif current and indent_level is not None: # Add line to the current block if appropriate
                # Add lines that are more indented or blank lines
                if stripped == '' or level > indent_level :
                    current.lines.append(line)
                # Handle lines at the same indent level (like decorators or parts of the block definition)
                elif level == indent_level and not is_new_block:
                     current.lines.append(line) 
            # else: line is outside any recognized block structure, skip

        # Append the last block if it exists
        if current:
            blocks.append(current)

    # Analyze each block for declarations and references
    for b in blocks:
        b.analyze()

    return ParsedAuraFile(blocks=blocks, standard_imports=standard_imports, aura_imports=aura_imports)


if __name__ == '__main__':
    # import sys # Already imported at top
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <aura_file>")
        sys.exit(1)
    parsed_data = parse_blocks(sys.argv[1])
    print("--- Parsed Data ---")
    print(f"Standard Imports: {len(parsed_data.standard_imports)}")
    for std_imp in parsed_data.standard_imports: print(f"  {std_imp}") # Use different loop var name
    print(f"Aura Imports:     {len(parsed_data.aura_imports)}")
    # Use different loop var name here to avoid confusing linter
    for aura_imp in parsed_data.aura_imports: 
        print(f"  Module: {aura_imp.module_name}, Alias: {aura_imp.alias}")
    print(f"Blocks:           {len(parsed_data.blocks)}")
    for b in parsed_data.blocks:
        print(f"  {b}")
        print(f"    Declarations: {b.declarations}")
        print(f"    References:   {b.references}")
        print("  Source preview:")
        for ln in b.lines[:3]: print(f"    {ln}")
        if len(b.lines) > 3: print("    ...")
        print() 
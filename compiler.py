# ─────────────────────────────────────────────────────────────────────────────
# Aura Toolchain – Enhanced Reference-Context Prototype (Fixed)
# Supports @-references, de-duplication, correct newline handling,
# and appends if __name__ == '__main__' entrypoint.
# Uses .env configuration for AURA_LLM_API_KEY and AURA_LLM_BASE_URL
# Drop into project root. Requires: python-dotenv, openai>=1.0
# ─────────────────────────────────────────────────────────────────────────────
"""
USAGE
-----
Create a .env file in your project root with:
  AURA_LLM_API_KEY="sk-..."
  AURA_LLM_BASE_URL="https://api.openai.com/v1"

Run the compiler:
  python aura_toolchain.py compile path/to/src_dir path/to/compiled_python
"""

from __future__ import annotations
import argparse
import dataclasses as dc
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
import difflib # For "did you mean" suggestions
from tqdm import tqdm # type: ignore # For progress bar

# Load .env configuration
from dotenv import load_dotenv
import openai
load_dotenv()



api_key = os.getenv("AURA_LLM_API_KEY")
if not api_key:
    raise RuntimeError("Missing AURA_LLM_API_KEY in environment (.env)")
base_url = os.getenv("AURA_LLM_BASE_URL", "https://api.openai.com/v1")

# Global verbose flag, to be set by main()
VERBOSE_LOGGING = False

def log_verbose(message):
    if VERBOSE_LOGGING:
        print(f"[VERBOSE] {message}", file=sys.stderr)

# ─────────────────────────────────────────────────────────────────────────────
# AST DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dc.dataclass
class ImportDecl:
    raw: str
    is_aura: bool
    alias: Optional[str]
    target: Optional[str]

@dc.dataclass
class TypeDecl:
    name: str
    doc: str

@dc.dataclass
class BlockDecl:
    kind: str   # "def" | "class"
    name: str
    signature: str
    doc: str
    body: str

@dc.dataclass
class AuraModule:
    name: str
    imports: List[ImportDecl]
    types: List[TypeDecl]
    blocks: List[BlockDecl]

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM COMPILER EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────
class AuraCompilationError(Exception):
    """Base class for Aura compilation errors."""
    pass

class AuraAmbiguityError(AuraCompilationError):
    """Raised when the LLM finds an Aura block too ambiguous to compile."""
    pass

class AuraImportNeededError(AuraCompilationError):
    """Raised when the LLM determines a necessary Python import is missing for a block."""
    def __init__(self, message, module_name=None):
        super().__init__(message)
        self.module_name = module_name

class AuraUncompilableError(AuraCompilationError):
    """Raised when the LLM deems an Aura block fundamentally uncompilable for a given reason."""
    pass

class AuraMissingDescriptionError(AuraCompilationError):
    """Raised when an Aura 'desc' declaration is missing its description."""
    def __init__(self, item_name: str, item_kind: str, filename: str):
        super().__init__(f"In file '{filename}', the {item_kind} '{item_name}' is missing a required description. All 'desc' declarations must be followed by a non-empty quoted string.")
        self.item_name = item_name
        self.item_kind = item_kind
        self.filename = filename

class AuraUnresolvedReferenceError(AuraCompilationError):
    """Raised when an @reference cannot be resolved."""
    def __init__(self, unresolved_ref: str, suggestions: List[str], block_name: str, filename: str):
        message = (
            f"In file '{filename}', within block '{block_name}', the @reference '{unresolved_ref}' could not be resolved. "
            f"Ensure it is defined with 'desc' in the current module or a correctly imported Aura module, and that its description is not empty."
        )
        if suggestions:
            message += f" Did you mean: { ', '.join(suggestions) }?"
        super().__init__(message)
        self.unresolved_ref = unresolved_ref
        self.suggestions = suggestions
        self.block_name = block_name
        self.filename = filename

# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────
class AuraParser:
    IMPORTS_RE = re.compile(r"^imports\s*{(?P<body>.*?)^}\s*", re.S | re.M)
    IMPORT_LINE_RE = re.compile(r"^\s*(?P<line>[^#\n]+)", re.M)
    TYPE_RE = re.compile(r"^desc\s+(?P<name>\w+)\s+\"(?P<doc>.*?)\"", re.M)
    BLOCK_HEADER_RE = re.compile(
        r"^desc\s+(?P<kind>def|class)\s+(?P<name>\w+)\((?P<sig>[^)]*)\):\s*\"(?P<doc>.*?)\"",
        re.M
    )

    def parse(self, text: str, filename: str) -> AuraModule:
        imports, types, blocks = [], [], []
        # Imports
        m_imp = self.IMPORTS_RE.search(text)
        if m_imp:
            for m_line in self.IMPORT_LINE_RE.finditer(m_imp.group('body')):
                line = m_line.group('line').strip()
                if not line:
                    continue
                is_aura = line.startswith('import aura')
                alias = None
                target = None
                if is_aura:
                    parts = line.split()
                    target = parts[2]
                    if 'as' in parts:
                        alias = parts[-1]
                imports.append(ImportDecl(raw=line, is_aura=is_aura, alias=alias, target=target))
        # Types
        for m in self.TYPE_RE.finditer(text):
            doc = m.group('doc').strip()
            if not doc:
                raise AuraMissingDescriptionError(item_name=m.group('name'), item_kind='type', filename=filename)
            types.append(TypeDecl(name=m.group('name'), doc=doc))
        # Blocks
        for m in self.BLOCK_HEADER_RE.finditer(text):
            doc = m.group('doc').strip()
            kind = m.group('kind')
            name = m.group('name')
            if not doc:
                raise AuraMissingDescriptionError(item_name=name, item_kind=kind, filename=filename)
            
            body_text, _ = self._extract_braces(text, m.end())
            blocks.append(BlockDecl(
                kind=kind, name=name,
                signature=m.group('sig'), doc=doc, body=body_text.strip()
            ))
        return AuraModule(Path(filename).stem, imports, types, blocks)

    def _extract_braces(self, text: str, start: int) -> tuple[str, int]:
        depth = 0; buf: List[str] = []
        i = start
        while i < len(text):
            ch = text[i]
            if ch == '{':
                depth += 1
                if depth == 1:
                    i += 1
                    continue
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return ''.join(buf), i+1
            buf.append(ch); i += 1
        raise SyntaxError("Unmatched '{'")

# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPTER
# ─────────────────────────────────────────────────────────────────────────────
LLM_SYSTEM = (
    "You are Aura-Compiler, an assistant that compiles Aura DSL into valid, executable Python. "
    "Respond in strict JSON. The JSON object should have the following keys:"
    "  'code': (string | null) The complete Python implementation, or null if reporting a fatal error."
    "  'error': (object | null) Null if successful, or an error object if compilation fails."
    "  'warnings': (array | null) An optional array of warning objects. Warnings are non-fatal."
    "Do NOT output any Aura syntax or DSL; include only real Python code in 'code'."
    "When considering imports, refer *only* to the '# Available Python imports:' section provided in the user prompt. Do not assume any other modules are available."
    "Fatal Errors (set 'code' to null and provide 'error' object):"
    "1. AmbiguityError: If the Aura block is too ambiguous, and you are unable to generate what the user is requesting. 'error': {'type': 'AmbiguityError', 'message': 'Reason...'}."
    "2. ImportNeededError: If a Python import is missing because the user wants to do something that isn't currently possible with the Python built-in functions AND the explicitly listed '# Available Python imports:'. "
    "   This often applies to tasks like GUI creation, networking, advanced scientific computing beyond listed imports, etc. "
    "   'error': {'type': 'ImportNeededError', 'message': 'Reason why [functionality] needs a module not listed...', 'details': {'module_name': 'suggested_module_or_type_of_module'}}. "
    "   Do NOT attempt to implement such functionality with placeholder/simplified code if a specific type of library is clearly missing. Do NOT add imports to 'code'."
    "3. UncompilableError: If the block is fundamentally uncompilable (e.g., logical inconsistencies, requests violating core Python principles). 'error': {'type': 'UncompilableError', 'message': 'Reason...'}."
    "4. LLMProcessingError: For any other internal processing error on your part. 'error': {'type': 'LLMProcessingError', 'message': 'Details...'}."
    "Warnings (generate 'code' for the parts you *can* implement, and optionally add to 'warnings' array):"
    "  Each warning in the array should be an object like: {'type': 'WarningTypeName', 'message': 'Explanation...'}. "
    "  Example warning types: 'PotentialPerformanceIssue', 'MinorDeviation', 'BestPracticeSuggestion', 'PartialImplementation'."
    "  If you issue a warning about partial implementation due to missing imports that you couldn't raise a fatal ImportNeededError for, be specific."
)

class LLMPrompter:
    def __init__(self, model: str = 'gpt-4.1'):
        if openai is None:
            raise RuntimeError('Please install openai package')
        self.model = model
        self.response_format = {'type': 'json_object'}
        self.client = openai.OpenAI(
            api_key=os.getenv("AURA_LLM_API_KEY"),
            base_url=os.getenv("AURA_LLM_BASE_URL", "https://api.openai.com/v1")
        )

    def compile_block(
        self,
        block: BlockDecl,
        local_mod: AuraModule,
        imported: Dict[str, AuraModule],
        filename: str
    ) -> tuple[str, list[dict[str, str]]]:
        log_verbose(f"Building prompt for block '{block.name}' in '{filename}'...")
        prompt = self._build_prompt(block, local_mod, imported, filename, block.name)
        # Optionally log a snippet of the prompt, or its length
        log_verbose(f"Prompt for '{block.name}' (length {len(prompt)} chars) starts with: {prompt[:100].replace('\n', ' ')}...")

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        
        log_verbose(f"Sending request to LLM for block '{block.name}'...")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
            response_format={"type": "json_object"}, # type: ignore
            temperature=.1
        )
        content = resp.choices[0].message.content
        log_verbose(f"LLM response content for '{block.name}':\n{content}") # Log full content if verbose

        if content is None:
            log_verbose(f"LLM API error: No content received for block {block.name} in {filename}")
            raise RuntimeError(f"LLM API error: No content received for block {block.name} in {filename}")
            
        data = json.loads(content)
        warnings_list = data.get('warnings', [])
        llm_error = data.get('error')

        if llm_error:
            error_type = llm_error.get('type')
            error_message = llm_error.get('message', 'No error message provided by LLM.')
            log_verbose(f"LLM reported error for '{block.name}': type='{error_type}', message='{error_message}'")
            if error_type == "AmbiguityError":
                raise AuraAmbiguityError(f"LLM Ambiguity Error for block '{block.name}' in {filename}: {error_message}")
            elif error_type == "ImportNeededError":
                module_name_detail = llm_error.get('details', {}).get('module_name')
                full_message = f"LLM Import Needed Error for block '{block.name}' in {filename}: {error_message}"
                if module_name_detail:
                    full_message += f" (Suggested module: {module_name_detail})"
                raise AuraImportNeededError(full_message, module_name=module_name_detail)
            elif error_type == "UncompilableError":
                raise AuraUncompilableError(f"LLM Uncompilable Error for block '{block.name}' in {filename}: {error_message}")
            else: 
                raise RuntimeError(f"LLM Processing Error for block '{block.name}' in {filename} (type: {error_type or 'Unknown'}): {error_message}")
        
        generated_code = data.get('code')
        if generated_code is None:
            log_verbose(f"LLM Error: Received null code for block '{block.name}' in {filename} without a corresponding fatal error object. This is unexpected if no LLM error was raised.")
            raise RuntimeError(f"LLM Error: Received null code for block '{block.name}' in {filename} without a corresponding fatal error object. LLM Response: {content}")
        elif not generated_code.strip():
            log_verbose(f"LLM Warning/Info: Received empty code string for block '{block.name}' in {filename}. This block will produce no Python output.")
            # Allow empty string, it might be intentional or an LLM way of saying "nothing to do"
            # The parameter validation below will catch issues if params were expected.

        generated_code = generated_code.rstrip()
        log_verbose(f"Generated code for '{block.name}' (length {len(generated_code)} chars). Validating params...")

        # Validate that all --params from signature are in the generated code
        declared_params = re.findall(r"--(\\\\w+)", block.signature)
        missing_params = []
        if generated_code or declared_params: # Only validate if there's code or params expected
            for param_name in declared_params:
                if not re.search(r'\\\\b' + re.escape(param_name) + r'\\\\b', generated_code):
                    missing_params.append(param_name)
        
        if missing_params:
            log_verbose(f"Validation Error for '{block.name}': Missing parameters: {missing_params}")
            error_message = (
                f"In file '{filename}', block '{block.name}', LLM output validation error: "
                f"The following parameters declared with '--' in the Aura signature "
                f"were not found in the generated Python code: {', '.join(missing_params)}. "
                "Please ensure the LLM uses these parameters or adjust the Aura signature."
            )
            raise AuraCompilationError(error_message)
        
        log_verbose(f"Block '{block.name}' processed successfully.")
        return generated_code, warnings_list

    def _get_all_defined_symbols(self, local_mod: AuraModule, imported: Dict[str, AuraModule]) -> List[str]:
        defined_symbols = set()
        # Local symbols
        for t in local_mod.types:
            if t.doc: defined_symbols.add(t.name)
        for b in local_mod.blocks:
            if b.doc: defined_symbols.add(b.name)
        
        # Imported symbols
        for alias, mod_obj in imported.items():
            for t in mod_obj.types:
                if t.doc: defined_symbols.add(f"{alias}.{t.name}")
            for b_imp in mod_obj.blocks:
                if b_imp.doc: defined_symbols.add(f"{alias}.{b_imp.name}")
        return sorted(list(defined_symbols))

    def _build_prompt(
        self,
        block: BlockDecl,
        local_mod: AuraModule,
        imported: Dict[str, AuraModule],
        filename: str,
        block_name_for_error: str
    ) -> str:
        lines: List[str] = []
        # 1) Block signature and body
        lines.append(f"desc {block.kind} {block.name}({block.signature}): \"{block.doc}\"")
        lines.append("{")
        lines.append(block.body)
        lines.append("}")
        
        # 2) Reference items
        refs_in_block_body_and_sig = set(re.findall(r"@([A-Za-z0-9_.]+)", block.signature + "\n" + block.body))
        ref_docs_for_prompt: List[str] = []
        resolved_ref_names: set[str] = set()
        all_defined_symbols = self._get_all_defined_symbols(local_mod, imported)

        def add_resolved_doc(ref_token: str, doc_content: str):
            ref_docs_for_prompt.append(f"- {ref_token}: {doc_content}")
            resolved_ref_names.add(ref_token)

        for ref_token in sorted(list(refs_in_block_body_and_sig)):
            item_found = False
            if '.' in ref_token: # Imported reference: module_alias.SymbolName
                mod_alias, name_in_import = ref_token.split('.', 1)
                mod_obj = imported.get(mod_alias)
                if mod_obj:
                    for t in mod_obj.types:
                        if t.name == name_in_import and t.doc: # Check t.doc is not empty
                            add_resolved_doc(ref_token, t.doc)
                            item_found = True; break
                    if item_found: continue
                    for b_imp in mod_obj.blocks:
                        if b_imp.name == name_in_import and b_imp.doc: # Check b_imp.doc is not empty
                            add_resolved_doc(ref_token, b_imp.doc)
                            item_found = True; break
            else: # Local reference: SymbolName
                for t in local_mod.types:
                    if t.name == ref_token and t.doc: # Check t.doc is not empty
                        add_resolved_doc(ref_token, t.doc)
                        item_found = True; break
                if item_found: continue
                for b_loc in local_mod.blocks:
                    if b_loc.name == ref_token and b_loc.doc: # Check b_loc.doc is not empty
                        add_resolved_doc(ref_token, b_loc.doc)
                        item_found = True; break
            
            if not item_found: # This ref_token is unresolved
                suggestions = difflib.get_close_matches(ref_token, all_defined_symbols, n=3, cutoff=0.6)
                raise AuraUnresolvedReferenceError(
                    unresolved_ref=ref_token, 
                    suggestions=suggestions,
                    block_name=block_name_for_error, 
                    filename=filename
                )

        if ref_docs_for_prompt:
            lines.append("# Reference items (context only – no code generation)")
            lines.extend(ref_docs_for_prompt)
        
        # 3) Python imports
        py_imports = [imp.raw for imp in local_mod.imports if not imp.is_aura]
        if py_imports:
            lines.append("# Available Python imports:")
            lines.extend(py_imports)
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# COMPILER / CLI
# ─────────────────────────────────────────────────────────────────────────────
def compile_file(src: Path, out_dir: Path, model: str, is_part_of_file_batch: bool):
    log_verbose(f"Starting compilation for file: {src} (part of batch: {is_part_of_file_batch})")
    parser = AuraParser()
    try:
        text = src.read_text()
        local_mod = parser.parse(text, src.name)
    except AuraCompilationError as e:
        print(f"Parsing Error in '{src.name}': {e}", file=sys.stderr)
        log_verbose(f"Parsing error in '{src.name}' caused early exit.")
        return
    except Exception as e:
        print(f"Unexpected Parsing Error in '{src.name}': {e}", file=sys.stderr)
        return

    log_verbose(f"Successfully parsed '{src.name}'. Importing Aura dependencies...")
    imported: Dict[str, AuraModule] = {}
    for imp in local_mod.imports:
        if imp.is_aura and imp.target:
            aura_path = Path(src.parent) / f"{imp.target}.aura"
            log_verbose(f"Processing Aura import: {imp.target} from {aura_path}")
            if aura_path.exists():
                try:
                    imported[imp.alias or imp.target] = parser.parse(aura_path.read_text(), aura_path.name)
                except AuraCompilationError as e:
                    print(f"Error parsing imported Aura module '{aura_path.name}': {e}", file=sys.stderr)
                    log_verbose(f"Error parsing import '{aura_path.name}'. Halting for current file '{src.name}'.")
                    return
                except Exception as e:
                    print(f"Unexpected error parsing imported Aura module '{aura_path.name}': {e}", file=sys.stderr)
                    return
            else:
                print(f"Error: Imported Aura module '{aura_path}' not found.", file=sys.stderr)
                log_verbose(f"Imported Aura module '{aura_path}' not found. Halting for '{src.name}'.")
                return

    prompter = LLMPrompter(model=model)
    sections: List[str] = []
    for imp in local_mod.imports:
        if not imp.is_aura:
            sections.append(imp.raw)
    sections.append("# --- GENERATED CODE ---")

    seen: set[str] = set()
    any_warnings_for_file = False
    
    log_verbose(f"Processing {len(local_mod.blocks)} blocks for '{src.name}'...")
    # Wrap block processing with tqdm
    for blk in tqdm(local_mod.blocks, desc=f"Compiling blocks in {src.name}", unit="block", disable=VERBOSE_LOGGING):
        log_verbose(f"Attempting to compile block: '{blk.name}'")
        try:
            code, block_warnings = prompter.compile_block(blk, local_mod, imported, src.name)
            if block_warnings:
                any_warnings_for_file = True
                print(f"Warnings for block '{blk.name}' in '{src.name}':", file=sys.stderr)
                for warning in block_warnings:
                    warn_type = warning.get('type', 'GenericWarning')
                    warn_msg = warning.get('message', 'No message provided.')
                    print(f"  ⚠️ ({warn_type}): {warn_msg}", file=sys.stderr)
                    log_verbose(f"  LLM Warning for '{blk.name}': type='{warn_type}', msg='{warn_msg}'")
            
            # Only add non-empty code. If LLM returns empty for a block, it means no output.
            if code.strip(): 
                m_fn = re.match(r"^def\\s+(\\w+)", code)
                m_cl = re.match(r"^class\\s+(\\w+)", code)
                name = m_fn.group(1) if m_fn else (m_cl.group(1) if m_cl else None)
                if name and name in seen:
                    log_verbose(f"Skipping duplicate block definition: '{name}' (from block '{blk.name}')")
                    continue
                if name:
                    seen.add(name)
                sections.append(code)
            else:
                log_verbose(f"Block '{blk.name}' resulted in empty generated code. Not added to output.")

        except AuraCompilationError as e:
            print(f"Compilation Error in '{src.name}' for block '{blk.name}': {e}", file=sys.stderr)
            log_verbose(f"Compilation error for block '{blk.name}' in '{src.name}'. Halting for this file.")
            return
        except RuntimeError as e:
            print(f"Runtime Error during compilation of '{src.name}' for block '{blk.name}': {e}", file=sys.stderr)
            return

    log_verbose(f"All blocks processed for '{src.name}'. Appending #MAIN section if present.")
    # Corrected Regex: Look for start of line, # MAIN, rest of line, newline, then capture everything after.
    main_section_match = re.search(r"^\s*# MAIN.*?\n(.*)", text, re.MULTILINE | re.DOTALL) # Allow leading whitespace
    if main_section_match:
        log_verbose(f"Found #MAIN section. Appending content.")
        sections.append("\n# --- UNCOMPILED CODE ---")
        captured_content = main_section_match.group(1).strip()
        log_verbose(f"Content captured from #MAIN (len={len(captured_content)}): {captured_content[:200].replace('\n', ' ')}...")
        sections.append(captured_content)
    else:
        log_verbose(f"No #MAIN section found in '{src.name}'. Regex match was None.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = out_dir / f"{local_mod.name}.py"
    log_verbose(f"Writing compiled output for '{src.name}' to '{out_file_path}'")
    out_file_path.write_text("\n".join(sections))
    
    status_emoji = "⚠️" if any_warnings_for_file else "✅"
    final_message_desc = f"Processed {src.name}"
    if VERBOSE_LOGGING:
        final_message_desc = f"Completed {src.name}"
        # tqdm handles its own line management when not disabled.
        # if not in a tqdm loop (e.g. single file), print directly.
        print(f"{status_emoji} {final_message_desc} → {out_file_path}")
    elif not is_part_of_file_batch: # If not in the file loop, print status using tqdm.write
         tqdm.write(f"{status_emoji} {final_message_desc} → {out_file_path}")
    # If it IS part of a file batch, the file-level tqdm will show overall progress.
    # Individual file status within a batch is implicitly handled by tqdm completing iterations.

def main():
    global VERBOSE_LOGGING
    ap = argparse.ArgumentParser(description="Aura compiler")
    ap.add_argument('command', choices=['compile'])
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging output to stderr.')
    args = ap.parse_args()

    if args.verbose:
        VERBOSE_LOGGING = True
        print("[INFO] Verbose logging enabled.", file=sys.stderr)

    inp, out = Path(args.input), Path(args.output)

    if args.command == 'compile':
        if inp.is_file():
            # Pass False for is_part_of_file_batch when compiling a single file
            compile_file(inp, out, args.model, is_part_of_file_batch=False)
        else:
            all_aura_files = list(inp.rglob('*.aura'))
            log_verbose(f"Found {len(all_aura_files)} Aura files in directory '{inp}'.")
            with tqdm(total=len(all_aura_files), desc="Compiling Aura files", unit="file") as pbar_files:
                for f_path in all_aura_files:
                    log_verbose(f"---Processing file from directory: {f_path} ---")
                    # Pass True for is_part_of_file_batch when compiling multiple files
                    compile_file(f_path, out / f_path.parent.relative_to(inp), args.model, is_part_of_file_batch=True)
                    pbar_files.update(1)

if __name__ == '__main__':
    main()
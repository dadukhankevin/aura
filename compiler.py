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

# Load .env configuration
from dotenv import load_dotenv
import openai
load_dotenv()



api_key = os.getenv("AURA_LLM_API_KEY")
if not api_key:
    raise RuntimeError("Missing AURA_LLM_API_KEY in environment (.env)")
base_url = os.getenv("AURA_LLM_BASE_URL", "https://api.openai.com/v1")

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
    def __init__(self, unresolved_refs: List[str], block_name: str, filename: str):
        super().__init__(
            f"In file '{filename}', within block '{block_name}', the following @references could not be resolved: {', '.join(sorted(unresolved_refs))}. "
            f"Ensure they are defined with 'desc' in the current module or a correctly imported Aura module, and that their descriptions are not empty."
        )
        self.unresolved_refs = unresolved_refs
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
        prompt = self._build_prompt(block, local_mod, imported, filename, block.name)
        
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages, # type: ignore
            response_format={"type": "json_object"} # type: ignore
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError(f"LLM API error: No content received for block {block.name} in {filename}")
            
        data = json.loads(content)
        warnings_list = data.get('warnings', [])
        llm_error = data.get('error')

        if llm_error:
            error_type = llm_error.get('type')
            error_message = llm_error.get('message', 'No error message provided by LLM.')
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
            raise RuntimeError(f"LLM Error: Received null code for block '{block.name}' in {filename} without a corresponding fatal error object. LLM Response: {content}")
        
        generated_code = generated_code.rstrip()

        # Validate that all --params from signature are in the generated code
        declared_params = re.findall(r"--(\\\w+)", block.signature)
        missing_params = []
        for param_name in declared_params:
            if not re.search(r'\\b' + re.escape(param_name) + r'\\b', generated_code):
                missing_params.append(param_name)
        
        if missing_params:
            error_message = (
                f"In file '{filename}', block '{block.name}', LLM output validation error: "
                f"The following parameters declared with '--' in the Aura signature "
                f"were not found in the generated Python code: {', '.join(missing_params)}. "
                "Please ensure the LLM uses these parameters or adjust the Aura signature."
            )
            raise AuraCompilationError(error_message)
            
        return generated_code, warnings_list

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
        
        unresolved_refs = refs_in_block_body_and_sig - resolved_ref_names
        if unresolved_refs:
            raise AuraUnresolvedReferenceError(
                unresolved_refs=list(unresolved_refs), 
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
def compile_file(src: Path, out_dir: Path, model: str):
    parser = AuraParser()
    try:
        text = src.read_text()
        local_mod = parser.parse(text, src.name)
    except AuraCompilationError as e:
        print(f"Parsing Error in '{src.name}': {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Unexpected Parsing Error in '{src.name}': {e}", file=sys.stderr)
        return

    imported: Dict[str, AuraModule] = {}
    for imp in local_mod.imports:
        if imp.is_aura and imp.target:
            aura_path = Path(src.parent) / f"{imp.target}.aura"
            if aura_path.exists():
                try:
                    imported[imp.alias or imp.target] = parser.parse(aura_path.read_text(), aura_path.name)
                except AuraCompilationError as e:
                    print(f"Error parsing imported Aura module '{aura_path.name}': {e}", file=sys.stderr)
                    return
                except Exception as e:
                    print(f"Unexpected error parsing imported Aura module '{aura_path.name}': {e}", file=sys.stderr)
                    return
            else:
                print(f"Error: Imported Aura module '{aura_path}' not found.", file=sys.stderr)
                return

    prompter = LLMPrompter(model=model)
    sections: List[str] = []
    for imp in local_mod.imports:
        if not imp.is_aura:
            sections.append(imp.raw)
    sections.append("# --- GENERATED CODE ---")

    seen: set[str] = set()
    any_warnings_for_file = False
    for blk in local_mod.blocks:
        try:
            code, block_warnings = prompter.compile_block(blk, local_mod, imported, src.name)
            if block_warnings:
                any_warnings_for_file = True
                print(f"Warnings for block '{blk.name}' in '{src.name}':", file=sys.stderr)
                for warning in block_warnings:
                    warn_type = warning.get('type', 'GenericWarning')
                    warn_msg = warning.get('message', 'No message provided.')
                    print(f"  ⚠️ ({warn_type}): {warn_msg}", file=sys.stderr)
            
            m_fn = re.match(r"^def\s+(\w+)", code)
            m_cl = re.match(r"^class\s+(\w+)", code)
            name = m_fn.group(1) if m_fn else (m_cl.group(1) if m_cl else None)
            if name and name in seen:
                continue
            if name:
                seen.add(name)
            sections.append(code)
        except AuraCompilationError as e:
            print(f"Compilation Error in '{src.name}' for block '{blk.name}': {e}", file=sys.stderr)
            return
        except RuntimeError as e:
            print(f"Runtime Error during compilation of '{src.name}' for block '{blk.name}': {e}", file=sys.stderr)
            return

    main_section_match = re.search(r"# MAIN\s*$(.*)", text, re.MULTILINE | re.DOTALL)
    if main_section_match:
        sections.append("\n# --- UNCOMPILED CODE ---")
        sections.append(main_section_match.group(1).strip())

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_path = out_dir / f"{local_mod.name}.py"
    out_file_path.write_text("\n".join(sections))
    
    status_emoji = "⚠️" if any_warnings_for_file else "✅"
    print(f"{status_emoji} Processed {src} → {out_file_path}")


def main():
    ap = argparse.ArgumentParser(description="Aura compiler")
    ap.add_argument('command', choices=['compile'])
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--model', default='gpt-4o-mini')
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    if args.command == 'compile':
        if inp.is_file():
            compile_file(inp, out, args.model)
        else:
            for f in inp.rglob('*.aura'):
                compile_file(f, out, args.model)

if __name__ == '__main__':
    main()
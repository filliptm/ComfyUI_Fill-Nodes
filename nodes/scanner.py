import os
import sys
import json
import inspect
import importlib.util
import time
from pathlib import Path

class NodeScanner:
    def __init__(self, comfy_path):
        self.comfy_path = os.path.abspath(comfy_path)
        if self.comfy_path not in sys.path:
            sys.path.append(self.comfy_path)
        
        # Get the directory containing this script
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set up cache directory paths
        self.cache_dir = os.path.join(self.base_dir, 'web', 'nodes')
        self.cache_file = os.path.join(self.cache_dir, 'node_definitions.txt')
        
        print(f"\nScanner initialized with:")
        print(f"Base directory: {self.base_dir}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Cache file: {self.cache_file}")
        
        self.ensure_cache_dir()

    def ensure_cache_dir(self):
        try:
            # First ensure the web directory exists
            web_dir = os.path.join(self.base_dir, 'web')
            if not os.path.exists(web_dir):
                os.makedirs(web_dir)
                print(f"Created web directory: {web_dir}")
            
            # Then create the nodes directory
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
                print(f"Created cache directory: {self.cache_dir}")
            
            # Test if we can write to the cache directory
            test_file = os.path.join(self.cache_dir, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Cache directory is writable: {self.cache_dir}")
            
            return True
        except Exception as e:
            print(f"ERROR: Failed to create directory structure:")
            print(f"  - {str(e)}")
            print(f"  - Base dir exists: {os.path.exists(self.base_dir)}")
            print(f"  - Web dir exists: {os.path.exists(web_dir if 'web_dir' in locals() else 'N/A')}")
            print(f"  - Cache dir exists: {os.path.exists(self.cache_dir)}")
            print(f"  - Have write permission: {os.access(self.base_dir, os.W_OK)}")
            print(f"  - Current working directory: {os.getcwd()}")
            return False

    def scan_nodes(self):
        if not os.path.exists(self.comfy_path):
            print(f"ERROR: ComfyUI path does not exist: {self.comfy_path}")
            return {}

        try:
            nodes = {}
            total_found = 0
            custom_nodes_path = os.path.join(self.comfy_path, 'custom_nodes')
            
            if not os.path.exists(custom_nodes_path):
                print(f"ERROR: Custom nodes directory not found: {custom_nodes_path}")
                return {}
                
            print(f"\nScanning custom nodes in: {custom_nodes_path}")
            
            for root, dirs, files in os.walk(custom_nodes_path):
                # Skip scanning ourselves and unwanted directories
                if 'scanner.py' in files:
                    files.remove('scanner.py')
                dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', 'web'}]
                
                for file in files:
                    if file.endswith('.py'):
                        try:
                            found_nodes = self.extract_node_info(os.path.join(root, file))
                            if found_nodes:
                                nodes.update(found_nodes)
                                total_found += len(found_nodes)
                                print(f"Found {len(found_nodes)} nodes in {file}")
                        except Exception as e:
                            if not str(e).startswith("No module named"):
                                print(f"Error processing {file}: {str(e)}")
            
            print(f"\nScan complete. Found {total_found} nodes.")
            if nodes and self.ensure_cache_dir():
                self.cache_definitions(nodes)
            return nodes
            
        except Exception as e:
            print(f"ERROR during scanning: {str(e)}")
            return {}

    def extract_node_info(self, file_path):
        """Extract node information by reading the file as text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract basic node info without importing
            nodes = {}
            
            # Look for NODE_CLASS_MAPPINGS
            if 'NODE_CLASS_MAPPINGS' in content:
                # Find class definitions
                class_blocks = content.split('class ')[1:]
                
                for block in class_blocks:
                    try:
                        # Get class name
                        class_name = block.split('(')[0].strip()
                        
                        # Extract category if exists
                        category = 'Unknown'
                        if 'CATEGORY' in block:
                            category_line = [l for l in block.split('\n') if 'CATEGORY' in l]
                            if category_line:
                                category = category_line[0].split('=')[1].strip().strip('"\'')
                        
                        # Extract docstring if exists
                        description = None
                        if '"""' in block or "'''" in block:
                            doc_start = block.find('"""') if '"""' in block else block.find("'''")
                            if doc_start > -1:
                                doc_end = block.find('"""', doc_start + 3) if '"""' in block else block.find("'''", doc_start + 3)
                                if doc_end > -1:
                                    description = block[doc_start+3:doc_end].strip()
                        
                        nodes[class_name] = {
                            'name': class_name,
                            'category': category,
                            'description': description,
                            'inputs': self.extract_input_types_from_text(block),
                            'outputs': self.extract_output_types_from_text(block)
                        }
                    except Exception as e:
                        print(f"Error parsing class in {os.path.basename(file_path)}: {str(e)}")
                        continue
                        
            return nodes
            
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {str(e)}")
            return None

    def get_input_types(self, node_class):
        """Extract input types from node class"""
        try:
            if hasattr(node_class, 'INPUT_TYPES'):
                if inspect.ismethod(node_class.INPUT_TYPES):
                    input_types = node_class.INPUT_TYPES()
                else:
                    input_types = node_class.INPUT_TYPES
                return input_types
        except Exception as e:
            print(f"ERROR getting input types: {str(e)}")
        return {}

    def extract_input_types_from_text(self, class_text):
        """Extract input types from class text with improved parsing"""
        inputs = {'required': {}, 'optional': {}}
        
        if 'INPUT_TYPES' in class_text:
            try:
                # Find the INPUT_TYPES block
                input_block = ""
                start = class_text.find('INPUT_TYPES')
                block_start = class_text.find('{', start)
                
                if block_start > -1:
                    # Count braces to find matching end
                    brace_count = 1
                    pos = block_start + 1
                    
                    while brace_count > 0 and pos < len(class_text):
                        if class_text[pos] == '{':
                            brace_count += 1
                        elif class_text[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    
                    input_block = class_text[block_start:pos]
                
                    # Parse required inputs
                    if '"required"' in input_block or "'required'" in input_block:
                        req_section = self._extract_section(input_block, 'required')
                        inputs['required'] = self._parse_input_section(req_section)
                    
                    # Parse optional inputs
                    if '"optional"' in input_block or "'optional'" in input_block:
                        opt_section = self._extract_section(input_block, 'optional')
                        inputs['optional'] = self._parse_input_section(opt_section)
                        
            except Exception as e:
                print(f"Error parsing INPUT_TYPES: {str(e)}")
                
        return inputs

    def _extract_section(self, text, section_name):
        """Helper to extract a section (required/optional) from INPUT_TYPES text"""
        try:
            start = text.find(f'"{section_name}"')
            if start == -1:
                start = text.find(f"'{section_name}'")
            
            if start > -1:
                start = text.find('{', start)
                if start > -1:
                    brace_count = 1
                    pos = start + 1
                    
                    while brace_count > 0 and pos < len(text):
                        if text[pos] == '{':
                            brace_count += 1
                        elif text[pos] == '}':
                            brace_count -= 1
                        pos += 1
                    
                    return text[start:pos]
        except Exception as e:
            print(f"Error extracting section {section_name}: {str(e)}")
        return ""

    def _parse_input_section(self, section_text):
        """Helper to parse individual input definitions"""
        inputs = {}
        
        try:
            # Split into lines and clean up
            lines = section_text.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and ('(' in line or '[' in line or '"' in line or "'" in line):
                    # Extract name
                    name = line.split(':')[0].strip().strip('"\'')
                    
                    # Extract type info
                    type_info = []
                    
                    # Look for common patterns
                    if '(' in line:  # Tuple/list definitions
                        start = line.find('(')
                        end = line.find(')', start)
                        if end > start:
                            types = line[start+1:end].split(',')
                            type_info = [t.strip().strip('"\'') for t in types if t.strip()]
                            
                    elif '[' in line:  # List definitions
                        start = line.find('[')
                        end = line.find(']', start)
                        if end > start:
                            types = line[start+1:end].split(',')
                            type_info = [t.strip().strip('"\'') for t in types if t.strip()]
                            
                    elif '"' in line or "'" in line:  # Simple string definitions
                        parts = line.split(':')[1].split(',')[0].strip()
                        if parts:
                            type_info = [parts.strip().strip('"\'')]
                    
                    if name and type_info:
                        inputs[name] = type_info
                        
        except Exception as e:
            print(f"Error parsing input section: {str(e)}")
            
        return inputs

    def extract_output_types_from_text(self, class_text):
        """Extract output types from class text"""
        outputs = {
            'return_types': (),
            'return_names': None
        }
        
        try:
            # Find RETURN_TYPES
            if 'RETURN_TYPES' in class_text:
                start = class_text.find('RETURN_TYPES')
                end = class_text.find('\n', start)
                if end > start:
                    types_text = class_text[start:end]
                    if '=' in types_text:
                        types_str = types_text.split('=')[1].strip().strip('()')
                        outputs['return_types'] = tuple(t.strip().strip('"\'') for t in types_str.split(',') if t.strip())
                        
            # Find RETURN_NAMES
            if 'RETURN_NAMES' in class_text:
                start = class_text.find('RETURN_NAMES')
                end = class_text.find('\n', start)
                if end > start:
                    names_text = class_text[start:end]
                    if '=' in names_text:
                        names_str = names_text.split('=')[1].strip().strip('()')
                        outputs['return_names'] = tuple(n.strip().strip('"\'') for n in names_str.split(',') if n.strip())
                        
        except Exception as e:
            print(f"Error parsing output types: {str(e)}")
            
        return outputs

    def cache_definitions(self, definitions):
        if not definitions:
            print("No definitions to cache - skipping cache write")
            return False
        
        print(f"\nWriting cache file to: {self.cache_file}")
        try:
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Convert to text format
            content = []
            for node_name, info in definitions.items():
                # Node header
                content.append(f"Node: {node_name}")
                content.append(f"Category: {info['category']}")
                if info.get('description'):
                    content.append(f"Description: {info['description']}")
                
                # Inputs
                content.append("Inputs:")
                if 'required' in info['inputs']:
                    content.append("  Required:")
                    for name, types in info['inputs']['required'].items():
                        content.append(f"    - {name}: {', '.join(types)}")
                if 'optional' in info['inputs']:
                    content.append("  Optional:")
                    for name, types in info['inputs']['optional'].items():
                        content.append(f"    - {name}: {', '.join(types)}")
                
                # Outputs
                content.append("Outputs:")
                return_types = info['outputs']['return_types']
                return_names = info['outputs']['return_names']
                if return_names:
                    for t, n in zip(return_types, return_names):
                        content.append(f"  - {n} ({t})")
                else:
                    for t in return_types:
                        content.append(f"  - {t}")
                
                content.append("-" * 50)  # Separator between nodes
            
            # Write to temporary file first
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
                f.flush()
                os.fsync(f.fileno())
            
            print(f"Temp file written: {temp_file}")
            print(f"Temp file exists: {os.path.exists(temp_file)}")
            print(f"Temp file size: {os.path.getsize(temp_file) if os.path.exists(temp_file) else 0}")
            
            # Safely move the temp file to final location
            if os.path.exists(self.cache_file):
                backup_file = self.cache_file + '.bak'
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(self.cache_file, backup_file)
                print(f"Created backup file: {backup_file}")
            
            # Move temp file to final location
            os.rename(temp_file, self.cache_file)
            
            # Verify the file was created
            if not os.path.exists(self.cache_file):
                raise Exception(f"File was not created at {self.cache_file}")
            
            # Try to open and read the file to verify it's accessible
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    test_content = f.read(100)  # Read first 100 chars as a test
                    print(f"File is readable. First 100 chars: {test_content[:100]}")
            except Exception as e:
                print(f"WARNING: File created but couldn't be read: {str(e)}")
                
            # Check server path
            comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.base_dir)))
            print(f"ComfyUI root directory: {comfy_dir}")
            
            # Generate the URL path that browsers would use
            relative_path = os.path.relpath(self.cache_file, comfy_dir)
            url_path = '/' + relative_path.replace('\\', '/')
            print(f"Relative web path for browser access: {url_path}")
                
            size = os.path.getsize(self.cache_file)
            print(f"Cache written successfully ({size:,} bytes)")
            print(f"Cache file absolute path: {os.path.abspath(self.cache_file)}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to write cache: {str(e)}")
            # Print more diagnostic information
            print(f"  - Temp file exists: {os.path.exists(temp_file if 'temp_file' in locals() else 'N/A')}")
            print(f"  - Cache dir exists: {os.path.exists(os.path.dirname(self.cache_file))}")
            print(f"  - Current working directory: {os.getcwd()}")
            print(f"  - Permission to write: {os.access(os.path.dirname(self.cache_file), os.W_OK)}")
            
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False

    def node_to_compact_format(self, node_name, info):
        """Convert node info to compact format"""
        lines = []
        
        # Node name and category
        lines.append(f"#N {node_name} #C {info['category']}")
        
        # Inputs
        input_parts = []
        if 'required' in info['inputs']:
            for name, types in info['inputs']['required'].items():
                input_parts.append(f"{name}:{','.join(types)}")
        if 'optional' in info['inputs']:
            for name, types in info['inputs']['optional'].items():
                input_parts.append(f"{name}:{','.join(types)}?")
        if input_parts:
            lines.append(f"#I {' '.join(input_parts)}")
        
        # Outputs
        if info['outputs']['return_types']:
            out_types = ','.join(info['outputs']['return_types'])
            if info['outputs']['return_names']:
                out_names = ','.join(info['outputs']['return_names'])
                lines.append(f"#O {out_types} #N {out_names}")
            else:
                lines.append(f"#O {out_types}")
        
        # Description
        if info.get('description'):
            lines.append(f"#D {info['description']}")
        
        lines.append("---")
        return '\n'.join(lines)

    def parse_compact_format(self, content):
        """Parse compact format back into node definitions"""
        definitions = {}
        current_node = None
        current_info = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line == '---':
                if current_node:
                    definitions[current_node] = current_info
                    current_node = None
                    current_info = {}
                continue
                
            parts = line.split()
            if line.startswith('#N') and '#C' in line:
                # Node name and category
                name_idx = line.index('#C')
                current_node = ' '.join(parts[1:name_idx]).strip()
                current_info['category'] = ' '.join(parts[name_idx+1:]).strip()
                current_info['inputs'] = {'required': {}, 'optional': {}}
                current_info['outputs'] = {'return_types': (), 'return_names': None}
                
            elif line.startswith('#I'):
                # Inputs
                for input_def in ' '.join(parts[1:]).split():
                    name, type_info = input_def.split(':')
                    if type_info.endswith('?'):
                        # Optional input
                        type_info = type_info[:-1]
                        current_info['inputs']['optional'][name] = type_info.split(',')
                    else:
                        # Required input
                        current_info['inputs']['required'][name] = type_info.split(',')
                        
            elif line.startswith('#O'):
                # Outputs
                if '#N' in line:
                    # Has named outputs
                    type_idx = line.index('#N')
                    types = ' '.join(parts[1:type_idx]).strip().split(',')
                    names = ' '.join(parts[type_idx+1:]).strip().split(',')
                    current_info['outputs']['return_types'] = tuple(types)
                    current_info['outputs']['return_names'] = tuple(names)
                else:
                    # Just types
                    current_info['outputs']['return_types'] = tuple(parts[1].split(','))
                    
            elif line.startswith('#D'):
                # Description
                current_info['description'] = ' '.join(parts[1:])
        
        return definitions

    def load_cache(self):
        """Load cached node definitions from compact format"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
        return None

    def is_cache_valid(self, max_age_hours=24):
        """Check if cache is valid based on age"""
        cache_data = self.load_cache()
        if not cache_data:
            return False
        
        cache_age = time.time() - cache_data['timestamp']
        return cache_age < (max_age_hours * 3600)

    def get_nodes_for_llm(self):
        """Get node definitions in LLM-friendly format"""
        cache_data = self.load_cache()
        if not cache_data:
            return "No node definitions available."

        llm_text = "Available Custom Nodes:\n\n"
        for name, info in cache_data['definitions'].items():
            llm_text += f"Node: {name}\n"
            if info['description']:
                llm_text += f"Description: {info['description']}\n"
            llm_text += f"Category: {info['category']}\n"
            
            # Inputs
            llm_text += "Inputs:\n"
            if 'required' in info['inputs']:
                for input_name, input_info in info['inputs']['required'].items():
                    llm_text += f"  - {input_name} ({input_info[0]})\n"
            if 'optional' in info['inputs']:
                for input_name, input_info in info['inputs']['optional'].items():
                    llm_text += f"  - {input_name} ({input_info[0]}) [Optional]\n"
            
            # Outputs
            llm_text += "Outputs:\n"
            return_types = info['outputs']['return_types']
            return_names = info['outputs']['return_names']
            if return_names:
                for i, (t, n) in enumerate(zip(return_types, return_names)):
                    llm_text += f"  - {n} ({t})\n"
            else:
                for t in return_types:
                    llm_text += f"  - {t}\n"
            
            llm_text += "\n"

        return llm_text


if __name__ == "__main__":
    try:
        # Get ComfyUI path from command line or use default parent directory
        default_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        comfy_path = sys.argv[1] if len(sys.argv) > 1 else default_path
        
        print(f"\n=== NodeScanner CLI ===")
        print(f"├── ComfyUI path: {comfy_path}")
        
        scanner = NodeScanner(comfy_path)
        print("├── Starting node scan...")
        
        results = scanner.scan_nodes()
        
        if results:
            print(f"└── Found {len(results)} nodes")
            print("\nNode Summary:")
            for node_name, info in results.items():
                print(f"├── Node: {node_name}")
                print(f"│   ├── Category: {info['category']}")
                if info.get('description'):
                    print(f"│   └── Description: {info['description']}")
        else:
            print("└── No nodes were found or errors occurred during scanning")
            
    except Exception as e:
        print(f"\nERROR: Script failed with error: {str(e)}")
        sys.exit(1)
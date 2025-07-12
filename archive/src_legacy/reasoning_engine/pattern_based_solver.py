import json
import re
from itertools import permutations


class PatternBasedSolver:
    def __init__(self, patterns_path):
        self.patterns = self.load_patterns(patterns_path)
        self.known_values = {}
        self.reasoning_steps = []

    def load_patterns(self, patterns_path):
        with open(patterns_path, 'r') as f:
            return json.load(f)['patterns']

    def solve(self, problem_text):
        self.known_values = {}
        self.reasoning_steps = []
        
        sentences = re.split(r'[.?!]', problem_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Loop until no new information can be derived
        new_info_found = True
        while new_info_found:
            new_info_found = False
            for sentence in sentences:
                if self._process_sentence(sentence):
                    new_info_found = True
        
        # For now, assume the last calculated value is the answer
        final_answer = None
        if self.reasoning_steps:
            last_step = self.reasoning_steps[-1]
            match = re.search(r'=\s*(-?\d+\.?\d*)$', last_step)
            if match:
                final_answer = float(match.group(1))

        return {"answer": final_answer, "reasoning_steps": self.reasoning_steps}

    def _process_sentence(self, sentence):
        for pattern_info in self.patterns:
            if self._apply_pattern(sentence, pattern_info):
                return True
        return False

    def _apply_pattern(self, sentence, pattern_info):
        pattern_regex = pattern_info['pattern']
        match = re.search(pattern_regex, sentence)
        if not match:
            return False

        groups = match.groups()
        
        if pattern_info['type'] == 'assignment':
            entity_name = groups[0]
            entity_value = float(groups[1])
            if entity_name not in self.known_values:
                self.known_values[entity_name] = entity_value
                step = f"Found value: {entity_name} = {entity_value}"
                self.reasoning_steps.append(step)
                print(f"DEBUG: {step}")
                return True
        
        elif pattern_info['type'] == 'binary_operation':
            arg1_name = groups[0]
            value = float(groups[1])
            arg3_name = groups[2]

            # Check if we are trying to calculate a value we already know
            if arg1_name in self.known_values:
                return False

            # Check if the dependency is known
            if arg3_name in self.known_values:
                arg3_value = self.known_values[arg3_name]
                
                # Build the expression from the template
                template = pattern_info['template']
                
                # The template string is like "{arg1} = {arg3} + {arg2}"
                # We only need the calculation part for eval()
                calculation_part = template.split('=')[1].strip()
                
                # Replace placeholders with actual values
                expr = calculation_part.replace('{arg3}', str(arg3_value)).replace('{arg2}', str(value))
                
                try:
                    result = eval(expr)
                    self.known_values[arg1_name] = result
                    
                    # Create a descriptive step for logging
                    reasoning_step = template.replace('{arg1}', arg1_name)\
                                             .replace('{arg2}', str(value))\
                                             .replace('{arg3}', arg3_name)
                    
                    step = f"Calculate {arg1_name}: {reasoning_step} => {arg1_name} = {expr} => {arg1_name} = {result}"
                    self.reasoning_steps.append(step)
                    print(f"DEBUG: {step}")
                    return True
                except Exception as e:
                    print(f"ERROR: eval failed for expression '{expr}': {e}")

        return False 
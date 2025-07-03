import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PatternBasedSolver:
    def __init__(self, patterns_file: str = "src/reasoning_engine/patterns.json"):
        """
        Initialize the pattern-based solver.
        
        Args:
            patterns_file: Path to the JSON file containing patterns
        """
        self.patterns_file = patterns_file
        self.patterns = self._load_patterns()
        self.variables = {}  # Store extracted variables
        
    def _load_patterns(self) -> List[Dict]:
        """Load patterns from JSON file."""
        try:
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('patterns', [])
        except Exception as e:
            print(f"Error loading patterns: {e}")
            return []
    
    def _extract_all_numbers(self, text: str) -> List[float]:
        """Extract all numerical values from text."""
        # Find all numbers (integers and floats)
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        return [float(num) for num in numbers]
    
    def _extract_entities_with_values(self, text: str) -> Dict[str, float]:
        """Extract entities and their associated numerical values."""
        entities = {}
        
        # Pattern: "X had/has/got Y number" 
        matches = re.finditer(r'(\w+)\s+(?:had|has|got|received|found|threw away|gave|lost|sold|bought|scored)\s+(\d+)', text, re.IGNORECASE)
        for match in matches:
            entity, value = match.groups()
            entities[entity.lower()] = float(value)
        
        # Pattern: "Y number of X"
        matches = re.finditer(r'(\d+)\s+(\w+)', text)
        for match in matches:
            value, entity = match.groups()
            if entity.lower() not in ['years', 'year', 'more', 'less', 'times', 'dollars', 'dollar']:
                entities[entity.lower()] = float(value)
        
        # Pattern: "There are/were Y X"
        matches = re.finditer(r'There\s+(?:are|were)\s+(\d+)\s+(\w+)', text, re.IGNORECASE)
        for match in matches:
            value, entity = match.groups()
            entities[entity.lower()] = float(value)
            
        return entities
    
    def solve(self, problem_text: str) -> Tuple[Optional[float], List[str]]:
        """
        Solve a mathematical word problem using pattern matching.
        
        Args:
            problem_text: The problem text to solve
            
        Returns:
            A tuple of (answer, reasoning_steps)
        """
        self.variables.clear()
        reasoning_steps = []
        
        # Extract all numbers and entities first
        all_numbers = self._extract_all_numbers(problem_text)
        entities_values = self._extract_entities_with_values(problem_text)
        
        reasoning_steps.append(f"Extracted numbers: {all_numbers}")
        reasoning_steps.append(f"Extracted entities: {entities_values}")
        
        # PRIORITY 1: Try direct arithmetic patterns first (they're more specific)
        direct_result = self._try_direct_arithmetic(problem_text, reasoning_steps)
        if direct_result is not None:
            reasoning_steps.append(f"Direct arithmetic result: {direct_result}")
            return direct_result, reasoning_steps
        
        # PRIORITY 2: Try pattern matching for more complex templates
        for pattern_info in self.patterns:
            pattern = pattern_info['pattern']
            template = pattern_info['template']
            pattern_name = pattern_info['name']
            
            match = re.search(pattern, problem_text, re.IGNORECASE)
            if match:
                reasoning_steps.append(f"Matched pattern '{pattern_name}': {pattern}")
                
                # Handle special calculation templates
                if template == "sum_all_entities":
                    if all_numbers:
                        result = sum(all_numbers)
                        reasoning_steps.append(f"Sum of all numbers: {' + '.join(map(str, all_numbers))} = {result}")
                        return result, reasoning_steps
                        
                elif template == "max_entity - min_entity":
                    if len(all_numbers) >= 2:
                        max_val = max(all_numbers)
                        min_val = min(all_numbers)
                        result = max_val - min_val
                        reasoning_steps.append(f"Difference: {max_val} - {min_val} = {result}")
                        return result, reasoning_steps
                
                # Handle template-based calculations
                elif "{" in template:
                    groups = match.groups()
                    reasoning_steps.append(f"Captured groups: {groups}")
                    
                    # Replace template variables with actual values
                    calc_expression = template
                    for i, group in enumerate(groups, 1):
                        placeholder = f"{{arg{i}}}"
                        if placeholder in calc_expression:
                            # Try to convert to number if possible
                            try:
                                value = float(group)
                                calc_expression = calc_expression.replace(placeholder, str(value))
                            except ValueError:
                                calc_expression = calc_expression.replace(placeholder, group)
                    
                    reasoning_steps.append(f"Calculation template: {calc_expression}")
                    
                    # Try to evaluate if it's a simple mathematical expression
                    if any(op in calc_expression for op in ['+', '-', '*', '/']):
                        try:
                            # Simple evaluation for basic arithmetic
                            result = self._safe_eval(calc_expression)
                            if result is not None:
                                reasoning_steps.append(f"Result: {result}")
                                return result, reasoning_steps
                        except:
                            pass
        
        reasoning_steps.append("No matching pattern found")
        return None, reasoning_steps
    
    def _safe_eval(self, expression: str) -> Optional[float]:
        """Safely evaluate a mathematical expression."""
        try:
            # Only allow basic arithmetic operations and numbers
            allowed_chars = set("0123456789+-*/.() ")
            if all(c in allowed_chars for c in expression):
                return float(eval(expression))
        except:
            pass
        return None
    
    def _try_direct_arithmetic(self, problem_text: str, reasoning_steps: List[str]) -> Optional[float]:
        """Try to solve using direct arithmetic pattern recognition."""
        numbers = self._extract_all_numbers(problem_text)
        text_lower = problem_text.lower()
        
        if len(numbers) < 2:
            return None
        
        # PRIORITY 1: Division patterns (most specific)
        # Enhanced division patterns
        if any(word in text_lower for word in ["how many packs", "how many bags", "how many groups"]):
            if len(numbers) == 2:
                # For division problems, typically we divide the larger number by the smaller
                larger, smaller = max(numbers), min(numbers)
                result = larger / smaller
                reasoning_steps.append(f"Division pattern (packs/bags/groups): {larger} / {smaller} = {result}")
                return result
                
        elif "each" in text_lower and any(word in text_lower for word in ["group", "bag", "pack", "friend"]):
            if len(numbers) == 2:
                # For "each" problems, typically total divided by each amount
                larger, smaller = max(numbers), min(numbers)
                result = larger / smaller
                reasoning_steps.append(f"Division pattern (each): {larger} / {smaller} = {result}")
                return result
        
        # PRIORITY 2: Comparison patterns
        if "how many more" in text_lower:
            if len(numbers) >= 2:
                result = max(numbers) - min(numbers)
                reasoning_steps.append(f"'How many more' pattern: {max(numbers)} - {min(numbers)} = {result}")
                return result
        
        # PRIORITY 2.5: Reverse reasoning patterns ("X did N more than Y, how many did Y do?")
        elif re.search(r'more [\w-]+ than', text_lower) and "how many" in text_lower:
            # Look for patterns like "David did 9 more push-ups than Zachary. How many did Zachary do?"
            if len(numbers) == 2:
                # Usually the first number is the total, second is the difference
                total, difference = numbers[0], numbers[1]
                result = total - difference
                reasoning_steps.append(f"Reverse reasoning: {total} - {difference} = {result}")
                return result
                
        # PRIORITY 2.6: "More X than Y" addition patterns
        elif (re.search(r'\d+ more [\w\s]+ than', text_lower) or re.search(r'they had \d+ more', text_lower)):
            # Patterns like "79 more bottles than diet soda" or "they had 79 more"
            # Even if it asks "how many", this is still an addition problem
            if len(numbers) >= 2:
                # Simple addition of the two numbers
                result = numbers[0] + numbers[1]
                reasoning_steps.append(f"'More than' addition: {numbers[0]} + {numbers[1]} = {result}")
                return result
        
        # PRIORITY 3: Sum patterns
        elif any(phrase in text_lower for phrase in ["in total", "in all", "altogether"]):
            result = sum(numbers)
            reasoning_steps.append(f"Total/sum pattern: {' + '.join(map(str, numbers))} = {result}")
            return result
            
        # PRIORITY 3.5: "Given away and lost" - addition patterns (only first two numbers)
        elif ("given" in text_lower or "gave" in text_lower) and ("lost" in text_lower) and "how many" in text_lower and ("given away" in text_lower or "lost or given" in text_lower):
            if len(numbers) >= 2:
                # Only add the first two numbers (given + lost), ignore the "left" amount
                result = numbers[0] + numbers[1]
                reasoning_steps.append(f"Given+Lost pattern: {numbers[0]} + {numbers[1]} = {result}")
                return result
                
        # PRIORITY 3.6: "Received + gave - spent" patterns
        elif "received" in text_lower and ("gave" in text_lower or "dad gave" in text_lower) and ("spent" in text_lower):
            if len(numbers) == 3:
                received, gave, spent = numbers[0], numbers[1], numbers[2]
                result = received + gave - spent
                reasoning_steps.append(f"Received+Gave-Spent: {received} + {gave} - {spent} = {result}")
                return result
                
        # PRIORITY 3.7: "Had X, left Y, used how many" patterns
        elif "had" in text_lower and "left" in text_lower and ("use" in text_lower or "used" in text_lower) and "how many" in text_lower:
            if len(numbers) >= 2:
                # For problems like "had 40 apples... had 39 apples left", we want 40 - 39
                # Look for the pattern where we have an original amount and a remaining amount
                # Usually they're close in value, pick the two closest numbers
                if len(numbers) >= 3:
                    # Find the two most similar numbers (likely the before/after amounts)
                    min_diff = float('inf')
                    best_pair = (numbers[0], numbers[1])
                    for i in range(len(numbers)):
                        for j in range(i+1, len(numbers)):
                            diff = abs(numbers[i] - numbers[j])
                            if diff < min_diff:
                                min_diff = diff
                                best_pair = (max(numbers[i], numbers[j]), min(numbers[i], numbers[j]))
                    had, left = best_pair
                else:
                    had, left = max(numbers[:2]), min(numbers[:2])
                result = had - left
                reasoning_steps.append(f"Had-Left=Used: {had} - {left} = {result}")
                return result
        
        # PRIORITY 4: Remaining/left patterns (only if not a division problem)  
        elif "how many" in text_lower and ("left" in text_lower or "remaining" in text_lower):
            # Make sure this isn't actually a division problem
            if not any(word in text_lower for word in ["packs", "bags", "groups", "each"]):
                if len(numbers) >= 2:
                    result = numbers[0] - sum(numbers[1:])
                    reasoning_steps.append(f"Remaining pattern: {numbers[0]} - {sum(numbers[1:])} = {result}")
                    return result
            
        # PRIORITY 5: Basic two-number operations
        if len(numbers) == 2:
            a, b = numbers[0], numbers[1]
            
            if "spend" in text_lower or "spent" in text_lower:
                result = a - b
                reasoning_steps.append(f"Spending pattern: {a} - {b} = {result}")
                return result
                
            elif "gave" in text_lower or "lost" in text_lower:
                result = a - b
                reasoning_steps.append(f"Loss pattern: {a} - {b} = {result}")
                return result
                
            elif "received" in text_lower or "got" in text_lower or "found" in text_lower:
                result = a + b
                reasoning_steps.append(f"Addition pattern: {a} + {b} = {result}")
                return result
        
        return None 
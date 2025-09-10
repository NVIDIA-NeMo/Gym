import re
from typing import Optional


def box_parser(output_str: str) -> Optional[str]:
    """Parse content from \\boxed{...} format commonly used in mathematical responses.
    
    Args:
        output_str: String that may contain boxed content
        
    Returns:
        The content inside the boxed format, or None if not found
    """
    if not isinstance(output_str, str):
        output_str = str(output_str)

    if output_str is None:
        return "N/A"
    try:
        match = re.search(r"\\boxed\{(.*?)\}", output_str)
        parsed_option = match.group(1) if match else "N/A"
        print(output_str[-100:])
        return parsed_option
    except Exception as e:
        print(f"Regex error: {e}")
        return "N/A"



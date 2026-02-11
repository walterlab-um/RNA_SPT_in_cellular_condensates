# Color_palette for plotting
import matplotlib.pyplot as plt
import seaborn as sns

color_palette_dict = {
    'dark_red' : '#9b2226',
    'red' : '#AE2012',
    'dark_orange': '#BB3E03',
    'orange': '#CA6702',
    'light_orange': '#EE9B00',
    'cream': '#E9D8A6',
    'light_blue': '#94D2BD',
    'blue': '#0A9396',
    'dark_blue': '#005F73',
    'black': '#001219',
}

def get_color_palette(color_palette_dict: dict = color_palette_dict,
                      n: int = 4) -> list[str]:
    """
    Get a list of colors from the color palette dictionary.
    
    Parameters:
    - color_palette_dict: Dictionary of color names and their hex codes.
    - n: Number of colors to retrieve (max 10).
    
    Returns:
    - List of hex color codes.
    
    """
    colors = list(color_palette_dict.values())
    
    if n > len(colors):
        raise ValueError(f"Requested number of colors {n} exceeds available colors {len(colors)}.")
    
    if n == 2:
        return [colors[1], colors[6]]  # red, blue
    
    elif n == 3:
        return [colors[1], colors[3], colors[6]]  # red, orange, blue
    
    elif n == 4:
        return [colors[1], colors[4], colors[6], colors[7]]  # dark_red, orange, light orange, blue
    
    elif n == 5:
        return [colors[1], colors[2], colors[3], colors[5], colors[6]]  # dark_red, dark orange, light orange, cream, blue

    else:
        return colors[:n]
# datacore/cleaning/html.py

from bs4 import BeautifulSoup


def clean_html(html_content: str, separator: str = ' ') -> str:
    """
    Clean HTML content to plain text.
    
    Args:
        html_content: Raw HTML
        separator: Separator for text elements
        
    Returns:
        Cleaned plain text
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=separator)
    
    # Collapse whitespace
    cleaned_text = ' '.join(text.split())
    
    return cleaned_text
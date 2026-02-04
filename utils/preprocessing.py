"""
Text Preprocessing Utilities

Provides text cleaning and normalization functions
for document processing pipeline.
"""

import re
import unicodedata
from typing import List, Optional


class TextPreprocessor:
    """
    Text preprocessing and cleaning utilities.
    
    Handles common text cleaning tasks like removing
    extra whitespace, normalizing unicode, and fixing
    common OCR errors.
    """
    
    # Common OCR error mappings
    OCR_CORRECTIONS = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '–': '-',
        '—': '-',
        '…': '...',
        '\u00a0': ' ',  # Non-breaking space
        '\u200b': '',   # Zero-width space
        '\ufeff': '',   # BOM
    }
    
    # Patterns for cleaning
    MULTIPLE_SPACES = re.compile(r' {2,}')
    MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
    HYPHEN_LINEBREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)')
    PAGE_NUMBERS = re.compile(r'\n\s*\d+\s*\n')
    HEADERS_FOOTERS = re.compile(
        r'^(?:page\s+\d+|confidential|draft|\d+\s+of\s+\d+).*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    def __init__(
        self,
        remove_headers_footers: bool = True,
        fix_hyphenation: bool = True,
        normalize_unicode: bool = True,
        fix_ocr_errors: bool = True,
        lowercase: bool = False
    ):
        """
        Initialize the preprocessor.
        
        Args:
            remove_headers_footers: Remove common header/footer patterns
            fix_hyphenation: Fix words split across lines
            normalize_unicode: Normalize unicode characters
            fix_ocr_errors: Apply common OCR error corrections
            lowercase: Convert text to lowercase
        """
        self.remove_headers_footers = remove_headers_footers
        self.fix_hyphenation = fix_hyphenation
        self.normalize_unicode = normalize_unicode
        self.fix_ocr_errors = fix_ocr_errors
        self.lowercase = lowercase
    
    def clean(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix OCR errors
        if self.fix_ocr_errors:
            text = self._fix_ocr_errors(text)
        
        # Normalize unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Fix hyphenation
        if self.fix_hyphenation:
            text = self._fix_hyphenation(text)
        
        # Remove headers/footers
        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)
        
        # Clean whitespace
        text = self._clean_whitespace(text)
        
        # Lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Apply common OCR error corrections."""
        for error, correction in self.OCR_CORRECTIONS.items():
            text = text.replace(error, correction)
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # NFKC normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(
            char for char in text
            if not unicodedata.category(char).startswith('C')
            or char in '\n\t'
        )
        
        return text
    
    def _fix_hyphenation(self, text: str) -> str:
        """Fix words split across lines with hyphens."""
        return self.HYPHEN_LINEBREAK.sub(r'\1\2', text)
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header and footer patterns."""
        # Remove page numbers
        text = self.PAGE_NUMBERS.sub('\n', text)
        
        # Remove header/footer patterns
        text = self.HEADERS_FOOTERS.sub('', text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace issues."""
        # Replace multiple spaces with single space
        text = self.MULTIPLE_SPACES.sub(' ', text)
        
        # Replace multiple newlines with double newline
        text = self.MULTIPLE_NEWLINES.sub('\n\n', text)
        
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence tokenization
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Split text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Remove punctuation and split
        word_pattern = re.compile(r'\b\w+\b')
        return word_pattern.findall(text.lower())
    
    def extract_entities(self, text: str) -> dict:
        """
        Extract simple entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types to lists of entities
        """
        entities = {
            'emails': [],
            'urls': [],
            'phone_numbers': [],
            'dates': [],
            'percentages': [],
            'currencies': []
        }
        
        # Email pattern
        email_pattern = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
        entities['emails'] = email_pattern.findall(text)
        
        # URL pattern
        url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w.-]*'
        )
        entities['urls'] = url_pattern.findall(text)
        
        # Phone pattern (various formats)
        phone_pattern = re.compile(
            r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        )
        entities['phone_numbers'] = phone_pattern.findall(text)
        
        # Date patterns
        date_pattern = re.compile(
            r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
            re.IGNORECASE
        )
        entities['dates'] = date_pattern.findall(text)
        
        # Percentage pattern
        pct_pattern = re.compile(r'\b\d+(?:\.\d+)?%')
        entities['percentages'] = pct_pattern.findall(text)
        
        # Currency pattern
        currency_pattern = re.compile(r'[\$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?')
        entities['currencies'] = currency_pattern.findall(text)
        
        return entities

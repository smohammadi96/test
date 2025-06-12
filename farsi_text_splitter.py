from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional, Any
import re

class FarsiTextSplitter(RecursiveCharacterTextSplitter):
    """A text splitter specifically designed for Persian (Farsi) text.
    
    This splitter handles Persian-specific characteristics including:
    1. Persian punctuation marks
    2. Common Persian text structures
    3. Zero-width non-joiner (ZWNJ) character handling
    4. Smart paragraph and sentence detection
    5. Preservation of semantic units
    """
    
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        keep_separator: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the Farsi text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            keep_separator: Whether to keep separator characters
            **kwargs: Additional arguments to pass to parent class
        """
        # Define Persian-specific separators with priority order
        separators = [
            # Major semantic breaks
            "\n\nماده ",  # New article/section
            "\n\nفصل ",   # New chapter
            "\n\nبخش ",   # New part
            "\n\n",       # Paragraph break
            "\n",         # Line break
            
            # Strong breaks
            ".",          # Period
            "؟",         # Persian question mark
            "!",         # Exclamation mark
            "؛",         # Persian semicolon
            
            # Medium breaks
            ":",         # Colon
            "،",         # Persian comma
            
            # Weak breaks
            " - ",       # Dash with spaces
            "-",         # Dash
            "]",         # Closing bracket
            "[",         # Opening bracket
            ")",         # Closing parenthesis
            "(",         # Opening parenthesis
            "»",         # Persian closing quotation mark
            "«",         # Persian opening quotation mark
            "...",       # Ellipsis
            "…",         # Single character ellipsis
            
            # Persian semantic connectors (only split if necessary)
            " و ",       # "and"
            " یا ",      # "or"
            " اما ",     # "but"
            " که ",      # "that"
            " را ",      # Object marker
            " در ",      # "in"
            " به ",      # "to"
            " از ",      # "from"
            " با ",      # "with"
            " بر ",      # "on"
            
            # Last resort
            " ",         # Space
            ""          # Empty string
        ]
        
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=keep_separator,
            **kwargs
        )
        
    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks.
        
        This method adds Persian-specific preprocessing before splitting:
        1. Normalizes different forms of Persian characters
        2. Handles zero-width non-joiner characters
        3. Cleans up excessive whitespace
        4. Preserves semantic units
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # Normalize Persian text
        text = self._normalize_persian_text(text)
        
        # Get chunks using parent class implementation
        chunks = super().split_text(text)
        
        # Post-process chunks to ensure semantic integrity
        processed_chunks = self._post_process_chunks(chunks)
        
        return processed_chunks
    
    def _normalize_persian_text(self, text: str) -> str:
        """Normalize Persian text for consistent processing.
        
        Performs:
        1. Character normalization (ي to ی, ك to ک, etc.)
        2. Zero-width non-joiner (ZWNJ) handling
        3. Whitespace normalization
        4. Punctuation spacing fixes
        """
        # Character normalization mapping
        chars = {
            'ي': 'ی',
            'ك': 'ک',
            '١': '۱',
            '٢': '۲',
            '٣': '۳',
            '٤': '۴',
            '٥': '۵',
            '٦': '۶',
            '٧': '۷',
            '٨': '۸',
            '٩': '۹',
            '٠': '۰',
            'ۀ': 'ه',
            'ة': 'ه',
        }
        
        # Apply character normalization
        for old, new in chars.items():
            text = text.replace(old, new)
        
        # Handle zero-width non-joiner
        text = re.sub(r'\u200c+', '\u200c', text)  # Replace multiple ZWNJs with single
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([،؛.؟!:»\]\)\}])\s*', r'\1 ', text)  # After marks
        text = re.sub(r'\s*([«\[\(\{])\s*', r' \1', text)  # Before marks
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single
        text = re.sub(r'[\n\r]+', '\n', text)  # Normalize line breaks
        
        return text.strip()
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process chunks to ensure semantic integrity.
        
        1. Merges very small chunks with neighbors
        2. Adjusts chunk boundaries to avoid breaking semantic units
        3. Ensures proper punctuation at chunk boundaries
        4. Preserves complete sentences where possible
        """
        processed = []
        current_chunk = ""
        min_chunk_size = self._chunk_size // 4  # Minimum chunk size threshold
        
        def is_complete_sentence(text: str) -> bool:
            """Check if text ends with a sentence-ending punctuation mark"""
            return bool(re.search(r'[.!?؟]\s*$', text.strip()))
        
        def can_merge_chunks(chunk1: str, chunk2: str) -> bool:
            """Determine if two chunks can be merged based on size and semantics"""
            combined_size = len(chunk1) + len(chunk2)
            if combined_size > self._chunk_size * 1.1:  # Allow 10% overflow
                return False
            return True
        
        # First pass: merge small chunks and fix boundaries
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # If we have a current chunk
            if current_chunk:
                # Try to merge if either chunk is too small or current doesn't end with sentence
                if (len(current_chunk) < min_chunk_size or 
                    len(chunk) < min_chunk_size or 
                    not is_complete_sentence(current_chunk)):
                    if can_merge_chunks(current_chunk, chunk):
                        current_chunk += " " + chunk
                        continue
                
                # If we can't merge, store current and start new
                processed.append(current_chunk)
                current_chunk = chunk
            else:
                current_chunk = chunk
        
        # Add the last chunk
        if current_chunk:
            processed.append(current_chunk)
        
        # Second pass: clean up chunk boundaries
        final_chunks = []
        for chunk in processed:
            # Remove leading/trailing spaces and fix Persian punctuation
            chunk = chunk.strip()
            chunk = re.sub(r'\s*([،؛.؟!])\s*', r'\1 ', chunk)  # Fix spacing around punctuation
            chunk = re.sub(r'\s+', ' ', chunk)  # Normalize spaces
            
            # Ensure chunk starts with a complete word
            chunk = re.sub(r'^\W+', '', chunk)
            
            # Ensure chunk ends with a complete sentence where possible
            if not is_complete_sentence(chunk) and len(final_chunks) > 0:
                # Try to merge with previous chunk if it makes sense
                if can_merge_chunks(final_chunks[-1], chunk):
                    final_chunks[-1] = final_chunks[-1] + " " + chunk
                    continue
            
            final_chunks.append(chunk)
        
        return final_chunks 

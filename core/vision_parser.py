"""
Vision Parser Module

Handles image analysis, chart extraction, and visual content
understanding using vision-language models.
"""

import os
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path

from openai import OpenAI
from PIL import Image
import io


class VisionParser:
    """
    Parser for visual content using vision-language models.
    
    Supports image captioning, chart data extraction,
    and document visual analysis.
    """
    
    CAPTION_PROMPT = """Analyze this image and provide a detailed description.
Include:
1. What type of visual this is (photo, chart, diagram, table, etc.)
2. Key information visible in the image
3. Any text or numbers shown
4. The main purpose or message of the visual

Be thorough but concise."""

    CHART_EXTRACTION_PROMPT = """This image contains a chart or graph.
Please extract the following information:
1. Chart type (bar, line, pie, scatter, etc.)
2. Title and axis labels
3. Data points or values shown
4. Key trends or insights
5. Any legends or annotations

Format the extracted data in a structured way."""

    TABLE_EXTRACTION_PROMPT = """This image contains a table.
Please extract all the data from this table:
1. Column headers
2. Row data
3. Any totals or summary rows
4. Notes or footnotes

Present the data in a clear, structured format."""

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 500
    ):
        """
        Initialize the vision parser.
        
        Args:
            model: Vision model to use
            max_tokens: Maximum tokens for response
        """
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze an image and extract information.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load and encode image
            image_data = self._encode_image(image_path)
            
            # Determine image type and select prompt
            prompt = self.CAPTION_PROMPT
            
            # Call vision model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens
            )
            
            description = response.choices[0].message.content
            
            return {
                "type": "image_analysis",
                "description": description,
                "source": image_path,
                "model": self.model
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
    
    def caption(self, image_element: Any) -> str:
        """
        Generate a caption for an image element.
        
        Args:
            image_element: Image element from document parser
            
        Returns:
            Generated caption string
        """
        try:
            # Handle different image element types
            if hasattr(image_element, 'image'):
                image_data = base64.b64encode(image_element.image).decode('utf-8')
            elif hasattr(image_element, 'metadata') and 'image_path' in image_element.metadata:
                image_data = self._encode_image(image_element.metadata['image_path'])
            else:
                return "[Image could not be processed]"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image concisely in 2-3 sentences, focusing on the key information it conveys."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[Image: {str(e)}]"
    
    def extract_chart_data(self, image_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a chart image.
        
        Args:
            image_path: Path to chart image
            
        Returns:
            Dictionary with extracted chart data
        """
        image_data = self._encode_image(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.CHART_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens
        )
        
        return {
            "type": "chart",
            "extracted_data": response.choices[0].message.content,
            "source": image_path
        }
    
    def extract_table_data(self, image_path: str) -> Dict[str, Any]:
        """
        Extract structured data from a table image.
        
        Args:
            image_path: Path to table image
            
        Returns:
            Dictionary with extracted table data
        """
        image_data = self._encode_image(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.TABLE_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens
        )
        
        return {
            "type": "table",
            "extracted_data": response.choices[0].message.content,
            "source": image_path
        }
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def resize_for_api(
        self, 
        image_path: str, 
        max_size: int = 2048
    ) -> bytes:
        """
        Resize image to fit API constraints.
        
        Args:
            image_path: Path to image
            max_size: Maximum dimension size
            
        Returns:
            Resized image bytes
        """
        with Image.open(image_path) as img:
            # Calculate new size
            ratio = min(max_size / img.width, max_size / img.height)
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Convert to bytes
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()

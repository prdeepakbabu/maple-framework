"""Web fetch tool for retrieving URL content."""

import re
from typing import Any, Dict
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
import html2text

from .base import BaseTool
from ..logging_config import get_logger

logger = get_logger(__name__)


class WebFetchTool(BaseTool):
    """Tool for fetching and extracting text content from URLs."""
    
    def __init__(
        self,
        timeout: int = 15,
        max_content_length: int = 10000,
        allowed_domains: list[str] | None = None
    ):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.allowed_domains = allowed_domains or []
        
        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # No wrapping
        
        # HTTP client
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PRISMBot/1.0; +https://prism.example.com)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        logger.info(
            "web_fetch_tool_initialized",
            timeout=timeout,
            max_content_length=max_content_length
        )
    
    @property
    def name(self) -> str:
        return "web_fetch"
    
    @property
    def description(self) -> str:
        return (
            "Fetch and read the content of a web page. "
            "Use this to read articles, documentation, READMEs, or any public web content. "
            "Returns the main text content in markdown format."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (must start with http:// or https://)"
            }
        }
    
    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate the URL."""
        # Check URL format
        if not url.startswith(("http://", "https://")):
            return False, "URL must start with http:// or https://"
        
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                return False, "Invalid URL format"
            
            # Check allowed domains if configured
            if self.allowed_domains:
                domain = parsed.netloc.lower()
                # Remove www. prefix for comparison
                if domain.startswith("www."):
                    domain = domain[4:]
                
                allowed = any(
                    domain == d.lower() or domain.endswith("." + d.lower())
                    for d in self.allowed_domains
                )
                if not allowed:
                    return False, f"Domain not in allowed list: {domain}"
            
            return True, ""
        except Exception as e:
            return False, f"URL parsing error: {str(e)}"
    
    def _extract_content(self, html: str, url: str) -> str:
        """Extract main content from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", 
                        "aside", "noscript", "iframe", "form"]):
            tag.decompose()
        
        # Try to find main content
        main_content = None
        
        # Look for common content containers
        for selector in ["main", "article", '[role="main"]', ".content", 
                        "#content", ".post-content", ".article-content"]:
            found = soup.select_one(selector)
            if found:
                main_content = found
                break
        
        # Fallback to body
        if main_content is None:
            main_content = soup.body or soup
        
        # Convert to markdown-like text
        text = self.html_converter.handle(str(main_content))
        
        # Clean up
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = text.strip()
        
        # Get page title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Prepend title and URL
        header = f"# {title}\n\nSource: {url}\n\n---\n\n" if title else f"Source: {url}\n\n---\n\n"
        text = header + text
        
        return text
    
    async def execute(self, **kwargs) -> str:
        """Fetch and return content from a URL."""
        url = kwargs.get("url", "").strip()
        
        # Validate URL
        valid, error = self._validate_url(url)
        if not valid:
            logger.warning("web_fetch_invalid_url", url=url, error=error)
            return f"Error: {error}"
        
        logger.info("web_fetch_starting", url=url)
        
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=self.headers
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    # For non-HTML, just return the text
                    if "text/" in content_type or "application/json" in content_type:
                        content = response.text[:self.max_content_length]
                        return f"Source: {url}\n\n---\n\n{content}"
                    else:
                        return f"Error: URL returned non-text content (type: {content_type})"
                
                # Extract content
                html = response.text
                content = self._extract_content(html, url)
                
                # Truncate if needed
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "\n\n... [content truncated]"
                
                logger.info(
                    "web_fetch_success",
                    url=url,
                    content_length=len(content)
                )
                
                return content
                
        except httpx.TimeoutException:
            logger.error("web_fetch_timeout", url=url)
            return f"Error: Request timed out after {self.timeout} seconds"
        
        except httpx.HTTPStatusError as e:
            logger.error("web_fetch_http_error", url=url, status=e.response.status_code)
            return f"Error: HTTP {e.response.status_code} - {e.response.reason_phrase}"
        
        except Exception as e:
            logger.error("web_fetch_error", url=url, error=str(e))
            return f"Error fetching URL: {str(e)}"

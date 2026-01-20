import re
from pathlib import Path
from typing import List, Tuple
from .schema import Document, Section, Chunk

class MarkdownLoader:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = file_path.read_text(encoding="utf-8")
        self.lines = self.content.splitlines()

    def parse(self) -> Document:
        title, authors, metadata = self._extract_metadata()
        sections = self._parse_sections()
        
        doc = Document(
            title=title,
            authors=authors,
            source_path=str(self.file_path),
            sections=sections,
            metadata=metadata
        )
        # Link sections to document
        for section in doc.sections:
            section.document_id = doc.id
            
        return doc

    def _extract_metadata(self) -> Tuple[str, List[str], dict]:
        """
        Heuristic extraction based on the Docling/MinerU output format.
        Assumes Title is the first ## Header that isn't a generic label.
        Assumes Authors are strictly text lines following the Title.
        """
        title = "Unknown Title"
        authors = []
        metadata = {}
        
        found_title = False
        ignored_titles = {"review", "review article", "abstract", "introduction", "open", "article"}
        
        for i, line in enumerate(self.lines[:50]): # Look at first 50 lines
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("## ") and not found_title:
                candidate_title = line.replace("## ", "").strip()
                
                # Check if this is a real title or just a label
                if candidate_title.lower() in ignored_titles or len(candidate_title) < 5:
                    continue
                    
                title = candidate_title
                found_title = True
                
                # Next non-empty lines are likely authors
                # Allow scanning a bit more lines for authors
                for j in range(i + 1, min(i + 10, len(self.lines))):
                    next_line = self.lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Stop if we hit another header or date info
                    if next_line.startswith("##") or next_line.startswith("Received") or next_line.startswith("Accepted"):
                        break
                        
                    # Remove citation numbers if present (e.g., "Name 1")
                    clean_author_line = re.sub(r'\s*\d+(\s*[·,]\s*\d+)*', '', next_line)
                    
                    # Heuristic: Authors usually don't have ":" or dates
                    if ":" in clean_author_line or "202" in clean_author_line:
                        continue
                        
                    # Split by dot or comma if multiple authors on one line
                    parts = re.split(r'[·,]', clean_author_line)
                    new_authors = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
                    authors.extend(new_authors)
                break
                
        return title, authors, metadata

    def _parse_sections(self) -> List[Section]:
        sections = []
        current_section = None
        current_content = []
        
        # Regex to match headers
        header_pattern = re.compile(r'^(#+)\s+(.+)$')
        
        for line in self.lines:
            match = header_pattern.match(line)
            if match:
                # If we have a current section, save it
                if current_section:
                    current_section.content = "\n".join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = Section(
                    title=title,
                    level=level,
                    content="", # Filled later
                    document_id="" # Filled by parent document
                )
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_section:
            current_section.content = "\n".join(current_content).strip()
            sections.append(current_section)
            
        return sections

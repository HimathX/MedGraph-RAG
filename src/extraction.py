import asyncio
from typing import List, Dict
from .schema import Document, Section
from .llm import get_llm, TripletList, Triplet
from .prompts import RELATION_EXTRACTION_PROMPT

class RelationExtractor:
    def __init__(self):
        self.llm = get_llm()
        self.chain = RELATION_EXTRACTION_PROMPT | self.llm.with_structured_output(TripletList)

    async def extract_from_section(self, section: Section) -> List[Triplet]:
        """Extracts triplets from a single section."""
        if not section.content or len(section.content) < 50:
            return []
        
        try:
            # For very long sections, we might want to split further, but for now we truncate or pass as is
            # Gemini Flash has a large context window so passing full section is usually fine
            result = await self.chain.ainvoke({"text": section.content})
            return result.triplets
        except Exception as e:
            print(f"Error extracting from section '{section.title}': {e}")
            return []

    async def process_document(self, doc_data: Dict) -> List[Dict]:
        """
        Process a document dictionary (from staging). 
        Returns a list of dicts representing the triplets enriched with source metadata.
        """
        # Convert dict back to objects if needed, or just iterate structure
        # Assuming doc_data is the JSON dict structure
        
        triplets_data = []
        doc_id = doc_data.get("id")
        doc_title = doc_data.get("title")
        
        sections = doc_data.get("sections", [])
        
        # Limit processing to important sections to save time/cost if needed
        # For now, let's process all sections that are likely to contain knowledge
        
        tasks = []
        for section_data in sections:
            section = Section(**section_data)
            # Skip boring sections
            if section.title.lower() in ["references", "acknowledgements", "declarations"]:
                continue
                
            tasks.append(self.extract_from_section(section))
            
        results = await asyncio.gather(*tasks)
        
        for section_idx, section_triplets in enumerate(results):
            section_title = sections[section_idx]["title"]
            for triplet in section_triplets:
                t_dict = triplet.model_dump()
                t_dict["source_doc_id"] = doc_id
                t_dict["source_doc_title"] = doc_title
                t_dict["source_section"] = section_title
                triplets_data.append(t_dict)
                
        return triplets_data

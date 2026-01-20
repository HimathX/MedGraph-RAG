import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class Triplet(BaseModel):
    head: str = Field(description="The source entity")
    head_type: str = Field(description="Type of the source entity")
    relation: str = Field(description="Relationship type, e.g., REGULATES, CAUSES")
    tail: str = Field(description="The target entity")
    tail_type: str = Field(description="Type of the target entity")

class TripletList(BaseModel):
    triplets: List[Triplet]

def get_llm():
    """Returns an instance of ChatGoogleGenerativeAI."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=api_key
    )

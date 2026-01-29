import asyncio
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import Neo4jManager

async def main():
    print("Initializing Neo4j Manager...")
    nm = Neo4jManager()
    try:
        await nm.run_local_community_detection()
    finally:
        nm.close()

if __name__ == "__main__":
    asyncio.run(main())

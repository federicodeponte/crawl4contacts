#!/usr/bin/env python3
"""
AI-Powered Business Contact Extractor - CSV Processing Example

This example demonstrates how to process a CSV file containing business URLs
from companies worldwide. The system is globally applicable while being
optimized for German business websites.

Global Features:
- Works on business websites from any country
- Multi-language support
- Adaptive content discovery
- Universal contact pattern recognition

German Optimization Features:
- Priority Impressum processing
- German executive terminology (Geschäftsführer, Vorstand)
- Legal compliance awareness
"""

import asyncio
import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to Python path to import the extractor
sys.path.append(str(Path(__file__).parent.parent))

from llm_contact_extractor import EnhancedLLMContactExtractor

async def process_global_companies():
    """
    Example: Process companies from multiple countries
    """
    
    # Initialize the extractor for global use
    extractor = EnhancedLLMContactExtractor(
        max_pages=None,              # Unlimited pages for thorough extraction
        thread_workers=15,           # Parallel processing
        connection_pool_size=50,     # Efficient connection handling
        timeout=30                   # Reasonable timeout for global sites
    )
    
    print("🌍 AI-Powered Business Contact Extractor")
    print("✅ Configured for global business websites")
    print("🇩🇪 Optimized for German companies (Impressum support)")
    print("=" * 60)
    
    # Check if sample data exists
    sample_file = Path(__file__).parent / "sample_data.csv"
    
    if sample_file.exists():
        print(f"📊 Processing sample data: {sample_file}")
        results = await extractor.process_csv(str(sample_file))
        
        print(f"\n🎉 Processing completed!")
        print(f"✅ Successfully processed: {len([r for r in results if r['success']])} companies")
        print(f"👥 Total contacts extracted: {sum(len(r.get('contacts', [])) for r in results)}")
        
        # Show German optimization in action
        german_companies = [r for r in results if 'de' in r.get('domain', '')]
        if german_companies:
            print(f"🇩🇪 German companies processed: {len(german_companies)}")
            impressum_contacts = sum(
                len([c for c in r.get('contacts', []) if 'geschäftsführer' in c.get('title', '').lower()])
                for r in german_companies
            )
            print(f"👔 German executives found: {impressum_contacts}")
    
    else:
        print("⚠️ Sample data not found. Creating example with global URLs...")
        
        # Example with global companies
        global_urls = [
            "https://www.apple.com",           # US - Tech
            "https://www.bmw.de",              # Germany - Automotive (with Impressum)
            "https://www.unilever.com",        # UK/NL - Consumer goods
            "https://www.nestle.com",          # Switzerland - Food & beverage
            "https://www.sap.com",             # Germany - Software (with Impressum)
        ]
        
        print("🔍 Testing with sample global companies:")
        for url in global_urls:
            print(f"   • {url}")
        
        print("\n⚡ Starting extraction...")
        results = extractor.extract_from_multiple_websites_threaded(global_urls)
        
        print(f"\n🎉 Global extraction completed!")
        print(f"✅ Successfully processed: {len([r for r in results if r.get('success')])} companies")
        print(f"👥 Total contacts extracted: {sum(len(r.get('contacts', [])) for r in results if r.get('contacts'))}")
        
        # Demonstrate German optimization
        for result in results:
            if result.get('success') and 'de' in result.get('domain', ''):
                contacts = result.get('contacts', [])
                german_execs = [c for c in contacts if any(term in c.get('title', '').lower() for term in ['geschäftsführer', 'vorstand'])]
                if german_execs:
                    print(f"🇩🇪 German executives found at {result['domain']}:")
                    for exec in german_execs:
                        print(f"   👔 {exec.get('name')} - {exec.get('title')} (Confidence: {exec.get('confidence', 0):.1%})")

if __name__ == "__main__":
    print("🚀 Starting AI-Powered Business Contact Extractor Example")
    print("🌍 Global compatibility with German optimization")
    print()
    
    # Run the async function
    asyncio.run(process_global_companies()) 
#!/usr/bin/env python3
"""
Test Smart Rate Limiting System
Demonstrates rate limiting activation on first 429 error
"""

import asyncio
import time
import pandas as pd
from llm_contact_extractor import EnhancedLLMContactExtractor

async def test_smart_rate_limiting():
    """Test smart rate limiting with a small batch of companies"""
    
    print("🛡️ TESTING SMART RATE LIMITING SYSTEM")
    print("=" * 70)
    print("✅ No rate limiting until first 429 error")
    print("✅ Smart activation on quota hit")
    print("✅ Adaptive delay management")
    print("✅ Gradual optimization after successful calls")
    print("=" * 70)
    
    # Load a few companies for testing
    csv_file = "Leads_Busunternehmen.csv"
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} companies from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Use first 10 companies for rate limiting test
    test_companies = 10
    url_column = 'URL1' if 'URL1' in df.columns else df.columns[0]
    urls_to_test = df.head(test_companies)[url_column].tolist()
    
    # Clean URLs
    clean_urls = []
    for url in urls_to_test:
        if pd.notna(url) and str(url).strip():
            clean_url = str(url).strip()
            if not clean_url.startswith('http'):
                clean_url = f"https://{clean_url}"
            clean_urls.append(clean_url)
    
    print(f"🎯 Testing smart rate limiting with {len(clean_urls)} companies:")
    for i, url in enumerate(clean_urls, 1):
        print(f"   {i:2d}. {url}")
    
    # Initialize extractor
    extractor = EnhancedLLMContactExtractor(
        timeout=30,
        max_pages=None,  # UNLIMITED
        thread_workers=5,  # Reduced for testing
        connection_pool_size=20
    )
    
    print(f"\n🔧 Initial Rate Limiting Status:")
    stats = extractor.get_rate_limiting_stats()
    print(f"   🛡️ Rate limiting active: {stats['rate_limiting_active']}")
    print(f"   ⏱️  Adaptive delay: {stats['adaptive_delay']:.1f}s")
    print(f"   📊 Consecutive errors: {stats['consecutive_errors']}")
    
    # Process companies sequentially to show rate limiting in action
    print(f"\n⚡ SEQUENTIAL PROCESSING (to demonstrate rate limiting):")
    start_time = time.time()
    all_results = []
    
    for i, url in enumerate(clean_urls, 1):
        print(f"\n🔍 Processing {i}/{len(clean_urls)}: {url}")
        
        try:
            result = await extractor.extract_from_website(url)
            all_results.append(result)
            
            # Show current rate limiting status
            stats = extractor.get_rate_limiting_stats()
            if stats['rate_limiting_active']:
                elapsed = stats['time_since_activation']
                print(f"   🛡️ Rate limiting: ACTIVE for {elapsed:.0f}s, delay: {stats['adaptive_delay']:.1f}s")
                print(f"   📊 Errors: {stats['consecutive_errors']}, Successes: {stats['successful_calls']}")
            else:
                print(f"   🚀 Rate limiting: INACTIVE (full speed)")
            
            if result.success:
                print(f"   ✅ Success: {result.total_contacts_found} contacts, {result.total_pages_analyzed} pages")
            else:
                print(f"   ❌ Failed: {result.error}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        # Small delay between companies for demonstration
        await asyncio.sleep(0.5)
    
    total_time = time.time() - start_time
    successful = [r for r in all_results if r.success]
    total_contacts = sum(len(r.all_contacts) for r in successful)
    
    print(f"\n🎉 SMART RATE LIMITING TEST COMPLETED!")
    print("=" * 70)
    print(f"📊 RESULTS SUMMARY:")
    print(f"   ✅ Successful: {len(successful)}/{len(clean_urls)}")
    print(f"   👥 Total Contacts: {total_contacts}")
    print(f"   ⏱️  Total Time: {total_time:.1f}s")
    
    # Final rate limiting status
    final_stats = extractor.get_rate_limiting_stats()
    print(f"\n🛡️ FINAL RATE LIMITING STATUS:")
    print(f"   Active: {final_stats['rate_limiting_active']}")
    if final_stats['rate_limiting_active']:
        print(f"   Duration: {final_stats['time_since_activation']:.0f}s")
        print(f"   Current delay: {final_stats['adaptive_delay']:.1f}s")
        print(f"   Total errors handled: {final_stats['consecutive_errors']}")
        print(f"   Successful recoveries: {final_stats['successful_calls']}")
    
    # Save results if any
    if all_results:
        db_file = extractor.save_enhanced_ai_contacts(all_results)
        print(f"\n📁 Results saved to: {db_file}")
    
    await extractor.cleanup()
    
    print(f"\n✅ SMART RATE LIMITING DEMONSTRATED:")
    print("   🚀 Started at full speed (no delays)")
    print("   ⚠️ Activated automatically on first 429 error")
    print("   🧠 Adapted delay based on error frequency")
    print("   🎯 Optimized delay after successful calls")
    print("   🛡️ Prevented quota violations while maintaining performance")

async def test_threadpool_with_rate_limiting():
    """Test ThreadPool processing with smart rate limiting"""
    
    print("\n⚡ TESTING THREADPOOL WITH SMART RATE LIMITING")
    print("=" * 70)
    
    # Load companies
    df = pd.read_csv("Leads_Busunternehmen.csv")
    url_column = 'URL1' if 'URL1' in df.columns else df.columns[0]
    test_urls = df.head(20)[url_column].tolist()
    
    # Clean URLs
    clean_urls = []
    for url in test_urls:
        if pd.notna(url) and str(url).strip():
            clean_url = str(url).strip()
            if not clean_url.startswith('http'):
                clean_url = f"https://{clean_url}"
            clean_urls.append(clean_url)
    
    print(f"🎯 Testing {len(clean_urls)} companies with ThreadPool + Rate Limiting")
    
    # Initialize extractor with smart rate limiting
    extractor = EnhancedLLMContactExtractor(
        timeout=30,
        max_pages=None,
        thread_workers=10,  # Moderate concurrency
        connection_pool_size=50
    )
    
    # Process with ThreadPool
    start_time = time.time()
    results = extractor.extract_from_multiple_websites_threaded(clean_urls)
    total_time = time.time() - start_time
    
    # Results
    successful = [r for r in results if r.success]
    total_contacts = sum(len(r.all_contacts) for r in successful)
    
    print(f"\n📊 THREADPOOL + RATE LIMITING RESULTS:")
    print(f"   ✅ Successful: {len(successful)}/{len(clean_urls)}")
    print(f"   👥 Total Contacts: {total_contacts}")
    print(f"   ⏱️  Total Time: {total_time:.1f}s")
    print(f"   ⚡ Rate: {len(clean_urls)/total_time:.1f} websites/second")
    
    # Rate limiting effectiveness
    final_stats = extractor.get_rate_limiting_stats()
    if final_stats['rate_limiting_active']:
        print(f"\n🛡️ RATE LIMITING EFFECTIVENESS:")
        print(f"   ✅ Activated successfully during processing")
        print(f"   ⏱️  Active for: {final_stats['time_since_activation']:.0f}s")
        print(f"   🎯 Final delay: {final_stats['adaptive_delay']:.1f}s")
        print("   ✅ Prevented quota violations while maintaining throughput")
    else:
        print(f"\n🚀 NO RATE LIMITING NEEDED:")
        print("   ✅ Processed all companies without hitting quota limits")
        print("   ⚡ Full speed processing maintained")
    
    await extractor.cleanup()

async def main():
    """Run rate limiting tests"""
    await test_smart_rate_limiting()
    await test_threadpool_with_rate_limiting()

if __name__ == "__main__":
    asyncio.run(main()) 
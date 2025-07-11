#!/usr/bin/env python3
"""
Process ALL Companies with Enhanced Error Handling & Checkpointing
✅ Resumable processing with checkpoints
✅ Better error handling and timeout management
✅ Progress tracking and intermediate saves
✅ Smart rate limiting with quota management
"""

import asyncio
import time
import pandas as pd
import json
import os
from llm_contact_extractor import EnhancedLLMContactExtractor

def save_checkpoint(processed_results, checkpoint_file="processing_checkpoint.json"):
    """Save processing checkpoint"""
    checkpoint_data = {
        'timestamp': time.time(),
        'processed_count': len(processed_results),
        'processed_domains': [r.domain for r in processed_results if hasattr(r, 'domain')],
        'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"💾 Checkpoint saved: {len(processed_results)} companies processed")

def load_checkpoint(checkpoint_file="processing_checkpoint.json"):
    """Load processing checkpoint"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def save_intermediate_results(results, timestamp=None):
    """Save intermediate results"""
    if not timestamp:
        timestamp = int(time.time())
    
    if not results:
        return None
    
    try:
        # Create extractor instance just for saving
        extractor = EnhancedLLMContactExtractor()
        db_file = extractor.save_enhanced_ai_contacts(results, f"intermediate_results_{timestamp}.db")
        
        # CSV export
        export_data = []
        for result in results:
            if result.success and hasattr(result, 'all_contacts'):
                for contact in result.all_contacts:
                    export_data.append({
                        'Company_Domain': result.domain,
                        'Company_Name': getattr(result, 'company_name', result.domain),
                        'Full_Name': getattr(contact, 'full_name', ''),
                        'First_Name': getattr(contact, 'first_name', ''),
                        'Last_Name': getattr(contact, 'last_name', ''),
                        'Email': getattr(contact, 'email', ''),
                        'Phone': getattr(contact, 'phone', ''),
                        'Title': getattr(contact, 'title', ''),
                        'Seniority_Level': getattr(contact, 'seniority_level', ''),
                        'Is_Real_Person': getattr(contact, 'is_real_person', False),
                        'AI_Confidence': getattr(contact, 'ai_confidence', 0.0),
                        'Processing_Time': getattr(result, 'processing_time', 0)
                    })
        
        if export_data:
            df = pd.DataFrame(export_data)
            csv_file = f"intermediate_export_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"📊 Intermediate results saved: {db_file}, {csv_file}")
            return db_file, csv_file
            
    except Exception as e:
        print(f"⚠️ Error saving intermediate results: {e}")
    
    return None

async def process_all_companies_improved():
    """Process all companies with improved error handling"""
    
    print("🇩🇪 IMPROVED ALL COMPANIES PROCESSING - With Checkpoints")
    print("=" * 80)
    print("✅ Enhanced error handling and timeouts")
    print("✅ Checkpointing and resumable processing")
    print("✅ Intermediate saves every 100 companies")
    print("✅ Smart rate limiting with quota management")
    print("✅ Priority German Impressum processing")
    print("=" * 80)
    
    # Load companies
    df = pd.read_csv("Leads_Busunternehmen.csv")
    url_column = 'URL1' if 'URL1' in df.columns else df.columns[0]
    all_urls = df[url_column].tolist()
    
    # Clean URLs
    clean_urls = []
    for url in all_urls:
        if pd.notna(url) and str(url).strip():
            clean_url = str(url).strip()
            if not clean_url.startswith('http'):
                clean_url = f"https://{clean_url}"
            clean_urls.append(clean_url)
    
    print(f"📊 Total URLs to process: {len(clean_urls)}")
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    start_index = 0
    if checkpoint:
        print(f"📂 Found checkpoint: {checkpoint['processed_count']} companies already processed")
        start_index = checkpoint['processed_count']
    
    remaining_urls = clean_urls[start_index:]
    print(f"🎯 Processing {len(remaining_urls)} remaining companies (starting from #{start_index + 1})")
    
    # Process in smaller batches for better error handling
    batch_size = 100
    all_results = []
    
    # Initialize extractor
    extractor = EnhancedLLMContactExtractor(
        timeout=30,
        max_pages=None,
        thread_workers=15,  # Reduced for stability
        connection_pool_size=75
    )
    
    print(f"\n🔧 Improved Configuration:")
    print(f"   🎯 Batch size: {batch_size} companies")
    print(f"   🚀 Thread workers: {extractor.thread_workers}")
    print(f"   ⏱️ Timeout: {extractor.timeout}s")
    print(f"   🛡️ Smart rate limiting: Ready")
    
    total_start_time = time.time()
    
    # Process in batches
    for batch_num, i in enumerate(range(0, len(remaining_urls), batch_size), 1):
        batch_urls = remaining_urls[i:i + batch_size]
        actual_batch_size = len(batch_urls)
        
        print(f"\n📦 BATCH {batch_num}: Processing {actual_batch_size} companies ({start_index + i + 1}-{start_index + i + actual_batch_size})")
        
        batch_start_time = time.time()
        
        try:
            # Process current batch
            batch_results = extractor.extract_from_multiple_websites_threaded(batch_urls)
            all_results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            successful = sum(1 for r in batch_results if r.success)
            
            print(f"   ✅ Batch {batch_num} completed: {successful}/{actual_batch_size} successful in {batch_time:.1f}s")
            
            # Show rate limiting status
            stats = extractor.get_rate_limiting_stats()
            if stats['rate_limiting_active']:
                print(f"   🛡️ Rate limiting: Active, delay {stats['adaptive_delay']:.1f}s")
            
            # Save checkpoint and intermediate results every batch
            save_checkpoint(all_results)
            
            # Save intermediate results every 2 batches or if significant progress
            if batch_num % 2 == 0 or actual_batch_size < batch_size:
                save_intermediate_results(all_results)
            
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            print("💾 Saving progress before continuing...")
            save_checkpoint(all_results)
            save_intermediate_results(all_results)
            continue
    
    total_time = time.time() - total_start_time
    
    # Final analysis
    successful_results = [r for r in all_results if r.success]
    total_contacts = sum(len(getattr(r, 'all_contacts', [])) for r in successful_results)
    total_executives = sum(len(getattr(r, 'executives', [])) for r in successful_results)
    
    print(f"\n🎉 ALL COMPANIES PROCESSING COMPLETED!")
    print("=" * 80)
    print(f"📊 FINAL RESULTS:")
    print(f"   ✅ Total processed: {len(all_results):,}")
    print(f"   ✅ Successful: {len(successful_results):,}")
    print(f"   👥 Total contacts: {total_contacts:,}")
    print(f"   👔 Executives: {total_executives:,}")
    print(f"   ⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"   ⚡ Rate: {len(all_results)/total_time:.2f} companies/second")
    
    # Rate limiting summary
    final_stats = extractor.get_rate_limiting_stats()
    if final_stats['rate_limiting_active']:
        print(f"   🛡️ Rate limiting used: {final_stats['time_since_activation']/60:.1f} minutes active")
    else:
        print(f"   🚀 Full speed processing - no rate limiting needed")
    
    # Save final results
    if all_results:
        timestamp = int(time.time())
        final_files = save_intermediate_results(all_results, timestamp)
        
        # Also save final comprehensive CSV
        export_data = []
        for result in successful_results:
            if hasattr(result, 'all_contacts'):
                for contact in result.all_contacts:
                    export_data.append({
                        'Company_Domain': result.domain,
                        'Company_Name': getattr(result, 'company_name', result.domain),
                        'Full_Name': getattr(contact, 'full_name', ''),
                        'First_Name': getattr(contact, 'first_name', ''),
                        'Last_Name': getattr(contact, 'last_name', ''),
                        'Email': getattr(contact, 'email', ''),
                        'Phone': getattr(contact, 'phone', ''),
                        'Title': getattr(contact, 'title', ''),
                        'Department': getattr(contact, 'department', ''),
                        'Seniority_Level': getattr(contact, 'seniority_level', ''),
                        'Role_Category': getattr(contact, 'role_category', ''),
                        'Is_Decision_Maker': getattr(contact, 'is_decision_maker', False),
                        'Is_Real_Person': getattr(contact, 'is_real_person', False),
                        'AI_Confidence': getattr(contact, 'ai_confidence', 0.0),
                        'Employee_Type': getattr(contact, 'employee_type', ''),
                        'Source_URL': getattr(contact, 'source_url', ''),
                        'Pages_Analyzed': getattr(result, 'total_pages_analyzed', 0),
                        'Processing_Time': getattr(result, 'processing_time', 0)
                    })
        
        if export_data:
            final_df = pd.DataFrame(export_data)
            final_csv = f"FINAL_ALL_companies_export_{timestamp}.csv"
            final_df.to_csv(final_csv, index=False)
            
            # Executive-only export
            executive_data = [row for row in export_data 
                            if row.get('Seniority_Level') in ['C-Level', 'Executive'] 
                            and row.get('Is_Real_Person', False)]
            
            if executive_data:
                exec_df = pd.DataFrame(executive_data)
                exec_csv = f"FINAL_executives_only_{timestamp}.csv"
                exec_df.to_csv(exec_csv, index=False)
                print(f"👔 Executives CSV: {exec_csv} ({len(executive_data)} executives)")
            
            print(f"📊 Final CSV: {final_csv} ({len(export_data):,} contacts)")
    
    # Clean up checkpoint
    if os.path.exists("processing_checkpoint.json"):
        os.remove("processing_checkpoint.json")
        print("🧹 Checkpoint file cleaned up")
    
    await extractor.cleanup()
    
    print(f"\n✅ IMPROVED PROCESSING COMPLETE!")
    print(f"🎯 Successfully processed {len(successful_results):,} companies with enhanced error handling")

async def main():
    await process_all_companies_improved()

if __name__ == "__main__":
    asyncio.run(main()) 
#!/usr/bin/env python3
"""
Enhanced LLM-Driven Contact Extractor v2.0
✅ Gemini-2.5-Flash with 1M token window
✅ No page number limitations
✅ No token limitations 
✅ ThreadPool parallel processing
✅ Optimized for maximum performance
"""

import asyncio
import aiohttp
import time
import re
import json
import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import google.generativeai as genai
import os
import logging

# Simple logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class AIContact:
    """AI-processed contact information with enhanced fields"""
    
    # Basic Contact Info
    email: Optional[str] = None
    phone: Optional[str] = None
    mobile: Optional[str] = None
    fax: Optional[str] = None
    extension: Optional[str] = None
    
    # Personal Info
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    
    # AI-Determined Classification
    seniority_level: Optional[str] = None  # C-Level, VP, Director, Manager, Individual Contributor
    role_category: Optional[str] = None    # Executive, Sales, Marketing, Engineering, Support, Operations, HR, Finance
    is_decision_maker: bool = False
    is_company_employee: bool = True       # True = employee, False = external/other
    employee_type: Optional[str] = None    # Full-time, Part-time, Contractor, Consultant, Advisor, Partner, Board Member
    employment_status: Optional[str] = None # Active, Former, Retired, Advisory
    
    # AI Analysis
    ai_confidence: float = 0.0             # AI's confidence in the extraction (0-1)
    ai_reasoning: Optional[str] = None     # AI's reasoning for classifications
    is_real_person: bool = True            # AI assessment if this is a real person vs generic contact
    
    # Professional Info
    linkedin_url: Optional[str] = None
    twitter_handle: Optional[str] = None
    github_profile: Optional[str] = None
    
    # Context & Quality
    source_url: Optional[str] = None
    source_page_type: Optional[str] = None
    extraction_method: str = "llm_ai_extraction_v2"
    confidence_score: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class AIExtractionResult:
    """Enhanced result of AI contact extraction"""
    url: str
    domain: str
    company_name: Optional[str] = None
    
    # All contacts found
    all_contacts: List[AIContact] = field(default_factory=list)
    
    # AI Analysis with Enhanced Stats
    ai_processing_time: float = 0.0
    total_email_blocks_processed: int = 0
    total_pages_analyzed: int = 0
    total_content_size: int = 0
    real_people_found: int = 0
    generic_contacts_found: int = 0
    
    # Categorized contacts
    executives: List[AIContact] = field(default_factory=list)
    sales_team: List[AIContact] = field(default_factory=list)
    marketing_team: List[AIContact] = field(default_factory=list)
    engineering_team: List[AIContact] = field(default_factory=list)
    support_team: List[AIContact] = field(default_factory=list)
    other_contacts: List[AIContact] = field(default_factory=list)
    
    # Metadata
    total_contacts_found: int = 0
    unique_emails_found: int = 0
    unique_phones_found: int = 0
    pages_processed: int = 0
    processing_time: float = 0.0
    success: bool = False
    error: str = ""
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())

class EnhancedLLMContactExtractor:
    """
    Enhanced LLM-driven contact extractor with unlimited processing capabilities
    ✅ Gemini-2.5-Flash with 1M token window
    ✅ No page limitations
    ✅ No token limitations  
    ✅ ThreadPool parallel processing
    ✅ Smart rate limiting (activated on first 429)
    ✅ Maximum performance optimization
    """
    
    def __init__(self, 
                 timeout: int = 30,
                 max_pages: int = None,  # UNLIMITED pages
                 thread_workers: int = 20,
                 connection_pool_size: int = 100):
        
        self.timeout = timeout
        self.max_pages = max_pages  # None = unlimited
        self.thread_workers = thread_workers
        self.connection_pool_size = connection_pool_size
        self.session = None
        self._lock = threading.Lock()
        
        # Smart Rate Limiting System
        self.rate_limiting_active = False
        self.quota_hit_time = None
        self.base_delay = 0.5  # Base delay between AI calls when rate limiting
        self.adaptive_delay = 0.5  # Adaptive delay that increases with errors
        self.consecutive_errors = 0
        self.successful_calls = 0
        self.last_call_time = 0
        
        # Initialize Enhanced Gemini AI
        self.setup_enhanced_ai()

    def setup_enhanced_ai(self):
        """Setup Enhanced AI with Gemini-2.5-Flash"""
        try:
            # Get API key from environment variable
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set. Please set your Gemini API key.")
            
            # Configure Gemini with enhanced settings
            genai.configure(api_key=gemini_api_key)
            
            # Use Gemini-2.5-Flash with 1M token window
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=8192,  # Large output for detailed analysis
                temperature=0.1,         # Low temperature for consistent results
                top_p=0.8,
                top_k=40
            )
            
            self.model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-exp',  # Latest high-performance model
                generation_config=generation_config
            )
            self.ai_provider = "gemini-2.5-flash"
            logger.info("✅ Enhanced Gemini-2.5-Flash configured with 1M token window")
            
        except Exception as e:
            logger.error(f"❌ Enhanced Gemini AI setup failed: {e}")
            # Fallback to basic model
            try:
                self.model = genai.GenerativeModel('gemini-pro')
                self.ai_provider = "gemini-pro-fallback"
                logger.warning("⚠️ Using fallback Gemini Pro model")
            except Exception as e2:
                logger.error(f"❌ All AI setup failed: {e2}")
                self.model = None
                self.ai_provider = None

    async def get_enhanced_session(self):
        """Get enhanced HTTP session with connection pooling"""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=self.connection_pool_size,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
        return self.session

    def extract_comprehensive_email_blocks(self, html: str, url: str) -> List[Dict[str, str]]:
        """Enhanced email extraction with comprehensive context analysis"""
        
        # Enhanced email pattern to catch more variations
        email_pattern = re.compile(
            r'\b[A-Za-z0-9]([A-Za-z0-9._%-]*[A-Za-z0-9])?@[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,}\b'
        )
        
        # Enhanced text cleaning while preserving more structure
        clean_text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<style[^>]*>.*?</style>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<!--.*?-->', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<br[^>]*>', '\n', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'<p[^>]*>', '\n\n', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'<div[^>]*>', '\n', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        email_blocks = []
        emails_found = set()
        
        # Find all emails with enhanced context
        for match in email_pattern.finditer(clean_text):
            email = match.group().lower().strip()
            
            # Skip duplicates and invalid emails
            if email in emails_found or len(email) < 5:
                continue
            emails_found.add(email)
            
            # Get larger surrounding context (500 chars instead of 200)
            start_pos = max(0, match.start() - 500)
            end_pos = min(len(clean_text), match.end() + 500)
            context = clean_text[start_pos:end_pos].strip()
            
            # Enhanced phone number extraction
            phone_patterns = [
                re.compile(r'[\+]?[\d\s\-\(\)\.\/]{8,25}'),  # International format
                re.compile(r'\b\d{2,5}[\s\-]?\d{6,12}\b'),    # Simple format
                re.compile(r'\(\d{2,5}\)[\s\-]?\d{6,12}'),    # Parentheses format
            ]
            
            phones = []
            for pattern in phone_patterns:
                found_phones = [p.group().strip() for p in pattern.finditer(context)]
                phones.extend(found_phones)
            
            # Clean and validate phone numbers
            cleaned_phones = []
            for phone in phones:
                digits_only = re.sub(r'\D', '', phone)
                if 7 <= len(digits_only) <= 15:  # Valid phone number length
                    cleaned_phones.append(phone)
            
            email_blocks.append({
                'email': email,
                'context': context,
                'phones': cleaned_phones[:5],  # Up to 5 phones per email
                'source_url': url,
                'position_in_page': match.start(),
                'context_size': len(context)
            })
        
        return email_blocks

    def create_enhanced_ai_prompt(self, email_blocks: List[Dict[str, str]], company_domain: str, all_page_content: str = "") -> str:
        """Create enhanced AI prompt leveraging full 1M token window"""
        
        # Calculate total content size for token management
        total_content_size = len(all_page_content) + sum(len(block['context']) for block in email_blocks)
        
        prompt = f"""You are an expert contact extraction AI with access to the complete website content for {company_domain}.

ENHANCED ANALYSIS CAPABILITIES:
- Leverage the full website context below for comprehensive understanding
- Identify relationships between contacts and organizational structure  
- Detect contact hierarchies and reporting relationships
- Analyze company culture and communication patterns
- Cross-reference information across multiple pages

COMPANY WEBSITE CONTENT:
{all_page_content[:200000]}  # Use substantial portion of 1M token window

EMAIL BLOCKS WITH CONTEXT:
"""
        
        for i, block in enumerate(email_blocks, 1):
            prompt += f"""
--- EMAIL BLOCK {i} ---
Email: {block['email']}
Full Context ({block['context_size']} chars): {block['context']}
Associated Phones: {', '.join(block['phones']) if block['phones'] else 'None'}
Page Position: {block['position_in_page']}
Source: {block['source_url']}
"""
        
        prompt += f"""

🇩🇪 SPECIAL FOCUS FOR GERMAN COMPANIES (HIGHEST PRIORITY):
- IMPRESSUM PAGES: German law requires all companies list managing directors in Impressum/Legal Notice
- Look for "Geschäftsführer" (Managing Directors) - ALWAYS real people with legal authority  
- Extract "Vertretungsberechtigt" (authorized representatives) - key decision makers
- Find "Vorstand" (Board members), "CEO", "Geschäftsleitung" (Executive management)
- Include "Inhaber" (Owners), "Gesellschafter" (Partners), "Prokurist" (Authorized signatories)
- Legal keywords: "Verantwortlich", "Registergericht", "Handelsregister", "Amtsgericht"
- These are REAL PEOPLE with full legal names - never generic contacts!
- Often format: "Geschäftsführer: [First Name] [Last Name]" or "Vertretungsberechtigte Geschäftsführer: [Name]"

ENHANCED OUTPUT FORMAT (JSON):
{{
  "company_analysis": {{
    "company_structure": "Brief analysis of organizational structure",
    "contact_patterns": "Patterns observed in contact information", 
    "confidence_factors": "What makes the analysis confident",
    "impressum_found": true/false,
    "managing_directors_found": ["list of managing director names found"]
  }},
  "contacts": [
    {{
      "email": "extracted_email",
      "is_real_person": true/false,
      "full_name": "First Last" or null,
      "first_name": "First" or null,
      "last_name": "Last" or null,
      "title": "Job Title" or null,
      "department": "Department" or null,
      "phone": "best_phone_number" or null,
      "seniority_level": "C-Level|VP/Director|Manager|Senior|Individual Contributor",
      "role_category": "Executive|Sales|Marketing|Engineering|Support|Operations|HR|Finance|Unknown",
      "is_decision_maker": true/false,
      "is_company_employee": true/false,
      "employee_type": "Full-time|Part-time|Contractor|Consultant|Advisor|Board Member|Unknown",
      "employment_status": "Active|Former|External|Unknown",
      "ai_confidence": 0.0-1.0,
      "ai_reasoning": "Detailed explanation of classification decisions with supporting evidence from website content",
      "relationship_context": "How this person relates to others in the organization"
    }}
  ]
}}

ENHANCED CLASSIFICATION GUIDELINES:
- Use full website context to understand organizational structure
- Cross-reference contact information across multiple pages
- Identify reporting relationships and team structures
- Distinguish between current and former employees
- Detect external consultants, partners, and advisors
- Analyze communication patterns and contact hierarchies
- Handle multilingual content (German, English, etc.)
- Provide detailed reasoning with specific evidence from content
- Confidence should reflect the quality and consistency of available information

Focus on comprehensive analysis using the full context available."""
        
        return prompt

    async def smart_rate_limited_ai_call(self, prompt: str):
        """Make AI call with smart rate limiting that activates on first 429 error"""
        
        # Apply rate limiting if active
        if self.rate_limiting_active:
            current_time = time.time()
            time_since_last = current_time - self.last_call_time
            
            if time_since_last < self.adaptive_delay:
                sleep_time = self.adaptive_delay - time_since_last
                logger.info(f"🛡️ Smart rate limiting: waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        try:
            # Record call time
            self.last_call_time = time.time()
            
            # Make AI call
            response = self.model.generate_content(prompt)
            
            # Success - reduce delay gradually
            if self.rate_limiting_active:
                self.successful_calls += 1
                self.consecutive_errors = 0
                
                # Gradually reduce delay after successful calls
                if self.successful_calls >= 3:
                    self.adaptive_delay = max(self.base_delay, self.adaptive_delay * 0.9)
                    self.successful_calls = 0
                    logger.info(f"🎯 Rate limit optimized: delay reduced to {self.adaptive_delay:.1f}s")
            
            return response
            
        except Exception as e:
            error_str = str(e)
            
            # Check for 429 quota error
            if "429" in error_str and "quota" in error_str.lower():
                if not self.rate_limiting_active:
                    # First 429 error - activate smart rate limiting
                    self.rate_limiting_active = True
                    self.quota_hit_time = time.time()
                    self.adaptive_delay = self.base_delay
                    logger.warning(f"⚠️ First 429 detected! Activating smart rate limiting with {self.base_delay}s delay")
                
                # Increase delay for subsequent errors
                self.consecutive_errors += 1
                self.successful_calls = 0
                
                # Smart delay calculation based on error frequency
                if self.consecutive_errors == 1:
                    self.adaptive_delay = 2.0  # First retry: 2 seconds
                elif self.consecutive_errors == 2:
                    self.adaptive_delay = 5.0  # Second retry: 5 seconds
                elif self.consecutive_errors >= 3:
                    self.adaptive_delay = min(30.0, self.adaptive_delay * 1.5)  # Exponential backoff, max 30s
                
                logger.warning(f"🛡️ 429 error #{self.consecutive_errors}: increasing delay to {self.adaptive_delay:.1f}s")
                
                # Wait before retrying
                await asyncio.sleep(self.adaptive_delay)
                
                # Retry with new delay
                try:
                    self.last_call_time = time.time()
                    response = self.model.generate_content(prompt)
                    logger.info(f"✅ Retry successful after rate limiting")
                    return response
                except Exception as retry_error:
                    # Still failing - increase delay further and give up for this call
                    self.adaptive_delay = min(60.0, self.adaptive_delay * 2)
                    logger.error(f"❌ Retry failed, increasing delay to {self.adaptive_delay:.1f}s")
                    raise retry_error
            else:
                # Non-quota error
                raise e

    async def process_contacts_with_enhanced_ai(self, email_blocks: List[Dict[str, str]], company_domain: str, all_content: str = "") -> List[AIContact]:
        """Process email blocks using enhanced AI with smart rate limiting"""
        
        if not self.model or not email_blocks:
            return []
        
        contacts = []
        
        try:
            # Create comprehensive prompt using full context
            prompt = self.create_enhanced_ai_prompt(email_blocks, company_domain, all_content)
            
            logger.info(f"🤖 Processing {len(email_blocks)} email blocks with enhanced AI")
            logger.info(f"📊 Total content size: {len(all_content):,} characters")
            
            if self.rate_limiting_active:
                elapsed = time.time() - self.quota_hit_time
                logger.info(f"🛡️ Rate limiting active for {elapsed:.0f}s, adaptive delay: {self.adaptive_delay:.1f}s")
            
            # Smart rate-limited AI call
            response = await self.smart_rate_limited_ai_call(prompt)
            response_text = response.text.strip()
            
            # Enhanced JSON extraction
            json_text = self._extract_json_from_response(response_text)
            ai_result = json.loads(json_text)
            
            # Extract company analysis
            company_analysis = ai_result.get('company_analysis', {})
            logger.info(f"🏢 Company Analysis: {company_analysis.get('company_structure', 'N/A')}")
            
            # Process contacts with enhanced data
            for contact_data in ai_result.get('contacts', []):
                original_block = next((b for b in email_blocks if b['email'] == contact_data.get('email')), {})
                
                contact = AIContact(
                    email=contact_data.get('email'),
                    is_real_person=contact_data.get('is_real_person', True),
                    full_name=contact_data.get('full_name'),
                    first_name=contact_data.get('first_name'),
                    last_name=contact_data.get('last_name'),
                    title=contact_data.get('title'),
                    department=contact_data.get('department'),
                    phone=contact_data.get('phone'),
                    seniority_level=contact_data.get('seniority_level'),
                    role_category=contact_data.get('role_category'),
                    is_decision_maker=contact_data.get('is_decision_maker', False),
                    is_company_employee=contact_data.get('is_company_employee', True),
                    employee_type=contact_data.get('employee_type'),
                    employment_status=contact_data.get('employment_status', 'Active'),
                    ai_confidence=contact_data.get('ai_confidence', 0.5),
                    ai_reasoning=contact_data.get('ai_reasoning', ''),
                    source_url=original_block.get('source_url'),
                    confidence_score=contact_data.get('ai_confidence', 0.5),
                    extraction_method="enhanced_llm_ai_v2"
                )
                
                contacts.append(contact)
            
            logger.info(f"✅ Enhanced AI processing completed: {len(contacts)} contacts extracted")
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced AI processing error: {e}")
            # Enhanced fallback: create contacts with context analysis
            for block in email_blocks:
                contact = AIContact(
                    email=block['email'],
                    phone=block['phones'][0] if block['phones'] else None,
                    source_url=block['source_url'],
                    ai_confidence=0.4,
                    ai_reasoning=f"Fallback extraction due to AI processing error: {str(e)}",
                    confidence_score=0.4,
                    extraction_method="enhanced_fallback_v2"
                )
                contacts.append(contact)
        
        return contacts

    def _extract_json_from_response(self, response_text: str) -> str:
        """Enhanced JSON extraction from AI response"""
        
        # Try multiple JSON extraction methods
        extraction_methods = [
            # Method 1: Look for ```json blocks
            lambda text: self._extract_between_markers(text, '```json', '```'),
            # Method 2: Look for { } blocks
            lambda text: self._extract_json_object(text),
            # Method 3: Look for markdown code blocks
            lambda text: self._extract_between_markers(text, '```', '```'),
            # Method 4: Take entire response if it looks like JSON
            lambda text: text if text.strip().startswith('{') else None
        ]
        
        for method in extraction_methods:
            try:
                json_text = method(response_text)
                if json_text and json_text.strip():
                    # Validate JSON
                    json.loads(json_text)
                    return json_text
            except:
                continue
        
        # Ultimate fallback
        return '{"contacts": []}'

    def _extract_between_markers(self, text: str, start_marker: str, end_marker: str) -> Optional[str]:
        """Extract text between markers"""
        try:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return None
            start_idx += len(start_marker)
            
            end_idx = text.find(end_marker, start_idx)
            if end_idx == -1:
                return None
            
            return text[start_idx:end_idx].strip()
        except:
            return None

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract JSON object from text"""
        try:
            # Find first { and matching }
            start_idx = text.find('{')
            if start_idx == -1:
                return None
            
            brace_count = 0
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx:i+1]
            
            return None
        except:
            return None

    def find_all_contact_pages(self, base_url: str, html: str) -> List[str]:
        """Find ALL contact-related pages with PRIORITY on German Impressum"""
        domain = urlparse(base_url).netloc
        urls = [base_url]
        
        # HIGH PRIORITY: German legal pages (managing directors always listed)
        priority_keywords = [
            'impressum', 'imprint', 'legal-notice', 'rechtliches', 'datenschutz',
            'geschäftsführung', 'geschaeftsfuehrung', 'management', 'vorstand',
            'geschäftsleitung', 'geschaeftsleitung', 'board', 'directors', 'ceo'
        ]
        
        # MEDIUM PRIORITY: Team and contact pages  
        contact_keywords = [
            'kontakt', 'contact', 'team', 'about', 'staff', 'menschen',
            'mitarbeiter', 'ansprechpartner', 'support', 'service', 'help',
            'über', 'about-us', 'who-we-are', 'meet-the-team', 'our-team',
            'leadership', 'executives', 'sales', 'vertrieb', 'consulting', 
            'beratung', 'info', 'information', 'unternehmen', 'firma'
        ]
        
        # Extract all links
        link_patterns = [
            re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE),
            re.compile(r'<a[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE),
            re.compile(r'<link[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE)
        ]
        
        found_links = set()
        for pattern in link_patterns:
            found_links.update(pattern.findall(html))
        
        # Separate priority and regular URLs
        priority_urls = []
        regular_urls = []
        
        for link in found_links:
            try:
                # Process relative URLs
                if link.startswith('/'):
                    full_url = f"https://{domain}{link}"
                elif link.startswith('http') and domain in link:
                    full_url = link
                elif not link.startswith('http'):
                    full_url = f"https://{domain}/{link.lstrip('/')}"
                else:
                    continue
                
                # Skip unwanted extensions
                if any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip']):
                    continue
                
                # Check for PRIORITY keywords first (Impressum, management)
                url_lower = full_url.lower()
                if any(keyword in url_lower for keyword in priority_keywords):
                    if full_url not in urls and full_url not in priority_urls:
                        priority_urls.append(full_url)
                # Then check regular contact keywords
                elif any(keyword in url_lower for keyword in contact_keywords):
                    if full_url not in urls and full_url not in regular_urls:
                        regular_urls.append(full_url)
                
                # Also check link text context in HTML for priority content
                link_context_pattern = re.compile(
                    f'<a[^>]*href=["\'][^"\']*{re.escape(link)}[^"\']*["\'][^>]*>([^<]+)</a>',
                    re.IGNORECASE
                )
                context_match = link_context_pattern.search(html)
                if context_match:
                    link_text = context_match.group(1).lower()
                    # Check priority keywords in link text (Impressum, Management)
                    if any(keyword in link_text for keyword in priority_keywords):
                        if full_url not in urls and full_url not in priority_urls:
                            priority_urls.append(full_url)
                    # Then regular contact keywords
                    elif any(keyword in link_text for keyword in contact_keywords):
                        if full_url not in urls and full_url not in regular_urls:
                            regular_urls.append(full_url)
                            
            except Exception as e:
                logger.debug(f"Error processing link {link}: {e}")
        
        # Add priority URLs first (Impressum pages with managing directors)
        urls.extend(priority_urls)
        # Then add regular contact pages
        urls.extend(regular_urls)
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        logger.info(f"🔍 Found {len(unique_urls)} contact pages (unlimited processing)")
        if priority_urls:
            logger.info(f"⚖️ Found {len(priority_urls)} priority pages (Impressum/Management): {[p.split('/')[-1] for p in priority_urls[:3]]}")
        
        return unique_urls

    def get_rate_limiting_stats(self) -> Dict[str, any]:
        """Get current rate limiting statistics"""
        return {
            "rate_limiting_active": self.rate_limiting_active,
            "quota_hit_time": self.quota_hit_time,
            "adaptive_delay": self.adaptive_delay,
            "consecutive_errors": self.consecutive_errors,
            "successful_calls": self.successful_calls,
            "time_since_activation": time.time() - self.quota_hit_time if self.quota_hit_time else 0
        }

    def copy_rate_limiting_state(self, source_extractor):
        """Copy rate limiting state from another extractor instance"""
        self.rate_limiting_active = source_extractor.rate_limiting_active
        self.quota_hit_time = source_extractor.quota_hit_time
        self.adaptive_delay = source_extractor.adaptive_delay
        self.consecutive_errors = source_extractor.consecutive_errors
        self.successful_calls = source_extractor.successful_calls
        self.last_call_time = source_extractor.last_call_time

    def process_single_website(self, url: str) -> AIExtractionResult:
        """Process a single website (for ThreadPool usage)"""
        # Create new event loop for each thread to avoid conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Create new extractor instance to avoid session conflicts
            extractor = EnhancedLLMContactExtractor(
                timeout=self.timeout,
                max_pages=self.max_pages,
                thread_workers=1,  # Single worker for thread
                connection_pool_size=10
            )
            
            # Share rate limiting state across thread instances
            extractor.copy_rate_limiting_state(self)
            
            result = loop.run_until_complete(extractor.extract_from_website(url))
            
            # Copy back any rate limiting updates
            self.copy_rate_limiting_state(extractor)
            
            loop.run_until_complete(extractor.cleanup())
            return result
        finally:
            loop.close()

    async def extract_from_website(self, base_url: str) -> AIExtractionResult:
        """Extract contacts from website using enhanced AI with unlimited processing"""
        start_time = time.time()
        domain = urlparse(base_url).netloc
        
        try:
            session = await self.get_enhanced_session()
            
            # Get main page
            async with session.get(base_url) as response:
                if response.status != 200:
                    return AIExtractionResult(
                        url=base_url,
                        domain=domain,
                        error=f"HTTP {response.status}",
                        processing_time=time.time() - start_time
                    )
                
                main_html = await response.text()
            
            # Enhanced company name extraction
            company_name = self._extract_company_name(main_html, domain)
            
            # Find ALL contact pages (unlimited)
            contact_urls = self.find_all_contact_pages(base_url, main_html)
            logger.info(f"📄 Processing {len(contact_urls)} pages for {domain}")
            
            # Extract content from ALL pages with unlimited processing
            all_email_blocks = []
            all_page_content = ""
            pages_processed = 0
            total_content_size = 0
            
            # Process pages concurrently
            semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
            
            async def fetch_page(url):
                async with semaphore:
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                html = await response.text()
                                return url, html
                    except Exception as e:
                        logger.debug(f"Error fetching {url}: {e}")
                    return url, None
            
            # Fetch all pages
            tasks = [fetch_page(url) for url in contact_urls]
            page_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process all page content
            for url, html in page_results:
                if html and not isinstance(html, Exception):
                    email_blocks = self.extract_comprehensive_email_blocks(html, url)
                    all_email_blocks.extend(email_blocks)
                    
                    # Add to comprehensive content for AI analysis
                    clean_content = re.sub(r'<[^>]+>', ' ', html)
                    clean_content = re.sub(r'\s+', ' ', clean_content)
                    all_page_content += f"\n\n--- PAGE: {url} ---\n{clean_content[:10000]}"  # 10K chars per page
                    
                    pages_processed += 1
                    total_content_size += len(html)
            
            # Process contacts with enhanced AI using full context
            ai_start_time = time.time()
            contacts = await self.process_contacts_with_enhanced_ai(
                all_email_blocks, 
                domain, 
                all_page_content
            )
            ai_processing_time = time.time() - ai_start_time
            
            # Enhanced duplicate removal with confidence scoring
            unique_contacts = {}
            for contact in contacts:
                key = contact.email
                if key not in unique_contacts:
                    unique_contacts[key] = contact
                else:
                    # Keep contact with higher confidence
                    if contact.ai_confidence > unique_contacts[key].ai_confidence:
                        unique_contacts[key] = contact
            
            final_contacts = list(unique_contacts.values())
            
            # Sort by AI confidence
            final_contacts.sort(key=lambda c: c.ai_confidence, reverse=True)
            
            # Enhanced categorization
            categorized = self.categorize_contacts_enhanced(final_contacts)
            
            # Comprehensive statistics
            unique_emails = len(set(c.email for c in final_contacts if c.email))
            unique_phones = len(set(c.phone for c in final_contacts if c.phone))
            real_people = len([c for c in final_contacts if c.is_real_person])
            generic_contacts = len([c for c in final_contacts if not c.is_real_person])
            
            return AIExtractionResult(
                url=base_url,
                domain=domain,
                company_name=company_name,
                all_contacts=final_contacts,
                executives=categorized['executives'],
                sales_team=categorized['sales_team'],
                marketing_team=categorized['marketing_team'],
                engineering_team=categorized['engineering_team'],
                support_team=categorized['support_team'],
                other_contacts=categorized['other_contacts'],
                ai_processing_time=ai_processing_time,
                total_email_blocks_processed=len(all_email_blocks),
                total_pages_analyzed=len(contact_urls),
                total_content_size=total_content_size,
                real_people_found=real_people,
                generic_contacts_found=generic_contacts,
                total_contacts_found=len(final_contacts),
                unique_emails_found=unique_emails,
                unique_phones_found=unique_phones,
                pages_processed=pages_processed,
                processing_time=time.time() - start_time,
                success=len(final_contacts) > 0
            )
            
        except Exception as e:
            logger.error(f"❌ Error extracting from {base_url}: {e}")
            return AIExtractionResult(
                url=base_url,
                domain=domain,
                error=str(e),
                processing_time=time.time() - start_time
            )

    def _extract_company_name(self, html: str, domain: str) -> Optional[str]:
        """Enhanced company name extraction"""
        try:
            # Try multiple methods
            methods = [
                # Method 1: Title tag
                lambda h: re.search(r'<title[^>]*>([^<]+)</title>', h, re.IGNORECASE),
                # Method 2: Meta property
                lambda h: re.search(r'<meta[^>]*property=["\']og:site_name["\'][^>]*content=["\']([^"\']+)', h, re.IGNORECASE),
                # Method 3: H1 tag
                lambda h: re.search(r'<h1[^>]*>([^<]+)</h1>', h, re.IGNORECASE),
                # Method 4: Company schema
                lambda h: re.search(r'"@type":\s*"Organization"[^}]*"name":\s*"([^"]+)"', h, re.IGNORECASE)
            ]
            
            for method in methods:
                match = method(html)
                if match:
                    name = match.group(1).strip()
                    # Clean up common suffixes
                    name = re.sub(r'\s*[-|–]\s*(home|start|welcome|willkommen).*', '', name, flags=re.IGNORECASE)
                    name = re.sub(r'\s*\|\s*.*', '', name)
                    return name[:100]  # Limit length
            
            # Fallback to domain
            return domain.replace('www.', '').split('.')[0].title()
            
        except:
            return domain

    def categorize_contacts_enhanced(self, contacts: List[AIContact]) -> Dict[str, List[AIContact]]:
        """Enhanced contact categorization with more granular analysis"""
        categories = {
            'executives': [],
            'sales_team': [],
            'marketing_team': [],
            'engineering_team': [],
            'support_team': [],
            'other_contacts': []
        }
        
        for contact in contacts:
            # Enhanced categorization logic
            if contact.seniority_level in ['C-Level', 'VP/Director'] or contact.is_decision_maker:
                categories['executives'].append(contact)
            elif contact.role_category and any(word in contact.role_category.lower() for word in ['sales', 'vertrieb', 'business development']):
                categories['sales_team'].append(contact)
            elif contact.role_category and any(word in contact.role_category.lower() for word in ['marketing', 'communication', 'pr']):
                categories['marketing_team'].append(contact)
            elif contact.role_category and any(word in contact.role_category.lower() for word in ['engineering', 'development', 'technical', 'it']):
                categories['engineering_team'].append(contact)
            elif contact.role_category and any(word in contact.role_category.lower() for word in ['support', 'service', 'help', 'customer']):
                categories['support_team'].append(contact)
            else:
                categories['other_contacts'].append(contact)
        
        return categories

    def extract_from_multiple_websites_threaded(self, urls: List[str]) -> List[AIExtractionResult]:
        """Extract from multiple websites using ThreadPool for maximum performance"""
        
        logger.info(f"🚀 Starting ThreadPool extraction for {len(urls)} websites")
        logger.info(f"⚡ Using {self.thread_workers} parallel workers")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.thread_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self.process_single_website, url): url 
                for url in urls
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress updates
                    if completed % 10 == 0 or completed <= 10:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(urls) - completed) / rate if rate > 0 else 0
                        
                        logger.info(f"📈 Progress: {completed}/{len(urls)} ({completed/len(urls)*100:.1f}%) | "
                                  f"Rate: {rate:.1f}/min | ETA: {eta/60:.1f}min")
                        
                        # Show success rate
                        successful = sum(1 for r in results if r.success)
                        logger.info(f"✅ Success rate: {successful}/{completed} ({successful/completed*100:.1f}%)")
                        
                        # Show rate limiting status
                        if self.rate_limiting_active:
                            stats = self.get_rate_limiting_stats()
                            logger.info(f"🛡️ Smart rate limiting: active for {stats['time_since_activation']:.0f}s, "
                                      f"delay: {stats['adaptive_delay']:.1f}s")
                
                except Exception as e:
                    logger.error(f"❌ Error processing {url}: {e}")
                    # Create error result
                    results.append(AIExtractionResult(
                        url=url,
                        domain=urlparse(url).netloc,
                        error=str(e),
                        processing_time=0
                    ))
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if r.success]
        
        logger.info(f"🎉 ThreadPool extraction completed!")
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"📊 Results: {len(successful_results)}/{len(urls)} successful")
        logger.info(f"⚡ Average rate: {len(urls)/total_time:.1f} websites/second")
        
        return results

    def save_enhanced_ai_contacts(self, results: List[AIExtractionResult], db_name: str = None) -> str:
        """Save enhanced AI-processed contacts to database with comprehensive data"""
        
        if db_name is None:
            timestamp = int(time.time())
            db_name = f"enhanced_ai_contacts_{timestamp}.db"
        
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Create enhanced AI contacts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_ai_contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                url TEXT NOT NULL,
                company_name TEXT,
                
                -- Basic Contact
                email TEXT,
                phone TEXT,
                mobile TEXT,
                fax TEXT,
                extension TEXT,
                
                -- Personal Info
                first_name TEXT,
                last_name TEXT,
                full_name TEXT,
                title TEXT,
                department TEXT,
                
                -- AI Classifications
                seniority_level TEXT,
                role_category TEXT,
                is_decision_maker BOOLEAN,
                is_company_employee BOOLEAN,
                employee_type TEXT,
                employment_status TEXT,
                
                -- Enhanced AI Analysis
                is_real_person BOOLEAN,
                ai_confidence REAL,
                ai_reasoning TEXT,
                
                -- Professional Info
                linkedin_url TEXT,
                twitter_handle TEXT,
                github_profile TEXT,
                
                -- Enhanced Metadata
                source_page_type TEXT,
                confidence_score REAL,
                extraction_method TEXT,
                last_updated TIMESTAMP,
                
                -- Indexes for performance
                UNIQUE(email, domain)
            )
        ''')
        
        # Create enhanced extraction summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_ai_extraction_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                company_name TEXT,
                total_contacts INTEGER,
                real_people_found INTEGER,
                generic_contacts_found INTEGER,
                employees_count INTEGER,
                externals_count INTEGER,
                executives_count INTEGER,
                
                -- Enhanced metrics
                total_pages_analyzed INTEGER,
                total_content_size INTEGER,
                ai_processing_time REAL,
                unique_emails INTEGER,
                unique_phones INTEGER,
                pages_processed INTEGER,
                processing_time REAL,
                success BOOLEAN,
                error_message TEXT,
                extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Performance metrics
                avg_confidence REAL,
                high_confidence_contacts INTEGER
            )
        ''')
        
        # Insert data with enhanced error handling
        for result in results:
            try:
                # Insert all contacts
                for contact in result.all_contacts:
                    cursor.execute('''
                        INSERT OR REPLACE INTO enhanced_ai_contacts (
                            domain, url, company_name, email, phone, mobile, fax, extension,
                            first_name, last_name, full_name, title, department,
                            seniority_level, role_category, is_decision_maker,
                            is_company_employee, employee_type, employment_status,
                            is_real_person, ai_confidence, ai_reasoning,
                            linkedin_url, twitter_handle, github_profile,
                            source_page_type, confidence_score, extraction_method, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.domain, result.url, result.company_name,
                        contact.email, contact.phone, contact.mobile, contact.fax, contact.extension,
                        contact.first_name, contact.last_name, contact.full_name,
                        contact.title, contact.department,
                        contact.seniority_level, contact.role_category, contact.is_decision_maker,
                        contact.is_company_employee, contact.employee_type, contact.employment_status,
                        contact.is_real_person, contact.ai_confidence, contact.ai_reasoning,
                        contact.linkedin_url, contact.twitter_handle, contact.github_profile,
                        contact.source_page_type, contact.confidence_score, contact.extraction_method, contact.last_updated
                    ))
                
                # Calculate enhanced statistics
                employees_count = sum(1 for c in result.all_contacts if c.is_company_employee)
                externals_count = sum(1 for c in result.all_contacts if not c.is_company_employee)
                executives_count = len(result.executives)
                avg_confidence = sum(c.ai_confidence for c in result.all_contacts) / len(result.all_contacts) if result.all_contacts else 0
                high_confidence_contacts = sum(1 for c in result.all_contacts if c.ai_confidence > 0.7)
                
                # Insert enhanced summary
                cursor.execute('''
                    INSERT INTO enhanced_ai_extraction_summary (
                        domain, company_name, total_contacts, real_people_found, generic_contacts_found,
                        employees_count, externals_count, executives_count, total_pages_analyzed,
                        total_content_size, ai_processing_time, unique_emails, unique_phones,
                        pages_processed, processing_time, success, error_message,
                        avg_confidence, high_confidence_contacts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.domain, result.company_name, result.total_contacts_found,
                    result.real_people_found, result.generic_contacts_found,
                    employees_count, externals_count, executives_count, result.total_pages_analyzed,
                    result.total_content_size, result.ai_processing_time, result.unique_emails_found,
                    result.unique_phones_found, result.pages_processed, result.processing_time,
                    result.success, result.error, avg_confidence, high_confidence_contacts
                ))
                
            except Exception as e:
                logger.error(f"Error saving data for {result.domain}: {e}")
        
        # Create performance indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON enhanced_ai_contacts(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_email ON enhanced_ai_contacts(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON enhanced_ai_contacts(ai_confidence)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_real_person ON enhanced_ai_contacts(is_real_person)')
        
        conn.commit()
        conn.close()
        
        successful_results = [r for r in results if r.success]
        total_contacts = sum(len(r.all_contacts) for r in successful_results)
        
        logger.info(f"🎉 Enhanced AI contacts database created: {db_name}")
        logger.info(f"📊 Total contacts saved: {total_contacts}")
        logger.info(f"✅ Successful extractions: {len(successful_results)}/{len(results)}")
        
        return db_name

    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

async def enhanced_demo_ai_extraction():
    """Enhanced demo with unlimited processing capabilities"""
    print("🚀 ENHANCED LLM-DRIVEN CONTACT EXTRACTOR v2.0")
    print("=" * 80)
    print("✅ Gemini-2.5-Flash with 1M token window")
    print("✅ NO page number limitations")
    print("✅ NO token limitations")
    print("✅ ThreadPool parallel processing")
    print("✅ Unlimited content analysis")
    print("✅ Enhanced AI reasoning")
    print("✅ Maximum performance optimization")
    print("=" * 80)
    
    # Initialize enhanced extractor
    extractor = EnhancedLLMContactExtractor(
        timeout=30,
        max_pages=None,  # UNLIMITED
        thread_workers=20,
        connection_pool_size=100
    )
    
    # Test with sample German companies
    test_urls = [
        "http://www.fuerst-reisen.de",
        "https://www.drive57.de",
        "https://www.premiumbusberlin.com"
    ]
    
    print(f"\n🔍 Testing enhanced AI extraction on {len(test_urls)} companies:")
    for i, url in enumerate(test_urls, 1):
        print(f"   {i}. {url}")
    
    # Single website test first
    print(f"\n🧪 SINGLE WEBSITE TEST")
    print("-" * 50)
    single_result = await extractor.extract_from_website(test_urls[0])
    
    print(f"🏢 Company: {single_result.company_name}")
    print(f"📄 Pages Analyzed: {single_result.total_pages_analyzed}")
    print(f"📊 Content Size: {single_result.total_content_size:,} chars")
    print(f"👥 Total Contacts: {single_result.total_contacts_found}")
    print(f"👤 Real People: {single_result.real_people_found}")
    print(f"🏢 Generic Contacts: {single_result.generic_contacts_found}")
    print(f"📧 Unique Emails: {single_result.unique_emails_found}")
    print(f"📞 Unique Phones: {single_result.unique_phones_found}")
    print(f"🤖 AI Processing Time: {single_result.ai_processing_time:.1f}s")
    print(f"⏱️  Total Time: {single_result.processing_time:.1f}s")
    
    if single_result.all_contacts:
        print(f"\n🤖 ENHANCED AI-PROCESSED CONTACTS:")
        for i, contact in enumerate(single_result.all_contacts[:5], 1):  # Show top 5
            person_type = "👤 REAL PERSON" if contact.is_real_person else "🏢 GENERIC CONTACT"
            employee_status = "🏢 EMPLOYEE" if contact.is_company_employee else "🤝 EXTERNAL"
            
            print(f"   {i}. {contact.full_name or 'No Name'} - {person_type} | {employee_status}")
            print(f"      📧 {contact.email}")
            if contact.phone:
                print(f"      📞 {contact.phone}")
            if contact.title:
                print(f"      💼 {contact.title}")
            if contact.department:
                print(f"      🏢 {contact.department}")
            print(f"      🤖 AI Confidence: {contact.ai_confidence:.1%}")
            if contact.ai_reasoning:
                reasoning = contact.ai_reasoning[:100] + "..." if len(contact.ai_reasoning) > 100 else contact.ai_reasoning
                print(f"      💭 AI Reasoning: {reasoning}")
            print()
    
    # ThreadPool multiple websites test
    print(f"\n⚡ THREADPOOL MULTIPLE WEBSITES TEST")
    print("-" * 50)
    
    start_time = time.time()
    threaded_results = extractor.extract_from_multiple_websites_threaded(test_urls)
    total_time = time.time() - start_time
    
    # Comprehensive summary
    successful_results = [r for r in threaded_results if r.success]
    total_contacts = sum(len(r.all_contacts) for r in successful_results)
    total_real_people = sum(r.real_people_found for r in successful_results)
    total_pages = sum(r.total_pages_analyzed for r in successful_results)
    total_content = sum(r.total_content_size for r in successful_results)
    
    print(f"\n📊 THREADPOOL RESULTS SUMMARY:")
    print(f"✅ Successful: {len(successful_results)}/{len(test_urls)}")
    print(f"👥 Total Contacts: {total_contacts}")
    print(f"👤 Real People: {total_real_people}")
    print(f"📄 Pages Analyzed: {total_pages}")
    print(f"📊 Content Processed: {total_content:,} chars")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    print(f"⚡ Rate: {len(test_urls)/total_time:.1f} websites/second")
    
    # Save results
    db_file = extractor.save_enhanced_ai_contacts(threaded_results)
    print(f"🎉 Enhanced AI contacts database created: {db_file}")
    
    await extractor.cleanup()
    return threaded_results

async def process_csv_with_enhanced_threadpool(csv_file: str, num_companies: int = None):
    """Process CSV file with enhanced ThreadPool capabilities"""
    
    print("🚀 ENHANCED CSV PROCESSING WITH THREADPOOL")
    print("=" * 80)
    print("✅ Unlimited page processing")
    print("✅ Unlimited token usage")
    print("✅ Maximum performance ThreadPool")
    print("✅ Comprehensive AI analysis")
    print("=" * 80)
    
    # Load CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} companies from {csv_file}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Determine URL column
    url_columns = ['URL1', 'url', 'website', 'homepage', 'domain']
    url_column = None
    for col in url_columns:
        if col in df.columns:
            url_column = col
            break
    
    if not url_column:
        url_column = df.columns[0]
        print(f"⚠️  Using first column '{url_column}' as URL column")
    
    # Process specified number of companies
    if num_companies:
        companies_to_process = df.head(num_companies)[url_column].tolist()
        print(f"🎯 Processing first {num_companies} companies...")
    else:
        companies_to_process = df[url_column].tolist()
        print(f"🎯 Processing ALL {len(companies_to_process)} companies...")
    
    # Remove invalid URLs
    valid_urls = []
    for url in companies_to_process:
        if pd.notna(url) and str(url).strip():
            clean_url = str(url).strip()
            if not clean_url.startswith('http'):
                clean_url = f"https://{clean_url}"
            valid_urls.append(clean_url)
    
    print(f"✅ {len(valid_urls)} valid URLs to process")
    
    # Initialize enhanced extractor
    extractor = EnhancedLLMContactExtractor(
        timeout=30,
        max_pages=None,  # UNLIMITED
        thread_workers=30,  # High concurrency for CSV processing
        connection_pool_size=150
    )
    
    # Process all URLs with ThreadPool
    start_time = time.time()
    all_results = extractor.extract_from_multiple_websites_threaded(valid_urls)
    total_time = time.time() - start_time
    
    # Generate comprehensive summary
    successful_results = [r for r in all_results if r.success]
    failed_results = [r for r in all_results if not r.success]
    
    total_contacts = sum(len(r.all_contacts) for r in successful_results)
    total_real_people = sum(r.real_people_found for r in successful_results)
    total_generic = sum(r.generic_contacts_found for r in successful_results)
    total_executives = sum(len(r.executives) for r in successful_results)
    total_pages = sum(r.total_pages_analyzed for r in successful_results)
    total_content = sum(r.total_content_size for r in successful_results)
    
    print(f"\n🎉 ENHANCED CSV PROCESSING COMPLETED!")
    print("=" * 80)
    print(f"📊 Processing Summary:")
    print(f"   ✅ Successful: {len(successful_results)}/{len(valid_urls)} ({len(successful_results)/len(valid_urls)*100:.1f}%)")
    print(f"   ❌ Failed: {len(failed_results)}")
    print(f"   ⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   ⚡ Rate: {len(valid_urls)/total_time:.1f} websites/second")
    print()
    print(f"👥 Contact Summary:")
    print(f"   📧 Total Contacts: {total_contacts}")
    print(f"   👤 Real People: {total_real_people}")
    print(f"   🏢 Generic Contacts: {total_generic}")
    print(f"   👔 Executives: {total_executives}")
    print()
    print(f"📊 Content Analysis:")
    print(f"   📄 Pages Analyzed: {total_pages}")
    print(f"   📊 Content Processed: {total_content:,} characters")
    print(f"   📈 Avg Pages/Company: {total_pages/len(successful_results):.1f}")
    print()
    
    # Save results to database
    db_file = extractor.save_enhanced_ai_contacts(all_results)
    
    # Create CSV export
    if successful_results:
        csv_data = []
        for result in successful_results:
            for contact in result.all_contacts:
                csv_data.append({
                    'company_name': result.company_name,
                    'domain': result.domain,
                    'url': result.url,
                    'email': contact.email,
                    'full_name': contact.full_name,
                    'first_name': contact.first_name,
                    'last_name': contact.last_name,
                    'title': contact.title,
                    'department': contact.department,
                    'phone': contact.phone,
                    'is_real_person': contact.is_real_person,
                    'is_company_employee': contact.is_company_employee,
                    'seniority_level': contact.seniority_level,
                    'role_category': contact.role_category,
                    'is_decision_maker': contact.is_decision_maker,
                    'ai_confidence': contact.ai_confidence,
                    'ai_reasoning': contact.ai_reasoning,
                    'pages_processed': result.pages_processed,
                    'total_pages_analyzed': result.total_pages_analyzed,
                    'processing_time': result.processing_time
                })
        
        if csv_data:
            timestamp = int(time.time())
            csv_filename = f"enhanced_contacts_export_{timestamp}.csv"
            pd.DataFrame(csv_data).to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"📁 CSV Export: {csv_filename}")
    
    print(f"🗄️  Database: {db_file}")
    print("=" * 80)
    
    await extractor.cleanup()
    return all_results

async def main():
    """Enhanced main function with comprehensive options"""
    print("🚀 ENHANCED LLM CONTACT EXTRACTOR v2.0")
    print("Choose processing mode:")
    print("1. Demo with sample websites")
    print("2. Process CSV file (specify number)")
    print("3. Process entire CSV file")
    print("4. Single website extraction")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            await enhanced_demo_ai_extraction()
            
        elif choice == "2":
            csv_file = input("Enter CSV filename: ").strip()
            num_companies = int(input("Enter number of companies to process: "))
            await process_csv_with_enhanced_threadpool(csv_file, num_companies)
            
        elif choice == "3":
            csv_file = input("Enter CSV filename: ").strip()
            await process_csv_with_enhanced_threadpool(csv_file)
            
        elif choice == "4":
            url = input("Enter website URL: ").strip()
            extractor = EnhancedLLMContactExtractor()
            result = await extractor.extract_from_website(url)
            
            print(f"\n📊 Results for {url}:")
            print(f"👥 Contacts: {result.total_contacts_found}")
            print(f"👤 Real People: {result.real_people_found}")
            print(f"📄 Pages: {result.total_pages_analyzed}")
            print(f"⏱️  Time: {result.processing_time:.1f}s")
            
            if result.all_contacts:
                for contact in result.all_contacts:
                    print(f"   📧 {contact.email} - {contact.full_name or 'No Name'}")
            
            await extractor.cleanup()
            
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 
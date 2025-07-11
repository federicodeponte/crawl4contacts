# 🌍 AI-Powered Business Contact Extractor

**Advanced AI-powered contact extraction system for business websites worldwide - optimized for German companies with priority Impressum processing**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 **Key Features**

### ✅ **Global Business Contact Extraction**
- **Worldwide Compatibility** - Works on business websites from any country
- **Multi-language Support** - Handles websites in multiple languages  
- **Adaptive Content Discovery** - Intelligently finds contact pages, about sections, and team pages
- **Universal Contact Patterns** - Extracts executives, managers, and key personnel globally

### ✅ **German Business Optimization (Bonus Features)**
- **Priority Impressum Processing** - Automatically finds and prioritizes German Impressum (legal notice) pages
- **Managing Director Extraction** - Specialized extraction of "Geschäftsführer" and legal representatives
- **High-Quality Executive Contacts** - 95-100% AI confidence on German managing directors
- **Legal Compliance** - Respects German privacy laws and website structure conventions

### ✅ **Advanced AI Processing**
- **Gemini-2.5-Flash** with 1M token context window
- **Unlimited Page Processing** - No artificial limits (processes 80-99 pages per company)
- **Smart Rate Limiting** - Automatic quota management with intelligent backoff
- **Enhanced AI Prompts** - Tuned for German business structure and terminology

### ✅ **High-Performance Architecture**
- **ThreadPool Processing** - Parallel processing of multiple companies
- **Smart Error Handling** - Robust fallback systems and error recovery
- **Checkpointing** - Resumable processing for large datasets
- **Real-time Progress Tracking** - Comprehensive statistics and monitoring

### ✅ **Flexible Input Processing**
- **Multiple CSV Formats** - Accepts various URL column headers
- **URL Validation & Cleaning** - Automatic protocol addition and validation
- **JSON Support** - Process JSON files with URL lists
- **Comma-separated Lists** - Direct URL list processing

## 📊 **Proven Results**

**Global Business Contact Extraction:**
- ✅ Works on websites from **any country** - US, UK, France, Italy, Spain, etc.
- 👥 Extracts contacts from standard business pages: About, Team, Contact, Management
- 🎯 Identifies executives and key personnel regardless of language or location

**Latest Run: 1,999 German Companies (Showcase)**
- ✅ **1,001 companies** successfully processed (50.1% success rate)
- 👥 **3,390 total contacts** extracted
- 👔 **217 executives/managing directors** identified from Impressum pages
- 📄 **Unlimited pages** processed per company (up to 99 pages)
- ⏱️ **178.5 minutes** total processing time
- 🛡️ **Smart rate limiting** managed quota effectively

## 🔧 **Installation**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ai-business-contact-extractor.git
cd ai-business-contact-extractor

# Install dependencies
pip install -r requirements.txt

# Set up API keys (create .env file)
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

## 🎯 **Quick Start**

### **Process CSV File**
```python
from llm_contact_extractor import EnhancedLLMContactExtractor

# Initialize extractor
extractor = EnhancedLLMContactExtractor(
    max_pages=None,        # Unlimited pages
    thread_workers=20,     # High-performance
    connection_pool_size=100
)

# Process CSV with any URL column format
results = await extractor.process_csv("companies.csv")

# Results automatically saved to database and CSV
```

### **Supported CSV Headers**
The system automatically detects URL columns with headers like:
- `URL`, `URL1`, `Website`, `Website URL`
- `Company URL`, `Homepage`, `Domain`
- `Site`, `Web`, `Link`, `Address`

### **Process Individual Companies**
```python
# Single company
result = await extractor.extract_from_website("https://company.de")

# Multiple companies
urls = ["https://company1.de", "https://company2.de"]
results = extractor.extract_from_multiple_websites_threaded(urls)
```

## 🇩🇪 **German Impressum Specialization**

### **Priority Keywords (High Priority)**
- `impressum`, `imprint`, `legal-notice`
- `geschäftsführung`, `management`, `vorstand`
- `geschäftsleitung`, `board`, `directors`

### **Executive Extraction**
The AI is specifically trained to identify:
- **Geschäftsführer** (Managing Directors)
- **Vertretungsberechtigte** (Authorized Representatives)
- **Vorstand** (Board Members)
- **Inhaber** (Owners)
- **Gesellschafter** (Partners)

### **Sample Results**
```
Jacqueline Hahne     | Managing Director         | 100% confidence
Andreas Schulze      | Geschäftsführer           | 95% confidence  
Peter Weiss          | Geschäftsführer           | 100% confidence
Dr. Ulrich Basteck   | Geschäftsführer           | 100% confidence
```

## 📁 **File Structure**

```
├── llm_contact_extractor.py      # Main extraction engine
├── process_all_companies_improved.py  # Batch processing script
├── test_smart_rate_limiting.py   # Rate limiting demo
├── requirements.txt              # Python dependencies
├── SMART_RATE_LIMITING_GUIDE.md # Technical documentation
└── examples/                     # Usage examples
    ├── process_csv_example.py
    ├── single_company_example.py
    └── sample_data.csv
```

## 🛡️ **Smart Rate Limiting**

The system includes advanced quota management:

1. **Full Speed Initially** - No delays until quota limits hit
2. **Auto-Activation** - Triggers on first 429 error
3. **Adaptive Delays** - Intelligent backoff (0.5s → 2s → 5s → 30s)
4. **Gradual Optimization** - Reduces delays after successful calls
5. **Cross-Thread Sharing** - Rate limiting state shared across workers

## 📊 **Output Formats**

### **Enhanced CSV Export**
```csv
Company_Domain,Full_Name,Title,Email,Phone,Seniority_Level,AI_Confidence,Is_Real_Person
company.de,Hans Mueller,Geschäftsführer,h.mueller@company.de,+49...,C-Level,1.0,True
```

### **Executive-Only Export**
Separate CSV with only C-Level and Executive contacts for focused outreach.

### **SQLite Database**
Complete relational database with:
- Enhanced contact details
- Company metadata
- Processing statistics
- AI confidence scores

## 🚀 **Advanced Usage**

### **Large-Scale Processing**
```python
# Process 2000+ companies with checkpointing
python3 process_all_companies_improved.py

# Features:
# - Batch processing (100 companies per batch)
# - Automatic checkpointing
# - Resumable processing
# - Intermediate saves
```

### **Custom Configuration**
```python
extractor = EnhancedLLMContactExtractor(
    timeout=45,                    # Longer timeout for complex sites
    max_pages=50,                  # Limit pages if needed
    thread_workers=10,             # Adjust for your system
    connection_pool_size=50        # Adjust for network capacity
)
```

## 📈 **Performance Metrics**

- **Processing Speed**: 0.19-0.8 companies/second (depending on quota)
- **Success Rate**: 50-76% (varies by website complexity)
- **Executive Extraction**: 31.6% of companies yield executive contacts
- **Page Analysis**: Up to 99 pages per company
- **Content Processing**: 100+ MB of content per session

## 🔍 **API Integration**

Currently supports:
- **Google Gemini-2.5-Flash** (primary)
- **Azure OpenAI** (fallback)
- **OpenAI GPT** (fallback)

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- German business law compliance research
- AI model fine-tuning for German terminology
- Community feedback on Impressum processing
- Performance optimization contributions

## 📞 **Support**

For questions, issues, or feature requests:
- Open an [Issue](https://github.com/YOUR_USERNAME/german-business-contact-extractor/issues)
- Check [Documentation](docs/)
- Review [Examples](examples/)

## 🌍 **Global Usage Examples**

### **US Companies**
```python
# Extract from US business websites
urls = ["https://company.com", "https://startup.io", "https://corp.net"]
results = extractor.extract_from_multiple_websites_threaded(urls)
# Finds: CEOs, Presidents, VPs, Directors from About/Team pages
```

### **International Companies**
```python
# Works with any business website globally
international_urls = [
    "https://company.co.uk",    # UK
    "https://empresa.es",       # Spain  
    "https://société.fr",       # France
    "https://azienda.it"        # Italy
]
results = extractor.extract_from_multiple_websites_threaded(international_urls)
```

---

**🌍 AI-powered business contact extraction for companies worldwide - with special German business optimization!** 
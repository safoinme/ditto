#!/usr/bin/env python3
"""
Demo script showing the difference between production and testing modes for DITTO matching.

Production Mode: Table has _left and _right columns for comparing different records
Testing Mode: Table has regular columns for self-matching records
"""

import pandas as pd
import json
import os
from hive_ditto_standalone import convert_to_ditto_format, detect_table_structure

def create_production_sample():
    """Create sample data with _left/_right columns (production mode)."""
    print("🏭 Creating Production Sample Data")
    print("=" * 50)
    
    # Production data: each row compares two different records
    data = {
        'record_id': [1, 2, 3, 4, 5],
        'firstname_left': ['Ahmed', 'Mohamed', 'Sarah', 'Omar', 'Fatima'],
        'lastname_left': ['Smith', 'Johnson', 'Williams', 'Brown', 'Garcia'],
        'email_left': ['ahmed.smith@company.com', 'mohamed.j@corp.com', 'sarah.w@business.com', 'omar.brown@firm.com', 'fatima.g@enterprise.com'],
        'phone_left': ['555-0101', '555-0202', '555-0303', '555-0404', '555-0505'],
        'firstname_right': ['Ahmed A.', 'Mohammed', 'Sarah J.', 'Omar B.', 'Fatima'],
        'lastname_right': ['Smith Jr.', 'Johnston', 'Williams-Jones', 'Brown', 'Garcia-Lopez'],
        'email_right': ['a.smith@company.com', 'mohammed.johnston@corp.com', 'sarah.jones@business.com', 'o.brown@firm.com', 'f.garcia@enterprise.com'],
        'phone_right': ['(555) 101-1234', '555.202.5678', '555-303-9012', '5550404', '555-505-6789']
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Sample production table:")
    print(df.to_string())
    
    return df

def create_testing_sample():
    """Create sample data with regular columns (testing mode)."""
    print("\n🧪 Creating Testing Sample Data")
    print("=" * 50)
    
    # Testing data: each record will be matched against itself
    data = {
        'person_id': [1, 2, 3, 4, 5],
        'firstname': ['Ahmed', 'Mohamed', 'Sarah', 'Omar', 'Fatima'],
        'lastname': ['Smith', 'Johnson', 'Williams', 'Brown', 'Garcia'],
        'email': ['ahmed.smith@example.com', 'mohamed.j@example.com', 'sarah.w@example.com', 'omar.b@example.com', 'fatima.g@example.com'],
        'phone': ['555-0101', '555-0202', '555-0303', '555-0404', '555-0505'],
        'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd', '321 Elm St', '654 Maple Dr']
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Sample testing table:")
    print(df.to_string())
    
    return df

def create_table_with_prefixes():
    """Create sample data with table name prefixes."""
    print("\n🏷️ Creating Data with Table Prefixes")
    print("=" * 50)
    
    # Data with table name prefixes (simulating Hive join results)
    data = {
        'customers.customer_id': [1, 2, 3],
        'customers.firstname_left': ['Ahmed', 'Mohamed', 'Sarah'],
        'customers.lastname_left': ['Smith', 'Johnson', 'Williams'],
        'customers.email_left': ['ahmed@company.com', 'mohamed@company.com', 'sarah@company.com'],
        'prospects.prospect_id': [101, 102, 103],
        'prospects.firstname_right': ['Ahmed A', 'Mohammed', 'Sara'],
        'prospects.lastname_right': ['Smith Jr', 'Johnston', 'Williams'],
        'prospects.email_right': ['a.smith@corp.com', 'mohammed@corp.com', 'sara@corp.com']
    }
    
    df = pd.DataFrame(data)
    
    print("📊 Sample table with prefixes:")
    print(df.to_string())
    
    return df

def demo_structure_detection():
    """Demonstrate automatic structure detection."""
    print("\n🔍 Structure Detection Demo")
    print("=" * 50)
    
    # Test production data
    prod_df = create_production_sample()
    prod_structure = detect_table_structure(prod_df)
    print(f"\nProduction table analysis: {prod_structure['message']}")
    print(f"Matching fields found: {prod_structure['matching_fields']}")
    
    # Test testing data
    test_df = create_testing_sample()
    test_structure = detect_table_structure(test_df)
    print(f"\nTesting table analysis: {test_structure['message']}")
    print(f"Clean columns: {test_structure['clean_columns']}")
    
    # Test table with prefixes
    prefix_df = create_table_with_prefixes()
    prefix_structure = detect_table_structure(prefix_df)
    print(f"\nPrefixed table analysis: {prefix_structure['message']}")
    print(f"Matching fields found: {prefix_structure['matching_fields']}")

def demo_conversion_modes():
    """Demonstrate DITTO format conversion for both modes."""
    print("\n🔄 DITTO Conversion Demo")
    print("=" * 50)
    
    # Production mode conversion
    print("\n🏭 PRODUCTION MODE CONVERSION:")
    print("-" * 40)
    prod_df = create_production_sample()
    prod_records = convert_to_ditto_format(prod_df, matching_mode='production')
    
    print("\n📋 Sample production DITTO records:")
    for i, record in enumerate(prod_records[:2]):
        print(f"\nRecord {i+1}:")
        print(f"  Left:  {record['left']}")
        print(f"  Right: {record['right']}")
        print(f"  Same:  {record['left'] == record['right']}")
    
    # Testing mode conversion
    print("\n\n🧪 TESTING MODE CONVERSION:")
    print("-" * 40)
    test_df = create_testing_sample()
    test_records = convert_to_ditto_format(test_df, matching_mode='testing')
    
    print("\n📋 Sample testing DITTO records:")
    for i, record in enumerate(test_records[:2]):
        print(f"\nRecord {i+1}:")
        print(f"  Left:  {record['left']}")
        print(f"  Right: {record['right']}")
        print(f"  Same:  {record['left'] == record['right']}")
    
    # Auto-detection mode
    print("\n\n🤖 AUTO-DETECTION MODE:")
    print("-" * 40)
    print("Testing auto-detection on production data...")
    auto_prod_records = convert_to_ditto_format(prod_df, matching_mode='auto')
    print(f"Detected as: {'Production' if auto_prod_records[0]['left'] != auto_prod_records[0]['right'] else 'Testing'}")
    
    print("\nTesting auto-detection on testing data...")
    auto_test_records = convert_to_ditto_format(test_df, matching_mode='auto')
    print(f"Detected as: {'Production' if auto_test_records[0]['left'] != auto_test_records[0]['right'] else 'Testing'}")

def save_sample_files():
    """Save sample JSONL files for testing."""
    print("\n💾 Saving Sample Files")
    print("=" * 50)
    
    os.makedirs('./demo_output', exist_ok=True)
    
    # Save production sample
    prod_df = create_production_sample()
    prod_records = convert_to_ditto_format(prod_df, matching_mode='production')
    
    with open('./demo_output/production_sample.jsonl', 'w', encoding='utf-8') as f:
        for record in prod_records:
            matcher_record = [record['left'], record['right']]
            f.write(json.dumps(matcher_record, ensure_ascii=False) + '\n')
    
    print("✅ Saved: ./demo_output/production_sample.jsonl")
    
    # Save testing sample
    test_df = create_testing_sample()
    test_records = convert_to_ditto_format(test_df, matching_mode='testing')
    
    with open('./demo_output/testing_sample.jsonl', 'w', encoding='utf-8') as f:
        for record in test_records:
            matcher_record = [record['left'], record['right']]
            f.write(json.dumps(matcher_record, ensure_ascii=False) + '\n')
    
    print("✅ Saved: ./demo_output/testing_sample.jsonl")
    
    # Show file contents
    print("\n📋 Production JSONL sample (first 2 lines):")
    with open('./demo_output/production_sample.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            data = json.loads(line)
            print(f"  Line {i+1}: Left='{data[0][:50]}...', Right='{data[1][:50]}...'")
    
    print("\n📋 Testing JSONL sample (first 2 lines):")
    with open('./demo_output/testing_sample.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            data = json.loads(line)
            print(f"  Line {i+1}: Left='{data[0][:50]}...', Right='{data[1][:50]}...'")

def main():
    """Run all demos."""
    print("🎭 DITTO Production vs Testing Mode Demo")
    print("=" * 60)
    
    print("""
📖 Overview:
    
🏭 PRODUCTION MODE:
   - Table has columns ending with '_left' and '_right'
   - Each row represents a comparison between two different records
   - Left columns are compared against corresponding right columns
   - Example: firstname_left vs firstname_right, email_left vs email_right
   - Use case: Pre-processed pairs from data linking systems
    
🧪 TESTING MODE:
   - Table has regular columns without _left/_right suffixes
   - Each record is compared against itself (self-matching)
   - All columns are included in both left and right sides
   - Example: firstname, email, phone (same values on both sides)
   - Use case: Testing, validation, or single-table deduplication
    
🤖 AUTO-DETECTION:
   - Automatically detects table structure
   - Looks for _left/_right column pairs
   - Falls back to testing mode if no pairs found
    """)
    
    # Run demos
    demo_structure_detection()
    demo_conversion_modes()
    save_sample_files()
    
    print("\n🎉 Demo completed!")
    print("\n📝 Next steps:")
    print("  1. Test with your own data using hive_ditto_standalone.py")
    print("  2. Use --matching-mode parameter to force specific mode")
    print("  3. Generate Kubeflow pipeline with matching_mode parameter")
    print("\n💡 Examples:")
    print("  # Force production mode")
    print("  python hive_ditto_standalone.py --input-table mytable --matching-mode production")
    print("  # Force testing mode")
    print("  python hive_ditto_standalone.py --input-table mytable --matching-mode testing")
    print("  # Auto-detect (default)")
    print("  python hive_ditto_standalone.py --input-table mytable --matching-mode auto")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Updated DITTO format converter that properly handles tablename.column format.
"""

import pandas as pd
import json
from typing import List, Dict, Any

def convert_to_ditto_format(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to DITTO format, removing table name prefixes from columns.
    
    Args:
        df: DataFrame with columns that may have tablename.column format
        
    Returns:
        List of DITTO records in JSONL format
    """
    records = []
    
    print(f"🔍 Original columns: {list(df.columns)}")
    
    # Show sample of table prefixes detected
    table_prefixes = set()
    for col in df.columns:
        if '.' in col:
            prefix = col.split('.')[0]
            table_prefixes.add(prefix)
    
    if table_prefixes:
        print(f"🏷️  Detected table prefixes: {list(table_prefixes)}")
    else:
        print("ℹ️  No table prefixes detected")
    
    for idx, row in df.iterrows():
        # Convert row to COL/VAL format
        col_val_parts = []
        
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                # Remove table name prefix if present
                if '.' in col:
                    # Split by first dot only, in case there are multiple dots
                    clean_col = col.split('.', 1)[1]
                    print(f"🔧 Column: {col} -> {clean_col}") if idx == 0 else None  # Show for first record only
                else:
                    clean_col = col
                
                value = str(row[col]).strip()
                col_val_parts.append(f"COL {clean_col} VAL {value}")
        
        record_text = " ".join(col_val_parts)
        
        # Create JSONL record for matching
        # For demo purposes, using same record as left and right
        # You can modify this logic based on your specific matching needs
        record = {
            "left": record_text,
            "right": record_text,  # Modify this for actual matching pairs
            "id": idx
        }
        records.append(record)
    
    print(f"✅ Successfully converted {len(records)} records")
    
    # Show column transformation summary
    if table_prefixes:
        print("\n📋 Column transformations applied:")
        sample_cols = [col for col in df.columns if '.' in col][:5]  # Show first 5
        for col in sample_cols:
            clean_col = col.split('.', 1)[1]
            print(f"  {col} → {clean_col}")
        if len([col for col in df.columns if '.' in col]) > 5:
            remaining = len([col for col in df.columns if '.' in col]) - 5
            print(f"  ... and {remaining} more columns")
    
    return records

def convert_to_ditto_format_enhanced(df: pd.DataFrame, 
                                   remove_prefixes: List[str] = None,
                                   create_pairs: bool = False) -> List[Dict[str, Any]]:
    """
    Enhanced DITTO format converter with additional options.
    
    Args:
        df: DataFrame with columns that may have tablename.column format
        remove_prefixes: Specific table prefixes to remove (optional)
        create_pairs: Whether to create actual matching pairs instead of self-pairs
        
    Returns:
        List of DITTO records in JSONL format
    """
    records = []
    
    print(f"🔍 Processing {len(df)} records with {len(df.columns)} columns")
    
    # Analyze column structure
    prefixed_cols = [col for col in df.columns if '.' in col]
    clean_cols = [col for col in df.columns if '.' not in col]
    
    print(f"📊 Column analysis:")
    print(f"  - Prefixed columns: {len(prefixed_cols)}")
    print(f"  - Clean columns: {len(clean_cols)}")
    
    if prefixed_cols:
        # Detect all prefixes
        all_prefixes = set()
        for col in prefixed_cols:
            prefix = col.split('.')[0]
            all_prefixes.add(prefix)
        
        print(f"🏷️  Detected prefixes: {sorted(all_prefixes)}")
        
        # Use specified prefixes or auto-detected ones
        if remove_prefixes:
            prefixes_to_remove = set(remove_prefixes)
            print(f"🎯 Removing specified prefixes: {prefixes_to_remove}")
        else:
            prefixes_to_remove = all_prefixes
            print(f"🎯 Removing all detected prefixes: {prefixes_to_remove}")
    
    # Process each row
    for idx, row in df.iterrows():
        col_val_parts = []
        
        for col in df.columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                # Clean column name
                if '.' in col:
                    parts = col.split('.', 1)
                    prefix = parts[0]
                    column_name = parts[1]
                    
                    # Remove prefix if it's in our removal list
                    if not remove_prefixes or prefix in remove_prefixes:
                        clean_col = column_name
                    else:
                        clean_col = col  # Keep original if prefix not in removal list
                else:
                    clean_col = col
                
                value = str(row[col]).strip()
                col_val_parts.append(f"COL {clean_col} VAL {value}")
        
        record_text = " ".join(col_val_parts)
        
        # Create record
        if create_pairs and idx < len(df) - 1:
            # Create pairs with next record for actual matching
            next_row = df.iloc[idx + 1]
            next_col_val_parts = []
            
            for col in df.columns:
                if pd.notna(next_row[col]) and str(next_row[col]).strip():
                    if '.' in col:
                        clean_col = col.split('.', 1)[1]
                    else:
                        clean_col = col
                    
                    value = str(next_row[col]).strip()
                    next_col_val_parts.append(f"COL {clean_col} VAL {value}")
            
            next_record_text = " ".join(next_col_val_parts)
            
            record = {
                "left": record_text,
                "right": next_record_text,
                "id": idx
            }
        else:
            # Self-pairing
            record = {
                "left": record_text,
                "right": record_text,
                "id": idx
            }
        
        records.append(record)
    
    return records

def save_ditto_records(records: List[Dict[str, Any]], 
                      output_file: str, 
                      show_samples: int = 3) -> str:
    """
    Save DITTO records to JSONL file and show samples.
    
    Args:
        records: List of DITTO records
        output_file: Output file path
        show_samples: Number of sample records to display
        
    Returns:
        Path to saved file
    """
    # Save to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved {len(records)} records to: {output_file}")
    
    # Show sample records
    if show_samples > 0:
        print(f"\n📋 Sample DITTO records (showing first {show_samples}):")
        for i, record in enumerate(records[:show_samples]):
            print(f"\nRecord {i+1}:")
            print(f"  ID:    {record['id']}")
            
            # Truncate long records for display
            left_display = record['left'][:150] + "..." if len(record['left']) > 150 else record['left']
            right_display = record['right'][:150] + "..." if len(record['right']) > 150 else record['right']
            
            print(f"  Left:  {left_display}")
            print(f"  Right: {right_display}")
            
            # Show if they're the same or different
            if record['left'] == record['right']:
                print(f"  Type:  Self-pair")
            else:
                print(f"  Type:  Different records")
    
    return output_file

# Example usage functions
def demo_table_prefix_removal():
    """Demonstrate table prefix removal with sample data."""
    
    # Create sample data with table prefixes
    sample_data = {
        'users.id': [1, 2, 3],
        'users.name': ['Ahmed', 'Fatima', 'Omar'],
        'users.email': ['ahmed@example.com', 'fatima@example.com', 'omar@example.com'],
        'profile.age': [25, 30, 35],
        'profile.city': ['Casablanca', 'Rabat', 'Marrakech'],
        'clean_column': ['A', 'B', 'C']  # Column without prefix
    }
    
    df = pd.DataFrame(sample_data)
    
    print("🧪 Demo: Table Prefix Removal")
    print("=" * 40)
    print(f"Original DataFrame columns: {list(df.columns)}")
    print("\nSample data:")
    print(df.to_string())
    
    print("\n" + "=" * 40)
    print("🔄 Converting to DITTO format...")
    
    # Convert using basic function
    records = convert_to_ditto_format(df)
    
    # Show results
    print("\n📋 Converted records:")
    for i, record in enumerate(records):
        print(f"\nRecord {i+1}:")
        print(f"  {record['left']}")
    
    return records

if __name__ == "__main__":
    # Run demo
    demo_table_prefix_removal()
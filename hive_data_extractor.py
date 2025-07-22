import os
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from pyhive import hive
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


class HiveDataExtractor:
    """
    Class to extract data from Hive tables and prepare for ditto matching pipeline.
    """
    
    def __init__(self, host: str, port: int = 10000, username: str = None, database: str = "default"):
        """Initialize Hive connection parameters."""
        self.host = host
        self.port = port
        self.username = username
        self.database = database
        self.connection = None
    
    def connect(self):
        """Establish connection to Hive."""
        try:
            self.connection = hive.Connection(
                host=self.host,
                port=self.port,
                username=self.username,
                database=self.database
            )
            print(f"Connected to Hive at {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Hive: {str(e)}")
    
    def close(self):
        """Close Hive connection."""
        if self.connection:
            self.connection.close()
    
    def extract_table_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Extract data from a Hive table.
        
        Args:
            table_name: Name of the Hive table
            limit: Optional limit for number of rows
            
        Returns:
            pandas DataFrame with table data
        """
        if not self.connection:
            self.connect()
        
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql(query, self.connection)
            print(f"Extracted {len(df)} rows from {table_name}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to extract data from {table_name}: {str(e)}")
    
    def convert_to_ditto_format(self, df: pd.DataFrame) -> List[str]:
        """
        Convert DataFrame to DITTO COL/VAL format, removing table name prefixes from columns.
        
        Args:
            df: DataFrame with columns that may have tablename.column format
            
        Returns:
            List of DITTO formatted strings
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
            records.append(record_text)
        
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

    def create_cartesian_pairs(self, table1_df: pd.DataFrame, table2_df: pd.DataFrame) -> List[List[str]]:
        """
        Create cartesian product pairs between two dataframes for ditto matching.
        
        Args:
            table1_df: First dataframe
            table2_df: Second dataframe
            
        Returns:
            List of [left_record, right_record] pairs in DITTO format
        """
        # Convert dataframes to DITTO format
        table1_records = self.convert_to_ditto_format(table1_df)
        table2_records = self.convert_to_ditto_format(table2_df)
        
        pairs = []
        
        for record1 in table1_records:
            for record2 in table2_records:
                # Each pair is a list with [left_record, right_record] as expected by matcher.py
                pairs.append([record1, record2])
        
        print(f"Created {len(pairs)} pairs from cartesian product")
        return pairs
    
    def save_pairs_to_jsonl(self, pairs: List[List[str]], output_path: str):
        """
        Save pairs to JSONL format for ditto processing.
        
        Args:
            pairs: List of [left_record, right_record] pairs to save
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                # Each pair is already a list [left_record, right_record] as expected by matcher.py
                json.dump(pair, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(pairs)} pairs to {output_path}")
        
        # Show sample records for verification
        print("\n📋 Sample JSONL format for matcher:")
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 lines
                    break
                sample_data = json.loads(line)
                print(f"\nLine {i+1}: {len(sample_data)} elements")
                print(f"  [0]: {sample_data[0][:80]}...")
                print(f"  [1]: {sample_data[1][:80]}...")
    
    def extract_and_pair(self, table1: str, table2: str, output_path: str, 
                        table1_limit: Optional[int] = None, table2_limit: Optional[int] = None):
        """
        Complete workflow to extract data from two tables and create pairs.
        
        Args:
            table1: First Hive table name
            table2: Second Hive table name
            output_path: Output JSONL file path
            table1_limit: Optional limit for first table
            table2_limit: Optional limit for second table
        """
        try:
            # Extract data from both tables
            print(f"Extracting data from {table1}...")
            df1 = self.extract_table_data(table1, table1_limit)
            
            print(f"Extracting data from {table2}...")
            df2 = self.extract_table_data(table2, table2_limit)
            
            # Create cartesian pairs
            print("Creating cartesian product pairs...")
            pairs = self.create_cartesian_pairs(df1, df2)
            
            # Save to JSONL format
            self.save_pairs_to_jsonl(pairs, output_path)
            
            print(f"Successfully created pairs file: {output_path}")
            
        except Exception as e:
            print(f"Error in extract_and_pair: {str(e)}")
            raise
        finally:
            self.close()


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Extract data from Hive and create pairs for ditto matching")
    
    parser.add_argument("--hive-host", required=True, help="Hive server host")
    parser.add_argument("--hive-port", type=int, default=10000, help="Hive server port")
    parser.add_argument("--hive-user", help="Hive username")
    parser.add_argument("--hive-database", default="default", help="Hive database name")
    
    parser.add_argument("--table1", required=True, help="First Hive table name")
    parser.add_argument("--table2", required=True, help="Second Hive table name")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    
    parser.add_argument("--table1-limit", type=int, help="Limit rows from first table")
    parser.add_argument("--table2-limit", type=int, help="Limit rows from second table")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = HiveDataExtractor(
        host=args.hive_host,
        port=args.hive_port,
        username=args.hive_user,
        database=args.hive_database
    )
    
    # Extract and create pairs
    extractor.extract_and_pair(
        table1=args.table1,
        table2=args.table2,
        output_path=args.output,
        table1_limit=args.table1_limit,
        table2_limit=args.table2_limit
    )


if __name__ == "__main__":
    main()
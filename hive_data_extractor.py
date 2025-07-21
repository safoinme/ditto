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
    
    def create_cartesian_pairs(self, table1_df: pd.DataFrame, table2_df: pd.DataFrame) -> List[Dict]:
        """
        Create cartesian product pairs between two dataframes for ditto matching.
        
        Args:
            table1_df: First dataframe
            table2_df: Second dataframe
            
        Returns:
            List of dictionaries in ditto format
        """
        pairs = []
        
        for _, row1 in table1_df.iterrows():
            for _, row2 in table2_df.iterrows():
                # Convert rows to dictionaries, handling NaN values
                dict1 = row1.to_dict()
                dict2 = row2.to_dict()
                
                # Clean up NaN values
                dict1 = {k: (str(v) if pd.notna(v) else "") for k, v in dict1.items()}
                dict2 = {k: (str(v) if pd.notna(v) else "") for k, v in dict2.items()}
                
                pairs.append([dict1, dict2])
        
        print(f"Created {len(pairs)} pairs from cartesian product")
        return pairs
    
    def save_pairs_to_jsonl(self, pairs: List[Dict], output_path: str):
        """
        Save pairs to JSONL format for ditto processing.
        
        Args:
            pairs: List of pairs to save
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pair in pairs:
                json.dump(pair, f)
                f.write('\n')
        
        print(f"Saved {len(pairs)} pairs to {output_path}")
    
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
# Hive-DITTO Entity Matching Pipeline Setup Guide

This guide provides step-by-step instructions to set up and run the Hive-DITTO entity matching pipeline that works exactly like the tested notebook implementation.

## 🚀 Quick Start Summary

1. **Test the pipeline** using the interactive notebook
2. **Run standalone script** for command-line execution  
3. **Generate Kubeflow pipeline** for production deployment
4. **Upload to Kubeflow UI** and run at scale

## 📋 Prerequisites

- Hive cluster accessible from your environment
- Docker registry for storing container images
- Kubeflow cluster with pipelines enabled
- DITTO model checkpoints available

## 🧪 Step 1: Test with Interactive Notebook

First, ensure everything works using the interactive notebook that we've already validated:

```bash
# Launch Jupyter notebook
jupyter notebook Hive_DITTO_Testing_Notebook.ipynb
```

**Key Configuration in Notebook:**
```python
HIVE_CONFIG = {
    'host': 'your-hive-host',      # Replace with your Hive host
    'port': 10000,
    'database': 'default',
    'username': 'hive',
    'auth': 'NOSASL'
}

DATA_CONFIG = {
    'input_table': 'base.table',    # Replace with your table
    'sample_size': 1000,            # Start small for testing
    'matching_mode': 'auto'         # Options: 'auto', 'production', 'testing'
}
```

**🏭 Production Mode vs 🧪 Testing Mode:**

**Production Mode** (columns with `_left` and `_right` suffixes):
```sql
-- Table structure for production mode
SELECT 
    record_id,
    firstname_left, firstname_right,
    lastname_left, lastname_right,
    email_left, email_right
FROM comparison_pairs_table;
```

**Testing Mode** (regular columns for self-matching):
```sql
-- Table structure for testing mode  
SELECT 
    person_id,
    firstname,
    lastname, 
    email,
    phone
FROM persons_table;
```

**✅ Validation Checklist:**
- [ ] Hive connection successful
- [ ] Data extraction working
- [ ] Table name prefixes properly removed (tablename.column → column)
- [ ] DITTO format conversion successful
- [ ] Matcher runs without errors
- [ ] Results analysis shows meaningful output

## 🖥️ Step 2: Run Standalone Script

Once the notebook works, test the equivalent standalone script:

```bash
# Test the modes first
python demo_production_testing_modes.py

# Run for production mode (with _left/_right columns)
python hive_ditto_standalone.py \
    --hive-host "your-hive-host" \
    --input-table "comparison_pairs_table" \
    --matching-mode "production" \
    --sample-size 100 \
    --use-gpu \
    --fp16

# Run for testing mode (self-matching)
python hive_ditto_standalone.py \
    --hive-host "your-hive-host" \
    --input-table "persons_table" \
    --matching-mode "testing" \
    --sample-size 100 \
    --use-gpu \
    --fp16

# Auto-detect mode (default)
python hive_ditto_standalone.py \
    --hive-host "your-hive-host" \
    --input-table "base.table" \
    --matching-mode "auto" \
    --sample-size 100
```

**Optional: Save results to Hive**
```bash
python hive_ditto_standalone.py \
    --hive-host "your-hive-host" \
    --input-table "base.table" \
    --output-table "results.ditto_matches" \
    --matching-mode "auto" \
    --sample-size 100
```

**✅ Validation Checklist:**
- [ ] Same results as notebook
- [ ] Proper table prefix handling
- [ ] Correct JSONL format for matcher
- [ ] Performance acceptable

## 🐳 Step 3: Build Docker Image

Build the container image for Kubeflow deployment:

```bash
# Build the image
docker build -f Dockerfile.kubeflow -t your-registry/ditto-hive:latest .

# Push to your registry
docker push your-registry/ditto-hive:latest
```

**Update the pipeline image reference:**
```bash
# Edit ditto_kubeflow_pipeline.py
sed -i 's/your-registry\/ditto-kubeflow:latest/your-registry\/ditto-hive:latest/g' ditto_kubeflow_pipeline.py
```

## ⚙️ Step 4: Generate Kubeflow Pipeline

Create the pipeline YAML file for Kubeflow:

```bash
# Generate pipeline YAML
python ditto_kubeflow_pipeline.py \
    --input-table "base.table" \
    --hive-host "your-hive-host" \
    --output "hive-ditto-pipeline.yaml"
```

**Example with custom name:**
```bash
python ditto_kubeflow_pipeline.py \
    --input-table "sales.customers" \
    --hive-host "hive.company.com" \
    --output "customer-matching-pipeline.yaml"
```

**✅ Generated Files:**
- [ ] `hive-ditto-pipeline.yaml` - Main pipeline definition
- [ ] No compilation errors
- [ ] Pipeline includes all 3 steps (extract, match, save)

## 🚀 Step 5: Deploy to Kubeflow

### Upload Pipeline to Kubeflow UI

1. **Access Kubeflow Dashboard**
   ```bash
   # If using port-forward
   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
   # Access: http://localhost:8080
   ```

2. **Upload Pipeline**
   - Go to **Pipelines** → **Upload Pipeline**
   - Select your generated `hive-ditto-pipeline.yaml`
   - Name: `Hive-DITTO-EntityMatching`
   - Description: `Entity matching using DITTO with Hive integration`

3. **Create Pipeline Run**
   - Click **Create Run**
   - Choose your uploaded pipeline
   - Configure parameters (see below)

### Pipeline Parameters Configuration

**Required Parameters:**
```yaml
hive_host: "your-hive-host"
hive_user: "hive"
input_table: "database.table_name"
```

**Optional Parameters:**
```yaml
hive_port: 10000
hive_database: "default"
sample_limit: 1000              # For testing
model_task: "person_records"
lm: "bert"
max_len: 64
use_gpu: true
fp16: true
save_to_hive: false            # Set true to save results
output_table: "results.ditto_matches"
```

**Resource Configuration:**
```yaml
# For GPU nodes
use_gpu: true
# Memory allocation will be automatic based on component definitions
```

## 📊 Step 6: Monitor and Validate

### Pipeline Execution Monitoring

1. **Check Pipeline Status**
   - Monitor in Kubeflow UI under **Runs**
   - Check each step completion
   - Review logs for any errors

2. **Component-Level Monitoring**
   - **Extract Step**: Verify data extraction and format conversion
   - **Matching Step**: Monitor GPU utilization and processing time
   - **Save Step**: Confirm results written to Hive (if enabled)

### Validation Steps

1. **Data Format Validation**
   ```bash
   # Check generated JSONL format
   head -n 3 /data/input/test_pairs.jsonl
   # Should show: ["COL field1 VAL value1...", "COL field1 VAL value1..."]
   ```

2. **Results Validation**
   ```bash
   # Check output format
   head -n 3 /data/output/matching_results.jsonl
   # Should include match confidence and decisions
   ```

3. **Hive Table Verification** (if save_to_hive=true)
   ```sql
   SELECT COUNT(*) FROM results.ditto_matches;
   SELECT * FROM results.ditto_matches LIMIT 5;
   ```

## 🔧 Configuration Templates

### Basic Configuration
```bash
# For simple self-matching
python ditto_kubeflow_pipeline.py \
    --input-table "customers.main_table" \
    --hive-host "hive.internal" \
    --output "basic-matching.yaml"
```

### Advanced Configuration
```bash
# For production with GPU and result saving
python ditto_kubeflow_pipeline.py \
    --input-table "products.catalog" \
    --hive-host "hive.prod.company.com" \
    --output "prod-product-matching.yaml"

# Then in Kubeflow UI, set:
# use_gpu: true
# fp16: true
# save_to_hive: true
# output_table: "analytics.product_matches"
```

## 🐛 Troubleshooting

### Common Issues

1. **Table Prefix Issues**
   ```
   Error: Columns still have table prefixes
   Solution: The pipeline now automatically handles tablename.column → column
   ```

2. **JSONL Format Errors**
   ```
   Error: IndexError: list index out of range
   Solution: Pipeline now generates correct [left, right] format for matcher
   ```

3. **Hive Connection Failures**
   ```bash
   # Test connection manually
   python -c "
   from pyhive import hive
   conn = hive.Connection(host='your-host', port=10000)
   print('Connected successfully')
   "
   ```

4. **GPU Not Available**
   ```yaml
   # In pipeline parameters, set:
   use_gpu: false
   fp16: false
   ```

5. **Memory Issues**
   ```yaml
   # Reduce data size for testing:
   sample_limit: 100
   max_len: 32
   ```

### Debug Commands

```bash
# Check Docker image
docker run --rm your-registry/ditto-hive:latest python --version

# Validate pipeline YAML
python -c "import yaml; yaml.safe_load(open('hive-ditto-pipeline.yaml'))"

# Test Hive connectivity from container
docker run --rm your-registry/ditto-hive:latest \
    python -c "from pyhive import hive; print('PyHive available')"
```

## 📈 Performance Tuning

### Data Volume Optimization
```yaml
# For large datasets:
sample_limit: null          # Process all data
use_gpu: true              # Enable GPU acceleration
fp16: true                 # Use half precision
```

### Resource Allocation
```yaml
# Pipeline automatically sets:
# Extract step: 2 CPU, 4GB RAM
# Matching step: 4 CPU, 8GB RAM, 1 GPU
# Save step: 2 CPU, 4GB RAM
```

## 🔄 Pipeline Updates

To update the pipeline with new features:

1. **Modify the standalone script** first
2. **Test with notebook** to validate
3. **Update Kubeflow pipeline** code
4. **Rebuild Docker image**
5. **Generate new YAML**
6. **Upload to Kubeflow**

## ✅ Success Criteria

Your pipeline is ready for production when:

- [ ] Notebook runs successfully end-to-end
- [ ] Standalone script produces same results as notebook
- [ ] Docker image builds without errors
- [ ] Pipeline YAML generates successfully
- [ ] Kubeflow accepts and runs the pipeline
- [ ] Results show meaningful match rates and confidence scores
- [ ] Table name prefixes are properly removed
- [ ] JSONL format is correct for DITTO matcher

## 📞 Support

For issues:
1. **Check logs** in Kubeflow UI pipeline runs
2. **Validate** with standalone script first
3. **Test** Hive connectivity separately
4. **Verify** Docker image includes all dependencies

The pipeline is designed to work exactly as tested in the notebook, ensuring consistent behavior across development and production environments.
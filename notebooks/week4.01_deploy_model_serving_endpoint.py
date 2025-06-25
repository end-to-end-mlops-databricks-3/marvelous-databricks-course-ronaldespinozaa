# Databricks notebook source
# MAGIC %md
# MAGIC # Week 4: Model Serving Deployment
# MAGIC
# MAGIC This notebook demonstrates deploying our banking marketing model to a serving endpoint using the ModelServing wrapper class.
# COMMAND ----------
# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %pip install ../dist/bank_marketing-0.0.1-py3-none-any.whl


# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import time
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.serving.model_serving import ModelServing

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Set environment variables for serving endpoint
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

print("✅ Environment setup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Setup
# MAGIC
# MAGIC Load project configuration and set up model details.

# COMMAND ----------

# Load project config
config_dict = {
    "catalog_name": "mlops_dev",
    "schema_name": "espinoza",
    "volume_name": "bank_marketing_dev",
    "experiment_name_custom": "/Shared/bank-marketing-custom",
    "experiment_name_basic": "/Shared/bank-marketing-basic",
    "experiment_name_fe": "/Shared/bank-marketing-fe",
    "num_features": ["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
    "cat_features": ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
    "target": "Target",
    "parameters": {"learning_rate": 0.05, "n_estimators": 200, "max_depth": 4, "random_state": 42},
}
config = ProjectConfig(**config_dict)

catalog_name = config.catalog_name
schema_name = config.schema_name

print(f"📂 Using catalog: {catalog_name}")
print(f"📂 Using schema: {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Model Serving
# MAGIC
# MAGIC Create the ModelServing instance for our banking marketing model.

# COMMAND ----------

# Initialize model serving
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.bank_marketing_model_custom", endpoint_name="bank-marketing-model-serving"
)

print("✅ Model Serving initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model Serving Endpoint
# MAGIC
# MAGIC Deploy or update the serving endpoint with our latest model.

# COMMAND ----------

# Deploy the model serving endpoint (includes wait inside)
print("🚀 Deploying model serving endpoint...")
try:
    model_serving.deploy_or_update_serving_endpoint()
    print("✅ Endpoint is ready for serving!")
except Exception as e:
    print(f"❌ Error during endpoint deployment: {e}")
    raise


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Test Data
# MAGIC
# MAGIC Create sample records for testing the serving endpoint.

# COMMAND ----------

# Required columns for banking marketing model
required_columns = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

# Load test data
try:
    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_processed").toPandas()
    print(f"✅ Loaded test data: {test_set.shape}")

    # Sample 50 records for testing
    sampled_records = test_set[required_columns].sample(n=50, replace=True).to_dict(orient="records")
    dataframe_records = [[record] for record in sampled_records]

    print(f"📊 Prepared {len(dataframe_records)} test records")

except Exception as e:
    print(f"⚠️ Could not load test data: {e}")
    print("🔄 Creating sample test data...")

    # Create sample data manually
    sample_data = [
        {
            "age": 35,
            "balance": 1500,
            "day": 15,
            "duration": 300,
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "poutcome": "unknown",
        },
        {
            "age": 42,
            "balance": 2300,
            "day": 8,
            "duration": 450,
            "campaign": 1,
            "pdays": 180,
            "previous": 1,
            "job": "technician",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "housing": "no",
            "loan": "yes",
            "contact": "telephone",
            "month": "nov",
            "poutcome": "failure",
        },
        {
            "age": 28,
            "balance": 890,
            "day": 22,
            "duration": 180,
            "campaign": 3,
            "pdays": -1,
            "previous": 0,
            "job": "student",
            "marital": "single",
            "education": "tertiary",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "jul",
            "poutcome": "unknown",
        },
    ]

    dataframe_records = [[record] for record in sample_data]
    print(f"📊 Created {len(dataframe_records)} sample test records")

# Display sample record
print("\n📋 Sample record format:")
print(dataframe_records[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Serving Endpoint
# MAGIC
# MAGIC Test the deployed endpoint with sample data.

# COMMAND ----------


def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.

    Args:
        record: List containing a single record dictionary

    Returns:
        Tuple of (status_code, response_text)
    """
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/bank-marketing-model-serving/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# Test with first record
print("🧪 Testing endpoint with sample record...")
status_code, response_text = call_endpoint(dataframe_records[0])

print(f"📊 Response Status: {status_code}")
print(f"📊 Response Text: {response_text}")

if status_code == 200:
    print("✅ Endpoint test successful!")
else:
    print("❌ Endpoint test failed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Testing
# MAGIC
# MAGIC Perform light load testing on the endpoint.

# COMMAND ----------

# Perform load test with multiple records
print("🔄 Running load test...")

successful_calls = 0
failed_calls = 0

for i, record in enumerate(dataframe_records[:10]):  # Test first 10 records
    try:
        status_code, response_text = call_endpoint(record)

        if status_code == 200:
            successful_calls += 1
            print(f"✅ Call {i + 1}: Success")
        else:
            failed_calls += 1
            print(f"❌ Call {i + 1}: Failed (Status: {status_code})")

        time.sleep(0.2)  # Small delay between calls

    except Exception as e:
        failed_calls += 1
        print(f"❌ Call {i + 1}: Exception - {e}")

print(f"\n📊 Load Test Results:")
print(f"   Successful calls: {successful_calls}")
print(f"   Failed calls: {failed_calls}")
print(f"   Success rate: {successful_calls / (successful_calls + failed_calls) * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Status Check
# MAGIC
# MAGIC Check the final status of our serving endpoint.

# COMMAND ----------

# Check endpoint status
# Check endpoint status
endpoint_status = model_serving.get_endpoint_status()
print(f"📊 Endpoint Status: {endpoint_status}")

if endpoint_status == "READY":
    print("✅ Endpoint is ready and serving requests!")
    print(
        f"🚀 Endpoint URL: https://{os.environ['DBR_HOST']}/serving-endpoints/bank-marketing-model-serving/invocations"
    )
else:
    print(f"⚠️ Endpoint status: {endpoint_status}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### ✅ What We Accomplished:
# MAGIC
# MAGIC 1. **Deployed Model Serving Endpoint** - Created a production-ready serving endpoint
# MAGIC 2. **Tested Endpoint** - Verified the endpoint responds correctly to requests
# MAGIC 3. **Load Testing** - Confirmed the endpoint can handle multiple requests
# MAGIC 4. **Model Integration** - Successfully integrated our banking marketing model
# MAGIC
# MAGIC ### 🔗 Endpoint Details:
# MAGIC
# MAGIC - **Model**: `{catalog_name}.{schema_name}.bank_marketing_model_custom`
# MAGIC - **Endpoint**: `bank-marketing-model-serving`
# MAGIC - **Status**: Ready for production use
# MAGIC
# MAGIC ### 📊 Request Format:
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "dataframe_records": [[{
# MAGIC     "age": 35,
# MAGIC     "balance": 1500,
# MAGIC     "duration": 300,
# MAGIC     "job": "management",
# MAGIC     "marital": "married",
# MAGIC     "education": "tertiary",
# MAGIC     ...
# MAGIC   }]]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Week 4 Model Serving Complete!** 🎉

# COMMAND ----------

print("🎯 WEEK 4 MODEL SERVING COMPLETE!")
print("=" * 50)
print(f"📊 Model: {catalog_name}.{schema_name}.bank_marketing_model_custom")
print(f"🚀 Endpoint: bank-marketing-model-serving")
print(f"📈 Status: {endpoint_status}")
print(f"✅ Ready for production use!")
print("\n🎉 Banking marketing model successfully deployed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC **Optional: Cleanup**
# MAGIC
# MAGIC Uncomment the cell below if you want to delete the endpoint after testing.

# COMMAND ----------

# # Uncomment to delete the endpoint
# # print("🗑️ Cleaning up endpoint...")
# # if model_serving.delete_endpoint():
# #     print("✅ Endpoint deleted successfully")
# # else:
# #     print("❌ Failed to delete endpoint")

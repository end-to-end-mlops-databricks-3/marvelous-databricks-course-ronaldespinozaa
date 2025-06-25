"""Model Serving implementation for Bank Marketing dataset.

This module implements model serving endpoints for the banking model
using Databricks Model Serving capabilities - simplified version matching house price pattern.
"""

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger


class ModelServing:
    """Model Serving implementation for Banking models - simplified version."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving."""
        self.model_name = model_name
        self.endpoint_name = endpoint_name
        self.workspace_client = WorkspaceClient()

        logger.info("✅ Model Serving initialized")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Endpoint: {self.endpoint_name}")

    def deploy_or_update_serving_endpoint(self, model_version: str | None = None) -> None:
        """Deploy or update the model serving endpoint."""
        logger.info(f"🚀 Deploying/updating serving endpoint: {self.endpoint_name}")

        # Use version 1 if not specified
        if model_version is None:
            model_version = "1"

        try:
            # Check if endpoint already exists
            try:
                # existing_endpoint = self.workspace_client.serving_endpoints.get(self.endpoint_name)
                logger.info(f"⚠️ Endpoint {self.endpoint_name} already exists - updating...")
                self._update_endpoint(model_version)
                self.wait_for_endpoint_ready()

                return
            except Exception:
                # Endpoint doesn't exist, create new one
                logger.info(f"📝 Creating new endpoint: {self.endpoint_name}")
                self._create_endpoint(model_version)
                self.wait_for_endpoint_ready()

                return
        except Exception as e:
            logger.error(f"❌ Error deploying endpoint: {e}")
            raise

    def _create_endpoint(self, model_version: str) -> None:
        """Create a new serving endpoint."""
        # Define the served entity
        served_entity = ServedEntityInput(
            entity_name=self.model_name, entity_version=model_version, workload_size="Small", scale_to_zero_enabled=True
        )

        # Define endpoint configuration
        endpoint_config = EndpointCoreConfigInput(name=self.endpoint_name, served_entities=[served_entity])

        # Create the endpoint
        response = self.workspace_client.serving_endpoints.create_and_wait(
            name=self.endpoint_name, config=endpoint_config
        )

        logger.info("✅ Serving endpoint created successfully")
        logger.info(f"   Status: {response.state}")

    def _update_endpoint(self, model_version: str) -> None:
        """Update an existing serving endpoint."""
        # Define the served entity
        served_entity = ServedEntityInput(
            entity_name=self.model_name, entity_version=model_version, workload_size="Small", scale_to_zero_enabled=True
        )

        # Update the endpoint
        response = self.workspace_client.serving_endpoints.update_config_and_wait(
            name=self.endpoint_name, served_entities=[served_entity]
        )

        logger.info("✅ Serving endpoint updated successfully")
        logger.info(f"   Status: {response.state}")

    def get_endpoint_status(self) -> str:
        """Get the status of the serving endpoint."""
        try:
            endpoint = self.workspace_client.serving_endpoints.get(self.endpoint_name)
            if endpoint.state and endpoint.state.ready:
                return "READY"
            else:
                return "NOT_READY"
        except Exception as e:
            logger.error(f"❌ Error getting endpoint status: {e}")
            return "ERROR"

    def delete_endpoint(self) -> bool:
        """Delete the serving endpoint."""
        logger.info(f"🗑️ Deleting serving endpoint: {self.endpoint_name}")

        try:
            self.workspace_client.serving_endpoints.delete(self.endpoint_name)
            logger.info("✅ Serving endpoint deleted successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting serving endpoint: {e}")
            return False

    def wait_for_endpoint_ready(self, timeout_minutes: int = 10) -> bool:
        """Wait for the endpoint to be ready."""
        logger.info("⏳ Waiting for endpoint to be ready...")

        timeout_seconds = timeout_minutes * 60
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            status = self.get_endpoint_status()

            if status == "READY":
                logger.info("✅ Endpoint is ready!")
                return True
            elif status in ["FAILED", "STOPPED"]:
                logger.error(f"❌ Endpoint failed with status: {status}")
                return False
            else:
                logger.info(f"   Current status: {status}")
                time.sleep(30)

        logger.error("❌ Timeout waiting for endpoint to be ready")
        return False

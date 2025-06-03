"""Table management utilities for working with Delta tables in Unity Catalog.

This module provides a TableManager class that simplifies working with tables and schemas
in Databricks Unity Catalog, supporting common operations needed in MLOps pipelines.
"""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from bank_marketing.config import ProjectConfig


class TableManager:
    """Manages Databricks Unity Catalog tables for MLOps projects."""

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """Initialize TableManager with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: Project configuration containing catalog and schema info

        """
        self.spark = spark
        self.config = config
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.full_schema = f"{self.catalog_name}.{self.schema_name}"

    def ensure_schema_exists(self) -> None:
        """Create the catalog and schema if they don't exist."""
        try:
            # Create catalog if it doesn't exist
            self.spark.sql(f"CREATE CATALOG IF NOT EXISTS {self.catalog_name}")
            logger.info(f"✅ Catalog {self.catalog_name} ensured")

            # Create schema if it doesn't exist
            self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.full_schema}")
            logger.info(f"✅ Schema {self.full_schema} ensured")

        except Exception as e:
            logger.error(f"❌ Failed to create catalog/schema: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the schema.

        Args:
            table_name: Name of the table (without schema prefix)

        Returns:
            True if table exists, False otherwise

        """
        full_table_name = f"{self.full_schema}.{table_name}"
        return self.spark.catalog.tableExists(full_table_name)

    def load_table(self, table_name: str) -> DataFrame:
        """Load a table from Unity Catalog.

        Args:
            table_name: Name of the table (without schema prefix)

        Returns:
            Spark DataFrame containing the table data

        Raises:
            ValueError: If table doesn't exist

        """
        full_table_name = f"{self.full_schema}.{table_name}"

        if not self.table_exists(table_name):
            available_tables = self.list_tables()
            raise ValueError(f"Table {full_table_name} does not exist. Available tables: {available_tables}")

        try:
            df = self.spark.table(full_table_name)
            logger.info(f"📊 Loaded table {full_table_name} with {df.count()} rows")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load table {full_table_name}: {e}")
            raise

    def save_table(
        self, df: DataFrame, table_name: str, mode: str = "overwrite", partition_by: list[str] | None = None
    ) -> None:
        """Save DataFrame as a Delta table in Unity Catalog.

        Args:
            df: DataFrame to save
            table_name: Name of the table (without schema prefix)
            mode: Write mode ('overwrite', 'append', 'error', 'ignore')
            partition_by: Optional list of columns to partition by

        """
        full_table_name = f"{self.full_schema}.{table_name}"

        try:
            writer = df.write.mode(mode).format("delta")

            if partition_by:
                writer = writer.partitionBy(*partition_by)

            writer.saveAsTable(full_table_name)
            logger.info(f"✅ Saved table {full_table_name} with mode '{mode}'")

        except Exception as e:
            logger.error(f"❌ Failed to save table {full_table_name}: {e}")
            raise

    def create_ml_ready_table(self, base_table_name: str, target_table_suffix: str = "_processed") -> DataFrame:
        """Create ML-ready version of a table with numeric target conversion.

        Args:
            base_table_name: Name of the base table
            target_table_suffix: Suffix to add to create processed table name

        Returns:
            DataFrame with processed data

        """
        # Load base table
        df = self.load_table(base_table_name)
        target_table_name = f"{base_table_name}{target_table_suffix}"

        # Convert target column if it's string ("yes"/"no" -> 1/0)
        if self.config.target in df.columns:
            target_col_type = dict(df.dtypes)[self.config.target]

            if target_col_type == "string":
                logger.info(f"🔄 Converting target column '{self.config.target}' from string to binary")
                df = df.withColumn(self.config.target, F.when(F.col(self.config.target) == "yes", 1).otherwise(0))

        # Cast numeric columns to appropriate types
        numeric_columns = self.config.num_features
        for col_name in numeric_columns:
            if col_name in df.columns:
                df = df.withColumn(col_name, F.col(col_name).cast("double"))

        # Save processed table
        self.save_table(df, target_table_name, mode="overwrite")
        logger.info(f"✅ Created ML-ready table: {target_table_name}")

        return df

    def get_table_metadata(self, table_name: str) -> dict[str, Any]:
        """Get metadata information about a table.

        Args:
            table_name: Name of the table (without schema prefix)

        Returns:
            Dictionary containing table metadata

        """
        full_table_name = f"{self.full_schema}.{table_name}"

        if not self.table_exists(table_name):
            raise ValueError(f"Table {full_table_name} does not exist")

        try:
            # Get basic table info
            df = self.spark.table(full_table_name)
            row_count = df.count()
            column_count = len(df.columns)

            # Get Delta table details
            try:
                detail_df = self.spark.sql(f"DESCRIBE DETAIL {full_table_name}")
                detail_row = detail_df.collect()[0]

                metadata = {
                    "table_name": full_table_name,
                    "num_rows": row_count,
                    "num_columns": column_count,
                    "num_files": detail_row.get("num_files", 0),
                    "size_bytes": detail_row.get("size_bytes", 0),
                    "format": detail_row.get("format", "unknown"),
                    "location": detail_row.get("location", None),
                    "columns": df.columns,
                    "schema": df.schema.simpleString(),
                }
            except Exception:
                # Fallback for non-Delta tables
                metadata = {
                    "table_name": full_table_name,
                    "num_rows": row_count,
                    "num_columns": column_count,
                    "num_files": 0,
                    "size_bytes": 0,
                    "format": "unknown",
                    "location": None,
                    "columns": df.columns,
                    "schema": df.schema.simpleString(),
                }

            logger.info(f"📊 Retrieved metadata for {full_table_name}")
            return metadata

        except Exception as e:
            logger.error(f"❌ Failed to get metadata for {full_table_name}: {e}")
            raise

    def optimize_table(self, table_name: str, zorder_by: list[str] | None = None) -> None:
        """Optimize a Delta table for better query performance.

        Args:
            table_name: Name of the table (without schema prefix)
            zorder_by: Optional list of columns to Z-order by

        """
        full_table_name = f"{self.full_schema}.{table_name}"

        if not self.table_exists(table_name):
            logger.warning(f"⚠️ Table {full_table_name} does not exist, skipping optimization")
            return

        try:
            # Run OPTIMIZE command
            optimize_sql = f"OPTIMIZE {full_table_name}"

            if zorder_by:
                # Validate that Z-order columns exist
                df = self.spark.table(full_table_name)
                existing_columns = df.columns
                valid_zorder_columns = [col for col in zorder_by if col in existing_columns]

                if valid_zorder_columns:
                    zorder_clause = ", ".join(valid_zorder_columns)
                    optimize_sql += f" ZORDER BY ({zorder_clause})"
                    logger.info(f"🚀 Optimizing {full_table_name} with Z-order: {valid_zorder_columns}")
                else:
                    logger.warning(f"⚠️ None of the Z-order columns {zorder_by} exist in {full_table_name}")

            # Execute optimization
            self.spark.sql(optimize_sql)
            logger.info(f"✅ Optimized table {full_table_name}")

        except Exception as e:
            logger.error(f"❌ Failed to optimize table {full_table_name}: {e}")
            # Don't raise - optimization failure shouldn't break the pipeline

    def combine_features_from_tables(
        self, output_table: str, source_tables: list[str], join_on: str, target_column: str | None = None
    ) -> DataFrame:
        """Combine features from multiple tables into a single feature table.

        Args:
            output_table: Name of the output table
            source_tables: List of source table names
            join_on: Column name to join tables on
            target_column: Optional target column to include

        Returns:
            Combined DataFrame

        """
        logger.info(f"🔄 Combining features from {len(source_tables)} tables")

        combined_df = None

        for _i, table_name in enumerate(source_tables):
            if not self.table_exists(table_name):
                logger.warning(f"⚠️ Table {table_name} does not exist, skipping")
                continue

            try:
                df = self.load_table(table_name)

                if combined_df is None:
                    # First table - use as base
                    combined_df = df
                    logger.info(f"📊 Base table: {table_name} ({df.count()} rows)")
                else:
                    # Join with existing data
                    # Avoid duplicate columns (except join key)
                    df_columns = [col for col in df.columns if col != join_on or col not in combined_df.columns]
                    df_selected = df.select(join_on, *[col for col in df_columns if col != join_on])

                    combined_df = combined_df.join(df_selected, on=join_on, how="left")
                    logger.info(f"📊 Joined table: {table_name}")

            except Exception as e:
                logger.error(f"❌ Failed to process table {table_name}: {e}")
                continue

        if combined_df is None:
            raise ValueError("No valid source tables found")

        # Include target column if specified and available
        if target_column and target_column not in combined_df.columns:
            logger.warning(f"⚠️ Target column '{target_column}' not found in combined data")

        # Save combined table
        self.save_table(combined_df, output_table, mode="overwrite")
        logger.info(f"✅ Created combined feature table: {output_table}")

        return combined_df

    def list_tables(self) -> list[str]:
        """List all tables in the schema.

        Returns:
            List of table names (without schema prefix)

        """
        try:
            tables_df = self.spark.sql(f"SHOW TABLES IN {self.full_schema}")
            table_names = [row["tableName"] for row in tables_df.collect()]
            return table_names
        except Exception as e:
            logger.warning(f"⚠️ Failed to list tables in {self.full_schema}: {e}")
            return []

    def drop_table(self, table_name: str, if_exists: bool = True) -> None:
        """Drop a table from Unity Catalog.

        Args:
            table_name: Name of the table (without schema prefix)
            if_exists: If True, don't raise error if table doesn't exist

        """
        full_table_name = f"{self.full_schema}.{table_name}"

        try:
            if if_exists:
                self.spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")
            else:
                self.spark.sql(f"DROP TABLE {full_table_name}")

            logger.info(f"✅ Dropped table {full_table_name}")

        except Exception as e:
            logger.error(f"❌ Failed to drop table {full_table_name}: {e}")
            if not if_exists:
                raise

    def vacuum_table(self, table_name: str, retention_hours: int = 168) -> None:
        """Vacuum a Delta table to remove old files.

        Args:
            table_name: Name of the table (without schema prefix)
            retention_hours: Number of hours to retain old files (default: 7 days)

        """
        full_table_name = f"{self.full_schema}.{table_name}"

        if not self.table_exists(table_name):
            logger.warning(f"⚠️ Table {full_table_name} does not exist, skipping vacuum")
            return

        try:
            self.spark.sql(f"VACUUM {full_table_name} RETAIN {retention_hours} HOURS")
            logger.info(f"✅ Vacuumed table {full_table_name} (retention: {retention_hours}h)")

        except Exception as e:
            logger.error(f"❌ Failed to vacuum table {full_table_name}: {e}")

    def get_table_history(self, table_name: str, limit: int = 10) -> DataFrame:
        """Get the history of a Delta table.

        Args:
            table_name: Name of the table (without schema prefix)
            limit: Maximum number of history entries to return

        Returns:
            DataFrame containing table history

        """
        full_table_name = f"{self.full_schema}.{table_name}"

        if not self.table_exists(table_name):
            raise ValueError(f"Table {full_table_name} does not exist")

        try:
            history_df = self.spark.sql(f"DESCRIBE HISTORY {full_table_name} LIMIT {limit}")
            logger.info(f"📊 Retrieved history for {full_table_name}")
            return history_df

        except Exception as e:
            logger.error(f"❌ Failed to get history for {full_table_name}: {e}")
            raise

    def clone_table(self, source_table: str, target_table: str, shallow: bool = False) -> None:
        """Clone a Delta table.

        Args:
            source_table: Name of the source table
            target_table: Name of the target table
            shallow: If True, create a shallow clone (metadata only)

        """
        source_full_name = f"{self.full_schema}.{source_table}"
        target_full_name = f"{self.full_schema}.{target_table}"

        if not self.table_exists(source_table):
            raise ValueError(f"Source table {source_full_name} does not exist")

        try:
            clone_type = "SHALLOW" if shallow else "DEEP"
            self.spark.sql(f"CREATE TABLE {target_full_name} {clone_type} CLONE {source_full_name}")
            logger.info(f"✅ Created {clone_type.lower()} clone: {target_full_name}")

        except Exception as e:
            logger.error(f"❌ Failed to clone table {source_full_name} to {target_full_name}: {e}")
            raise

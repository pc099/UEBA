# threat_feed_ingestion.py
import csv
# import requests  # if fetching from an external API
from src.Knowledge_pillar.knowledge_manager import KnowledgeManager

def ingest_threat_feed(csv_file_path: str):
    """
    Reads a CSV of threat indicators and updates them in the knowledge base.
    CSV format (example): threat_name,ioc_type,value,source,confidence
    """
    knowledge_manager = KnowledgeManager(uri="bolt://localhost:7687", user="neo4j", password="your_neo4j_password")

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            threat_node = knowledge_manager.upsert_threat_indicator({
                "threat_name": row.get("threat_name"),
                "ioc_type": row.get("ioc_type"),
                "value": row.get("value"),
                "source": row.get("source"),
                "confidence": row.get("confidence"),
            })

            # Optionally correlate newly ingested threat with known anomalies or hosts
            if threat_node:
                # Link to a host if you know the host IP matches the threat's IP
                if row.get("ioc_type") == "ip":
                    knowledge_manager.link_threat_to_host(threat_node, row.get("value"))

                # Optionally correlate with anomalies
                correlated_anomalies = knowledge_manager.correlate_threat_with_anomalies(threat_node)
                if correlated_anomalies:
                    print(f"Threat {threat_node['value']} correlates with {len(correlated_anomalies)} anomalies in the last 7 days!")

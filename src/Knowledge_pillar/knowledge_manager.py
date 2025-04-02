# knowledge_manager.py
from py2neo import Graph, Node, Relationship
from datetime import datetime
from typing import Dict, Any

class KnowledgeManager:

    def __init__(self, uri="Host", user="neo4j", password="password"):
        """
        Connect to Neo4j. Replace with your own credentials/URI.
        """
        self.graph = Graph(uri, auth=(user, password))

    def create_or_get_host_node(self, ip: str) -> Node:
        """
        Create or retrieve a Host node by IP.
        """
        existing = self.graph.nodes.match("Host", ip=ip).first()
        if existing:
            return existing
        else:
            host_node = Node("Host", ip=ip, created_at=str(datetime.now()))
            self.graph.create(host_node)
            return host_node

    def create_anomaly_node(self, anomaly_data: Dict[str, Any]) -> Node:
        """
        Store an anomaly with relevant info. Returns the newly created Node.
        anomaly_data can contain keys like:
            - 'timestamp'
            - 'anomaly_score'
            - 'label'
            - 'source_ip'
            - 'destination_ip'
            etc.
        """
        anomaly_node = Node("Anomaly", **anomaly_data)
        anomaly_node["created_at"] = datetime.now()
        self.graph.create(anomaly_node)
        return anomaly_node

    def link_anomaly_to_host(self, anomaly_node: Node, ip: str):
        """
        Link an Anomaly node to a host IP (source or destination) and
        store label, timestamp, source/destination IP in the relationship.
        """
        host_node = self.create_or_get_host_node(ip)

        # Safely retrieve fields from the anomaly_node (use .get if not guaranteed present)
        label = anomaly_node["label"] if "label" in anomaly_node else "Unknown"
        timestamp = anomaly_node["timestamp"] if "timestamp" in anomaly_node else "Unknown"
        source_ip = anomaly_node["source_ip"] if "source_ip" in anomaly_node else "Unknown"
        destination_ip = anomaly_node["destination_ip"] if "destination_ip" in anomaly_node else "Unknown"

        # Create the relationship with extra properties
        rel = Relationship(
            host_node,
            "ASSOCIATED_WITH",
            anomaly_node,
            label=label,
            timestamp=timestamp,
            source_ip=source_ip,
            destination_ip=destination_ip
        )
        self.graph.create(rel)


    def create_or_get_cluster_node(self, cluster_id: int) -> Node:
        cluster_node = self.graph.nodes.match("Cluster", cluster_id=cluster_id).first()
        if cluster_node:
            return cluster_node
        else:
            cluster_node = Node("Cluster", cluster_id=cluster_id, created_at=str(datetime.now()))
            self.graph.create(cluster_node)
            return cluster_node

    def link_user_to_cluster(self, user_id: str, cluster_id: int):
        """
        Example relationship: (User {user_id}) -[:MEMBER_OF]-> (Cluster {cluster_id})
        """
        user_node = self.graph.nodes.match("User", user_id=user_id).first()
        if not user_node:
            user_node = Node("User", user_id=user_id, created_at=str(datetime.now()))
            self.graph.create(user_node)

        cluster_node = self.create_or_get_cluster_node(cluster_id)
        rel = Relationship(user_node, "MEMBER_OF", cluster_node)
        self.graph.create(rel)

    def correlate_anomalies(self, anomaly_node: Node):
        """
        Simple placeholder method that can find anomalies with similar characteristics.
        Example query: find anomalies with same label or close timestamps.
        """
        # Example: find anomalies from the past 24 hours with the same label
        label = anomaly_node.get("label", "Unknown")
        query = f"""
        MATCH (a:Anomaly)
        WHERE a.label = '{label}'
        AND datetime(a.timestamp) > datetime() - duration('P1D')
        RETURN a
        """
        results = self.graph.run(query)
        similar = [record["a"] for record in results]
        return similar

    def upsert_threat_indicator(self, threat_info: Dict[str, Any]) -> Node:
            """
            Creates or updates a Threat node in the knowledge graph.
            threat_info can contain fields like:
            - 'threat_name'
            - 'ioc_type' (e.g. 'ip', 'domain', 'url', 'hash')
            - 'value' (the actual IP or domain)
            - 'source' (where you got the intel)
            - 'confidence' or 'severity'
            """
            ioc_value = threat_info.get("value")
            if not ioc_value:
                return None

            # Attempt to find an existing Threat node with the same value
            existing_threat = self.graph.nodes.match("Threat", value=ioc_value).first()

            if existing_threat:
                # Update any new fields on the existing threat node
                for key, val in threat_info.items():
                    existing_threat[key] = val
                existing_threat["updated_at"] = str(datetime.now())
                self.graph.push(existing_threat)
                return existing_threat
            else:
                # Create a new Threat node
                threat_node = Node("Threat", **threat_info)
                threat_node["created_at"] = str(datetime.now())
                self.graph.create(threat_node)
                return threat_node
            

    def link_threat_to_host(self, threat_node: Node, host_ip: str):
            """
            Creates a relationship between a Threat node and a Host node, if the host IP matches the threat IoC.
            Relationship type can be something like :THREATENS, :INDICATES, etc.
            """
            host_node = self.graph.nodes.match("Host", ip=host_ip).first()
            if host_node:
                rel = Relationship(threat_node, "INDICATES", host_node)
                self.graph.merge(rel, "Threat", "value")
            else:
                # Optionally create a new Host node if it doesn't exist
                new_host = Node("Host", ip=host_ip, created_at=str(datetime.now()))
                self.graph.create(new_host)
                rel = Relationship(threat_node, "INDICATES", new_host)
                self.graph.create(rel)


    def correlate_threat_with_anomalies(self, threat_node: Node):
            """
            Example correlation: find anomalies with a matching IP in the last 7 days.
            """
            ioc_value = threat_node.get("value", None)
            if not ioc_value:
                return []

            query = f"""
            MATCH (a:Anomaly)
            WHERE a.timestamp >= date() - duration('P7D')  // anomalies from last 7 days
            AND (a."Source IP" = '{ioc_value}' OR a."Destination IP" = '{ioc_value}')
            RETURN a
            """
            results = self.graph.run(query)
            anomalies = [record["a"] for record in results]
            return anomalies
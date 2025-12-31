import logging
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

logger = logging.getLogger(__name__)

class MilvusService:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.text_collection_name = "text_chunks"
        self.image_collection_name = "image_embeddings"
        self._connected = False

    def connect(self):
        if not self._connected:
            try:
                connections.connect("default", host=self.host, port=self.port)
                self._connected = True
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
                self._ensure_collections()
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                raise

    def _ensure_collections(self):
        """Creates collections if they don't exist."""
        # 1. Text Collection
        if not utility.has_collection(self.text_collection_name):
            self._create_text_collection()
        
        # 2. Image Collection
        if not utility.has_collection(self.image_collection_name):
            self._create_image_collection()

    def _create_text_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # Milvus has limits, but 65k is decent
            # Metadata as JSON string if needed, or separate fields. 
            # For simplicity, let's keep it minimal for now.
        ]
        schema = CollectionSchema(fields, "Text chunks from documents")
        collection = Collection(self.text_collection_name, schema)
        # Create index for faster search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection: {self.text_collection_name}")

    def _create_image_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1152),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="caption", dtype=DataType.VARCHAR, max_length=2000),
        ]
        schema = CollectionSchema(fields, "Image embeddings from documents")
        collection = Collection(self.image_collection_name, schema)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection: {self.image_collection_name}")

    def insert_text(self, data: list[dict]):
        """
        data: list of dicts with keys: id, embedding, doc_id, text
        """
        if not data:
            return
        
        collection = Collection(self.text_collection_name)
        # Transform list of dicts to columnar format for Milvus
        # Milvus insert expects: [[id1, id2], [emb1, emb2], ...]
        ids = [d["id"] for d in data]
        embeddings = [d["embedding"] for d in data]
        doc_ids = [d["doc_id"] for d in data]
        texts = [d["text"] for d in data]

        collection.insert([ids, embeddings, doc_ids, texts])
        collection.flush()
        logger.info(f"Inserted {len(data)} text chunks into Milvus.")

    def insert_images(self, data: list[dict]):
        """
        data: list of dicts with keys: id, embedding, doc_id, image_path, caption
        """
        if not data:
            return

        collection = Collection(self.image_collection_name)
        ids = [d["id"] for d in data]
        embeddings = [d["embedding"] for d in data]
        doc_ids = [d["doc_id"] for d in data]
        paths = [d["image_path"] for d in data]
        captions = [d["caption"] for d in data]

        collection.insert([ids, embeddings, doc_ids, paths, captions])
        collection.flush()
        logger.info(f"Inserted {len(data)} image embeddings into Milvus.")

    def search_text(self, query_embeddings: list[list[float]], top_k=5):
        """
        Searches the text collection.
        :param query_embeddings: List of embedding vectors (1024d)
        """
        collection = Collection(self.text_collection_name)
        collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=query_embeddings,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "doc_id"]
        )
        return results

    def search_images(self, query_embeddings: list[list[float]], top_k=5):
        """
        Searches the image collection.
        :param query_embeddings: List of embedding vectors (1152d)
        """
        collection = Collection(self.image_collection_name)
        collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=query_embeddings,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["image_path", "caption", "doc_id"]
        )
        return results

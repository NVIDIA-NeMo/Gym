import re
from typing import List, Optional, Union

class CitationIf:
    def __init__(
        self,
        document_keys: Optional[List[str]] = None,
        citation_pattern: str = r"\[\d+\]",
        threshold: float = 0.5
    ):
        self.document_keys = document_keys or []
        self.citation_pattern = re.compile(citation_pattern)
        self.threshold = threshold

    def __call__(
        self,
        response: Optional[str] = None,
        documents: Optional[List[dict]] = None
    ) -> float:
        if response is None:
            response = ""
        else:
            response = str(response).strip()

        if documents:
            # Extract raw citation references like [1], [1-2]
            found_citations = self.citation_pattern.findall(response)
            
            # Normalize found citations to indices
            cited_indices = [int(c[1:]) for c in found_citations if len(c) > 1]
            
            # Determine how many unique documents we have
            available_keys = set()
            if documents and len(documents) > 0:
                for idx, doc in enumerate(documents):
                    if isinstance(doc, dict):
                        key = doc.get("key", idx)
                        available_keys.add(key)
                    else:
                        available_keys.add(idx)
                
                # Calculate coverage based on available keys
                total_available = len(available_keys)
                if total_available > 0:
                    # Determine which citations matched available keys
                    # If specific keys provided, match against them
                    # If generic, match based on presence
                    matched_count = 0
                    
                    # Strategy: Check if found citations exist within available range or set
                    if document_keys:
                        # Specific key matching
                        for idx in cited_indices:
                            if idx in document_keys:
                                matched_count += 1
                    else:
                        # Fallback to numeric presence (0-based or 1-based)
                        max_key = max(documents[0].get("key", 0), default=0) if documents else 0
                        for idx in cited_indices:
                            if str(idx) in map(str, available_keys):
                                matched_count += 1
                    
                    # Normalize score
                    if total_available:
                        score = min(matched_count / total_available, 1.0) if total_available else 1.0
                    else:
                        score = 1.0 if found_citations else 0.0
                else:
                    score = 1.0 if found_citations else 0.0
            else:
                # No specific documents provided, just check if citations exist or are empty
                score = 1.0 if not found_citations else 0.5 
        else:
            # Empty documents list, default to presence or count check
            cited_indices = [int(c[1:]) for c in found_citations if len(c) > 1]
            # If no specific keys, assume 1 if any found, else 0
            score = 1.0 if found_citations else 0.0

        # Apply threshold scaling
        if score > self.threshold:
            return score
        return score

    def parse_documents(self, docs: Union[List[dict], List[str]]) -> List[str]:
        """Helper to normalize document list for internal use."""
        if not docs:
            return []
        normalized = []
        for d in docs:
            if isinstance(d, str):
                normalized.append(d.strip())
            elif isinstance(d, dict) and "key" in d:
                normalized.append(str(d["key"]))
            else:
                normalized.append(str(d))
        return normalized

    def set_documents(self, docs: Union[List[dict], List[str]]) -> "CitationIf":
        self.document_keys = self.parse_documents(docs)
        return self

    def get_score(self, response: Optional[str] = None) -> float:
        return self.__call__(response=response, documents=self.document_keys if hasattr(self, 'document_keys') else None)
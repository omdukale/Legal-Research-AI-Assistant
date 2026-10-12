from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Citation(BaseModel):
    source_type: Literal["case", "statute", "regulation", "treatise", "article", "other"]
    title: str
    citation: str
    jurisdiction: Optional[str] = None
    link: Optional[str] = None
    citation_summary: str
    SCC_citation: bool = Field(..., description="Whether the citation is from the SCC or not")

class LegalResearchAnswer(BaseModel):
    issue: str = Field(..., description="Clearly defined legal issue or question.")
    short_answer: str = Field(..., description="Concise, plain-language answer to the legal question.")
    rule: str = Field(..., description="Relevant legal rule(s), including statute or precedent.")
    analysis: str = Field(..., description="Detailed explanation applying the rule to the issue.")
    conclusion: str = Field(..., description="Final conclusion, considering all analysis.")
    citations: List[Citation] = Field(default_factory=list, description="Primary and secondary legal sources relied on.")
    judgement: Optional[str] = Field(None, description="Whether the judgement was overruled or not")
    confidence_score: float = Field(..., description="A score from 0.0 to 1.0 indicating confidence in the answer based on the clarity and completeness of the retrieved documents.")



def lra_to_markdown(lra) -> str:
    """Format a LegalResearchAnswer as structured Markdown."""
    if hasattr(lra, "model_dump"):
        data = lra.model_dump()
    elif hasattr(lra, "dict"):
        data = lra.dict()
    elif isinstance(lra, dict):
        data = lra
    else:
        return "Unable to format the legal research response."
    sections = [
        ("Issue", "issue"),
        ("Short Answer", "short_answer"),
        ("Applicable Legal Rule", "rule"),
        ("Analysis", "analysis"),
        ("Conclusion", "conclusion"),
    ]
    md = ["# Legal Research Report"]
    for heading, key in sections:
        content = data.get(key)
        if content:
            md.append(f"## {heading}\n\n{str(content).strip()}")

    citations = data.get("citations") or []
    if citations:
        md.append("## References")
        citation_lines = []
        for citation in citations:
            if hasattr(citation, "model_dump"):
                citation = citation.model_dump()
            elif hasattr(citation, "dict"):
                citation = citation.dict()
            if not isinstance(citation, dict):
                continue

            title = citation.get("title") or "Untitled Source"
            link = citation.get("link") or ""
            citation_text = citation.get("citation") or ""
            jurisdiction = citation.get("jurisdiction") or ""
            metadata = [
                value for value in (
                    citation_text,
                    jurisdiction,
                    "SCC Citation" if citation.get("SCC_citation") else "",
                )
                if value
            ]
            label = f"[{title}]({link})" if link else title
            line = f"- {label}"
            if metadata:
                line += f" — {' | '.join(metadata)}"
            citation_lines.append(line)
            
        if citation_lines:
            md.append("\n".join(citation_lines))

    metadata = []
    confidence = data.get("confidence_score")
    if confidence is not None:
        try:
            score = float(confidence)
            metadata.append(f"**Confidence Score:** {score:.2f}")
        except (TypeError, ValueError):
            pass
    judgement = data.get("judgement")
    if judgement:
        metadata.append(f"**Judgment Status:** {judgement}")
    if metadata:
        md.append("## Assessment\n\n" + "\n\n".join(metadata))

    return "\n\n---\n\n".join(md)

def format_retrieved_context(context) -> str:
    """Convert retrieved context to readable Markdown."""
    if isinstance(context, dict):
        lines = ["**Source documents:**"]
        for key, value in context.items():
            lines.append(f"- **{key}:** {value}")
        return "\n".join(lines)
    
    elif isinstance(context, list):
        lines = ["**Source documents:**"]
        for i, item in enumerate(context, 1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("source") or f"Document {i}"
                content = item.get("content") or item.get("text") or str(item)
                lines.append(f"\n**{i}. {title}**\n> {content}")
            else:
                lines.append(f"\n**{i}.**\n> {str(item)}")
        return "\n".join(lines)
    
    else:
        return str(context)

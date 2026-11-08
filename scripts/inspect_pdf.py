from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_multimodal import extraire_pdf

manifeste = extraire_pdf("pdfs/attention is all you need.pdf")
print(json.dumps({
    "pages": len(manifeste["pages"]),
    "images": [{"page": i["page"], "source": i["source"], "origine": i["origine"]}
               for i in manifeste["images"]],
    "avertissements": manifeste["avertissements"],
}, ensure_ascii=False, indent=2))

import PyPDF2
import spacy
import json

pdf_path = './info.pdf'

with open(pdf_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    number_of_pages = len(pdf_reader.pages)
    full_text = ''
    for page in pdf_reader.pages:
        full_text += page.extract_text()
chunk_size = 1000  # Adjust the size as needed
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

nlp = spacy.load("en_core_web_sm")

embeddings = []
for chunk in chunks:
    doc = nlp(chunk)
    chunk_embedding = doc.vector.tolist()
    embeddings.append(chunk_embedding)

json_data = json.dumps(embeddings)

with open('output.json', 'w') as json_file:
    json_file.write(json_data)


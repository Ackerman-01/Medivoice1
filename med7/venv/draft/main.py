import spacy
from med7 import Med7

# 1. Load spaCy's English model, disabling the default NER
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# 2. Initialize med7 and add it to the pipeline
med7 = Med7()
nlp.add_pipe(med7, name="med7", last=True)

# 3. Sample text (In practice, you'll replace this with OCR-extracted text)
sample_text = "Patient should take Paracetamol 500 mg twice daily and Ibuprofen 200 mg if needed."

# 4. Process the text
doc = nlp(sample_text)

# 5. Print recognized entities
for ent in doc.ents:
    print(ent.text, ent.label_)

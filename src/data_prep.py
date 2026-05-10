import os
from datasets import load_dataset, DatasetDict


def load_and_prepare_dataset(dataset_name: str = "Renicames/turkish-law-chatbot") -> DatasetDict:
    """
    Loads the dataset from Hugging Face Hub and returns it as a DatasetDict
    with 'train' and 'test' splits.

    Args:
        dataset_name: The Hugging Face dataset identifier.

    Returns:
        A DatasetDict containing at least 'train' and 'test' splits.
    """
    dataset = load_dataset(dataset_name)
    return dataset


def apply_prompt_template(dataset: DatasetDict) -> DatasetDict:
    """
    Applies the Turkish legal QA prompt template to every split in the dataset.
    Adds a 'text' column that combines system prompt, question, optional context,
    and reference answer into a single training/inference string.

    Supported field name variants:
        - Question : 'instruction', 'question', 'Soru', 'soru'
        - Context  : 'input', 'context', 'Bagam', 'bagam'
        - Answer   : 'output', 'answer', 'Cevap', 'cevap'

    Args:
        dataset: A DatasetDict whose splits will be formatted.

    Returns:
        A new DatasetDict where every split has an additional 'text' column.
    """
    system_prompt = (
        "Sen bir Türk hukuk asistanısın. "
        "Kullanıcının hukuki sorularını doğru ve eksiksiz bir şekilde yanıtla."
    )

    def _format_row(row):
        question = (
            row.get("instruction")
            or row.get("question")
            or row.get("Soru")
            or row.get("soru")
            or ""
        )
        context = (
            row.get("input")
            or row.get("context")
            or row.get("Bağlam")
            or row.get("Baglam")
            or row.get("bağlam")
            or row.get("baglam")
            or row.get("Bagam")
            or row.get("bagam")
        )
        answer = (
            row.get("output")
            or row.get("answer")
            or row.get("Cevap")
            or row.get("cevap")
            or ""
        )

        if context and context.strip():
            text = (
                f"Sistem: {system_prompt}\n\n"
                f"Soru: {question}\n\n"
                f"Bağlam: {context}\n\n"
                f"Cevap: {answer}"
            )
        else:
            text = (
                f"Sistem: {system_prompt}\n\n"
                f"Soru: {question}\n\n"
                f"Cevap: {answer}"
            )

        return {"text": text}

    return dataset.map(_format_row)


def main():
    print("⏳ Hugging Face'ten veri seti indiriliyor (Renicames/turkish-law-chatbot)...")
    dataset = load_dataset("Renicames/turkish-law-chatbot")
    
    print("\n📊 Veri Seti Bilgileri:")
    print(dataset)
    
    # data klasörünü oluştur (yoksa hata vermesin diye exist_ok=True yapıyoruz)
    os.makedirs("data", exist_ok=True)
    
    # Test setini direkt kaydet (Eğitimden TAMAMEN izole)
    print("\n🔒 Test seti (1.5k) izole ediliyor: data/test_corpus.jsonl")
    dataset['test'].to_json("data/test_corpus.jsonl", force_ascii=False)
    
    # Train setini kaydet
    print("📝 Eğitim seti (13.4k) kaydediliyor: data/train_corpus.jsonl")
    dataset['train'].to_json("data/train_corpus.jsonl", force_ascii=False)
    
    print("\n✅ İşlem Tamam! data/ klasörü oluşturuldu ve dosyalar içine yazıldı.")

if __name__ == "__main__":
    main()

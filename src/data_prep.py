import os
from datasets import load_dataset

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
